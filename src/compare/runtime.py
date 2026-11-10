"""固定上限预算下的计时、模块记录与统一autograd目标。"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from basis import flips_for, spin_table, target_index
from compare.autodiff import probability, probability_gradient
from echo.core import validate_depth
from instances.benchmark import energy_table
from records import dumps


class BudgetExpired(Exception):
    """候选完成时间越过预算，禁止回填成绩。"""


class Objective:
    """从已知系数开始计时，纯Echo可按需避免建立完整状态向量。"""

    def __init__(self,J,h,s_star,depth,stream,budget=None):
        """创建统一零角度起点、模块日志与候选历史。"""
        validate_depth(depth)
        self.started = perf_counter()
        self.J,self.h,self.s_star = J,h,s_star
        self.n,self.depth = len(h),depth
        self.G,self.budget,self.stream = 10*np.pi,budget,stream
        self.target = target_index(s_star)
        self.best_p,self.best_angles = 2.**(-self.n),np.zeros(2*depth)
        # 首条历史是均匀分布的起点，不是真实候选；截断分析必须剔除它。
        self.history = [{"time":0.,"probability":self.best_p,"angles":self.best_angles.tolist(),"synthetic":True}]
        self.modules,self.current_stage = [],None
        self.calls,self.gradient_calls = 0,0
        self.local_starts = {}
        self.ready = False

    def elapsed(self):
        """返回从输入系数开始经过的墙钟时间。"""
        return perf_counter()-self.started

    def event(self,kind,**fields):
        """把可观测记录写入逐行JSON，写入时间属于算法成本。"""
        self.stream.write(dumps({"kind":kind,"elapsed":self.elapsed(),**fields})+"\n")
        self.stream.flush()

    def check(self):
        """在计算边界执行预算截止。"""
        if self.budget is not None and self.elapsed() >= self.budget:
            raise BudgetExpired

    def prepare(self):
        """首次需要完整模拟时构造能量表和翻转索引。"""
        if self.ready:
            return
        self.check()
        spins = spin_table(self.n)
        self.energy = energy_table(spins,self.J,self.h,0.)
        self.flips = flips_for(self.n)
        self.ready = True
        self.check()

    def start(self,optimizer,x):
        """记录一次本地优化的起点，供结果汇总看出每行真正跑了几次。"""
        self.local_starts[optimizer] = self.local_starts.get(optimizer,0)+1
        self.event("local_start",optimizer=optimizer,depth=len(x)//2,initial_angles=x.tolist())

    def finish_stage(self):
        """完成模块并记录其耗时、前后最好概率和真实已评估候选数。"""
        if self.current_stage is None:
            return
        row = self.current_stage
        row.update(end=self.elapsed(),after_probability=self.best_p,
                   seconds=self.elapsed()-row["start"],candidate_count=self.calls-row.pop("start_calls"),
                   best_angles=self.best_angles.tolist())
        self.modules.append(row)
        self.current_stage = None
        self.event("module",**row)

    def stage(self,name,**fields):
        """切换模块，尚未生成可执行候选时最好分数保持原值。"""
        self.finish_stage()
        self.current_stage = dict(name=name,start=self.elapsed(),before_probability=self.best_p,start_calls=self.calls,**fields)

    def record(self,angles,p,stage,**fields):
        """检查参数与概率，按候选真正完成时刻判断是否允许更新最优值。"""
        done = self.elapsed()
        if not np.isfinite(p) or not np.all(np.isfinite(angles)):
            raise ValueError("非有限概率或角度")
        if len(angles)!=2*self.depth or np.any(abs(angles[:self.depth])>self.G) or np.any(abs(angles[self.depth:])>np.pi/2):
            raise ValueError("角度越界")
        self.calls += 1
        eligible = self.budget is None or done <= self.budget
        improved = eligible and p>self.best_p
        if improved:
            self.best_p,self.best_angles = float(p),angles.copy()
            self.history.append(dict(time=done,probability=float(p),angles=angles.tolist(),synthetic=False))
        self.event("evaluation",completed_at=done,eligible=eligible,improved=improved,
                   probability=float(p),angles=angles.tolist(),stage=stage,**fields)
        if not eligible:
            raise BudgetExpired

    def evaluate(self,x,gradient=False):
        """所有优化器共享任意深度autograd线路；较低深度补零嵌入。"""
        self.check()
        self.prepare()
        p = len(x)//2
        if gradient:
            self.gradient_calls += 1
            value,grad = probability_gradient(x,self.energy,self.flips,self.target)
        else:
            value = probability(x,self.energy,self.flips,self.target)
        full = np.r_[x[:p],np.zeros(self.depth-p),x[p:],np.zeros(self.depth-p)]
        self.record(full,float(value),self.current_stage["name"] if self.current_stage else "optimizer",gradient=gradient)
        return (float(value),np.asarray(grad)) if gradient else float(value)

    def result(self):
        """冻结计时和结果，随后执行的复算不得修改这些字段。"""
        self.finish_stage()
        return dict(probability=self.best_p,angles=self.best_angles.tolist(),elapsed_seconds=self.elapsed(),
                    budget_seconds=self.budget,depth=self.depth,history=self.history,modules=self.modules,
                    candidate_count=self.calls,gradient_calls=self.gradient_calls,local_starts=self.local_starts)
