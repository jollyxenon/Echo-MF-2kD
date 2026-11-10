"""用 Daubechies 谱因子与逐层旋转分解构造任意阶平顶 mixer。"""
from math import comb

from mpmath import mp
import numpy as np
from scipy.special import betainc

from echo.core import spinors


def synthesize(pairs):
    """高精度构造后转为双精度角度；报告分解残差与响应复核误差。"""
    digits = 40+3*pairs
    with mp.workdps(digits):
        # S(y) 的每个根对应一对互为倒数的谱因子根，固定选择单位圆内根。
        coefficients = [mp.mpf(comb(pairs-1+j,j)) for j in range(pairs)]
        roots = mp.polyroots(coefficients[::-1], maxsteps=300, extraprec=20*pairs) if pairs>1 else []
        spectral_roots = []
        for root in roots:
            center = 1-2*root
            delta = mp.sqrt(center*center-1)
            spectral_roots.append(min((center+delta,center-delta),key=abs))
        polynomial = [mp.mpf(1)]
        for root in [-1]*pairs+spectral_roots:
            polynomial = ([-root*polynomial[0]]
                          +[polynomial[j-1]-root*polynomial[j] for j in range(1,len(polynomial))]
                          +[polynomial[-1]])
        scale = mp.sqrt(2)/sum(polynomial)
        polynomial = [value*scale for value in polynomial][::-1]
        imaginary_residual = max(abs(mp.im(value)) for value in polynomial)
        polynomial = [mp.re(value) for value in polynomial]

        # 从最后一个 Echo 对向前消元；每次消去两个系数并提取一个实旋转。
        mixers = [mp.mpf(0)]*pairs
        peel_residual = mp.mpf(0)
        for j in range(pairs,1,-1):
            angle = mp.atan2((-1)**(j+1)*polynomial[-1],polynomial[0])
            angle = (angle+mp.pi/2)%mp.pi-mp.pi/2
            mixers[j-1] = angle
            flipped = [value*(-1)**index for index,value in enumerate(polynomial)][::-1]
            reduced = [-mp.sin(angle)*a-(-1)**(j-1)*mp.cos(angle)*b
                       for a,b in zip(polynomial,flipped)]
            peel_residual = max(peel_residual,abs(reduced[0]),abs(reduced[1]))
            polynomial = reduced[2:]
        norm_residual = abs(sum(value*value for value in polynomial)-1)
        mixers[0] = (mp.atan2(polynomial[1],polynomial[0])+mp.pi/2)%mp.pi-mp.pi/2
        decomposition_residual = float(max(imaginary_residual,peel_residual,norm_residual))
        mixers = np.array([float(value) for value in mixers])

    # 复核实际双精度线路；这是数值诊断，存在性由谱因子与旋转分解证明。
    phases = np.linspace(-np.pi,np.pi,max(257,32*pairs+1))
    minus,plus = spinors(np.array([1]+[2]*(pairs-1)),mixers,phases/4)
    expected = betainc(pairs,pairs,(1+np.sin(phases))/2)
    response_residual = float(np.max(abs(abs(plus)**2-expected)))
    normalization_residual = float(np.max(abs(abs(minus)**2+abs(plus)**2-1)))
    residuals = np.array([decomposition_residual,response_residual,normalization_residual])
    if not np.all(np.isfinite(residuals)) or np.max(residuals)>1e-10:
        raise ArithmeticError(f"平顶角度构造未通过数值复核：k={pairs}，残差={residuals.tolist()}")
    return mixers,dict(construction="Daubechies spectral factorization",working_digits=digits,
                       decomposition_residual=decomposition_residual,
                       response_residual=response_residual,normalization_residual=normalization_residual,
                       response_samples=len(phases))
