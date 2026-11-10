# Nature Computational Science 首次投稿材料包 (Initial Submission Package)

本文档提供向 ***Nature Computational Science*** 或 ***Nature Communications*** 提交 Article 稿件《Training-free mean-field adjoint manifold initialization for constrained variational quantum optimization》（双语底稿为 `paper/paper2/paper2-ncs-zh.md` 与 `paper/paper2/paper2-ncs-en.md`）的完整配套材料与格式合规核验清单。

---

## 1. 投稿信 (Cover Letter to the Editors of *Nature Computational Science*)

**Dear Chief Editor and Associate Editors of *Nature Computational Science*,**

We are pleased to submit our manuscript entitled **"Training-free mean-field adjoint manifold initialization for constrained variational quantum optimization"** for consideration as an **Article** in *Nature Computational Science*.

Variational quantum algorithms such as the Quantum Approximate Optimization Algorithm (QAOA) are central to solving constrained combinatorial optimization problems on near-term quantum processors. However, when canonical NP-hard problems (such as Maximum Weight Independent Set, MWIS) are mapped to Ising Hamiltonians, hard-constraint penalties induce strong, heterogeneous local external fields that compete fiercely with quadratic spin-spin couplings. Existing parameter initialization methods are caught in a dilemma:
1. **Homogeneous annealing schedules** (such as Trotterized Quantum Annealing, TQA, and layerwise interpolation) apply uniform phase envelopes across all qubits, failing to resolve site-specific local-field phases and requiring dozens of online quantum circuit measurements in Stage 1 for candidate selection that are slow and vulnerable to projective shot noise.
2. **Circuit ansatz modifications** (such as SDP Warm-Start QAOA) require site-customized mixers at every layer, which forfeit the native hardware advantage of neutral Rydberg atom arrays and trapped-ion processors driven by uniform global laser or microwave transverse pulses, while risking spin-freezing dead zones.
3. **One-dimensional single-scalar spin-echo filters** (Yao, 2026) decouple quadratic couplings within the standard QAOA ansatz, but encounter severe **Diophantine frequency crowding** under incommensurate local fields when system size scales beyond $n>2k$, as well as **local effective-field inversions** the instant multi-body couplings are re-activated in Stage 2.

In this work, we resolve this bottleneck from first principles—requiring **zero training data, zero data leakage, and zero online quantum selection measurements**—by bridging **$2k$-dimensional spin-echo operator parity decoupling**, **statistical-physics temperature-continuation cluster orientation**, and **exact reverse-mode $\mathrm{SU}(2)$-adjoint manifold control**:

- **Exact $2k$-D Intra-Circuit Parity Decoupling & Quantum–Classical Product-Energy Identity**: We prove that within the unmodified standard $p$-layer global QAOA circuit ansatz at positive even depth $p=2k$, exact cancellation of quadratic spin-spin couplings holds across the entire **$2k$-dimensional complete spin-echo pair submanifold** $(\boldsymbol{g},\boldsymbol{\theta})\in\mathbb{R}^{2k}$ without small-angle or weak-coupling approximations. This confines the circuit output strictly to a product state computable in $O(kn)$ classical $2\times 2$ matrix multiplications and guarantees that the **multi-body quantum expected energy equals the classical product-state energy identically**: $\langle\psi_p(\boldsymbol{g},\boldsymbol{\theta})|H_C|\psi_p(\boldsymbol{g},\boldsymbol{\theta})\rangle \equiv E_{\mathrm{prod}}(\boldsymbol{g},\boldsymbol{\theta})$.
- **Training-Free Mean-Field Adjoint Product-Manifold Initialization (`Echo-MF-2kD`)**: Exploiting this exact operator identity, we formulate **`Echo-MF-2kD`** (`Echo-Flat-2kD`), which takes only $(J,h)$, locks onto a self-consistent binary spin orientation via deterministic temperature-continuation damped mean-field inference with single-edge cluster flips ($\beta\in\{2.5,6.0,16.0\}$), extracts 1D Daubechies maximally flat convex-interval roots as anchors, and performs staged $O(kn)$ $\mathrm{SU}(2)$-adjoint L-BFGS-B optimization within a non-degenerate mixer bound $|\theta_j|\le 1.40<\pi/2$ (jointly minimizing flip-sensitivity-weighted negative log-likelihood, local effective-field inversion guards, and classical product energy $E_{\mathrm{prod}}$).
- **Doubled Ground-State Probability with Nearly Half the Quantum Search Cost on MWIS**: On held-out MWIS center test instances ($n=10, p\in\{8,10,12\}$, 1,024 instances), training-free `Echo-MF-2kD` requires **zero Stage-1 quantum evaluations** yet achieves a cross-depth mean final ground-state probability of **$46.16\%$** (reaching **$46.70\%$** at $p=10$ and **$60.80\%$** at $p=12$)—decisively outperforming 32-evaluation online **TQA** ($22.42\%$, a paired net gain of **$+23.74$ percentage points**, $2.06\times$ TQA) and 32-evaluation **RI (32)** ($9.73\%$, $+36.43$ pp), while lifting high-probability instance coverage ($P_{\mathrm{gs}}\ge 50\%$) to **$45.7\%$** ($12.7\times$ TQA's $3.6\%$). Remarkably, `Echo-MF-2kD`'s Stage-1 purely classical product initial probability (mean **$41.77\%$**) already reaches $1.86\times$ TQA's final post-optimization probability, while cutting Stage-2 L-BFGS-B gradient quantum evaluations from TQA's mean of $118.3$ down to **$64.8$ (a $45.2\%$ reduction)**.
- **Cross-Scale Dominance Across 5,120 Test Instances ($n=6\text{--}14$) and $100\%$ Stage-1 Shot-Noise Immunity**: Across a 2,750-instance five-family benchmark ($n=6\ldots 16$, up to $65{,}536$ states) and large-sample MWIS scaling tests ($n\in\{6,8,10,12,14\}$, 1,024 unseen test instances per scale, 5,120 total), offline `Echo-Flat-2kD` outperforms equal-candidate TQA across all system sizes by $+7.61\text{--}+41.39$ pp—reaching $96.79\% / 97.44\%$, $85.01\% / 87.18\%$, $64.29\% / 67.41\%$, $41.41\% / 45.51\%$, and $24.84\% / 27.86\%$ (Top-1 / Top-2) across $n=6,8,10,12,14$—while online training-free `Echo-MF-2kD` maintains **$93.75\%\text{--}99.61\%$** oriented spin accuracy and beats 1D Echo and traditional online baselines with roughly half the Stage-2 quantum calls. Under realistic finite-shot projective measurement noise ($N_{\mathrm{shots}}=16\text{--}4096$), `Echo-MF-2kD` eliminates Stage-1 Winner's Curse misselection entirely ($100\%$ hit rate vs. $36.86\%\text{--}78.20\%$ for TQA) and reaches **$46.77\%\text{--}46.98\%$** under 64-step parameter-shift optimization ($+27.87\text{--}+28.25$ pp over 64-step TQA).

Why *Nature Computational Science*? Our manuscript exemplifies the journal's core mission: introducing a rigorous, training-free mathematical and algorithmic framework that fuses quantum coherent control, statistical mechanics, and wavelet signal processing to overcome a fundamental optimization bottleneck in near-term quantum computing, accompanied by fully reproducible open-source software and verified benchmark datasets.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests.

Sincerely,

**Jianxuan Yao**  
ShanghaiTech University, Shanghai, China  
Email: `yaojx2025@shanghaitech.edu.cn`

---

## 2. *Nature Computational Science* 版面与结构合规核验表 (Formatting & Length Audit)

| 版面模块 | *Nature Computational Science* Article 规范要求 | 本稿件实际配置（由 `paper/paper2/paper2-ncs-en.md` / `paper2-ncs-zh.md` 导出） | 合规状态 |
| :--- | :--- | :--- | :---: |
| **Title** | 简洁明确，不超 ~15 词，无冒号或空洞修饰词 | *Training-free mean-field adjoint manifold initialization for constrained variational quantum optimization* (10 词，无冒号) | PASS |
| **Abstract** | 单段，无文献引用，~150–200 词，零复杂公式，含核心定量结果 | 单段 171 词，零引用、零内联复杂公式，涵盖 $2k$D 奇偶消耦、直积能量恒等式、`Echo-MF-2kD` 在 $n=10$（$46.16\%$ vs. TQA $22.42\%$，调用减 $45.2\%$）与 $n=6\text{--}14$（5,120 题）跨规模及抗散粒噪声结果 | PASS |
| **Introduction** | 无二级小标题，严格四段式漏斗推进（约 600–850 词） | 4 段无子标题漏斗结构（边界与通用模型 $\rightarrow$ 三大范式演进与结构性权衡 $\rightarrow$ $2k$D 直积能量恒等式破局洞察 $\rightarrow$ 技术路线、假设前提与核心实证），零底层数值超参污染 | PASS |
| **Results** | 不标章节号，陈述句式短标题，遵循“核心现象 $\rightarrow$ 物理机制归因 $\rightarrow$ 理论预言印证”倒金字塔 | 4 个无编号陈述句式结果子节（`Exact product-energy identity on the 2k-dimensional spin-echo manifold` $\rightarrow$ `Doubling ground-state probability while halving quantum search cost` $\rightarrow$ `Cross-scale scaling laws across 5,120 instances and five Ising families` $\rightarrow$ `Shot-noise immunity and early-stopping efficiency`） | PASS |
| **Main Display Items** | 正文图表总数 **$\le 6$ 个**（Figures + Tables） | **3 张多面板主图（Figs. 1–3）+ 3 张核心汇总主表（Tables 1–3）= 6 个 Display Items**（全部采用首句加粗点明核心结论的 NCS 自包含图注） | PASS |
| **Discussion** | 无二级小标题，逆漏斗综合讨论与显式边界分析 | 3 段无子标题结构，涵盖对称受限可积子流形的跨领域方法论迁移价值、零外场边界（$h=0$）与超完备区间（$n>2k$）激发态长尾的 CVaR 扩展 | PASS |
| **Methods** | 置于 Discussion 之后，不计入正文字数上限，按五维抽象度严格拆分为 4 个独立小节 | 严格拆分为 4 个独立小节：`Problem Formulation and Theoretical Preliminaries`、`Module Design and Algorithmic Procedure`、`Experimental Protocol and Evaluation Setup`、`Implementation Details` | PASS |
| **Extended Data** | 最多 **10 个** Extended Data 图/表 | **Extended Data Tables 1–2**（五家族 $n=11,12$ 机制边界、散粒噪声模式 A/B 详表） | PASS |
| **Source Data & Code** | 每张主图需提供 Source Data CSV 及完整复现代码 | `paper/paper2/figures/*.csv` 与 `runs/canonical/*.json` 包含全部源数据；`src/` 与 `scripts/` 提供完整可执行源码（公开仓库：[https://github.com/jollyxenon/Echo-MF-2kD](https://github.com/jollyxenon/Echo-MF-2kD)） | PASS |
