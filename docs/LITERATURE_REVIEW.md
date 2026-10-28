# Monotone and contractive constraints, multi-start voting and initial-state sensitivity in recursive reasoning models: literature review

*Working review, 27 Oct 2026. It synthesises 26 verified papers in detail (scope and limits in §6) and lists 46
further verified papers, not synthesised, in Appendix A. It may miss relevant work, several sources are only days or
weeks old, and some figures came through summarised reads: check the primary source before citing anything from it.*

**Scope.** Prior work related to the regularizers benchmarked in this repository: monotone (one-sided Lipschitz) and
contractive constraints on the latent update of recursive and looped reasoners (TRM, HRM, looped transformers, deep
equilibrium models), majority voting and confidence over several trajectories, sensitivity of the answer to the
initial latent state, and uses of the fixed-point residual for stopping, selection and error detection. "Ours", "our
condition" and "our numbers" refer to the benchmark in this repository: V1 is the two-sided Lipschitz cap on TRM's
cycle map, V2 the monotone (one-sided Lipschitz) penalty on Euler-TRM's vector field, and the decoder-fibre variant is
the Dγ penalty (definitions in the [README](../README.md), *The idea*; results in
[benchmark.md](../results/sudoku-contractive/report/benchmark.md)).

**Method.** A multi-agent search. Search agents queried the literature by topic (contractive constraints, monotone
constraints, voting and confidence, initial-state sensitivity, TRM variants, other) in a first round and a gap round; candidates above a
relevance cutoff, up to a cap, went to verification. Each paper was then fetched and checked against its full text
(three workshop papers only against their abstracts: [Landscape], [UPI] and xRRMs), and a structured record was kept
of what it imposes, assumes or measures, how it votes or selects, how it handles the initial state, how it uses the
fixed-point residual, its main results and its overlap with this work. This produced 72 verified records. Sections 1–5
synthesise the first 26, the 26th only in part (§6); the other 46 are listed in Appendix A without synthesis.

**Notation.** $s=(z_H,z_L)$ is the joint latent state, $F$ the TRM cycle map, $V=F-I$, $J_F$ and $J_V$ their Jacobians, $\operatorname{sym}A=(A+A^\top)/2$, and $\mu_2(A)=\lambda_{\max}(\operatorname{sym}A)$ the Euclidean log-norm.

**Tags used throughout:**
- ***[imposes]***: the condition is enforced by a loss or by the architecture.
- ***[assumes]***: the condition is a hypothesis of a theorem.
- ***[measures]***: the quantity is estimated post hoc.
- ***[mentions]***: the condition is only cited or named.

---

## 1. Bottom line

- **Has anyone proposed a monotone (one-sided Lipschitz) constraint for TRM or similar models? No, not in the verified set.**
  - No paper imposes $\langle V(s)-V(s'),s-s'\rangle\le-\mu\|s-s'\|^2$, equivalently $\mu_2(J_V)\le-\mu$, on TRM, HRM or any looped reasoner. This holds for losses, architectural constraints and theorem hypotheses alike.
  - "Monotone operators" appear only as related-work pointers to monotone-operator DEQs, in [EqR] and [PI] [mentions].
  - The closest *imposed* conditions are weaker or of a different kind:
    - a Jacobian-trace penalty on the TRM drift $F-I$ [CMM];
    - a power-iteration spectral-radius penalty on looped LMs [STARS];
    - an architectural two-sided Lipschitz bound [DT-L];
    - a tunable contraction on TRM [UPI] (verified from the abstract only);
    - a terminal fixed-point displacement penalty [CoRe];
    - a trajectory anti-oscillation hinge [TAPS].

- **V2 monotone penalty on Euler-TRM: novel within the searched set, but the framing is not.**
  - The closest work is [CMM]. It reads TRM as forward Euler ($dt=1$) on $\dot z=F(z)-z$ and penalises $\tfrac1K\operatorname{tr}J_F>1$, i.e. $\operatorname{tr}J_V>0$, pointwise at segment-end states.
  - Our condition implies theirs, since $\operatorname{tr}J_V/K\le\mu_2(J_V)\le-\mu$. The converse does not hold.
  - [FPRM] and [SFPRM] [assume] that $\operatorname{Re}\lambda(J_F)<1$ at the fixed point in order to prove local convergence of their damped iteration. V2 enforces a global sufficient condition for exactly that hypothesis.
  - Damped or relaxed updates are themselves prior art:
    - \[EqR]: $z+0.95(f(z)-z)$;
    - [FPRM]: adaptive $\eta$;
    - [TAPS]: online $\eta_k$;
    - [DT-L]: convex-interpolation residual;
    - \[SFPRM]: Slerp.

- **Invariance to the initial state via monotonicity: the goal is prior art, the mechanism is not.**
  - [PI] defines path independence and obtains it by training with random initialisations.
  - [EqR] obtains it with randomized initial latents alone (RI): its gap $\Delta_{PI}$ drops from 3.58 to 0.10 on Sudoku-Lite.
  - [SFPRM] obtains 72% all-8-start agreement architecturally, and shows that uniqueness trades off against accuracy.
  - No paper obtains invariance from a constraint on the vector field.
  - **The essential baseline is TRM trained with random $z_0$**, ideally [EqR]'s RI+NI (random init plus a damped, noisy update).

- **Population majority voting is not novel. Vote share as a confidence score was not found, but there are close neighbours.**
  - Majority or mode voting over trajectories of TRM-family models appears in [EqR], [GRAM], [PTRM], [ERM], [Guess] and [SMC]. In [GRAM], random-init TRM goes from 79.9% to 91.2% at N=20.
  - Nobody uses the decoded-answer vote share as a confidence score. The nearest equivalents:
    - [SMC]: token-marginal entropy over a particle cloud (AUROC 0.834; abstention reaches 100% retained accuracy);
    - [PI]: per-instance fixed-point alignment;
    - [ICS]: an acceptance rule that requires at least two outputs to agree.
  - The LLM self-consistency literature was not searched. Expect reviewers to treat agreement-rate confidence as standard.

- **Min-residual selection: anticipated.**
  - [EqR]'s Top-1-Converged selector takes, among B random restarts of a TRM-style model, the restart with the smallest mean fixed-point residual over the last 3 iterations.
  - It beats majority vote after landscape shaping, and can lose to it on baseline TRM.
  - What we can still claim: evidence that a *monotone penalty* is what makes the residual informative. Parallels in the literature:
    - [EqR]: the residual is only reliable after RI+NI;
    - [FRM] v3: residual AUROC goes from 0.50 to 1.00 after its training fix, Fixed-Point Forcing.

- **Residual-based stopping time: not novel.** The following all halt on a residual:
  - [FPRM]: relative $L_\infty$ residual below 0.1;
  - [SFPRM]: worst-token chordal residual below 0.01;
  - [AttrM]: relative residual;
  - [TAPS]: relative residual energy plus prediction stability.

- **Decoder-fibre semi-contraction: no regularizer found, but there are conceptual neighbours.** All of the following are analyses or diagnostics, not training terms:
  - [CoRe] Prop. 1a: the decoded answer freezes once the remaining path length is below the decoder margin.
  - [EqR] App. A.1: a decoding bound $\|R\|<(1-L)\gamma/G_{gap}$.
  - [Finish]: measured anisotropic stability, $\gamma_{nat}<1<\sigma_{\max}$.

- **V1's accuracy loss is consistent with prior evidence:**
  - architectural stabilisation alone drops TRM to 19.8–39.5% on Sudoku-Extreme ([FPRM] Table 3);
  - a strong fixed-point penalty kills extrapolation [CoRe];
  - a unique fixed point stops solving puzzles once $\lambda\ge0.7$ [SFPRM];
  - trained TRMs keep $\sigma_{\max}\approx27$ [Finish].
  - Counterexample: [DT-L] works on prefix sums.

---

## 2. Themes

### 2.1 Monotonicity (one-sided Lipschitz)

**Inventory**
- **[imposes]:** none.
- **[assumes]:** [FPRM] Thm 3 and [SFPRM] Thm 1.
  - Both require every eigenvalue of $J_F$ at the fixed point to satisfy $\operatorname{Re}\lambda<1$.
  - Under that hypothesis, $z+\eta(F(z)-z)$ converges locally for $\eta<\min_i 2(1-\operatorname{Re}\lambda_i)/|\lambda_i-1|^2$.
  - [SFPRM] adds a remark: damping only tames oscillatory or spiralling modes, and expansive modes must be removed by training. It gives no mechanism for doing so.
- **[mentions]:**
  - [EqR]: monotone or well-posed operators, cited for conditioning implicit gradients in DEQ layers.
  - [PI]: monotone-operator equilibrium networks, cited as one route to global convergence.
- **Unrelated uses of "monotone":** accuracy rising with depth ([PI], [DiffCurr]); descent of the correct answer's logit-lens rank ([CoRe] R3); a toy example ([TAPS]); a guide transform ([SMC]); coverage growing with the number of samples ([GRAM]).

**How the literature's conditions relate to V2**
1. **Two-sided cap implies V2.** $\|J_F\|_2\le\rho<1$ (V1, [DT-L], [FPRM] Thm 2, [AttrM] Assumption 1, [UPI]) gives $\mu_2(J_V)\le\rho-1$, i.e. V2 with $\mu=1-\rho$. V2 is therefore a strict relaxation of V1, which fits V2 costing less accuracy.
2. **V2 gives the FPRM/SFPRM hypothesis globally.** V2 implies $\operatorname{Re}\lambda(J_F)\le1-\mu$ everywhere.
3. **V2 implies CMM's condition.** V2 gives $\operatorname{tr}J_V\le-K\mu$. [CMM]'s condition is the averaged, $\mu=0$ version, and it still allows expanding directions.
4. **V2 and $\rho(J_F)<1$ are incomparable.** The latter is [STARS]' penalty target and appears in [Looped-Stab] and [ICS].
   - V2 allows eigenvalues with large imaginary part, so $|\lambda|>1$ is possible.
   - $\rho<1$ allows non-normal $J_F$ with $\mu_2(J_F)>1$.
5. **Euler step.** For $E_h=I+hV$, V2 plus $\operatorname{Lip}(V)\le L$ gives $\|E_h s-E_h s'\|^2\le(1-2h\mu+h^2L^2)\|s-s'\|^2$. This is a contraction when $h<2\mu/L^2$, the discrete analogue of [FPRM]'s $\eta_0$.
   - With a time embedding the field is non-autonomous. The correct conclusion is then incremental stability (trajectories converge to one another), not a unique equilibrium.

**Caveat.** The penalty is a soft hinge evaluated on adversarial or sampled pairs, so it certifies nothing globally. Reaching about 57% unanimity across 16 starts (1 canonical + 15 random; A3: 56.70 ± 6.88% from fully random starts) rather than about 100% suggests two possible causes:
- the condition holds only approximately; or
- $h^2L^2$ is comparable to $2h\mu$.

**Empirical hook.** [Finish] finds completed TRM states with trajectory gain $\gamma_{nat}<1$ but $\sigma_{\max}(J)\approx27$ (Attention-A) and about 6.5 (MLP).
- This is compatible with a monotone $F$ only if the expansion is mainly in the skew part.
- No paper in the synthesised set measures $\mu_2(J_F)$ on trained TRMs. This repository's benchmark reports the median $\lambda_{\max}(\operatorname{sym}J_V)$ of its joint field $V$ for each evaluated arm (256 test puzzles; for example 1.82 ± 0.78 for TRM at 35k steps and 2.43 ± 0.10 for the V2 arm A3), next to $\sigma_{\max}$. The median is positive in all of them, including A3, whereas V2 targets $\le-\mu$. Adding $\gamma_{nat}$ to these measurements, and extending them, would be a clean contribution.

### 2.2 Majority voting and confidence over trajectories

**Where diversity comes from**

| Source | Papers |
|---|---|
| Random initial latents | [EqR], [C-voting], [GRAM] baseline, [ERM], [Fractal] |
| Per-step latent noise | [PTRM], [SMC], [GRAM] (learned noise), [DiffCurr] (single trajectory) |
| Resampled answer state | [GAR], [Landscape] |
| Input transforms | [Guess] (digit relabelling plus checkpoint bootstrap), [ICS] (hypothesis injection, restatements) |
| Output-space noise | [LoopFlow], [FRM] |

**Aggregation rules used**
- **Mode or plurality:** [EqR], [GRAM], [PTRM] mode@K, [ERM], [Guess]; [SMC] uses a weighted plurality.
- **Max mean top-1 softmax:** [C-voting].
- **Halting logit $q$:** [PTRM], [GAR], [LoopFlow], [ERM].
- **Minimum residual:** [EqR].
- **Re-noise stability score:** [FRM] v1.
- **Hand-coded rule energy:** [ERM].

**Results**

When trajectories are diverse, majority vote is often the weakest label-free selector:
- [ERM] (EqR, K=128) on Sudoku-Lite: majority 97.90, $q$ 98.54, confidence 98.58, rule energy 98.83 (= oracle).
- [FRM] v1 on Sudoku-Extreme: plurality top-1 0.69–0.84 against about 1.0 for the stability score, with gold injected into the pool.
- [EqR] Fig. 8: Top1Conv overtakes majority vote after RI+NI.

It is not uniformly worse:
- On Maze-Hard, [PTRM]'s mode@K (86.73) beats best $q$ (85.17).
- [GRAM]'s voting curve is steep: 86.6 at N=1, 96.1 at N=20.

**Confidence.** Vote share is never used as a confidence score. The label-free signals that are used:
- [SMC]: normalised token-marginal entropy over 16 particles. AUROC is 0.834 on Test and 0.710 on the split of puzzles deterministic TRM fails, and abstaining on the top-entropy decile gives 100.0% retained exact-solve.
- [PI]: per-instance alignment of fixed points across about 10 initialisations separates correct from incorrect predictions.
- [CoRe]: a single-trajectory logistic read-out (AUC 0.80 on Sudoku).
- [SFPRM]: the final angle flags wrong answers on Sudoku, but not on Maze.
- \[PTRM]: $q\approx\pm6$.
- [C-voting]: temperature scaling changes ECE but not selection accuracy, so ranking matters, not calibration.

**Gap**
- No paper reports calibration or risk–coverage for decoded-answer vote share across random latent starts on TRM.
- Specific to V2: invariance concentrates the vote, so the informative quantity becomes accuracy *conditional on unanimity*.
- [ICS] shows that vanilla TRM has locally contractive wrong attractors, so unanimity can be confidently wrong.

### 2.3 Sensitivity to random initial latent states

**Two camps**
- **Exploit diversity:** [C-voting], [GAR], [PTRM], [GRAM], [Landscape], [ICS], [SMC], [LoopFlow], [DiffCurr].
- **Remove it:** [PI], [SFPRM], [DT-L] (a Banach claim, never tested), [Looped-Stab] (theory only) and V2.
  - [EqR] belongs to both: its stated motivation is path independence, yet it exploits breadth.

**Measurements**
- **[PI]**
  - Asymptotic Alignment (AA) score: cosine similarity between latent fixed points reached from different starts.
  - An L-BFGS search for adversarial initial states drives AA of non-path-independent models from 0.32 to 0.09 and their accuracy to 0. Path-independent models stay at 1.00.
  - This is the nearest precedent for our adversarial pairs, but it is an evaluation attack, not a training term.
- **[EqR]**
  - $\Delta_{PI}(B)=\mathbb E_x|\mathrm{Acc}_B-\mathrm{Acc}_1|$, an accuracy gap rather than answer agreement.
  - Init ablation, single runs: deterministic 84.06%, best random init 87.30%, learnable $z_0=g(x)$ 83.99%.
- **[SFPRM]**
  - Setup: 8 starts (6 random, $x$, $-x$), the raw undamped map, 400 steps.
  - All 8 starts coincide on 72% of Sudoku puzzles at the trained $\lambda=0.47$ (App. C.3; the uniqueness test of Sec. 4.2 uses 32 puzzles, but the sample size of the 72% figure is not stated), and on every puzzle at $\lambda\ge0.8$.
  - But from $\lambda\ge0.7$ the unique fixed point no longer solves the puzzle.
  - This is the closest published number to our 57% (16 starts, 1,000-puzzle test subset), though the protocols differ.
- **[Fractal]**
  - On hard instances, EqR and FPRM have fractal settling-time basins, caused by transient chaos near saddles at nearly-correct answers.
  - Slices where endpoints disagree are excluded, so this measures sensitivity of the route, not of the final answer.
- **[Guess] (HRM):** a true and a spurious attractor. Which basin the initial latent sits in decides the outcome, and arbitrary latent perturbations destroy coherence.
- **Which state is perturbed matters a lot.**
  - [GAR]: resampling $y$ removes 56.8% of the Sudoku failures that depth alone leaves; resampling $z$ removes 1.8%.
  - [TAPS]: noise on $z_H$ drops best $q$ accuracy from 97.70 to 82.37, while noise on $z_L$ is harmless.
- **[ICS]:** restarting from a fresh $z_0$ recovers far more than restarting from a saved failed state (73.9% vs 0% of compatible near-fixed pairs). Noisy restarts yield at least one valid candidate for 24.5–30.9% of first-pass failures.
- **[Landscape]** (abstract only): trajectories split within a few recursions and settle into minima determined by the initialisation.

**Gap**
- Nobody measures decoded-answer unanimity across random latent starts as a function of a trained constraint.
- Nobody obtains invariance through a condition on the vector field.
- The diversity camp implies a possible cost: V2 may remove recoverable diversity.

### 2.4 Contractive and monotone constraints actually imposed

**[imposes]**
- **[DT-L]** (closest analogue to V1)
  - Spectral normalisation of every recurrent conv, 1-Lipschitz activations, and a convex-interpolation residual $(1-\gamma)I+\gamma\,\mathrm{block}$, giving $K<1$ approximately. The input-recall path is exempt.
  - Training is more reliable and extrapolates better on prefix sums; chess is at parity.
- **[STARS]**
  - Penalty $\|Jv\|^2$, with $v$ from power iteration on $J$ using JVPs; weight 0.1; random log-normal training depth.
  - Models: Ouro-1.4B and 4-digit addition.
  - It targets $\rho(J)$. It argues, without testing, that a full-norm constraint is too restrictive.
- **[CMM]**
  - Trace (first Routh–Hurwitz) penalty, weight $10^4$, plus a repeller term at $x$, an equilibrium loss and a repulsion loss.
  - 93.7% on Sudoku-Extreme, from a single run with confounded ablations.
- **[CoRe]**
  - Terminal displacement penalty with weight 0.05. At weight 1.0 it over-contracts and gives zero extrapolation.
  - Its Jacobian-gain probe is harmless at $\tau=0.5$ and breaks training at $\tau\ge2$.
- **[TAPS]:** $\mu\sum_k[S_k-\kappa P_k]_+$, a penalty on sign-alternating consecutive updates along the realised trajectory. It is not a pairwise or Jacobian condition.
- **[UPI]** (abstract only): contraction strength and a latent projection on TRM in an RL setting, with an $L_z^n$ truncation bias.
- **Architectural stability without any penalty:**
  - [FPRM]: pre-norm and coupled residual scalars;
  - [SFPRM]: sphere constraint plus input injection;
  - [Looped-Stab]: outer normalisation and recall.

**[assumes] or [measures]**
- [FPRM] Thm 2: $\alpha_2\operatorname{Lip}(f)<1$.
- [AttrM]: local contraction; the authors "generally notice" $\|J_F(y^{\ast})\|<1$.
- [EqR] App. A.1.
- [SMC] Prop. 5.1: local Lipschitz bound in a tube, estimated by a secant ratio over 16 random probes.
- [ICS]: spectral radius at wrong TRM near-fixed points, median 0.919 over 239 points.
- [Finish].
- [Looped-Stab] Prop. 3: an autonomous loop with $\rho<1$ forgets its start. Recall keeps input dependence through $(I-J)^{-1}\partial f/\partial x_0$, which explains why a uniform cap limits expressivity.

**Gap**
- No one-sided condition anywhere.
- No adversarial pairs found by power iteration on $\operatorname{sym}J$:
  - [STARS] power-iterates $J$ itself;
  - [SMC] uses random probes;
  - [PI] attacks initial states, and only for evaluation.

### 2.5 The fixed-point residual: stopping, selection, error detection

- **Stopping [imposes]:** [FPRM], [SFPRM], [AttrM], [TAPS].
  - Halting on output stability instead: [DiffCurr], [FRM] v1, and [Fractal]'s settling time.
- **Selection:** only [EqR] Top1Conv. [FRM] v1's perturbation-return score is a close cousin.
- **Error detection [measures]:**
  - [SFPRM]: 99.7–100% of halted Sudoku puzzles are correct, but this fails on Maze.
  - [FRM] v3: AUROC 0.74 (base), 0.50 (self-conditioning), 1.00 (Fixed-Point Forcing).
  - [Finish]: final $Z_L$ update is 0.025 for solved puzzles vs 0.76 for persistent failures, but at step 16 late solvers look like failures.
  - [LoopFlow]: 88.3% of TRM failures are non-convergent and 11.7% are spurious attractors.
  - [PI] App. F.
- **Counter-evidence:**
  - [ICS]: wrong near-fixed points have relative residual around 1e-7.
  - [EqR]: the residual is unreliable on unshaped TRM.
  - [Landscape]: failed runs sit at stable high-loss attractors.

**Gap.** It is open whether a monotone penalty eliminates low-residual wrong attractors, as the theory predicts if the condition held globally. Testing this is the cleanest way to differentiate from [EqR].

### 2.6 Decoder-aware semi-contraction

- **[CoRe] Prop. 1a:** with $R_t=\sum_{u\ge t}\|z_{u+1}-z_u\|$, if $R_t$ is below the radius on which the decoder is constant, the answer is frozen. The weak form of its criterion R4 allows the latent to keep moving in directions the readout ignores.
- **[EqR] App. A.1:** decoding is correct if $\|R\|<(1-L)\gamma(z^{\ast})/G_{gap}$.
- **[Finish]:** contraction holds only along the trajectory direction.

These works point to contraction modulo decoder-invisible directions as the right object, but none trains for it. Our decoder-fibre variant appears novel as a regularizer.

---

## 3. Comparison table

| Paper | Year | Constraint type | Voting / selection | Init handling | Residual use | Overlap with ours |
|---|---|---|---|---|---|---|
| EqR | 2026 | none; damped + noisy update; contraction only in analysis | mean, MajVote, Top1Conv (min residual), B≤128 | random Gaussian z0 in train and test; Δ_PI | selection, diagnostic | **High**: protocol and min-residual selection |
| FPRM | 2026 | [assumes] Reλ<1, Lip<1; architectural | none | unspecified | stopping, step adaptation | damped update, residual stopping |
| C-voting | 2026 | none | best-of-K by top-1 probability | random Gaussian, exploited | none | multi-start selection |
| GAR | 2026 | none | halting-q gate / max q | resamples y | none | multi-start on TRM; y vs z |
| CMM | 2026 | [imposes] trace (RH) + equilibrium loss | none | z_H=x ablation | training loss | **closest constraint** (weaker) |
| Finish | 2026 | [measures] γ_nat, σ_max | none | not varied | diagnostic | explains V1; detection |
| Landscape | 2026 | none | multiple guesses (rule unknown) | noise on initial answer | not stated | same phenomenon |
| SFPRM | 2026 | [assumes] Reλ<1; architectural | none | 8-start uniqueness test (72%) | stopping, detection | invariance test |
| PTRM | 2026 | none | best-Q@K, mode@K | per-step noise | none | voting on TRM |
| Guess | 2026 | none | majority over relabel / bootstrap | fixed; basins | none | attractors, voting |
| ERM | 2026 | none | majority / conf / q / energy | EqR random | none | selector baselines |
| PI | 2022 | none; [mentions] monDEQ | AA score as confidence | mixed init training; adversarial init attack | diagnostic | invariance concept |
| SMC | 2026 | [assumes] local Lipschitz (secant diagnostic) | Q-weighted plurality; entropy confidence | learned h0 + noise | none | vote-share analogue |
| GRAM | 2026 | none | majority, N≤50 | fixed z0; random-init TRM baseline | none | voting baseline |
| LoopFlow | 2026 | none | best-Q of 5 | random x0, fixed z0 | diagnostic | time embedding + Euler |
| CoRe | 2026 | [imposes] terminal residual penalty | single-trajectory confidence | single init | loss, diagnostic | decoder margin |
| FRM | 2026 | none (FPF) | stability score vs plurality (v1) | random noise | detection (v3) | residual informative after training fix |
| STARS | 2026 | [imposes] ρ(J) penalty via power iteration | none | deterministic | none | V1/V2 analogue in LMs |
| ICS | 2026 | [assumes] local contraction | lexicographic; agreement rule | cold restarts | diagnostic | wrong low-residual attractors |
| Fractal | 2026 | none | none | ~40k inits per slice | FPRM halting | motivation |
| TAPS | 2026 | [imposes] anti-oscillation hinge | best-Q (with PTRM) | z_H fragile | adaptive step, exit | relaxed update, exit |
| DT-L | 2024 | [imposes] architectural Lipschitz K<1 | none | deterministic | none | V1 analogue |
| UPI | 2026 | contraction L_z<1 (abstract) | none | none | none | V1 on TRM |
| AttrM | 2026 | [assumes] local contraction | none | warm start | stopping | TRM as fixed point |
| Looped-Stab | 2026 | architectural; theory ρ<1 | none | theory: locally independent | none | theory for V1/V2 |
| DiffCurr | 2026 | none | single trajectory | fresh noise each step | output-stability halting | invariance via other means |

---

## 4. Background theory (brief)

**Standard facts (no specific source needed)**
- For $C^1$ $V$ on a convex domain, $\mu_2(J_V)\le-\mu$ everywhere is equivalent to $-V$ being $\mu$-strongly monotone.
  - If $V$ is also Lipschitz, the Browder–Minty / Zarantonello theory gives a unique zero.
  - The flow $\dot s=V(s)$ then contracts at rate $\mu$ (log-norm bound), so trajectories forget their initial condition.
  - If $V$ depends on $t$ but satisfies the bound uniformly in $t$, trajectories still converge to one another (incremental stability).
- Explicit Euler preserves contraction only for $h<2\mu/L^2$.
- Banach (two-sided contraction) underlies [DT-L], [FPRM] Thm 2, [SFPRM] Thm 2 and [AttrM].

**Covered by verified papers**
- DEQ training and path independence: [PI]; [AttrM] (Anderson acceleration, implicit-function-theorem and phantom gradients); [EqR].
- Local spectral stability of looped transformers: [Looped-Stab].
- Finite-time Lyapunov regimes: [CoRe], [Fractal].
- Bifurcations: [ICS] (saddle-node removal of wrong attractors); [Fractal] (wrong stable fixed points becoming saddles during training).

**Named inside the synthesised papers but not synthesised here (read the primary source before citing)**
- Winston & Kolter 2020, monotone operator equilibrium networks (monDEQ); cited by [PI]. Verified in the wider search (Appendix A, 2006.08591).
- Revay et al. 2020, Lipschitz-bounded equilibrium networks; cited by [PI]. Verified in the wider search (Appendix A, 2010.01732).
- Bai et al. 2019 (DEQ) and 2021 (Jacobian regularisation); cited by [PI], [FPRM], [CoRe], [STARS], [Finish]. The 2021 paper was verified in the wider search (Appendix A, 2106.14342).
- Gu 2020, Geng 2021, Fung 2022 on well-posedness and implicit gradients; cited by [EqR].
- Bansal et al. 2022, Deep Thinking; cited by [DT-L] and [Looped-Stab].
- Wang et al., self-consistency; cited by [STARS], [Fractal], [FRM]. Verified in the wider search (Appendix A, 2203.11171).

---

## 5. Open risks: where a reviewer could say "this was done"

1. **"EqR already does random restarts, majority vote and min-residual selection on TRM."** This is true.
   - Cite [EqR] as the source of the protocol.
   - Claim only three things: the training-time mechanism, answer-level unanimity, and vote-share confidence.
   - Add TRM+RI and RI+NI baselines, report both $\Delta_{PI}$ and unanimity, and place our 16-start numbers on their Top1Conv and MajVote curves.
2. **"monDEQ already imposes monotonicity."** This paper is not synthesised here (it is among the verified papers in Appendix A), and is the most likely objection.
   - Differentiate: monDEQ uses a parametrisation of a single implicit layer to guarantee uniqueness. V2 is a soft one-sided penalty on a deep, weight-tied, two-level TRM field with a time embedding, enforced on adversarial secant pairs, and in this benchmark it costs far less accuracy than V1 (−4.85 ± 4.65 pp against Euler-TRM, versus −53.56 ± 0.85 pp for V1 against TRM at equal steps; exact@16, 100k subset, 3 seeds).
3. **"CMM already puts a Jacobian stability penalty on the TRM ODE."**
   - Show the implication chain in §2.1.
   - Note that their condition is averaged and pointwise, and does not yield invariance to the initial state.
   - Consider running their trace penalty as an ablation.
4. **"STARS already penalises the Jacobian with power iteration."**
   - Their target is $\rho(J)$: two-sided, local and unhinged. Ours is $\mu_2(J_V)$ with a margin, on secant pairs.
   - The two conditions are incomparable (§2.1, point 4).
5. **Residual stopping** belongs to [FPRM], [SFPRM], [AttrM] and [TAPS]. Do not claim it as ours.
6. **"Vote share is just self-consistency."**
   - On the same pool, report AUROC, risk–coverage and ECE for vote share against token entropy [SMC], top-1 confidence [C-voting], halting $q$ ([PTRM], [GAR]) and the residual.
7. **"Invariance destroys useful diversity"** (per [GAR], [ICS], [C-voting], [PTRM]; [SFPRM] shows the uniqueness–accuracy trade-off).
   - Report oracle coverage@16 and accuracy conditional on unanimity, with and without V2.
8. **"Wrong attractors also have low residual"** (per [ICS], [Landscape], and the 11.7% spurious attractors in [LoopFlow]).
   - Report residual distributions for correct and for wrong *selected* trajectories.
9. **Baseline strength.** Our numbers sit well below published ones:

   | Comparison | Ours | Published |
   |---|---|---|
   | Single run | 62–69% (exact@16 on the 100k subset for TRM, Euler-TRM and the V2 arm: 61.93 for A3 to 69.01 for TRM at 45k steps) | TRM-MLP reproductions 83.6–87.3% (87.4% published) |
   | 16 starts | 81–82% from random starts (A3 min-residual 81.00 ± 5.74; TRM at 45k steps, majority vote, 82.37 ± 3.93; 1,000-puzzle subset, 64 ACT steps) | [EqR] 99.8 (B=128), [GAR] 99.43, [PTRM] 98.75, [GRAM] 96.1 (N=20); single-trajectory [LoopFlow] 97.9, [SFPRM] 96.6, [FPRM] 94.2 |

   - Matched-compute point: in [Finish], running stock TRM for 512 instead of 16 steps raises accuracy from 59.2% to 87.5%. Our population protocol runs 16 starts × 64 ACT steps, about 1,024 steps of compute per puzzle (fewer with residual stopping).
   - So the claim must be about the mechanism, supported by same-checkpoint, matched-compute comparisons.
10. **Which state is randomised.** State whether $y$, $z_H$ or $z_L$ is randomised; [GAR] and [TAPS] show this matters a great deal.
11. **The Euler-TRM framing is prior art** ([CMM], [LoopFlow], [EqR], [FPRM], [TAPS], [DT-L]). Present it as the vehicle, not the contribution.
12. **Decoder-fibre variant.** Cite [CoRe] Prop. 1a, [EqR] App. A.1 and [Finish]. Check the control-theory literature on seminorm contraction, which this search did not cover.
13. **Theoretical overclaim.** The penalty holds only on sampled pairs. Report held-out estimates of $\mu_2(J_V)$ (power iteration on $\operatorname{sym}J_V$) and the fraction of pairs that violate the condition. The current benchmark reports the first on test states (positive medians, §2.1) and the violation rate only on training pairs (held near the 10% target).
14. **Recency and versions.**
    - [FRM] changed substantially between versions; cite the specific version.
    - Several sources are only days or weeks old: [GAR] v2 is from 26 Oct 2026, [SFPRM] from 19 Oct, [ICS] from 13 Oct. Recheck them before submission.

---

## 6. What was searched, and its limits

**Topics searched:** contractive constraints, monotone constraints, voting and confidence, initial-state sensitivity,
TRM variants, other.

**Communities covered:**
- the TRM/HRM recursive-reasoning line;
- looped transformers and looped LMs;
- DEQ and implicit models;
- flow- and diffusion-based reasoners;
- dynamical-systems analyses.

Venues are mostly arXiv cs.LG/cs.AI from 2026, plus NeurIPS 2022 and 2024, ICML 2026, and the ICLR 2026 workshops.

**Not covered by this synthesis** (a few papers in these areas were verified and are listed in Appendix A, e.g. 2005.09774, 2303.11273, 2402.13092, 2106.14342, 2203.11171 and 2402.13904, but they are not synthesised):
- control-theory contraction and monotone-systems work, including seminorm contraction;
- primary sources in monotone operator theory;
- neural-ODE stability regularisers;
- LLM self-consistency and calibration;
- classical DEQ stability papers. These are named in §4 but were not verified.

**Recall cap.** 126 round-1 candidates were dropped (below the relevance cutoff or over the cap), and 1 gap-round candidate was dropped over the cap. None of these were reviewed.

**Verification limits:**
- [Landscape] and [UPI] were verified from their abstracts only, because OpenReview returned a bot challenge. Semantic Scholar returned HTTP 429.
- Some figures came through summarised reads and should be checked against the PDFs before quoting: [ICS] figures and [Looped-Stab] tables.
- The synthesis covers entry 26 ([DiffCurr]) only in part, so its overlap assessment is partial. Entries 27–72 of the verified set are not synthesised; they are listed in Appendix A.

---

## References

- [EqR] Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning. B. Huang, Z. Geng, Z. Kolter, 2026. https://arxiv.org/abs/2605.21488
- [FPRM] Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers. S. Movahedi, V. Milovanović, S. L. Feigin, A. Theus, T. Hofmann, V. Boeva, T. K. Rusch, A. Orvieto, 2026. https://arxiv.org/abs/2606.18206
- [C-voting] C-voting: Confidence-Based Test-Time Voting without Explicit Energy Functions. K. Kubo, S. Kamiya, M. Koyama, K. Hayashi, Y. Iwasawa, Y. Matsuo, 2026. https://arxiv.org/abs/2604.13521
- [GAR] Resample the Answer: Where to Spend Test-Time Compute in Recursive Reasoning. M. Illouz, 2026. https://arxiv.org/abs/2610.16427
- [CMM] Dynamical Systems Theory Behind a Hierarchical Reasoning Model. V. A. Es'kin, M. E. Smorkalov, 2026. https://arxiv.org/abs/2603.22871
- [Finish] When Recursive Models Finish Computing. H. Krishna, S. Singh, S. Ebert, H.-Y. Sun, 2026. https://arxiv.org/abs/2609.26487
- [Landscape] Recursive Reasoning as Attractor Landscape Search: Mechanistic Dynamics of the Tiny Recursive Model. A. Efstathiou, A. Balwani, 2026. https://iclr.cc/virtual/2026/10016652
- [SFPRM] Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere. A. Zela, A. Maranjyan, P. Puigdemont, D. Berghaus, V. Boreiko, V. Cevher, 2026. https://arxiv.org/abs/2610.25646
- [PTRM] Probabilistic Tiny Recursive Model. A. Sghaier, A. Parviz, A. Jolicoeur-Martineau, 2026. https://arxiv.org/abs/2605.19943
- [Guess] Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models. Z. Ren, Z. Liu, 2026. https://arxiv.org/abs/2601.10679
- [ERM] Energy-guided Recursive Model. Y. Zhao, Y. Tang, 2026. https://arxiv.org/abs/2607.10128
- [PI] Path Independent Equilibrium Models Can Better Exploit Test-Time Computation. C. Anil, A. Pokle, K. Liang, J. Treutlein, Y. Wu, S. Bai, J. Z. Kolter, R. Grosse, 2022. https://arxiv.org/abs/2211.09961
- [SMC] Boosting Inference with Guided Reasoning: Stochastic Exploration for Recursive Models. A. Corbett, A. Sood, A. Tzatzopoulou, S.-A. Ramesh, T. Dodwell, 2026. https://arxiv.org/abs/2605.25230
- [GRAM] Generative Recursive Reasoning. J. Baek, M. Jo, M. Kim, M. Ren, Y. Bengio, S. Ahn, 2026. https://arxiv.org/abs/2605.19376
- [LoopFlow] Thinking with Looped Flows. A. Suleymanzade, C. Lee, F. Eijkelboom, N. M. Boffi, I. I. Ceylan, J. Kim, 2026. https://arxiv.org/abs/2609.11801
- [CoRe] Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth. I. Viakhirev, K. Borodin, A. Almutairi, S. Barannikov, M. Abramov, G. Mkrtchian, 2026. https://arxiv.org/abs/2608.18222
- [FRM] Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners (v3; v1 had a different title and content). A. Helbling, A. Bryutkin, M. Martino, D. H. Chau, N. Dehmamy, H. Strobelt, 2026. https://arxiv.org/abs/2606.29150
- [STARS] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models. X.-W. Yang, Z. Han, X.-H. Zhang, W.-D. Wei, J.-J. Shao, L.-Z. Guo, Y.-F. Li, 2026. https://arxiv.org/abs/2605.26733
- [ICS] Input-Conditioned Search for Efficient Test-Time Scaling of Recursive Reasoners. H. Shi, T. Fu, Q. Ding, Y. Lu, Y. Li, H. Zhang, Y. Zhao, X. Yuan, H. Lu, B. Liu, 2026. https://arxiv.org/abs/2610.18547
- [Fractal] Fractal basins trap latent reasoning. J. Lai, A. Bao, J. Quinn, W. Gilpin, 2026. https://arxiv.org/abs/2609.04963
- [TAPS] Scheduling Recursive Reasoning in Looped Transformers. B. Wang, C. Yu, J. Ren, H. Wei, B. Jing, Y. Tao, 2026. https://arxiv.org/abs/2609.36653
- [DT-L] Rethinking Deep Thinking: Stable Learning of Algorithms using Lipschitz Constraints. J. Bear, A. Prügel-Bennett, J. Hare, 2024. https://arxiv.org/abs/2410.23451
- [UPI] Unrolled Policy Iteration for Tiny Recursive Models. B. Behzadian, B. Daley, G. R. Subbaraj, H. Nassif, 2026. https://iclr.cc/virtual/2026/10018635
- [AttrM] Solve the Loop: Attractor Models for Language and Reasoning. J. Fein-Ashley, P. Rashidinejad, 2026. https://arxiv.org/abs/2605.12466
- [Looped-Stab] Stability and Generalization in Looped Transformers. A. Labovich, 2026. https://arxiv.org/abs/2604.15259
- [DiffCurr] Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning. M. Drozdova, A. Sirbu, P. Miotti, R. Obryk, M. Etcheverry, E. Niklasson, B. Richards, 2026. https://arxiv.org/abs/2609.01449

---

## Appendix A. Verified papers not synthesised in this version

These 46 papers passed the same verification as the 26 above (each fetched and checked; xRRMs against its abstract only),
but the synthesis did not see them, so nothing in §1–5 accounts for them. Several bear directly on §4 and §5
(monDEQ, LBEN, Jacobian regularisation, non-Euclidean contraction and monotone operator theory, self-consistency), and
the benchmark plan ([CVPR_BENCHMARK_PLAN.md](CVPR_BENCHMARK_PLAN.md), §1.2 and §11) cites some of them.

| # | Paper | Year | Link |
|---|---|---|---|
| 27 | Artificial Kuramoto Oscillatory Neurons | 2024 | [arXiv:2410.13821](https://arxiv.org/abs/2410.13821) |
| 28 | Reasoning on the Simplex: Geometric Fixed-Point Models | 2026 | [arXiv:2609.33540](https://arxiv.org/abs/2609.33540) |
| 29 | Looped Reasoning Finishes Earlier Than You Think | 2026 | [arXiv:2610.18174](https://arxiv.org/abs/2610.18174) |
| 30 | Monotone operator equilibrium networks | 2020 | [arXiv:2006.08591](https://arxiv.org/abs/2006.08591) |
| 31 | Stabilizing Equilibrium Models by Jacobian Regularization | 2021 | [arXiv:2106.14342](https://arxiv.org/abs/2106.14342) |
| 32 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach | 2025 | [arXiv:2502.05171](https://arxiv.org/abs/2502.05171) |
| 33 | Invariant-Measure Reasoners: Stable Representations for Latent Reasoning | 2026 | [arXiv:2610.10996](https://arxiv.org/abs/2610.10996) |
| 34 | Unrolled Flow Models for Reasoning | 2026 | [arXiv:2610.09759](https://arxiv.org/abs/2610.09759) |
| 35 | A Tilted Bowl Is Not a Slippery Slope: Compressing Looped Models | 2026 | [arXiv:2609.39277](https://arxiv.org/abs/2609.39277) |
| 36 | Speed is Confidence | 2026 | [arXiv:2601.19085](https://arxiv.org/abs/2601.19085) |
| 37 | What Limits Recursive Reasoning Models: Optimization, Architecture and Test-Time Scaling | 2026 | [arXiv:2609.39967](https://arxiv.org/abs/2609.39967) |
| 38 | Robust Implicit Networks via Non-Euclidean Contractions | 2021 | [arXiv:2106.03194](https://arxiv.org/abs/2106.03194) |
| 39 | LayerNorm as Implicit Gain Control in Looped Transformers | 2026 | [arXiv:2607.10681](https://arxiv.org/abs/2607.10681) |
| 40 | Less is More: Recursive Reasoning with Tiny Networks | 2025 | [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) |
| 41 | SCORE: Replacing Layer Stacking with Contractive Recurrent Depth | 2026 | [arXiv:2603.10544](https://arxiv.org/abs/2603.10544) |
| 42 | Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling | 2026 | [arXiv:2609.37351](https://arxiv.org/abs/2609.37351) |
| 43 | Non-Euclidean Monotone Operator Theory and Applications | 2023 | [arXiv:2303.11273](https://arxiv.org/abs/2303.11273) |
| 44 | Reasoning with Neural Cellular Automata | 2026 | [arXiv:2609.36126](https://arxiv.org/abs/2609.36126) |
| 45 | Learning Maximally Monotone Operators for Image Recovery | 2020 | [arXiv:2012.13247](https://arxiv.org/abs/2012.13247) |
| 46 | Learning truly monotone operators with applications to nonlinear inverse problems | 2024 | [arXiv:2404.00390](https://arxiv.org/abs/2404.00390) |
| 47 | Multi-Attractor GNNs: Set-Valued Expressivity Beyond Unique Equilibria | 2026 | [arXiv:2609.35274](https://arxiv.org/abs/2609.35274) |
| 48 | xRRMs: Explorative Recurrent Reasoning Models | 2026 | [NeurIPS 2026 workshop, abstract only](https://neurips.cc/virtual/2026/156920) |
| 49 | Weak and Semi-Contraction for Network Systems and Diffusively-Coupled Oscillators | 2020 | [arXiv:2005.09774](https://arxiv.org/abs/2005.09774) |
| 50 | Dynamical systems' based neural networks | 2022 | [arXiv:2210.02373](https://arxiv.org/abs/2210.02373) |
| 51 | Calibrating Large Language Models with Sample Consistency | 2024 | [arXiv:2402.13904](https://arxiv.org/abs/2402.13904) |
| 52 | Memory-efficient model-based deep learning with convergence and robustness guarantees | 2022 | [arXiv:2206.04797](https://arxiv.org/abs/2206.04797) |
| 53 | Lipschitz-Bounded Equilibrium Networks | 2020 | [arXiv:2010.01732](https://arxiv.org/abs/2010.01732) |
| 54 | Contractivity of neural ODEs: an eigenvalue optimization problem | 2024 | [arXiv:2402.13092](https://arxiv.org/abs/2402.13092) |
| 55 | Parcae: Scaling Laws For Stable Looped Language Models | 2026 | [arXiv:2604.12946](https://arxiv.org/abs/2604.12946) |
| 56 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 2022 | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) |
| 57 | Hierarchical Reasoning Model | 2025 | [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) |
| 58 | Monotone, Bi-Lipschitz, and Polyak-Lojasiewicz Networks | 2024 | [arXiv:2402.01344](https://arxiv.org/abs/2402.01344) |
| 59 | Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness | 2023 | [arXiv:2104.05942](https://arxiv.org/abs/2104.05942) |
| 60 | Deep Think with Confidence | 2025 | [arXiv:2508.15260](https://arxiv.org/abs/2508.15260) |
| 61 | Designing Stable Neural Networks using Convex Analysis and ODEs | 2024 | [arXiv:2306.17332](https://arxiv.org/abs/2306.17332) |
| 62 | Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks | 2021 | [arXiv:2110.12976](https://arxiv.org/abs/2110.12976) |
| 63 | Robust Classification using Contractive Hamiltonian Neural ODEs | 2022 | [arXiv:2203.11805](https://arxiv.org/abs/2203.11805) |
| 64 | Deep Equilibrium Algorithmic Reasoning | 2024 | [arXiv:2410.15059](https://arxiv.org/abs/2410.15059) |
| 65 | Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute | 2025 | [arXiv:2512.11847](https://arxiv.org/abs/2512.11847) |
| 66 | Non-Euclidean Contraction Analysis of Continuous-Time Neural Networks | 2021 | [arXiv:2110.08298](https://arxiv.org/abs/2110.08298) |
| 67 | Contracting Implicit Recurrent Neural Networks: Stable Models with Improved Trainability | 2019 | [arXiv:1912.10402](https://arxiv.org/abs/1912.10402) |
| 68 | Stable Recurrent Models | 2018 | [arXiv:1805.10369](https://arxiv.org/abs/1805.10369) |
| 69 | On Logical Extrapolation for Mazes with Recurrent and Implicit Networks | 2024 | [arXiv:2410.03020](https://arxiv.org/abs/2410.03020) |
| 70 | Parallel Test-Time Scaling for Latent Reasoning Models | 2025 | [arXiv:2510.07745](https://arxiv.org/abs/2510.07745) |
| 71 | Not All Noises Are Created Equally: Diffusion Noise Selection and Optimization | 2024 | [arXiv:2407.14041](https://arxiv.org/abs/2407.14041) |
| 72 | Confidence Improves Self-Consistency in LLMs | 2025 | [arXiv:2502.06233](https://arxiv.org/abs/2502.06233) |
