# Target Identifiability in Single-Frame JEPA World Models — Formal Statement

This is a reviewer-checkable formalization of the proposition underlying
TI-JEPA's motivation: it is meant to be read near-verbatim as the paper's
Theory section. Everything here is standard
measure-theoretic conditional-independence / data-processing reasoning; the
contribution is not proof difficulty, it is *pointing the argument at the
specific target-construction choice* (`z_{t+1} = E(o_{t+1})`, single frame)
that every next-frame JEPA world model (LeWM and its 2026 follow-ups) makes,
and making the informal empirical Protocol A–C findings a *consequence* of a
clean formal fact rather than a pile of independent measurements.

## 1. Setup

Let $(\Omega,\mathcal F,\mathbb P)$ be the underlying probability space and
fix a (possibly stochastic, possibly action-conditioned) trajectory
$(q_t,\dot q_t)_{t\ge 0}$ on $\mathcal Q\times\mathbb R^{d}$, $\mathcal Q
\subseteq\mathbb R^d$ open, together with actions $(a_t)_{t\ge0}$ on a
measurable action space $\mathcal A$. We make one physical assumption and
one architectural assumption, both satisfied by every next-frame JEPA world
model we are aware of (LeWM, PSG-JEPA, Delta-JEPA, Sub-JEPA, Fast-LeWM):

**(P) Instantaneous rendering.** There is a measurable *rendering map*
$g:\mathcal Q\to\mathcal O$ (a rasterizer / renderer) such that the
observation at time $t$ satisfies $o_t = g(q_t)$ a.s. — i.e. the pixel image
depends on the trajectory only through the *instantaneous configuration*
$q_t$, not through $\dot q_t$ (no motion blur, no stroboscopic multi-exposure
frames, no explicit velocity overlay/HUD).

**(A) Single-frame encoder.** The representation is produced by a
deterministic, Borel-measurable map $E:\mathcal O\to\mathcal Z$ applied to
one frame at a time: $z_t := E(o_t)$. ($E$ may be an arbitrarily expressive
neural network; measurability is the only requirement, so this covers every
practically realizable encoder.)

Write $h := E\circ g:\mathcal Q\to\mathcal Z$, itself Borel-measurable as a
composition of Borel maps.

**Definition (identifiability).** A random variable $v_t$ (here, a candidate
"physical quantity") is *identifiable from* a random variable $x_t$ if there
exists a measurable function $\phi$ with $v_t=\phi(x_t)$ a.s. Equivalently
(and this is the form we use below), $v_t$ is a $\sigma(x_t)$-measurable
random variable.

## 2. Proposition (single-frame targets cannot carry instantaneous rate)

> **Proposition 1.** Under (P)+(A), $z_t = h(q_t)$ a.s.; in particular $z_t$
> is $\sigma(q_t)$-measurable. Consequently, for *any* choice of $E$:
>
> 1. **(Data processing)** every quantity identifiable from $z_t$ is
>    identifiable from $q_t$: $\sigma(z_t)\subseteq\sigma(q_t)$.
> 2. **(Non-identifiability of $\dot q$)** if $\dot q_t$ is *not* a.s. a
>    measurable function of $q_t$ alone — equivalently, the conditional law
>    $\mathcal L(\dot q_t\mid q_t)$ is non-degenerate (puts mass on more than
>    one value) for a set of $q_t$ of positive probability — then $\dot q_t$
>    is not identifiable from $z_t$, for *any* encoder $E$, however
>    expressive.
> 3. **(Conditional independence)** $z_t \perp \dot q_t \mid q_t$.

*Proof.* (1) is immediate: $z_t=E(g(q_t))=h(q_t)$ is by construction a
function of $q_t$, so any $\phi(z_t)=\phi(h(q_t))=:\psi(q_t)$ exhibiting
identifiability from $z_t$ also exhibits identifiability from $q_t$ with
$\psi=\phi\circ h$. (2) is the contrapositive of (1) specialized to
$v_t=\dot q_t$: if $\dot q_t$ were identifiable from $z_t=h(q_t)$ via some
measurable $\phi$, then $\dot q_t=\phi(h(q_t))$ a.s. would exhibit $\dot q_t$
as a measurable function of $q_t$ alone, contradicting the hypothesis. (3)
Since $z_t$ is $\sigma(q_t)$-measurable, conditioning on $q_t$ makes $z_t$
degenerate (a.s. constant), so it trivially carries no further information
about $\dot q_t$ beyond $\sigma(q_t)$ itself: for any bounded measurable
$f,\phi$, $\mathbb E[f(z_t)\phi(\dot q_t)\mid q_t]=f(h(q_t))\,\mathbb
E[\phi(\dot q_t)\mid q_t]=\mathbb E[f(z_t)\mid q_t]\,\mathbb E[\phi(\dot
q_t)\mid q_t]$. $\blacksquare$

**Remark (when the hypothesis of (2) fails).** The only escape is the
degenerate case in which $\dot q_t$ is *itself* a deterministic function of
$q_t$ — e.g. a first-order system disguised as second-order, or a
trajectory confined to a curve in phase space on which $q$ determines
$\dot q$ (a single fixed-energy level set of a 1-DOF conservative system
traversed in one direction only, for instance). This does not hold for any
of our environments or for PushT/Reacher/Cube/TwoRoom: in InertiaBall/
Pendulum/CartPole the initial-condition distribution independently
randomizes $q_0$ and $\dot q_0$ (so the *same* $q$ is visited with many
different $\dot q$ across episodes — this is exactly what Protocol B's
opposite-velocity pairs exploit), and for a pendulum in particular the same
angle is revisited each half-period with *opposite-signed* velocity within a
single trajectory. The hypothesis of Proposition 1(2) is not an idealized
worst case; it is the generic, typically-satisfied condition.

## 3. Corollary (the JEPA predictor cannot use velocity, even though its numeric target does)

> **Corollary 1.** Let $\mathrm{pred}:\mathcal Z\times\mathcal A\to\mathcal
> Z$ range over all measurable functions and let
> $\mathrm{pred}^\star=\arg\min_{\mathrm{pred}}\mathbb E\|\mathrm{pred}(z_t,a_t)-z_{t+1}\|^2$
> be the population-risk minimizer of the JEPA prediction loss (LeWM's
> $L_{\text{pred}}$). Then
> $$\mathrm{pred}^\star(z,a) = \mathbb E\big[z_{t+1}\mid z_t=z,\,a_t=a\big]
> = \mathbb E\big[h(q_{t+1})\mid h(q_t)=z,\,a_t=a\big],$$
> i.e. $\mathrm{pred}^\star$ is the $\dot q_t$-marginalized (averaged over
> the conditional law of $\dot q_t$ given $z_t,a_t$) one-step transition. If
> the true transition kernel $q_{t+1}\mid(q_t,\dot q_t,a_t)$ genuinely
> depends on $\dot q_t$ (true for any non-degenerate second-order/Newtonian
> system — velocity determines where you end up next, that is what "second
> order" means) and the hypothesis of Proposition 1(2) holds, then
> $\mathrm{pred}^\star(z_t,a_t)\ne\mathbb E[z_{t+1}\mid q_t,\dot q_t,a_t]$
> with positive probability: the *L2-optimal* single-frame-target predictor
> is provably not the correct one-step dynamics model.

*Proof.* The first equality is the standard fact that the conditional mean
is the $L^2$-risk minimizer for squared-error regression, restricted to the
$\sigma(z_t,a_t)$-measurable function class (which is exactly the
hypothesis class of a predictor that only sees $(z_t,a_t)$). The second
equality substitutes $z_{t+1}=h(q_{t+1})$ and $z_t=h(q_t)$. For the final
claim: by the tower property, $\mathbb E[h(q_{t+1})\mid z_t,a_t] = \mathbb
E\big[\,\mathbb E[h(q_{t+1})\mid q_t,\dot q_t,a_t]\;\big|\;z_t,a_t\big]$,
i.e. $\mathrm{pred}^\star$ is an *average*, over the conditional law of
$\dot q_t$ given $(z_t,a_t)$, of the true $\dot q_t$-dependent one-step
map. When that map genuinely varies with $\dot q_t$ (non-degenerate
second-order dynamics) and the averaging is over a non-degenerate law
(Prop. 1(2)'s hypothesis), the average differs from each individual branch
on a positive-probability set — this is exactly Jensen's inequality applied
to the (generically) non-constant map $\dot q_t\mapsto\mathbb
E[h(q_{t+1})\mid q_t,\dot q_t,a_t]$. $\blacksquare$

**This is the formal statement behind Protocol B (kill experiment).** Two
states with identical $q_t$ (hence identical $o_t$, hence identical $z_t$)
but opposite $\dot q_t=\pm v$ have, by Corollary 1, the *same*
$\mathrm{pred}^\star(z_t,a_t)$ — the model's best possible one-step
prediction cannot distinguish them, and an open-loop rollout from them must
coincide (or, for a finite-capacity network trained to approximate
$\mathrm{pred}^\star$, "branch separation" measured empirically should be
$\approx 0$, decaying toward the population value as capacity/data grow).
This is precisely the empirical finding, on every environment tested
(InertiaBall/Pendulum/CartPole synthetic + PushT official checkpoint):
$\text{branch-sep}(\text{baseline}_{k=1})\approx 0$, matching the theory's
prediction *exactly*, not just "qualitatively similarly."

## 4. Corollary (a single static goal-image cannot specify a target velocity)

> **Corollary 2.** For any goal-conditioned planning objective of the form
> $J(a_{0:H-1}) = \|z_H - E(o_g)\|_2^2$ with a single goal frame $o_g=g(q_g)$,
> $J$ depends on the planned trajectory's terminal state only through
> $h(q_H)$ (Prop. 1), hence $J$ is constant on the fiber $\{(q,\dot
> q):q=q_H\}$ for any fixed $q_H$ — such an objective cannot express a
> preference over terminal velocity (e.g. "arrive *and stop*" vs. "arrive at
> speed $v$") and is minimized identically by any terminal velocity whatsoever.

*Proof.* Immediate from $z_H=h(q_H)$ depending on $\dot q_H$ only vacuously.
$\blacksquare$

This is the formal reason Protocol C's "stop-at-goal" task (§4.3) is
diagnostic: a planner whose cost is literally $\|z_H-E(o_g)\|^2$ has no
way to prefer "arrive slow" over "arrive fast and coast through," and must
get whatever stopping behavior it exhibits from *predictor* accidents, not
from the stated objective — which is exactly the mechanism by which
`baseline_k1`'s planning failures in Protocol C are explained, not merely
observed.

## 5. Three identifiability-restoring constructions (formal, matching §1's informal list)

Each of the following replaces assumption (A) and is enough to break
Proposition 1's negative conclusion — i.e. each makes $\dot q_t$
identifiable *in principle* (Definition in §1) from the new target. We state
the (trivial but necessary) formal reason for each, then note which property
is *not* implied by identifiability-in-principle (this distinction is what
Protocol A's probe ladder is actually measuring, and is exactly why
CartPole's weak-but-nonzero $v_t$ linear decodability, $r\approx0.30$, is
not a contradiction of the theory — see Remark below).

1. **Multi-frame encoding**, $z_t = E(o_{t-k:t})$ for $k\ge1$. Now $z_t =
   \tilde h(q_{t-k:t})$ for $\tilde h := E\circ(g,\dots,g)$, a function of
   $k+1$ configurations, not one. If the dynamics are (piecewise-)
   differentiable and $\Delta t$ is the frame interval, $\dot q_t = (q_t -
   q_{t-1})/\Delta t + O(\Delta t)$ is a measurable (continuous, in fact) function
   of the window — so the *necessary condition* of Prop. 1(1)
   ($\sigma(z_t)\subseteq\sigma(q_{t-k:t})$, no longer $\sigma(q_t)$ alone)
   no longer forecloses identifiability. This is exactly LeWM's own
   `history_size=3` predictor input — but note (Corollary 1's "not merely
   observed" point in reverse): history reaching the *predictor* does not
   put velocity *in $z$*, since LeWM's target is still $z_{t+1}=E(o_{t+1})$,
   a **single** frame — this is precisely the "history crutch" distinction
   Protocol B's `baseline_k1` vs. `baseline_k3` arms are built to separate,
   and is why this construction alone (multi-frame *encoder input*, without
   also changing what the *target* is) is insufficient on its own without
   pairing it with (2) or (3).
2. **Structured target**, $z_t=(q_t,v_t)$ with $q_t=E_q(o_t)$ single-frame
   and $v_t=E_v(o_{t-k+1:t})$ an explicit function of a finite-difference
   window, *and*, critically, the JEPA prediction target becomes $z_{t+1} =
   (q_{t+1},v_{t+1})$ rather than $z_{t+1}=E(o_{t+1})$ alone — i.e. the
   velocity head's own output enters what is being predicted, not only what
   is being predicted *from*. This is TI-JEPA (§3.2/§5 Method); formally it
   is a special case of construction (1) that additionally forces the
   $\dot q$-carrying part of $z$ to be exposed as its own coordinate (rather
   than left implicit inside an opaque multi-frame $\tilde h$), which is
   what makes it *linearly* probeable by design (Protocol A) rather than
   merely information-theoretically present.
3. **Predict the transition itself**: supervise $\Delta z_t := z_{t+1}-z_t$
   (or a transition/flow representation) rather than $z_{t+1}$ directly.
   Since $\Delta q_t \approx \dot q_t\,\Delta t$, a target built from
   *consecutive* frame differences is, by the same argument as (1), a
   function of a two-frame window and hence not subject to Prop. 1's
   single-frame hypothesis.

This project implements the lightweight combination **(1)+(2)**: frame
stacking into the encoder together with an explicit, structurally-separated
pose/motion split in $z$ (§1's suggested default), not all three.

**Remark (identifiability-in-principle vs. linear decodability — the two
things Protocol A actually separates).** Constructions 1–3 each restore
identifiability *in the Definition-of-§1 sense* (some measurable $\phi$
exists), but say nothing about whether a specific *finite* encoder, trained
with a specific loss and optimizer on finite data, actually represents
$\dot q_t$ in an easily-decodable (e.g. linear) way. This is exactly the gap
Protocol A (linear probe ladder) is designed to expose empirically, and it
is why CartPole's $v_t$ Pearson $r\approx0.30$ (present, well above the
single-frame floor of $\approx0$, but weaker than InertiaBall's $0.57$ or
Pendulum's $0.88$) is not a theoretical anomaly: the *theorem* only
guarantees $\dot q_t$ is no longer measure-theoretically excluded from $z_t$
once assumption (A) is dropped; whether a small bias-free MLP acting on a
4-layer CNN's features finds an easily *linear* encoding of two coupled
physical degrees of freedom (cart translation + pole rotation) from 64×64
pixels is an orthogonal, empirical, capacity/optimization question — which
is exactly what we observe empirically in CartPole's probe numbers, and which the "bigger backbone doesn't fix it much"
finding (see below) further narrows down to backbone spatial
resolution / motion-MLP expressivity rather than embedding dimensionality.

## 6. What this section does *not* claim (scope discipline for Related Work)

- It does **not** claim single-frame JEPA representations are useless for
  physical understanding — only that a specific, narrow, well-defined
  quantity (instantaneous rate of change of configuration) is excluded from
  a specific, narrow, well-defined target construction (next single-frame
  embedding). Configuration variables ($q$: position, angle, joint pose)
  are not covered by Proposition 1 and are exactly what LeWM's own probing
  tables measure well.
- It does **not** claim PSG-JEPA/Delta-JEPA's pair-based constructions are
  wrong — construction (1) above is a formalization of what they do, and
  Proposition 1 is precisely why their approach (grounding $\Delta q$ in a
  *pair* $(z_t,z_{t+k})$) is a valid fix. The distinction we draw (§2 of the
  informal spec) is that grounding a pair is not the same claim as "the
  Markov state $z_t$ itself contains velocity" — a pair-decodable quantity
  need not be single-$z_t$-decodable, which is the distinction Protocol
  A's "single / pair / window / explicit-$v$" ladder is built to expose as
  four *different, ordered* empirical claims rather than one.
- It does **not** require any particular parametric family for $E$ — the
  proof only uses Borel measurability, so it applies unchanged to a ViT-Tiny
  ImageNet-pretrained encoder (the official LeWM architecture) exactly as
  much as to our small CNN. This is the formal reason the official-checkpoint
  Protocol B result on real PushT (branch-sep $\approx0$, sign accuracy
  $\approx$ chance, measured on the *actual released weights*, not a
  re-implementation) is not a mere "our toy model is too small" artifact —
  Proposition 1 says no single-frame-target encoder, of any size, escapes
  this, and the official-checkpoint measurement is the empirical
  confirmation at the one scale we did not train ourselves.
