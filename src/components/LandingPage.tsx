import type { MouseEvent } from 'react'
import 'katex/dist/katex.min.css'
import { BlockMath, InlineMath } from 'react-katex'

const toc = [
  { id: 'attention-as-memory', label: 'Attention as Memory Retrieval' },
  { id: 'parallelism-over-state', label: 'Why Parallelism Won' },
  { id: 'smarter-quadratic', label: 'Smarter Quadratic' },
  { id: 'linear-attention', label: 'Linear Attention as State' },
  { id: 'forgetting', label: 'Forgetting Becomes Central' },
  { id: 'fast-weight-programmers', label: 'Fast Weight Programmers' },
  { id: 'delta-rule', label: 'Delta Rule Write-Back' },
  { id: 'family-tree', label: 'A Unifying Family Tree' },
  { id: 'learning-at-test-time', label: 'Learning at Test Time' },
  { id: 'risks', label: 'Why This Matters (and Risks)' },
]

const tags = ['Transformers', 'Memory', 'Linear attention', 'Fast weights', 'Continual learning']

const sectionTitleClass =
  'text-2xl sm:text-3xl font-semibold font-[var(--font-display)] text-[var(--ink)]'
const paragraphClass = 'mt-4 text-base sm:text-lg leading-relaxed text-[var(--ink-muted)]'
const minorHeadingClass =
  'mt-8 text-xl font-semibold font-[var(--font-display)] text-[var(--ink)]'

function EquationBlock({ label, math }: { label?: string; math: string }) {
  return (
    <div className="mt-4 rounded-2xl border border-black/10 bg-white/70 p-4 shadow-sm">
      {label && (
        <div className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
          {label}
        </div>
      )}
      <div className="mt-2 text-[var(--ink)]">
        <BlockMath math={math} />
      </div>
    </div>
  )
}

export default function LandingPage() {
  const tocOffset = 24
  const tocTop = 'calc(var(--tab-header-height, 72px) + 1.5rem)'

  const getHeaderHeight = () => {
    if (typeof window === 'undefined') return 72
    const raw = getComputedStyle(document.documentElement).getPropertyValue('--tab-header-height')
    const parsed = parseFloat(raw)
    return Number.isFinite(parsed) ? parsed : 72
  }

  const handleTocClick = (event: MouseEvent<HTMLAnchorElement>, id: string) => {
    event.preventDefault()
    const target = document.getElementById(id)
    if (!target) return
    const headerHeight = getHeaderHeight()
    const top = target.getBoundingClientRect().top + window.scrollY - headerHeight - tocOffset
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    window.scrollTo({ top, behavior: prefersReducedMotion ? 'auto' : 'smooth' })
  }

  return (
    <div className="relative min-h-screen bg-[var(--paper)] text-[var(--ink)] font-[var(--font-body)]">
      <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
        <div className="absolute -top-24 right-[-120px] h-72 w-72 rounded-full bg-emerald-200/50 blur-3xl" />
        <div className="absolute bottom-[-120px] left-[-80px] h-80 w-80 rounded-full bg-amber-200/50 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.9)_0%,_rgba(247,241,232,0.7)_45%,_rgba(239,229,216,0.6)_100%)]" />
      </div>

      <div className="relative z-0 mx-auto max-w-6xl px-6 py-16 lg:py-20">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_260px]">
          <article className="space-y-12">
            <header className="space-y-6">
              <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.35em] text-[var(--accent-strong)] font-[var(--font-display)]">
                <span className="rounded-full border border-black/10 bg-white/70 px-3 py-1">Memory Lab</span>
                <span className="text-[var(--ink-muted)]">Field Notes</span>
              </div>
              <h1 className="text-4xl sm:text-5xl font-semibold font-[var(--font-display)] leading-tight text-[var(--ink)]">
                Memory, Forgetting, and the Road to Continual Learning
              </h1>
              <p className="text-lg sm:text-xl leading-relaxed text-[var(--ink-muted)]">
                The transformer story can be told as a sequence of answers to one question: how do we
                represent the past so we can retrieve what we need now?
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-[var(--ink-muted)]">
                {tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-black/10 bg-white/70 px-3 py-1"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <div className="rounded-2xl border border-black/10 bg-white/80 p-5 shadow-sm">
                <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                  Editor note
                </p>
                <p className="mt-3 text-sm leading-relaxed text-[var(--ink-muted)]">
                  Below is a math-forward rewrite that keeps the slide mapping but adds inline
                  derivations where the algebra matters. Kernelization or feature-map steps are
                  treated as modeling choices (not theorems), and any hidden assumptions are called
                  out explicitly.
                </p>
              </div>
            </header>

            <div className="space-y-4 text-base sm:text-lg leading-relaxed text-[var(--ink-muted)]">
              <p>
                Vanilla attention stores the past explicitly (all tokens) and retrieves via
                similarity. This is expressive but quadratic. The next wave (sparse, grouped-query,
                latent) reduces constants. The wave after that (linear attention, fast weights,
                delta nets, learning at test time) changes the representation of the past into a
                state that is updated, which reintroduces the classic RNN problem: forgetting.
              </p>
            </div>

            <section id="attention-as-memory" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Attention as Memory Retrieval (vanilla transformer)</h2>
              <p className={paragraphClass}>
                For a single head with sequence length <InlineMath math="n" /> and head dimension{' '}
                <InlineMath math="d" />, self-attention is the familiar matrix recipe.
              </p>
              <EquationBlock
                label="Single-head attention"
                math={String.raw`\begin{aligned}
Q, K, V &\in \mathbb{R}^{n \times d} \\
A &= \frac{1}{\sqrt{d}} Q K^\top \\
P &= \operatorname{softmax}(A) \\
O &= P V
\end{aligned}`}
              />
              <p className={paragraphClass}>
                Per-token view. For token <InlineMath math="i" />, let{' '}
                <InlineMath math="q_i, k_j, v_j \in \mathbb{R}^d" />. Then
              </p>
              <EquationBlock
                label="Per-token"
                math={String.raw`\begin{aligned}
o_i &= \sum_{j=1}^n p_{ij} v_j \\
p_{ij} &= \frac{\exp(\langle q_i, k_j \rangle / \sqrt{d})}{\sum_{\ell=1}^n \exp(\langle q_i, k_\ell \rangle / \sqrt{d})}
\end{aligned}`}
              />
              <p className={paragraphClass}>
                This is literally weighted interpolation over memory contents, where the memory
                slots are the values <InlineMath math="v_j" />.
              </p>
              <div className="mt-6 rounded-2xl border border-black/10 bg-[var(--paper-deep)] p-5">
                <h3 className="text-sm font-semibold uppercase tracking-[0.25em] text-[var(--ink-muted)]">
                  Why quadratic?
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-[var(--ink-muted)]">
                  Computing <InlineMath math="QK^\top" /> materializes an{' '}
                  <InlineMath math="n \times n" /> matrix. Time and memory scale like{' '}
                  <InlineMath math="\mathcal{O}(n^2 d)" /> and{' '}
                  <InlineMath math="\mathcal{O}(n^2)" /> respectively (modulo implementation
                  tricks).
                </p>
              </div>
            </section>

            <section id="parallelism-over-state" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Why Transformers Were Built This Way: parallelism over state</h2>
              <p className={paragraphClass}>
                The goal was to remove sequential state updates so training can be parallel. In an
                RNN-like model you have a serial dependency chain:
              </p>
              <EquationBlock label="RNN state" math={String.raw`h_t = f(h_{t-1}, x_t)`} />
              <p className={paragraphClass}>
                Attention replaces that with content-addressed retrieval over all positions,
                enabling parallel computation across t.
              </p>
            </section>

            <section id="smarter-quadratic" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>"Smarter Quadratic": GQA / sparse / latent attention</h2>
              <p className={paragraphClass}>
                Grouped-query attention reduces KV bandwidth by sharing keys and values across
                multiple query heads. Sparse attention restricts j in the sum{' '}
                <InlineMath math="o_i = \sum_j p_{ij} v_j" /> to a subset. Latent attention
                compresses the set of memory slots. These are important engineering wins, but
                mathematically they preserve the same shape: explicit storage of many past items
                and explicit retrieval.
              </p>
            </section>

            <section id="linear-attention" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Linear Attention: kernelizing softmax (and why it becomes an RNN)</h2>
              <p className={paragraphClass}>
                The core trick is to replace softmax similarity with something that factorizes:
              </p>
              <EquationBlock
                label="Kernelization idea"
                math={String.raw`\exp(\langle q, k \rangle) \approx \phi(q)^\top \phi(k)`}
              />
              <p className={paragraphClass}>
                This is an approximation or design choice, not an identity, unless you pick a
                kernel with an explicit finite-dimensional feature map. Many linear-attention
                variants differ mainly in the choice of <InlineMath math="\phi" />. Write the
                feature map as <InlineMath math="\phi: \mathbb{R}^d \to \mathbb{R}^m" />, with
                feature dimension <InlineMath math="m" />.
              </p>

              <h3 className={minorHeadingClass}>Deriving the running state form</h3>
              <p className={paragraphClass}>
                Start from attention without the scale (absorbed into <InlineMath math="q" /> and{' '}
                <InlineMath math="k" />) and work forward.
              </p>
              <EquationBlock
                label="Start from causal attention"
                math={String.raw`o_i = \frac{\sum_{j=1}^i \exp(\langle q_i, k_j \rangle) v_j}{\sum_{j=1}^i \exp(\langle q_i, k_j \rangle)}`}
              />
              <EquationBlock
                label="Substitute the factorized kernel"
                math={String.raw`o_i \approx \frac{\sum_{j=1}^i (\phi(q_i)^\top \phi(k_j)) v_j}{\sum_{j=1}^i \phi(q_i)^\top \phi(k_j)}`}
              />
              <EquationBlock
                label="Factor out phi(q_i)^T"
                math={String.raw`o_i \approx \frac{\phi(q_i)^\top \left(\sum_{j=1}^i \phi(k_j) v_j^\top\right)}{\phi(q_i)^\top \left(\sum_{j=1}^i \phi(k_j)\right)}`}
              />
              <EquationBlock
                label="Running state and recurrence"
                math={String.raw`\begin{aligned}
S_i &= \sum_{j=1}^i \phi(k_j) v_j^\top \\
z_i &= \sum_{j=1}^i \phi(k_j) \\
o_i &\approx \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i} \\
S_i &= S_{i-1} + \phi(k_i) v_i^\top \\
z_i &= z_{i-1} + \phi(k_i)
\end{aligned}`}
              />
              <p className={paragraphClass}>
                This is the sense in which linear attention is an RNN: the past is compressed into
                constant-size state (<InlineMath math="S_i" />, <InlineMath math="z_i" />) that is
                updated once per token.
              </p>
              <div className="mt-6 rounded-2xl border border-black/10 bg-[var(--paper-deep)] p-5">
                <h3 className="text-sm font-semibold uppercase tracking-[0.25em] text-[var(--ink-muted)]">
                  Complexity: you did not kill quadratic, you moved it
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-[var(--ink-muted)]">
                  Vanilla attention is quadratic in sequence length <InlineMath math="n" />. Linear
                  attention is linear in <InlineMath math="n" />, but each update involves an outer
                  product <InlineMath math="\phi(k_i) v_i^\top" /> costing{' '}
                  <InlineMath math="\mathcal{O}(m d)" />. If <InlineMath math="m" /> is large (or
                  effectively scales with <InlineMath math="d" />), you can end up quadratic in
                  feature dimension instead. The real trade is <InlineMath math="n^2" /> vs{' '}
                  <InlineMath math="m d" /> per step.
                </p>
              </div>
            </section>

            <section id="forgetting" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Why forgetting becomes central once you have state</h2>
              <p className={paragraphClass}>
                If <InlineMath math="S_i" /> only ever grows by addition, it becomes a dump of
                everything ever written. In the recurrence{' '}
                <InlineMath math="S_i = S_{i-1} + \phi(k_i) v_i^\top" />, every token contributes
                permanently. That is write-only memory. For long horizons, the model needs a
                mechanism to remove or overwrite state; otherwise retrieval degrades as unrelated
                content accumulates.
              </p>
              <p className={paragraphClass}>
                This is exactly the historical role of LSTM forget gates: keep state bounded and
                contextually relevant.
              </p>
            </section>

            <section id="fast-weight-programmers" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Linear Transformers as Fast Weight Programmers (FWP)</h2>
              <p className={paragraphClass}>
                With <InlineMath math="S_i = \sum_{j \le i} \phi(k_j) v_j^\top" />, you can
                interpret <InlineMath math="S_i" /> as a data-dependent weight matrix. Given a
                query feature <InlineMath math="\phi(q_i)" />, the model computes{' '}
                <InlineMath math="\phi(q_i)^\top S_i" />, meaning it applies weights constructed
                online from the sequence.
              </p>
              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                    Slow weights
                  </p>
                  <p className="mt-2 text-sm text-[var(--ink-muted)]">
                    Parameterize how <InlineMath math="\phi(\cdot)" /> is computed and how{' '}
                    <InlineMath math="v" /> is produced.
                  </p>
                </div>
                <div className="rounded-2xl border border-black/10 bg-white/70 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                    Fast weights
                  </p>
                  <p className="mt-2 text-sm text-[var(--ink-muted)]">
                    The state <InlineMath math="S" />, written as outer products based on the input
                    stream.
                  </p>
                </div>
              </div>
            </section>

            <section id="delta-rule" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Read-write access: the delta rule derivation</h2>
              <p className={paragraphClass}>
                If state is write-only, it clogs. We want read-write updates. A simple and
                ubiquitous pattern is error-correcting overwrite: read a prediction from the
                current state, compare it to what you want to write, then update state by a
                fraction of the error.
              </p>

              <div className="mt-6 grid gap-4">
                <div className="rounded-2xl border border-black/10 bg-white/70 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                    Step 1
                  </p>
                  <h3 className="mt-2 text-lg font-semibold font-[var(--font-display)] text-[var(--ink)]">
                    Read from state
                  </h3>
                  <EquationBlock
                    label="Readout"
                    math={String.raw`\bar{v}_i = S_{i-1}^\top \phi(k_i)`}
                  />
                  <p className="mt-3 text-sm text-[var(--ink-muted)]">
                    Here <InlineMath math="S \in \mathbb{R}^{m \times d}" /> and{' '}
                    <InlineMath math="\phi(k_i) \in \mathbb{R}^m" />. This matches the intuition
                    of reading the previous state with the current key.
                  </p>
                </div>

                <div className="rounded-2xl border border-black/10 bg-white/70 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                    Step 2
                  </p>
                  <h3 className="mt-2 text-lg font-semibold font-[var(--font-display)] text-[var(--ink)]">
                    Define a target write
                  </h3>
                  <EquationBlock
                    label="Gated interpolation"
                    math={String.raw`\begin{aligned}
v_i^{\text{new}} &= \beta_i v_i + (1 - \beta_i) \bar{v}_i \\
\beta_i &\in [0, 1]
\end{aligned}`}
                  />
                  <p className="mt-3 text-sm text-[var(--ink-muted)]">
                    <InlineMath math="\beta_i = 0" /> means do not update;{' '}
                    <InlineMath math="\beta_i = 1" /> means fully overwrite.
                  </p>
                </div>

                <div className="rounded-2xl border border-black/10 bg-white/70 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                    Step 3
                  </p>
                  <h3 className="mt-2 text-lg font-semibold font-[var(--font-display)] text-[var(--ink)]">
                    Update by outer-product error correction
                  </h3>
                  <EquationBlock
                    label="Delta rule"
                    math={String.raw`\begin{aligned}
S_i &= S_{i-1} + \phi(k_i) (v_i^{\text{new}} - \bar{v}_i)^\top \\
S_i &= S_{i-1} + \beta_i \phi(k_i) (v_i - \bar{v}_i)^\top
\end{aligned}`}
                  />
                  <p className="mt-3 text-sm text-[var(--ink-muted)]">
                    That is write and remove in one expression: add what you want and subtract what
                    the old memory predicted, gated by <InlineMath math="\beta_i" />.
                  </p>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50/70 p-5">
                <h3 className="text-sm font-semibold uppercase tracking-[0.25em] text-amber-800">
                  Hidden assumption
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-amber-900">
                  This update only corrects along the direction <InlineMath math="\phi(k_i)" />.
                  If many keys are correlated, updates interfere. This is the associative memory
                  capacity problem in modern form.
                </p>
              </div>
            </section>

            <section id="family-tree" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>A unifying family tree</h2>
              <ul className="mt-5 space-y-3 text-base sm:text-lg text-[var(--ink-muted)]">
                <li>Quadratic attention: explicit memory + content-based retrieval.</li>
                <li>Linear attention: implicit memory state + content-based retrieval through state.</li>
                <li>Delta nets / gated delta: implicit memory state + learned overwrite and forget.</li>
                <li>LSTMs: implicit state + gates, with different parameterization and inductive bias.</li>
              </ul>
            </section>

            <section id="learning-at-test-time" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Learning at Test Time: state updates as gradient descent</h2>
              <p className={paragraphClass}>
                A more general view is that the state update can be interpreted as one step of
                online optimization. Let <InlineMath math="S_t" /> be the state and{' '}
                <InlineMath math="z_t = f(x_t; S_t)" /> the output. If we define a per-token loss{' '}
                <InlineMath math="\ell(S; x_t)" />, online gradient descent is:
              </p>
              <EquationBlock
                label="Online update"
                math={String.raw`S_t = S_{t-1} - \eta \nabla_S \ell(S_{t-1}; x_t)`}
              />
              <p className={paragraphClass}>
                Several fast-weight and linear-attention-like updates can be derived as exact or
                approximate solutions to such a step under certain choices of loss and
                parameterization.
              </p>
            </section>

            <section id="risks" className="scroll-mt-24">
              <h2 className={sectionTitleClass}>Why this is a big deal (and a risk)</h2>
              <ul className="mt-5 space-y-3 text-base sm:text-lg text-[var(--ink-muted)]">
                <li>It turns long-context inference into continual learning: the model adapts as it reads.</li>
                <li>It introduces classic online learning trade-offs: stability vs plasticity, step size schedules, and sample efficiency (it may take many tokens to regress effectively).</li>
                <li>If the test-time learning signal is misaligned or adversarial, you can poison the state.</li>
                <li>Gating and forgetting become security and robustness primitives, not just efficiency tricks.</li>
              </ul>
            </section>

            <section className="scroll-mt-24">
              <h2 className={sectionTitleClass}>References</h2>
              <div className="mt-5 space-y-3 text-sm text-[var(--ink-muted)]">
                <a
                  href="https://arxiv.org/pdf/2407.04620"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 text-[var(--accent-strong)] hover:text-[var(--accent)] underline underline-offset-4"
                >
                  TTT paper (arXiv:2407.04620)
                </a>
              </div>
              <p className="mt-6 text-sm leading-relaxed text-[var(--ink-muted)]">
                If you want to see these ideas in motion, the demos live in the tabs above.
              </p>
            </section>
          </article>

          <aside className="hidden lg:block">
            <div className="sticky space-y-6 z-0" style={{ top: tocTop }}>
              <div className="rounded-2xl border border-black/10 bg-white/70 p-5 shadow-sm">
                <h3 className="text-sm font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                  On this page
                </h3>
                <ul className="mt-4 space-y-3 text-sm text-[var(--ink-muted)]">
                  {toc.map((item) => (
                    <li key={item.id}>
                      <a
                        href={`#${item.id}`}
                        className="transition-colors hover:text-[var(--accent-strong)]"
                        onClick={(event) => handleTocClick(event, item.id)}
                      >
                        {item.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded-2xl border border-black/10 bg-white/70 p-5 shadow-sm">
                <h3 className="text-sm font-semibold uppercase tracking-[0.3em] text-[var(--ink-muted)]">
                  Notation
                </h3>
                <div className="mt-4 space-y-2 text-xs text-[var(--ink-muted)]">
                  <div>
                    <InlineMath math="n" />: sequence length
                  </div>
                  <div>
                    <InlineMath math="d" />: head dimension
                  </div>
                  <div>
                    <InlineMath math="m" />: feature dimension
                  </div>
                  <div>
                    <InlineMath math="S_i" />: state matrix
                  </div>
                  <div>
                    <InlineMath math="z_i" />: normalizer state
                  </div>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}
