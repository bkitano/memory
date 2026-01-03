const highlights = [
  {
    title: 'Fast Weight Programmers',
    description:
      'I finally grokked fast weights after playing with this view. It shows how linear transformers stash updates.',
  },
  {
    title: 'Modern Hopfield Networks',
    description:
      'This one helped me see the bridge between associative memory and attention without the math wall.',
  },
  {
    title: 'State Space Models',
    description:
      'If SSMs feel abstract, this makes the dynamics feel tangible (at least it did for me).',
  },
]

const learningGoals = [
  'Get a feel for the ideas before diving into papers.',
  'Compare memory tricks across fast weights, Hopfield nets, and SSMs.',
  'Make the diagrams match the intuition in your head.',
]

const demoCards = [
  {
    title: 'Linear Attention',
    description: 'Walk through kernelized attention and see the pieces click.',
    tag: 'Hands-on',
  },
  {
    title: 'TTT',
    description: 'Token-to-token training made less mysterious.',
    tag: 'Deep dive',
  },
  {
    title: 'Fast Weight SSM',
    description: 'Play with recurrence and see when memory sticks.',
    tag: 'Try it',
  },
]

export default function LandingPage() {
  return (
    <div className="bg-gradient-to-b from-slate-50 via-white to-slate-100">
      <section className="max-w-6xl mx-auto px-6 py-16">
        <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] items-center">
          <div>
            <p className="text-sm font-semibold text-blue-600 tracking-wide uppercase">Memory Lab</p>
            <h1 className="mt-4 text-4xl sm:text-5xl font-semibold text-slate-900 leading-tight">
              I built this to make memory models feel less abstract.
            </h1>
            <p className="mt-4 text-lg text-slate-600 leading-relaxed">
              These are the demos I wish I had when I was reading the papers. If any of them help
              you, great — poke around, tweak the sliders, and keep whatever clicks.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <div className="rounded-full bg-blue-600 px-5 py-2 text-sm font-semibold text-white shadow">
                Take a look around
              </div>
              <div className="rounded-full border border-blue-200 bg-blue-50 px-5 py-2 text-sm font-semibold text-blue-700">
                Start with whichever topic you’re curious about
              </div>
            </div>
          </div>
          <div className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">What helped me most</h2>
            <ul className="mt-4 space-y-3 text-sm text-slate-600">
              {learningGoals.map((goal) => (
                <li key={goal} className="flex items-start gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-500" />
                  <span>{goal}</span>
                </li>
              ))}
            </ul>
            <div className="mt-6 rounded-2xl bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">For</p>
              <p className="mt-2 text-sm text-slate-700">
                Anyone who wants a quick intuition check before getting lost in equations.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-16">
        <div className="grid gap-6 md:grid-cols-3">
          {highlights.map((highlight) => (
            <div
              key={highlight.title}
              className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
            >
              <h3 className="text-lg font-semibold text-slate-900">{highlight.title}</h3>
              <p className="mt-3 text-sm text-slate-600">{highlight.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-20">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900">Demos I keep coming back to</h2>
            <p className="mt-2 text-sm text-slate-600">
              Pick a topic from the navigation bar and take it for a spin.
            </p>
          </div>
          <div className="hidden md:flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <span className="h-2 w-2 rounded-full bg-blue-500" />
            Go explore
          </div>
        </div>
        <div className="mt-6 grid gap-6 md:grid-cols-3">
          {demoCards.map((card) => (
            <div
              key={card.title}
              className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
            >
              <div className="text-xs font-semibold uppercase tracking-wide text-blue-600">
                {card.tag}
              </div>
              <h3 className="mt-2 text-lg font-semibold text-slate-900">{card.title}</h3>
              <p className="mt-3 text-sm text-slate-600">{card.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
