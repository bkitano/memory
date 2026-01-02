// @ts-nocheck
import React, { useMemo, useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { ArrowRight, RefreshCw, SlidersHorizontal, Calculator, Target, Info, Sparkles, BookOpen } from 'lucide-react';

const D = 2;

const DEFAULT_S = [
  [0.2, -0.1],
  [0.4, 0.3],
];

const DEFAULT_K = [0.6, -0.2];
const DEFAULT_V = [0.8, 0.1];

const RULES = [
  {
    id: 'la',
    label: 'LA',
    name: 'Linear Attention',
    accent: {
      text: 'text-sky-700',
      border: 'border-sky-200',
      bg: 'bg-sky-50',
      pill: 'bg-sky-100 text-sky-700',
      ring: 'ring-sky-400',
    },
    objectiveLines: [
      'J(S_t) = \\lVert S_t - S_{t-1} \\rVert_F^2 - 2\\langle S_t k_t, v_t \\rangle',
      '\\nabla_{S_t} J = 2(S_t - S_{t-1}) - 2 v_t k_t^{\\top}',
    ],
    updateLines: ['S_t = S_{t-1} + v_t k_t^{\\top}'],
    note: 'Stay-close term plus a linear fit term (Hebbian).',
    usesAlpha: false,
    usesBeta: false,
  },
  {
    id: 'mamba2',
    label: 'Mamba2',
    name: 'Scaled State Update',
    accent: {
      text: 'text-emerald-700',
      border: 'border-emerald-200',
      bg: 'bg-emerald-50',
      pill: 'bg-emerald-100 text-emerald-700',
      ring: 'ring-emerald-400',
    },
    objectiveLines: [
      'J(S_t) = \\lVert S_t - \\alpha_t S_{t-1} \\rVert_F^2 - 2\\langle S_t k_t, v_t \\rangle',
      '\\nabla_{S_t} J = 2(S_t - \\alpha_t S_{t-1}) - 2 v_t k_t^{\\top}',
    ],
    updateLines: ['S_t = \\alpha_t S_{t-1} + v_t k_t^{\\top}'],
    note: 'Same as LA, but with a gated carry-over alpha_t.',
    usesAlpha: true,
    usesBeta: false,
  },
  {
    id: 'longhorn',
    label: 'Longhorn',
    name: 'RLS-ish',
    accent: {
      text: 'text-amber-700',
      border: 'border-amber-200',
      bg: 'bg-amber-50',
      pill: 'bg-amber-100 text-amber-700',
      ring: 'ring-amber-400',
    },
    objectiveLines: [
      'J(S_t) = \\lVert S_t - S_{t-1} \\rVert_F^2 + \\beta_t \\lVert S_t k_t - v_t \\rVert_2^2',
      'S_t (I + \\beta_t k_t k_t^{\\top}) = S_{t-1} + \\beta_t v_t k_t^{\\top}',
    ],
    updateLines: [
      'S_t = S_{t-1}(I - \\varepsilon_t k_t k_t^{\\top}) + \\varepsilon_t v_t k_t^{\\top}',
      '\\varepsilon_t = \\beta_t / (1 + \\beta_t k_t^{\\top} k_t)',
    ],
    note: 'Uses + beta for a well-posed reconstruction term.',
    usesAlpha: false,
    usesBeta: true,
  },
  {
    id: 'deltanet',
    label: 'DeltaNet',
    name: 'Error-Driven',
    accent: {
      text: 'text-rose-700',
      border: 'border-rose-200',
      bg: 'bg-rose-50',
      pill: 'bg-rose-100 text-rose-700',
      ring: 'ring-rose-400',
    },
    objectiveLines: [
      'u_t = \\beta_t (v_t - S_{t-1} k_t)',
      'J(S_t) = \\lVert S_t - S_{t-1} \\rVert_F^2 - 2\\langle S_t k_t, u_t \\rangle',
    ],
    updateLines: ['S_t = S_{t-1}(I - \\beta_t k_t k_t^{\\top}) + \\beta_t v_t k_t^{\\top}'],
    note: 'Equivalent to a single gradient step on reconstruction error.',
    usesAlpha: false,
    usesBeta: true,
  },
  {
    id: 'gated-delta',
    label: 'Gated Delta',
    name: 'Gated Error-Driven',
    accent: {
      text: 'text-teal-700',
      border: 'border-teal-200',
      bg: 'bg-teal-50',
      pill: 'bg-teal-100 text-teal-700',
      ring: 'ring-teal-400',
    },
    objectiveLines: [
      'u_t = \\beta_t (v_t - \\alpha_t S_{t-1} k_t)',
      'J(S_t) = \\lVert S_t - \\alpha_t S_{t-1} \\rVert_F^2 - 2\\langle S_t k_t, u_t \\rangle',
    ],
    updateLines: ['S_t = \\alpha_t S_{t-1}(I - \\beta_t k_t k_t^{\\top}) + \\beta_t v_t k_t^{\\top}'],
    note: 'Adds carry-over gate alpha_t to the DeltaNet step.',
    usesAlpha: true,
    usesBeta: true,
  },
];

const randomVal = () => Number((Math.random() * 1.6 - 0.8).toFixed(1));

const zeros = (rows: number, cols: number) =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));

const identity = (n: number) =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

const dot = (a: number[], b: number[]) => a.reduce((sum, val, i) => sum + val * b[i], 0);
const vecSub = (a: number[], b: number[]) => a.map((val, i) => val - b[i]);
const vecScale = (a: number[], s: number) => a.map(val => val * s);
const vecNormSq = (a: number[]) => dot(a, a);

const matAdd = (A: number[][], B: number[][]) => A.map((row, i) => row.map((val, j) => val + B[i][j]));
const matSub = (A: number[][], B: number[][]) => A.map((row, i) => row.map((val, j) => val - B[i][j]));
const matScale = (A: number[][], s: number) => A.map(row => row.map(val => val * s));
const matMul = (A: number[][], B: number[][]) => {
  const result = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < B[0].length; j += 1) {
      for (let k = 0; k < B.length; k += 1) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
};
const matVecMul = (A: number[][], v: number[]) =>
  A.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));

const outer = (a: number[], b: number[]) => a.map(ai => b.map(bj => ai * bj));

const frobNormSq = (A: number[][]) =>
  A.reduce((sum, row) => sum + row.reduce((rowSum, val) => rowSum + val * val, 0), 0);

const formatNum = (n: number) => {
  if (!Number.isFinite(n)) return '0.00';
  const clamped = Math.abs(n) < 0.005 ? 0 : n;
  return clamped.toFixed(2);
};

const MatrixGrid = ({ data, title, editable = false, onChange, className = '' }: any) => (
  <div className={`flex flex-col items-center ${className}`}>
    {title && <div className="text-xs font-semibold text-slate-500 mb-2">{title}</div>}
    <div className="grid gap-px bg-slate-200 p-1 rounded-lg">
      {data.map((row: number[], i: number) => (
        <div key={i} className="flex gap-px">
          {row.map((val: number, j: number) =>
            editable ? (
              <input
                key={`${i}-${j}`}
                type="number"
                step="0.1"
                value={Number.isFinite(val) ? val : 0}
                onChange={(e) => onChange(i, j, parseFloat(e.target.value))}
                className="w-14 h-10 text-xs font-mono text-center bg-white border border-slate-200 rounded focus:ring-2 focus:ring-slate-300 outline-none"
              />
            ) : (
              <div
                key={`${i}-${j}`}
                className="w-14 h-10 flex items-center justify-center text-xs font-mono bg-white text-slate-700"
              >
                {formatNum(val)}
              </div>
            )
          )}
        </div>
      ))}
    </div>
  </div>
);

const VectorGrid = ({ data, title, editable = false, onChange, className = '' }: any) => (
  <div className={`flex flex-col items-center ${className}`}>
    {title && <div className="text-xs font-semibold text-slate-500 mb-2">{title}</div>}
    <div className="flex gap-px bg-slate-200 p-1 rounded-lg">
      {data.map((val: number, i: number) =>
        editable ? (
          <input
            key={i}
            type="number"
            step="0.1"
            value={Number.isFinite(val) ? val : 0}
            onChange={(e) => onChange(i, parseFloat(e.target.value))}
            className="w-14 h-10 text-xs font-mono text-center bg-white border border-slate-200 rounded focus:ring-2 focus:ring-slate-300 outline-none"
          />
        ) : (
          <div
            key={i}
            className="w-14 h-10 flex items-center justify-center text-xs font-mono bg-white text-slate-700"
          >
            {formatNum(val)}
          </div>
        )
      )}
    </div>
  </div>
);

const MetricCard = ({ title, prev, next, lowerIsBetter = true }: any) => {
  const diff = next - prev;
  const improved = lowerIsBetter ? diff <= 0 : diff >= 0;
  const diffLabel = `${diff >= 0 ? '+' : ''}${formatNum(diff)}`;
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-400">{title}</div>
      <div className="mt-2 flex items-center justify-between text-xs font-mono text-slate-600">
        <span>prev {formatNum(prev)}</span>
        <span>next {formatNum(next)}</span>
      </div>
      <div className={`mt-1 text-xs font-semibold ${improved ? 'text-emerald-600' : 'text-rose-600'}`}>
        change {diffLabel}
      </div>
    </div>
  );
};

const ParamSlider = ({ label, value, min, max, step, onChange, disabled = false }: any) => (
  <div className={`flex flex-col gap-1 ${disabled ? 'opacity-50' : ''}`}>
    <div className="flex justify-between text-xs text-slate-500 font-mono">
      <span>{label}</span>
      <span>{formatNum(value)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-2 rounded-lg bg-slate-200 appearance-none cursor-pointer accent-slate-700"
    />
  </div>
);

export default function LinearAttentionUpdatesDemo() {
  const [activeRule, setActiveRule] = useState('la');
  const [Sprev, setSprev] = useState(DEFAULT_S);
  const [k, setK] = useState(DEFAULT_K);
  const [v, setV] = useState(DEFAULT_V);
  const [alpha, setAlpha] = useState(0.9);
  const [beta, setBeta] = useState(0.8);

  const rule = RULES.find(r => r.id === activeRule);

  const computed = useMemo(() => {
    const kkT = outer(k, k);
    const vkT = outer(v, k);
    const kNorm = dot(k, k);
    const eps = beta / (1 + beta * kNorm);
    const I = identity(D);
    const IminusBeta = matSub(I, matScale(kkT, beta));
    const IminusEps = matSub(I, matScale(kkT, eps));

    let nextS = Sprev;
    if (activeRule === 'la') {
      nextS = matAdd(Sprev, vkT);
    } else if (activeRule === 'mamba2') {
      nextS = matAdd(matScale(Sprev, alpha), vkT);
    } else if (activeRule === 'longhorn') {
      nextS = matAdd(matMul(Sprev, IminusEps), matScale(vkT, eps));
    } else if (activeRule === 'deltanet') {
      nextS = matAdd(matMul(Sprev, IminusBeta), matScale(vkT, beta));
    } else if (activeRule === 'gated-delta') {
      nextS = matAdd(matMul(matScale(Sprev, alpha), IminusBeta), matScale(vkT, beta));
    }

    const delta = matSub(nextS, Sprev);

    const objective = (S: number[][]) => {
      if (activeRule === 'la') {
        return frobNormSq(matSub(S, Sprev)) - 2 * dot(matVecMul(S, k), v);
      }
      if (activeRule === 'mamba2') {
        return frobNormSq(matSub(S, matScale(Sprev, alpha))) - 2 * dot(matVecMul(S, k), v);
      }
      if (activeRule === 'longhorn') {
        return frobNormSq(matSub(S, Sprev)) + beta * vecNormSq(vecSub(matVecMul(S, k), v));
      }
      if (activeRule === 'deltanet') {
        const u = vecScale(vecSub(v, matVecMul(Sprev, k)), beta);
        return frobNormSq(matSub(S, Sprev)) - 2 * dot(matVecMul(S, k), u);
      }
      const u = vecScale(vecSub(v, matVecMul(matScale(Sprev, alpha), k)), beta);
      return frobNormSq(matSub(S, matScale(Sprev, alpha))) - 2 * dot(matVecMul(S, k), u);
    };

    const reconPrev = vecNormSq(vecSub(matVecMul(Sprev, k), v));
    const reconNext = vecNormSq(vecSub(matVecMul(nextS, k), v));
    const alignPrev = dot(matVecMul(Sprev, k), v);
    const alignNext = dot(matVecMul(nextS, k), v);

    return {
      nextS,
      delta,
      eps,
      kNorm,
      objectivePrev: objective(Sprev),
      objectiveNext: objective(nextS),
      reconPrev,
      reconNext,
      alignPrev,
      alignNext,
    };
  }, [Sprev, k, v, alpha, beta, activeRule]);

  const randomize = () => {
    setSprev(Array.from({ length: D }, () => Array.from({ length: D }, randomVal)));
    setK(Array.from({ length: D }, randomVal));
    setV(Array.from({ length: D }, randomVal));
  };

  const handleMatrixChange = (row: number, col: number, value: number) => {
    if (!Number.isFinite(value)) return;
    setSprev(prev => prev.map((r, i) => r.map((v2, j) => (i === row && j === col ? value : v2))));
  };

  const handleVectorChange = (index: number, value: number, setter: (v: number[]) => void) => {
    if (!Number.isFinite(value)) return;
    setter(prev => prev.map((v2, i) => (i === index ? value : v2)));
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6 text-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-slate-900"></div>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-2">Online Update Rules</div>
              <h1 className="text-3xl font-bold text-slate-900 mb-2">Linear Attention State Updates</h1>
              <p className="text-slate-600 max-w-3xl">
                Each rule below is the closed-form minimizer of a small quadratic objective. The demo
                shows the objective, the update, and how the reconstruction loss fits into the same view.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-500 font-mono bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
              <Calculator size={16} />
              <InlineMath math={`d_v = ${D},\\; d_k = ${D}`} />
            </div>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-4">
          <div className="flex items-center gap-2 mb-3 text-xs uppercase tracking-wider text-slate-400 font-semibold">
            <Sparkles size={14} />
            Choose a rule
          </div>
          <div className="flex flex-wrap gap-3">
            {RULES.map((item) => {
              const active = item.id === activeRule;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveRule(item.id)}
                  className={`flex-1 min-w-[180px] text-left p-4 rounded-xl border transition-all ${
                    active ? `${item.accent.border} ${item.accent.bg} shadow-sm` : 'border-slate-200 bg-white hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-semibold uppercase tracking-wider ${item.accent.text}`}>
                      {item.label}
                    </span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full ${item.accent.pill}`}>Update</span>
                  </div>
                  <div className="text-sm font-bold text-slate-900 mt-2">{item.name}</div>
                  <div className="text-[11px] text-slate-500 mt-1">
                    <InlineMath math={item.updateLines[0]} />
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.05fr_1.25fr] gap-6">
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-700 mb-4">
                <BookOpen size={16} className={rule.accent.text} />
                Objective and Solve
              </div>
              <div className="space-y-3">
                <div className={`rounded-lg border ${rule.accent.border} ${rule.accent.bg} p-3`}>
                  <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-2">Objective</div>
                  <div className="text-xs text-slate-700 space-y-1">
                    {rule.objectiveLines.map((line: string) => (
                      <BlockMath key={line} math={line} />
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-2">Update</div>
                  <div className="text-xs text-slate-700 space-y-1">
                    {rule.updateLines.map((line: string) => (
                      <BlockMath key={line} math={line} />
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                    <Info size={14} />
                    Notes
                  </div>
                  <div className="text-sm text-slate-600">{rule.note}</div>
                </div>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-700 mb-4">
                <Target size={16} className="text-slate-600" />
                Reconstruction Objective View
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
                <BlockMath math={'L(S) = \\lVert S k_t - v_t \\rVert_2^2'} />
              </div>
              <div className="mt-3 space-y-2 text-sm text-slate-600">
                <div>
                  - Longhorn: closed-form minimizer of <InlineMath math={'\\lVert S - S_{t-1} \\rVert^2 + \\beta L(S)'} />.
                </div>
                <div>- DeltaNet and Gated: one gradient step on L(S) with a stay-close term.</div>
                <div>
                  - LA and Mamba2: linearized fit term (drops the quadratic in <InlineMath math={'S k_t'} />).
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                <SlidersHorizontal size={16} className="text-slate-600" />
                State Playground
              </div>
              <button
                onClick={randomize}
                className="flex items-center gap-2 text-xs font-semibold text-slate-600 border border-slate-200 px-3 py-1.5 rounded-lg hover:bg-slate-50"
              >
                <RefreshCw size={14} />
                Randomize
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <MatrixGrid data={Sprev} title="S_{t-1}" editable onChange={handleMatrixChange} />
              <div className="flex flex-col gap-3 items-center">
                <VectorGrid data={k} title="k_t" editable onChange={(i: number, val: number) => handleVectorChange(i, val, setK)} />
                <VectorGrid data={v} title="v_t" editable onChange={(i: number, val: number) => handleVectorChange(i, val, setV)} />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-3">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">Parameters</div>
                <ParamSlider
                  label="alpha"
                  value={alpha}
                  min={0}
                  max={1.2}
                  step={0.05}
                  onChange={setAlpha}
                  disabled={!rule.usesAlpha}
                />
                <ParamSlider
                  label="beta"
                  value={beta}
                  min={0}
                  max={2}
                  step={0.05}
                  onChange={setBeta}
                  disabled={!rule.usesBeta}
                />
                {activeRule === 'longhorn' && (
                  <div className="text-[11px] text-slate-500 font-mono">
                    eps = beta / (1 + beta k^T k) = {formatNum(computed.eps)}
                  </div>
                )}
                {(activeRule === 'deltanet' || activeRule === 'gated-delta') && (
                  <div className="text-[11px] text-slate-500 font-mono">
                    k^T k = {formatNum(computed.kNorm)}
                  </div>
                )}
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-3">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">Update Preview</div>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  <MatrixGrid data={Sprev} title="S_{t-1}" className="scale-90" />
                  <span className="text-lg text-slate-400">+</span>
                  <MatrixGrid data={computed.delta} title="Delta" className="scale-90" />
                  <ArrowRight size={16} className="text-slate-400" />
                  <MatrixGrid data={computed.nextS} title="S_t" className={`scale-90 ring-2 ${rule.accent.ring} rounded-lg`} />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <MetricCard
                title="Objective J(S)"
                prev={computed.objectivePrev}
                next={computed.objectiveNext}
                lowerIsBetter
              />
              <MetricCard
                title="Recon Loss L(S)"
                prev={computed.reconPrev}
                next={computed.reconNext}
                lowerIsBetter
              />
              <MetricCard
                title="Alignment dot(S k, v)"
                prev={computed.alignPrev}
                next={computed.alignNext}
                lowerIsBetter={false}
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Target size={16} className="text-slate-600" />
              Why the updates look like reconstruction
            </div>
            <div className="text-sm text-slate-600">
              Expand the reconstruction loss:
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
              <BlockMath math={'\\lVert S k - v \\rVert^2 = k^{\\top} S^{\\top} S k - 2\\langle S k, v \\rangle + v^{\\top} v'} />
            </div>
            <div className="text-sm text-slate-600 space-y-2">
              <div>- Keep the quadratic term: Longhorn solves it exactly.</div>
              <div>
                - Drop the quadratic term: LA and Mamba2 keep only <InlineMath math={'-2\\langle S k, v \\rangle'} />.
              </div>
              <div>
                - Take a gradient step: DeltaNet uses <InlineMath math={'v - S_{t-1} k'} /> as the error.
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Sparkles size={16} className="text-slate-600" />
              Weights-as-state view (second screenshot)
            </div>
            <div className="text-sm text-slate-600">
              Same idea, but you do a gradient step instead of solving the quadratic exactly.
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700 space-y-1">
              <BlockMath math={'W_t = W_{t-1} - \\eta \\nabla_W \\ell(W_{t-1}; x_t)'} />
              <BlockMath math={'\\ell(W; x_t) = \\lVert f(\\tilde{x}_t; W) - x_t \\rVert^2'} />
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-mono">
              <div className="px-2 py-1 rounded bg-slate-100 border border-slate-200">x_t_tilde</div>
              <ArrowRight size={14} className="text-slate-400" />
              <div className="px-2 py-1 rounded bg-slate-100 border border-slate-200">
                f(.; <InlineMath math={'W_{t-1}'} />)
              </div>
              <ArrowRight size={14} className="text-slate-400" />
              <div className="px-2 py-1 rounded bg-slate-100 border border-slate-200">loss</div>
              <ArrowRight size={14} className="text-slate-400" />
              <div className="px-2 py-1 rounded bg-slate-100 border border-slate-200">W_t</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
