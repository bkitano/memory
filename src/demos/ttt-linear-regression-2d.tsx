// @ts-nocheck
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { Pause, Play, RefreshCw, RotateCcw, SkipForward } from 'lucide-react';

const TRAIN_SIZE = 28;
const PROBE_SIZE = 28;
const MAX_HISTORY = 120;
const EMA_ALPHA = 0.2;

const DEFAULT_NOISE = 0.05;
const DEFAULT_LR = 0.35;

const MAIN_PLOT = { size: 440, pad: 54 };
const SIGNATURE_PLOT = { size: 220, pad: 38 };
const LOSS_PLOT = { width: 320, height: 160, pad: 20 };

const rand = (min: number, max: number) => min + Math.random() * (max - min);

const zeroMatrix = () => [
  [0, 0],
  [0, 0],
];

const randomMatrix = () => [
  [Number(rand(-1.2, 1.2).toFixed(2)), Number(rand(-1.2, 1.2).toFixed(2))],
  [Number(rand(-1.2, 1.2).toFixed(2)), Number(rand(-1.2, 1.2).toFixed(2))],
];

const matVecMul = (M: number[][], v: number[]) => [
  M[0][0] * v[0] + M[0][1] * v[1],
  M[1][0] * v[0] + M[1][1] * v[1],
];

const outer = (a: number[], b: number[]) => [
  [a[0] * b[0], a[0] * b[1]],
  [a[1] * b[0], a[1] * b[1]],
];

const matAdd = (A: number[][], B: number[][]) =>
  A.map((row, i) => row.map((val, j) => val + B[i][j]));

const matScale = (A: number[][], s: number) => A.map(row => row.map(val => val * s));

const formatNum = (val: number, digits = 2) =>
  Number.isFinite(val) ? val.toFixed(digits) : '0.00';

const makePoint = (matrix: number[][], noise: number) => {
  const k = [Number(rand(-1, 1).toFixed(2)), Number(rand(-1, 1).toFixed(2))];
  const cleanV = matVecMul(matrix, k);
  const noisy = [cleanV[0] + rand(-noise, noise), cleanV[1] + rand(-noise, noise)];
  const v = noisy.map(val => Number(val.toFixed(2)));
  return { k, v };
};

const computeAxisLimit = (targets: number[][], matrix: number[][]) => {
  const basis = [
    matVecMul(matrix, [1, 0]),
    matVecMul(matrix, [0, 1]),
  ];
  const all = [...targets, ...basis];
  const maxAbs = Math.max(
    1,
    ...all.map(vec => Math.max(Math.abs(vec[0]), Math.abs(vec[1])))
  );
  return maxAbs * 1.25;
};

const generateDataset = (matrix: number[][], noise: number) => {
  const train = Array.from({ length: TRAIN_SIZE }, (_, i) => ({
    id: i,
    ...makePoint(matrix, noise),
  }));
  const probe = Array.from({ length: PROBE_SIZE }, (_, i) => ({
    id: i,
    ...makePoint(matrix, noise),
  }));
  const axisLimit = computeAxisLimit(probe.map(point => point.v), matrix);
  return { train, probe, axisLimit };
};

const sgdUpdate = (S: number[][], sample: any, learningRate: number) => {
  const pred = matVecMul(S, sample.k);
  const err = [sample.v[0] - pred[0], sample.v[1] - pred[1]];
  const loss = err[0] * err[0] + err[1] * err[1];
  const grad = outer(err, sample.k);
  const delta = matScale(grad, learningRate);
  const nextS = matAdd(S, delta);
  return { nextS, loss, delta, err };
};

const MatrixGrid = ({ title, data, tone = 'text-slate-600' }: any) => (
  <div className="flex flex-col gap-2">
    <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">{title}</div>
    <div className="grid grid-cols-2 gap-px bg-slate-200 rounded-lg overflow-hidden">
      {data.map((row: number[], i: number) =>
        row.map((val: number, j: number) => (
          <div key={`${i}-${j}`} className={`w-14 h-12 flex items-center justify-center text-xs font-mono bg-white ${tone}`}>
            {formatNum(val, 2)}
          </div>
        ))
      )}
    </div>
  </div>
);

const VectorGrid = ({ title, data, tone = 'text-slate-600' }: any) => (
  <div className="flex flex-col gap-2">
    <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">{title}</div>
    <div className="grid grid-cols-2 gap-px bg-slate-200 rounded-lg overflow-hidden">
      {data.map((val: number, idx: number) => (
        <div key={idx} className={`w-14 h-10 flex items-center justify-center text-xs font-mono bg-white ${tone}`}>
          {formatNum(val, 2)}
        </div>
      ))}
    </div>
  </div>
);

const ResidualPlot = ({
  targets,
  predictions,
  axisLimit,
  showResiduals,
}: any) => {
  const min = -axisLimit;
  const max = axisLimit;
  const plotX = (x: number) =>
    MAIN_PLOT.pad + ((x - min) / (max - min)) * (MAIN_PLOT.size - MAIN_PLOT.pad * 2);
  const plotY = (y: number) =>
    MAIN_PLOT.size - MAIN_PLOT.pad - ((y - min) / (max - min)) * (MAIN_PLOT.size - MAIN_PLOT.pad * 2);
  const ticks = Array.from({ length: 5 }, (_, i) => min + (i / 4) * (max - min));

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Value Residuals (Probe)</div>
          <div className="text-xs text-slate-500">Predictions S_t k chase fixed values v; arrows are residuals.</div>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-500" /> values
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-orange-400" /> predictions
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-4 h-0.5 bg-slate-300" /> residual
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${MAIN_PLOT.size} ${MAIN_PLOT.size}`} className="w-full h-auto bg-slate-50 rounded-lg">
        <defs>
          <marker id="arrowhead-residual" markerWidth="8" markerHeight="6" refX="6" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="rgba(148, 163, 184, 0.8)" />
          </marker>
        </defs>

        {ticks.map((t, idx) => (
          <g key={`tick-${idx}`} className="text-slate-200">
            <line x1={plotX(t)} y1={MAIN_PLOT.pad} x2={plotX(t)} y2={MAIN_PLOT.size - MAIN_PLOT.pad} stroke="currentColor" />
            <line x1={MAIN_PLOT.pad} y1={plotY(t)} x2={MAIN_PLOT.size - MAIN_PLOT.pad} y2={plotY(t)} stroke="currentColor" />
          </g>
        ))}

        <g className="text-slate-400">
          <line x1={plotX(0)} y1={MAIN_PLOT.pad} x2={plotX(0)} y2={MAIN_PLOT.size - MAIN_PLOT.pad} stroke="currentColor" />
          <line x1={MAIN_PLOT.pad} y1={plotY(0)} x2={MAIN_PLOT.size - MAIN_PLOT.pad} y2={plotY(0)} stroke="currentColor" />
        </g>

        <text x={MAIN_PLOT.size - MAIN_PLOT.pad} y={MAIN_PLOT.size - 12} textAnchor="end" className="text-[10px] fill-slate-500">
          v1
        </text>
        <text x={MAIN_PLOT.pad - 16} y={MAIN_PLOT.pad - 12} textAnchor="start" className="text-[10px] fill-slate-500">
          v2
        </text>

        {showResiduals &&
          predictions.map((pred: any, idx: number) => {
            const target = targets[idx];
            if (!target) return null;
            return (
              <line
                key={`res-${idx}`}
                x1={plotX(pred.x)}
                y1={plotY(pred.y)}
                x2={plotX(target.x)}
                y2={plotY(target.y)}
                stroke="rgba(148, 163, 184, 0.6)"
                strokeDasharray="4 4"
                markerEnd="url(#arrowhead-residual)"
              />
            );
          })}

        {targets.map((point: any, idx: number) => (
          <circle key={`target-${idx}`} cx={plotX(point.x)} cy={plotY(point.y)} r={5} className="fill-blue-500" />
        ))}

        {predictions.map((point: any, idx: number) => (
          <circle key={`pred-${idx}`} cx={plotX(point.x)} cy={plotY(point.y)} r={4} className="fill-orange-400" />
        ))}
      </svg>
    </div>
  );
};

const SignaturePlot = ({
  basis,
  trueBasis,
  axisLimit,
  showTrueBasis,
}: any) => {
  const min = -axisLimit;
  const max = axisLimit;
  const plotX = (x: number) =>
    SIGNATURE_PLOT.pad + ((x - min) / (max - min)) * (SIGNATURE_PLOT.size - SIGNATURE_PLOT.pad * 2);
  const plotY = (y: number) =>
    SIGNATURE_PLOT.size - SIGNATURE_PLOT.pad - ((y - min) / (max - min)) * (SIGNATURE_PLOT.size - SIGNATURE_PLOT.pad * 2);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Model Signature</div>
      <svg viewBox={`0 0 ${SIGNATURE_PLOT.size} ${SIGNATURE_PLOT.size}`} className="w-full h-auto bg-slate-50 rounded-lg">
        <defs>
          <marker id="arrowhead-basis" markerWidth="8" markerHeight="6" refX="6" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="rgba(249, 115, 22, 0.9)" />
          </marker>
        </defs>

        <g className="text-slate-200">
          <line x1={plotX(min)} y1={plotY(0)} x2={plotX(max)} y2={plotY(0)} stroke="currentColor" />
          <line x1={plotX(0)} y1={plotY(min)} x2={plotX(0)} y2={plotY(max)} stroke="currentColor" />
        </g>

        {showTrueBasis &&
          trueBasis.map((vec: any, idx: number) => (
            <g key={`true-basis-${idx}`} className="text-slate-300">
              <line
                x1={plotX(0)}
                y1={plotY(0)}
                x2={plotX(vec.x)}
                y2={plotY(vec.y)}
                stroke="currentColor"
                strokeDasharray="6 4"
              />
              <circle cx={plotX(vec.x)} cy={plotY(vec.y)} r={4} className="fill-slate-300" />
            </g>
          ))}

        {basis.map((vec: any, idx: number) => (
          <g key={`basis-${idx}`} className="text-orange-500">
            <line
              x1={plotX(0)}
              y1={plotY(0)}
              x2={plotX(vec.x)}
              y2={plotY(vec.y)}
              stroke="currentColor"
              strokeWidth="2"
              markerEnd="url(#arrowhead-basis)"
            />
            <circle cx={plotX(vec.x)} cy={plotY(vec.y)} r={4} className="fill-orange-500" />
            <text
              x={plotX(vec.x) + 6}
              y={plotY(vec.y) - 6}
              className="text-[10px] fill-orange-500"
            >
              e{idx + 1}
            </text>
          </g>
        ))}
      </svg>
      <div className="text-[11px] text-slate-500 mt-2">
        Arrows show S_t e1 and S_t e2 in value space.
      </div>
    </div>
  );
};

const LossPlot = ({ raw, ema, maxValue }: any) => {
  const safeMax = Math.max(maxValue, 1e-6);
  const plotX = (idx: number, total: number) =>
    LOSS_PLOT.pad + (idx / Math.max(1, total - 1)) * (LOSS_PLOT.width - LOSS_PLOT.pad * 2);
  const plotY = (val: number) =>
    LOSS_PLOT.height - LOSS_PLOT.pad - (val / safeMax) * (LOSS_PLOT.height - LOSS_PLOT.pad * 2);
  const makePath = (values: number[]) =>
    values
      .map((val, idx) => `${idx === 0 ? 'M' : 'L'} ${plotX(idx, values.length)} ${plotY(val)}`)
      .join(' ');

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Loss Trace</div>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5 bg-slate-300" /> raw
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5 bg-emerald-500" /> ema
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${LOSS_PLOT.width} ${LOSS_PLOT.height}`} className="w-full h-auto bg-slate-50 rounded-lg">
        <rect x="0" y="0" width={LOSS_PLOT.width} height={LOSS_PLOT.height} fill="transparent" />
        <line
          x1={LOSS_PLOT.pad}
          y1={LOSS_PLOT.height - LOSS_PLOT.pad}
          x2={LOSS_PLOT.width - LOSS_PLOT.pad}
          y2={LOSS_PLOT.height - LOSS_PLOT.pad}
          stroke="rgba(148, 163, 184, 0.6)"
        />
        {raw.length > 1 && (
          <path d={makePath(raw)} fill="none" stroke="rgba(148, 163, 184, 0.7)" strokeWidth="2" />
        )}
        {ema.length > 1 && (
          <path d={makePath(ema)} fill="none" stroke="rgba(16, 185, 129, 0.9)" strokeWidth="2.5" />
        )}
      </svg>
      <div className="text-[11px] text-slate-500 mt-2">
        Per-step loss with EMA smoothing to expose trend.
      </div>
    </div>
  );
};

const initialTrueMatrix = randomMatrix();
const initialData = generateDataset(initialTrueMatrix, DEFAULT_NOISE);

export default function TTTLinearRegression2DDemo() {
  const [trueMatrix, setTrueMatrix] = useState(initialTrueMatrix);
  const [trainSet, setTrainSet] = useState(initialData.train);
  const [probeSet, setProbeSet] = useState(initialData.probe);
  const [axisLimit, setAxisLimit] = useState(initialData.axisLimit);

  const [matrix, setMatrix] = useState(zeroMatrix());
  const [step, setStep] = useState(0);
  const [trainIndex, setTrainIndex] = useState(0);
  const [lastSample, setLastSample] = useState<any>(null);
  const [lastDelta, setLastDelta] = useState(zeroMatrix());
  const [lastErr, setLastErr] = useState([0, 0]);
  const [isPlaying, setIsPlaying] = useState(false);

  const [learningRate, setLearningRate] = useState(DEFAULT_LR);
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [showTrueBasis, setShowTrueBasis] = useState(true);
  const [showResiduals, setShowResiduals] = useState(true);

  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [emaHistory, setEmaHistory] = useState<number[]>([]);
  const [lossMax, setLossMax] = useState(1);

  const resetModel = useCallback(() => {
    setMatrix(zeroMatrix());
    setStep(0);
    setTrainIndex(0);
    setLastSample(null);
    setLastDelta(zeroMatrix());
    setLastErr([0, 0]);
    setLossHistory([]);
    setEmaHistory([]);
    setLossMax(1);
    setIsPlaying(false);
  }, []);

  const regenerate = useCallback(
    (nextMatrix: number[][], nextNoise: number) => {
      const nextData = generateDataset(nextMatrix, nextNoise);
      setTrueMatrix(nextMatrix);
      setTrainSet(nextData.train);
      setProbeSet(nextData.probe);
      setAxisLimit(nextData.axisLimit);
      resetModel();
    },
    [resetModel]
  );

  const stepOnce = useCallback(() => {
    if (!trainSet.length) return;
    const sample = trainSet[trainIndex % trainSet.length];
    const { nextS, loss, delta, err } = sgdUpdate(matrix, sample, learningRate);
    setMatrix(nextS);
    setLastSample(sample);
    setLastDelta(delta);
    setLastErr(err);
    setStep(prev => prev + 1);
    setTrainIndex(prev => (prev + 1) % trainSet.length);
    setLossHistory(prev => {
      const next = [...prev, loss];
      if (next.length > MAX_HISTORY) next.shift();
      return next;
    });
    setEmaHistory(prev => {
      const last = prev.length ? prev[prev.length - 1] : loss;
      const nextEma = last + EMA_ALPHA * (loss - last);
      const next = [...prev, nextEma];
      if (next.length > MAX_HISTORY) next.shift();
      setLossMax(prevMax => Math.max(prevMax, loss, nextEma));
      return next;
    });
  }, [learningRate, matrix, trainIndex, trainSet]);

  useEffect(() => {
    if (!isPlaying) return;
    const id = window.setTimeout(() => {
      stepOnce();
    }, 1200);
    return () => window.clearTimeout(id);
  }, [isPlaying, stepOnce]);

  const targets = useMemo(
    () => probeSet.map(point => ({ x: point.v[0], y: point.v[1] })),
    [probeSet]
  );
  const predictions = useMemo(
    () => probeSet.map(point => {
      const pred = matVecMul(matrix, point.k);
      return { x: pred[0], y: pred[1] };
    }),
    [matrix, probeSet]
  );
  const visibleProbeCount = Math.min(step, probeSet.length);

  const probeLoss = useMemo(() => {
    if (!probeSet.length) return 0;
    let sum = 0;
    probeSet.forEach(point => {
      const pred = matVecMul(matrix, point.k);
      const err0 = point.v[0] - pred[0];
      const err1 = point.v[1] - pred[1];
      sum += err0 * err0 + err1 * err1;
    });
    return sum / probeSet.length;
  }, [matrix, probeSet]);

  const basis = useMemo(
    () => [
      { x: matVecMul(matrix, [1, 0])[0], y: matVecMul(matrix, [1, 0])[1] },
      { x: matVecMul(matrix, [0, 1])[0], y: matVecMul(matrix, [0, 1])[1] },
    ],
    [matrix]
  );

  const trueBasis = useMemo(
    () => [
      { x: matVecMul(trueMatrix, [1, 0])[0], y: matVecMul(trueMatrix, [1, 0])[1] },
      { x: matVecMul(trueMatrix, [0, 1])[0], y: matVecMul(trueMatrix, [0, 1])[1] },
    ],
    [trueMatrix]
  );

  const lastLoss = lossHistory[lossHistory.length - 1] ?? 0;
  const lastEma = emaHistory[emaHistory.length - 1] ?? 0;
  const currentIndex = trainSet.length
    ? (step === 0 ? 0 : (trainIndex - 1 + trainSet.length) % trainSet.length)
    : 0;
  const queryIndex = Math.min(step, Math.max(0, probeSet.length - 1));
  const querySample = probeSet[queryIndex];
  const queryPred = querySample ? matVecMul(matrix, querySample.k) : [0, 0];

  const handleTogglePlay = () => {
    setIsPlaying(prev => !prev);
  };

  const handleStep = () => {
    setIsPlaying(false);
    stepOnce();
  };

  const handleReset = () => {
    resetModel();
  };

  const handleNewData = () => {
    regenerate(randomMatrix(), noise);
  };

  const handleNoiseChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextNoise = parseFloat(event.target.value);
    setNoise(nextNoise);
    regenerate(trueMatrix, nextNoise);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold">Online SGD Linear Regression (2D)</h1>
            <p className="text-xs text-slate-500">
              State matrix S_t (aka A) maps keys k to values v. Probe set is fixed; residual arrows shrink as fit improves.
            </p>
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-2">
            <span className="bg-slate-100 px-2 py-1 rounded font-mono">train: {TRAIN_SIZE}</span>
            <span className="bg-slate-100 px-2 py-1 rounded font-mono">probe: {PROBE_SIZE}</span>
            <span className="bg-slate-100 px-2 py-1 rounded font-mono">fixed axis</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6">
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm mb-6">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Incoming Sequence (k_t, v_t)
            </div>
            <div className="text-[11px] text-slate-500">Highlight = current update</div>
          </div>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {trainSet.map((item, idx) => {
              const isActive = idx === currentIndex;
              return (
                <div
                  key={item.id}
                  className={`min-w-[170px] rounded-lg border px-3 py-2 text-[11px] font-mono ${
                    isActive
                      ? 'border-blue-500 bg-blue-50 text-blue-700 shadow-sm'
                      : 'border-slate-200 bg-white text-slate-600'
                  }`}
                >
                  <div className="text-[10px] uppercase tracking-wider mb-1">
                    t = {idx + 1}
                  </div>
                  <div>k = [{formatNum(item.k[0])}, {formatNum(item.k[1])}]</div>
                  <div>v = [{formatNum(item.v[0])}, {formatNum(item.v[1])}]</div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Playback</div>
              <div className="flex flex-wrap gap-2 mb-4">
                <button
                  onClick={handleTogglePlay}
                  className="px-3 py-2 rounded-md bg-blue-600 text-white text-xs font-semibold flex items-center gap-2 shadow-sm"
                >
                  {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <button
                  onClick={handleStep}
                  className="px-3 py-2 rounded-md bg-white border border-slate-200 text-xs font-semibold flex items-center gap-2 text-slate-600"
                >
                  <SkipForward size={14} />
                  Step
                </button>
                <button
                  onClick={handleReset}
                  className="px-3 py-2 rounded-md bg-white border border-slate-200 text-xs font-semibold flex items-center gap-2 text-slate-600"
                >
                  <RotateCcw size={14} />
                  Reset
                </button>
                <button
                  onClick={handleNewData}
                  className="px-3 py-2 rounded-md bg-white border border-slate-200 text-xs font-semibold flex items-center gap-2 text-slate-600"
                >
                  <RefreshCw size={14} />
                  New Data
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Step</div>
                  <div className="text-lg font-mono text-slate-800">t = {step}</div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Step Loss</div>
                  <div className="text-lg font-mono text-slate-800">{formatNum(lastLoss, 3)}</div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">EMA Loss</div>
                  <div className="text-lg font-mono text-slate-800">{formatNum(lastEma, 3)}</div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Probe MSE</div>
                  <div className="text-lg font-mono text-slate-800">{formatNum(probeLoss, 3)}</div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3 col-span-2">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Last Update (k_t, v_t)</div>
                  <div className="text-[11px] font-mono text-slate-700">
                    k_t = [{lastSample ? formatNum(lastSample.k[0]) : '--'}, {lastSample ? formatNum(lastSample.k[1]) : '--'}], v_t = [
                    {lastSample ? formatNum(lastSample.v[0]) : '--'}, {lastSample ? formatNum(lastSample.v[1]) : '--'}], e_t = [
                    {formatNum(lastErr[0])}, {formatNum(lastErr[1])}]
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
                    <span>Learning rate (eta)</span>
                    <span className="font-mono">{learningRate.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.8"
                    step="0.05"
                    value={learningRate}
                    onChange={event => setLearningRate(parseFloat(event.target.value))}
                    className="w-full accent-blue-600"
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
                    <span>Noise</span>
                    <span className="font-mono">{noise.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="0.2"
                    step="0.01"
                    value={noise}
                    onChange={handleNoiseChange}
                    className="w-full accent-blue-600"
                  />
                </div>
                <label className="flex items-center gap-2 text-xs text-slate-600">
                  <input
                    type="checkbox"
                    checked={showResiduals}
                    onChange={event => setShowResiduals(event.target.checked)}
                    className="accent-blue-600"
                  />
                  Show residual arrows
                </label>
                <label className="flex items-center gap-2 text-xs text-slate-600">
                  <input
                    type="checkbox"
                    checked={showTrueBasis}
                    onChange={event => setShowTrueBasis(event.target.checked)}
                    className="accent-blue-600"
                  />
                  Show true basis
                </label>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">State Matrix S_t</div>
              <div className="grid grid-cols-2 gap-4">
                <MatrixGrid title="S_t (state)" data={matrix} tone="text-orange-600" />
                <MatrixGrid title="ΔS_t (grad)" data={lastDelta} tone="text-emerald-600" />
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <VectorGrid title="k_t (key)" data={lastSample ? lastSample.k : [0, 0]} tone="text-slate-600" />
                <VectorGrid title="v_t (value)" data={lastSample ? lastSample.v : [0, 0]} tone="text-slate-600" />
                <VectorGrid title="e_t" data={lastErr} tone="text-rose-600" />
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <VectorGrid title="q (query)" data={querySample ? querySample.k : [0, 0]} tone="text-slate-600" />
                <VectorGrid title="S_t q" data={queryPred} tone="text-orange-600" />
                <VectorGrid title="v(q)" data={querySample ? querySample.v : [0, 0]} tone="text-blue-600" />
              </div>
              <div className="text-xs text-slate-500">
                <InlineMath math={'\\Delta S_t = \\eta (v_t - S_{t-1} k_t) k_t^{\\top}'} /> ·{' '}
                <InlineMath math={'S_t = S_{t-1} + \\Delta S_t'} />
              </div>
              <div className="text-[11px] text-slate-500">
                k_t / v_t are the online update pair; q is a probe key used for the readout.
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Objective</div>
              <div className="text-xs text-slate-600 space-y-2">
                <BlockMath math={'\\mathcal{L}(S) = \\lVert S k - v \\rVert^2'} />
                <div className="text-[11px] text-slate-500">
                  Online SGD uses one key/value pair at a time; the probe plot stays fixed across steps.
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <ResidualPlot
              targets={targets.slice(0, visibleProbeCount)}
              predictions={predictions.slice(0, visibleProbeCount)}
              axisLimit={axisLimit}
              showResiduals={showResiduals}
            />

            <div className="grid md:grid-cols-2 gap-4">
              <SignaturePlot
                basis={basis}
                trueBasis={trueBasis}
                axisLimit={axisLimit}
                showTrueBasis={showTrueBasis}
              />
              <LossPlot raw={lossHistory} ema={emaHistory} maxValue={lossMax} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
