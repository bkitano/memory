// @ts-nocheck
import React, { useEffect, useMemo, useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { Pause, Play, RefreshCw, RotateCcw, SkipForward } from 'lucide-react';

const DEFAULT_SEQ_LEN = 6;
const DEFAULT_NOISE = 0.05;
const DEFAULT_LR = 0.4;
const CHART = {
  width: 640,
  height: 420,
  pad: 52,
};

const rand = (min: number, max: number) => min + Math.random() * (max - min);
const randomSlope = () => Number(rand(0.8, 1.4).toFixed(2));

const generateSequence = (count: number, slope: number, noise: number) =>
  Array.from({ length: count }, (_, i) => {
    const k = Number(rand(0.15, 1.0).toFixed(2));
    const rawV = slope * k + rand(-noise, noise);
    const v = Number(Math.max(0, rawV).toFixed(2));
    return { id: i, k, v };
  });

const formatNum = (val: number, digits = 3) =>
  Number.isFinite(val) ? val.toFixed(digits) : '0.000';

const initialSlope = randomSlope();
const initialSequence = generateSequence(DEFAULT_SEQ_LEN, initialSlope, DEFAULT_NOISE);

export default function TTTLinearRegressionDemo() {
  const [trueSlope, setTrueSlope] = useState(initialSlope);
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [learningRate, setLearningRate] = useState(DEFAULT_LR);
  const [sequence, setSequence] = useState(initialSequence);
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTrueLine, setShowTrueLine] = useState(true);

  const weights = useMemo(() => {
    let w = 0;
    const history = [w];
    sequence.forEach(point => {
      w += learningRate * point.k * point.v;
      history.push(w);
    });
    return history;
  }, [sequence, learningRate]);

  const deltas = useMemo(
    () => sequence.map(point => learningRate * point.k * point.v),
    [sequence, learningRate]
  );

  useEffect(() => {
    if (!isPlaying) return;
    if (step >= sequence.length) {
      setIsPlaying(false);
      return;
    }
    const id = window.setTimeout(() => {
      setStep(current => Math.min(current + 1, sequence.length));
    }, 1200);
    return () => window.clearTimeout(id);
  }, [isPlaying, step, sequence.length]);

  const currentSlope = weights[step] ?? 0;
  const currentPoint = step > 0 ? sequence[step - 1] : null;
  const currentDelta = step > 0 ? deltas[step - 1] : 0;

  const { xMax, yMax } = useMemo(() => {
    if (!sequence.length) {
      return { xMax: 1, yMax: 1 };
    }
    const maxK = Math.max(...sequence.map(point => point.k), 1);
    const maxV = Math.max(...sequence.map(point => point.v), 1);
    const maxSlope = Math.max(Math.abs(trueSlope), Math.abs(currentSlope), 1);
    const nextX = Math.max(1, maxK) * 1.1;
    const nextY = Math.max(maxV, maxSlope * nextX, 1) * 1.1;
    return { xMax: nextX, yMax: nextY };
  }, [sequence, currentSlope, trueSlope]);

  const xMin = 0;
  const yMin = 0;
  const rangeX = xMax - xMin || 1;
  const rangeY = yMax - yMin || 1;
  const plotX = (x: number) =>
    CHART.pad + ((x - xMin) / rangeX) * (CHART.width - CHART.pad * 2);
  const plotY = (y: number) =>
    CHART.height - CHART.pad - ((y - yMin) / rangeY) * (CHART.height - CHART.pad * 2);

  const pointsSeen = sequence.slice(0, step);
  const pointsFuture = sequence.slice(step);

  const handleReset = () => {
    setIsPlaying(false);
    setStep(0);
  };

  const handleStep = () => {
    setIsPlaying(false);
    setStep(current => Math.min(current + 1, sequence.length));
  };

  const handleRegenerate = () => {
    const nextSlope = randomSlope();
    setTrueSlope(nextSlope);
    setSequence(generateSequence(DEFAULT_SEQ_LEN, nextSlope, noise));
    setIsPlaying(false);
    setStep(0);
  };

  const handleNoiseChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextNoise = parseFloat(event.target.value);
    setNoise(nextNoise);
    setSequence(generateSequence(DEFAULT_SEQ_LEN, trueSlope, nextNoise));
    setIsPlaying(false);
    setStep(0);
  };

  const togglePlay = () => {
    if (step >= sequence.length) {
      setStep(0);
    }
    setIsPlaying(current => !current);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold">TTT Linear Regression (1D)</h1>
            <p className="text-xs text-slate-500">
              Batch GD update at w0=0 matches the linear attention state update.
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              <a
                href="https://arxiv.org/pdf/2407.04620"
                target="_blank"
                rel="noreferrer"
                className="text-blue-600 hover:text-blue-700 underline underline-offset-2"
              >
                TTT paper (arXiv:2407.04620)
              </a>
            </p>
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-2">
            <span className="bg-slate-100 px-2 py-1 rounded font-mono">state: S_t</span>
            <span className="bg-slate-100 px-2 py-1 rounded font-mono">k on x-axis</span>
            <span className="bg-slate-100 px-2 py-1 rounded font-mono">v on y-axis</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6">
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Playback</div>
              <div className="flex flex-wrap gap-2 mb-4">
                <button
                  onClick={togglePlay}
                  className="px-3 py-2 rounded-md bg-blue-600 text-white text-xs font-semibold flex items-center gap-2 shadow-sm"
                >
                  {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <button
                  onClick={handleStep}
                  disabled={step >= sequence.length}
                  className="px-3 py-2 rounded-md bg-white border border-slate-200 text-xs font-semibold flex items-center gap-2 text-slate-600 disabled:text-slate-300 disabled:border-slate-100"
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
                  onClick={handleRegenerate}
                  className="px-3 py-2 rounded-md bg-white border border-slate-200 text-xs font-semibold flex items-center gap-2 text-slate-600"
                >
                  <RefreshCw size={14} />
                  New Data
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Step</div>
                  <div className="text-lg font-mono text-slate-800">
                    S_{step} / {sequence.length}
                  </div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">State S_t</div>
                  <div className="text-lg font-mono text-slate-800">{formatNum(currentSlope)}</div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Delta S_t</div>
                  <div className="text-lg font-mono text-slate-800">{formatNum(currentDelta)}</div>
                </div>
                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">True Slope</div>
                  <div className="text-lg font-mono text-slate-800">{formatNum(trueSlope, 2)}</div>
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
                    checked={showTrueLine}
                    onChange={event => setShowTrueLine(event.target.checked)}
                    className="accent-blue-600"
                  />
                  Show ground truth line
                </label>
              </div>

              <div className="mt-4 text-xs text-slate-500">
                Current point: k_t = {currentPoint ? formatNum(currentPoint.k, 2) : '--'} , v_t ={' '}
                {currentPoint ? formatNum(currentPoint.v, 2) : '--'}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Update Rule</div>
              <div className="space-y-3 text-xs text-slate-600">
                <BlockMath math={'\\mathcal{L}(w; k_t, v_t) = (w k_t - v_t)^2'} />
                <div className="flex flex-col gap-2">
                  <span className="font-semibold text-slate-500">Batch GD at w0=0:</span>
                  <InlineMath math={'\\Delta S_t = \\eta v_t k_t'} />
                </div>
                <div className="flex flex-col gap-2">
                  <span className="font-semibold text-slate-500">State update:</span>
                  <InlineMath math={'S_t = S_{t-1} + \\eta v_t k_t'} />
                </div>
                <div className="flex flex-col gap-2">
                  <span className="font-semibold text-slate-500">Prediction line:</span>
                  <InlineMath math={'\\hat{v} = S_t k'} />
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">1D Regression Fit</div>
                  <div className="text-sm text-slate-600">
                    Points appear over time; the orange line is the current state S_t.
                  </div>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-500">
                  <div className="flex items-center gap-2">
                    <span className="inline-block w-6 h-0.5 bg-orange-500 rounded-full" />
                    State line
                  </div>
                  {showTrueLine && (
                    <div className="flex items-center gap-2">
                      <span className="inline-block w-6 border-t-2 border-dashed border-slate-400" />
                      True line
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <span className="inline-block w-2 h-2 bg-blue-500 rounded-full" />
                    Observed points
                  </div>
                </div>
              </div>

              <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="w-full h-auto bg-slate-50 rounded-lg">
                <rect x="0" y="0" width={CHART.width} height={CHART.height} fill="transparent" />

                <g className="text-slate-300" stroke="currentColor" strokeWidth="1">
                  <line x1={CHART.pad} y1={CHART.height - CHART.pad} x2={CHART.width - CHART.pad} y2={CHART.height - CHART.pad} />
                  <line x1={CHART.pad} y1={CHART.pad} x2={CHART.pad} y2={CHART.height - CHART.pad} />
                </g>

                {Array.from({ length: 5 }, (_, i) => {
                  const t = i / 4;
                  const xVal = xMin + t * rangeX;
                  const yVal = yMin + t * rangeY;
                  const xPos = plotX(xVal);
                  const yPos = plotY(yVal);
                  return (
                    <g key={`tick-${i}`} className="text-slate-400">
                      <line x1={xPos} y1={CHART.height - CHART.pad} x2={xPos} y2={CHART.height - CHART.pad + 6} stroke="currentColor" />
                      <text x={xPos} y={CHART.height - CHART.pad + 18} textAnchor="middle" className="text-[10px] fill-slate-400">
                        {xVal.toFixed(1)}
                      </text>
                      <line x1={CHART.pad - 6} y1={yPos} x2={CHART.pad} y2={yPos} stroke="currentColor" />
                      <text x={CHART.pad - 10} y={yPos + 3} textAnchor="end" className="text-[10px] fill-slate-400">
                        {yVal.toFixed(1)}
                      </text>
                    </g>
                  );
                })}

                <text x={CHART.width - CHART.pad} y={CHART.height - 12} textAnchor="end" className="text-xs fill-slate-500">
                  k
                </text>
                <text x={CHART.pad - 26} y={CHART.pad - 12} textAnchor="start" className="text-xs fill-slate-500">
                  v
                </text>

                {showTrueLine && (
                  <line
                    x1={plotX(0)}
                    y1={plotY(0)}
                    x2={plotX(xMax)}
                    y2={plotY(trueSlope * xMax)}
                    strokeWidth="2"
                    strokeDasharray="6 6"
                    className="stroke-slate-400"
                  />
                )}

                <line
                  x1={plotX(0)}
                  y1={plotY(0)}
                  x2={plotX(xMax)}
                  y2={plotY(currentSlope * xMax)}
                  strokeWidth="3"
                  className="stroke-orange-500"
                />

                {pointsFuture.map(point => (
                  <circle
                    key={`future-${point.id}`}
                    cx={plotX(point.k)}
                    cy={plotY(point.v)}
                    r={5}
                    className="fill-white stroke-slate-300"
                    strokeWidth="2"
                  />
                ))}
                {pointsSeen.map(point => (
                  <circle
                    key={`seen-${point.id}`}
                    cx={plotX(point.k)}
                    cy={plotY(point.v)}
                    r={6}
                    className="fill-blue-500"
                  />
                ))}
                {currentPoint && (
                  <circle
                    cx={plotX(currentPoint.k)}
                    cy={plotY(currentPoint.v)}
                    r={10}
                    className="fill-transparent stroke-blue-300"
                    strokeWidth="2"
                  />
                )}

                <text x={CHART.pad + 8} y={CHART.pad + 16} className="text-xs font-mono fill-slate-400">
                  S_{step}
                </text>
              </svg>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Sequence</div>
              <div className="grid grid-cols-4 gap-2 text-[11px] font-mono text-slate-500 mb-2">
                <div>step</div>
                <div>k</div>
                <div>v</div>
                <div>eta * k * v</div>
              </div>
              <div className="space-y-1">
                {sequence.map((point, index) => {
                  const isActive = step === index + 1;
                  return (
                    <div
                      key={point.id}
                      className={`grid grid-cols-4 gap-2 text-[11px] font-mono px-2 py-1 rounded ${
                        isActive ? 'bg-amber-50 text-amber-700' : 'text-slate-600'
                      }`}
                    >
                      <div>S_{index + 1}</div>
                      <div>{formatNum(point.k, 2)}</div>
                      <div>{formatNum(point.v, 2)}</div>
                      <div>{formatNum(deltas[index], 3)}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
