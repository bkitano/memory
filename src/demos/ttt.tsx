

// @ts-nocheck
import React, { useState, useEffect, useMemo } from 'react';
import { Play, Pause, SkipForward, RotateCcw, ArrowRight, Activity, Zap, BookOpen, Box, Info, ChevronRight, Target, Sparkles } from 'lucide-react';

// --- Math & Simulation Logic ---

const D = 4;
const SEQ_LEN = 6; 

const randomVec = () => Array.from({ length: D }, () => Math.random() * 2 - 1);
const randomMat = () => Array.from({ length: D }, () => Array.from({ length: D }, () => (Math.random() * 0.5 - 0.25)));
const zeroMat = () => Array.from({ length: D }, () => Array.from({ length: D }, () => 0));

const matVecMul = (M: number[][], v: number[]) => M.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
const outerProduct = (v1: number[], v2: number[]) => v1.map(val1 => v2.map(val2 => val1 * val2));
const matAdd = (A: number[][], B: number[][], scalar = 1) => A.map((row, i) => row.map((val, j) => val + scalar * B[i][j]));

const generateSequence = () => {
  const seq = [];
  for (let i = 0; i < SEQ_LEN; i++) {
    const u = randomVec(); 
    seq.push({ u, id: i });
  }
  return seq;
};

// --- Components ---

const HeatmapCell = ({ value, label, showValue, isHighlight, size = "w-10 h-10" }: any) => {
  const intensity = Math.min(Math.abs(value), 1);
  const color = value > 0 
    ? `rgba(59, 130, 246, ${intensity})` 
    : value < 0 ? `rgba(239, 68, 68, ${intensity})` : 'transparent';
  
  return (
    <div 
      className={`
        relative ${size} flex items-center justify-center text-[10px] font-mono transition-all duration-300 border
        ${isHighlight ? 'border-yellow-400 border-2 z-10 scale-110' : 'border-slate-200 dark:border-slate-700'}
        ${value === 0 ? 'bg-slate-50 dark:bg-slate-900/50' : ''}
      `}
      style={{ backgroundColor: color }}
    >
      {showValue && (
        <span className={Math.abs(value) > 0.5 ? 'text-white' : 'text-slate-900 dark:text-slate-100'}>
          {value === 0 ? "0" : value.toFixed(1)}
        </span>
      )}
      {label && (
        <span className="absolute -bottom-4 text-[8px] text-slate-400 whitespace-nowrap">{label}</span>
      )}
    </div>
  );
};

const MatrixView = ({ data, title, subtitle, formula, highlightIndices = [], showValues = true, small = false }: any) => {
  return (
    <div className="flex flex-col items-center space-y-1">
      <div className="text-center">
        <h3 className={`font-bold text-slate-800 dark:text-slate-100 ${small ? 'text-[10px]' : 'text-sm'}`}>{title}</h3>
        {formula && <p className="text-[9px] font-mono text-indigo-500 mb-1">{formula}</p>}
        {subtitle && <p className="text-[10px] text-slate-500 dark:text-slate-400">{subtitle}</p>}
      </div>
      <div className="grid gap-px p-1 bg-slate-100 dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
        {data.map((row: any[], i: number) => (
          <div key={i} className="flex gap-px">
            {row.map((val, j) => (
              <HeatmapCell 
                key={`${i}-${j}`} 
                value={val} 
                showValue={showValues && !small}
                size={small ? "w-4 h-4" : "w-10 h-10"}
                isHighlight={highlightIndices.some(([r, c]: any) => r === i && c === j)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

const VectorView = ({ data, title, formula, orientation = 'vertical', showValues = true, labelPrefix = '', small = false, color = 'text-slate-600' }: any) => {
  return (
    <div className="flex flex-col items-center space-y-2">
       <div className="text-center">
         <h3 className={`font-bold text-[10px] uppercase tracking-wider ${color} dark:text-slate-300`}>{title}</h3>
         {formula && <p className="text-[9px] font-mono text-indigo-500 opacity-80">{formula}</p>}
       </div>
      <div className={`flex ${orientation === 'vertical' ? 'flex-col' : 'flex-row'} gap-px p-1 bg-slate-100 dark:bg-slate-800 rounded`}>
        {data.map((val: any, i: number) => (
          <HeatmapCell 
            key={i} 
            value={val} 
            showValue={showValues && !small} 
            size={small ? "w-4 h-4" : "w-10 h-10"}
            label={labelPrefix ? `${labelPrefix}${i}` : null} 
          />
        ))}
      </div>
    </div>
  );
};

const MathFormula = ({ tex }: { tex: string }) => (
  <span className="font-serif italic bg-slate-100 dark:bg-slate-800 px-1 rounded text-slate-800 dark:text-slate-200">
    {tex}
  </span>
);

export default function TTTDemo() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0); 
  const [subStep, setSubStep] = useState(0); 
  const [learningRate, setLearningRate] = useState(0.5); 
  const [showValues] = useState(true);
  
  const [sequence, setSequence] = useState<any[]>([]);
  const [weights, setWeights] = useState<number[][][]>([]); 
  const [initialW, setInitialW] = useState<number[][]>([]);
  
  const [thetaK, setThetaK] = useState<number[][]>([]);
  const [thetaV, setThetaV] = useState<number[][]>([]);
  const [thetaQ, setThetaQ] = useState<number[][]>([]);

  useEffect(() => {
    const seq = generateSequence();
    const w0 = zeroMat();
    setThetaK(randomMat());
    setThetaV(randomMat());
    setThetaQ(randomMat());
    setSequence(seq);
    setInitialW(w0);
    setWeights([w0]); 
  }, []);

  const currentW = weights[step] || initialW;
  const currentRawInput = sequence[step]?.u || Array(D).fill(0);

  const x_train = useMemo(() => thetaK.length ? matVecMul(thetaK, currentRawInput) : [], [thetaK, currentRawInput]);
  const z_train = useMemo(() => thetaV.length ? matVecMul(thetaV, currentRawInput) : [], [thetaV, currentRawInput]);
  const x_test  = useMemo(() => thetaQ.length ? matVecMul(thetaQ, currentRawInput) : [], [thetaQ, currentRawInput]);

  const reconstruction = useMemo(() => initialW.length && x_train.length ? matVecMul(initialW, x_train) : [], [initialW, x_train]);
  const errorVec = useMemo(() => reconstruction.length ? reconstruction.map((p, i) => p - z_train[i]) : [], [reconstruction, z_train]);
  const gradient = useMemo(() => errorVec.length ? outerProduct(errorVec, x_train) : [], [errorVec, x_train]);
  
  const nextW = useMemo(() => currentW.length ? matAdd(currentW, gradient, -learningRate) : [], [currentW, gradient, learningRate]);
  const finalOutput = useMemo(() => nextW.length && x_test.length ? matVecMul(nextW, x_test) : [], [nextW, x_test]);

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(handleNext, 1400);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step, subStep, weights, nextW]);

  const handleNext = () => {
    if (step >= SEQ_LEN) {
      setIsPlaying(false);
      return;
    }
    if (subStep < 3) {
      setSubStep(s => s + 1);
    } else {
      const newWeights = [...weights];
      newWeights[step + 1] = nextW;
      setWeights(newWeights);
      if (step < SEQ_LEN - 1) {
        setStep(s => s + 1);
        setSubStep(0);
      } else {
        setIsPlaying(false);
        setSubStep(4); 
      }
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    setStep(0);
    setSubStep(0);
    setWeights([initialW]);
  };

  const subStepLabels = [
    "1. Projections",
    "2. Reconstruction at W_0",
    "3. Batch GD Update (Linear Attention Path)",
    "4. Test Pass Output"
  ];

  if (!sequence.length || !thetaK.length) return <div className="p-10 text-center text-slate-500">Initializing...</div>;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100 font-sans pb-20">
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 p-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-600 p-2 rounded-lg text-white">
              <Activity size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">TTT-Linear: Unrolled RNN View</h1>
              <p className="text-xs text-slate-500 dark:text-slate-400 font-medium flex items-center gap-2">
                <span>Update:</span>
                <MathFormula tex="W_t = W_{t-1} - \eta \nabla \mathcal{L}(W_0; x_t)" />
                <span className="bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 px-2 py-0.5 rounded text-[10px] font-bold">BATCH GD MODE</span>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4 bg-slate-100 dark:bg-slate-800 p-2 rounded-xl">
             <div className="flex flex-col px-2">
                <span className="text-[10px] uppercase font-bold text-slate-400">Step η</span>
                <input 
                  type="range" min="0" max="1.0" step="0.05" 
                  value={learningRate} 
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-20 h-2 bg-slate-300 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
             </div>
             <button onClick={() => setIsPlaying(!isPlaying)} className="p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition">
               {isPlaying ? <Pause size={20} className="fill-current"/> : <Play size={20} className="fill-current"/>}
             </button>
             <button onClick={handleNext} disabled={step >= SEQ_LEN - 1 && subStep >= 3} className="p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition">
               <SkipForward size={20} />
             </button>
             <button onClick={handleReset} className="p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition text-red-500">
               <RotateCcw size={20} />
             </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 space-y-6">
        
        {/* Objective Section */}
        <section className="bg-indigo-900 text-white rounded-2xl p-6 shadow-xl border border-indigo-700 overflow-hidden relative">
           <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
              <Target size={120} />
           </div>
           <div className="relative z-10">
              <div className="flex items-center gap-2 mb-4">
                <Target className="text-indigo-300" size={20} />
                <h2 className="text-sm font-bold uppercase tracking-widest text-indigo-200">Test-Time Objective (Self-Supervision)</h2>
              </div>
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <div>
                  <p className="text-sm leading-relaxed text-indigo-100 mb-4">
                    In TTT-Linear, the hidden state <MathFormula tex="W" /> is a weight matrix. At each step <MathFormula tex="t" />, we perform gradient descent to solve a <strong>Reconstruction Task</strong>: 
                    predicting the value (V) from the key (K).
                  </p>
                  <div className="bg-indigo-950/50 p-4 rounded-xl border border-indigo-500/30">
                    <p className="text-xs font-mono mb-2 text-indigo-300 uppercase tracking-tighter font-bold">Self-supervised Loss:</p>
                    <div className="text-lg font-serif">
                       <MathFormula tex="\mathcal{L}(W; x_t) = \| W x_t - z_t \|^2" />
                    </div>
                  </div>
                </div>
                <div className="text-xs text-indigo-200/80 space-y-3 bg-white/5 p-4 rounded-xl">
                   <p><strong className="text-white">Goal:</strong> Find weights <MathFormula tex="W" /> that best map Key space to Value space for the current token.</p>
                   <p><strong className="text-white">Mechanism:</strong> Every state update <MathFormula tex="W_t" /> is literally a training step. We use <strong>Batch GD</strong>, meaning we always calculate the gradient relative to a fixed starting point <MathFormula tex="W_0" /> to maintain linearity.</p>
                   <p><strong className="text-white">Result:</strong> The "memory" of the RNN is stored in the weights of this tiny internal model.</p>
                </div>
              </div>
           </div>
        </section>

        {/* Unrolled RNN View */}
        <section className="bg-white dark:bg-slate-900 rounded-2xl p-6 shadow-sm border border-slate-200 dark:border-slate-800 overflow-x-auto">
          <div className="flex items-center gap-2 mb-6 border-b pb-3">
            <BookOpen className="text-indigo-500" size={18} />
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-500">Temporal Unrolling (Sequence Flow)</h2>
          </div>
          
          <div className="flex items-start gap-4 min-w-[900px] py-4">
             {sequence.map((item, idx) => {
               const isActive = idx === step;
               const isPast = idx < step;
               
               return (
                 <div key={item.id} className="flex items-center gap-4">
                    <div className={`flex flex-col items-center gap-3 p-4 rounded-xl border transition-all duration-500 ${isActive ? 'bg-indigo-50 border-indigo-200 scale-105 shadow-md ring-2 ring-indigo-400/20' : isPast ? 'bg-slate-50 border-slate-200 opacity-60' : 'bg-transparent border-dashed border-slate-200 opacity-30'}`}>
                       <span className={`text-[10px] font-bold ${isActive ? 'text-indigo-600' : 'text-slate-400'}`}>t = {idx}</span>
                       
                       <div className="flex flex-col gap-2">
                          <VectorView data={item.u} title="u" small showValues={false} />
                          <div className="h-4 flex items-center justify-center">
                            <ChevronRight size={12} className="rotate-90 text-slate-300" />
                          </div>
                          <MatrixView data={weights[idx] || initialW} title={`W_${idx}`} small showValues={false} />
                          <div className="h-4 flex items-center justify-center">
                            <ChevronRight size={12} className="rotate-90 text-slate-300" />
                          </div>
                          <VectorView 
                            data={idx === step ? finalOutput : (idx < step ? matVecMul(weights[idx+1], matVecMul(thetaQ, item.u)) : Array(D).fill(0))} 
                            title="y" 
                            small 
                            showValues={false} 
                            color={isActive ? "text-emerald-600" : "text-slate-400"}
                          />
                       </div>
                    </div>
                    {idx < SEQ_LEN - 1 && (
                      <div className="flex flex-col items-center">
                        <ArrowRight className={`transition-colors duration-500 ${isPast ? 'text-indigo-400' : 'text-slate-200'}`} size={20} />
                        <span className="text-[8px] mt-1 font-mono text-slate-400">W_t+1</span>
                      </div>
                    )}
                 </div>
               );
             })}
          </div>
        </section>

        {/* Detailed Step Visualization */}
        <div className="flex items-center justify-center">
          <span className={`px-6 py-2 rounded-full text-sm font-bold shadow-sm transition-all duration-500 border
            ${subStep === 0 ? 'bg-slate-100 text-slate-600' : 
              subStep === 1 ? 'bg-blue-50 text-blue-800 border-blue-200' :
              subStep === 2 ? 'bg-orange-50 text-orange-800 border-orange-200' :
              'bg-emerald-50 text-emerald-800 border-emerald-200'}
          `}>
            Step {step}: {subStepLabels[subStep]}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Projection Stage */}
          <div className="lg:col-span-3 flex flex-col gap-6">
             <div className="bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                <div className="text-xs font-bold uppercase text-slate-400 mb-4">Input Projections</div>
                <div className="space-y-4">
                  <VectorView data={x_train} title="Key (x_t)" formula="u θ_K" small color="text-indigo-600" />
                  <VectorView data={z_train} title="Value (z_t)" formula="u θ_V" small color="text-indigo-600" />
                  <VectorView data={x_test} title="Query (x_test)" formula="u θ_Q" small color="text-emerald-600" />
                </div>
             </div>
          </div>

          {/* Core State Update */}
          <div className="lg:col-span-6 flex flex-col gap-6">
            <div className="bg-white dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-800 shadow-md relative">
               <div className="flex items-center gap-2 mb-6 border-b pb-3">
                 <Zap size={18} className="text-orange-500" />
                 <h2 className="text-sm font-bold uppercase tracking-wider">Inner Loop Mechanism</h2>
               </div>

               {/* Forward at W0 */}
               <div className={`flex justify-center items-center gap-4 mb-10 transition-all duration-300 ${subStep >= 1 ? 'opacity-100' : 'opacity-20'}`}>
                  <MatrixView data={initialW} title="W_0" subtitle="Initialization (Zero)" showValues={false} />
                  <span className="text-xl font-bold text-slate-300">·</span>
                  <VectorView data={x_train} title="x_t" showValues={false} small />
                  <ArrowRight size={20} className="text-slate-300" />
                  <div className="flex flex-col items-center gap-2">
                    <VectorView data={reconstruction} title="f(x_t)" formula="W_0 x_t = 0" showValues={false} color="text-slate-400" />
                  </div>
               </div>

               {/* Update Rule */}
               {subStep >= 2 && (
                 <div className="bg-orange-50/50 dark:bg-orange-950/10 p-4 rounded-lg border border-orange-100 dark:border-orange-900 animate-in fade-in slide-in-from-bottom-2">
                    <div className="flex items-center justify-between mb-2">
                       <div className="flex flex-col">
                         <span className="text-[10px] font-bold text-orange-600 uppercase">Weight Accumulation (Batch GD Mode)</span>
                         <code className="text-[11px] font-mono text-orange-800 dark:text-orange-300">ΔW = -η ∇L(W_0; x_t) = η (z_t x_tᵀ)</code>
                       </div>
                       <div className="flex items-center gap-1 bg-blue-100 dark:bg-blue-900/50 px-2 py-1 rounded border border-blue-200 text-blue-700 dark:text-blue-300">
                          <Sparkles size={12} />
                          <span className="text-[9px] font-bold uppercase tracking-tighter">Linear Attention Identity</span>
                       </div>
                    </div>

                    <p className="text-[10px] text-slate-500 italic mb-4">
                      When <MathFormula tex="W_0=0" />, the gradient update is <strong>identical</strong> to the Key-Value outer product used in Linear Attention.
                    </p>
                    
                    <div className="flex justify-center items-center gap-4">
                       <div className="group relative">
                        <VectorView data={z_train} title="Value (z_t)" small showValues={false} />
                        <span className="absolute -top-4 left-0 text-[8px] font-bold text-blue-500 whitespace-nowrap opacity-100 transition-opacity">v_t</span>
                       </div>
                       <span className="text-lg text-slate-300">⊗</span>
                       <div className="group relative">
                        <VectorView data={x_train} title="Key (x_tᵀ)" orientation="horizontal" small showValues={false} />
                        <span className="absolute -top-4 right-0 text-[8px] font-bold text-blue-500 whitespace-nowrap opacity-100 transition-opacity">k_tᵀ</span>
                       </div>
                       <ArrowRight size={16} className="text-slate-300" />
                       <MatrixView data={gradient} title="ΔW" small showValues={false} />
                    </div>

                    <div className="mt-6 pt-4 border-t border-orange-100 text-center">
                       <div className="inline-block bg-white dark:bg-slate-800 px-4 py-2 rounded-lg border border-orange-200 shadow-sm">
                         <p className="text-xs font-mono font-bold text-indigo-600">
                           W_{step + 1} = W_t + η (z_t x_tᵀ)
                         </p>
                         <p className="text-[8px] uppercase tracking-widest text-slate-400 mt-1">This is exactly the Linear Transformers memory update</p>
                       </div>
                    </div>
                 </div>
               )}
            </div>
          </div>

          {/* Generation Pass */}
          <div className="lg:col-span-3 flex flex-col gap-6">
             <div className={`bg-emerald-50 dark:bg-emerald-950/10 p-6 rounded-xl border border-emerald-100 dark:border-emerald-900 h-full transition-all duration-500 shadow-sm ${subStep === 3 ? 'opacity-100 ring-2 ring-emerald-400' : 'opacity-20 grayscale'}`}>
                <div className="flex items-center gap-2 mb-6">
                   <Box size={20} className="text-emerald-600" />
                   <h3 className="font-bold text-emerald-800 dark:text-emerald-200 uppercase text-xs">Final Result</h3>
                </div>

                <div className="flex flex-col items-center gap-6">
                   <MatrixView data={nextW} title={`W_{${step+1}}`} formula="New State" small showValues={false} />
                   <span className="text-emerald-500 font-bold">·</span>
                   <VectorView data={x_test} title="Query (x_test)" showValues={false} color="text-emerald-600" />
                   <ArrowRight className="rotate-90 text-emerald-500" />
                   <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-emerald-200">
                      <VectorView data={finalOutput} title="y_t" formula="f_W(x_test)" showValues={showValues} color="text-emerald-700" />
                   </div>
                </div>
             </div>
          </div>
        </div>

        {/* Theoretical Footer */}
        <section className="bg-slate-900 text-slate-300 p-6 rounded-xl border border-slate-800">
           <div className="flex items-center gap-2 mb-4 text-white">
             <Info size={16} />
             <h4 className="font-bold text-sm uppercase tracking-widest">The Batch GD / Linear Attention Duality</h4>
           </div>
           <p className="text-xs leading-relaxed mb-4">
             In TTT-Linear, calculating the gradient at the <strong>initial weights</strong> (<MathFormula tex="W_0" />) for every step $t$ is equivalent to <strong>Batch Gradient Descent</strong> on a dataset size of 1. Because <MathFormula tex="W_0 = 0" />, the state $W_t$ becomes an unbiased accumulation of independent features:
           </p>
           <div className="grid md:grid-cols-2 gap-4 text-[10px] font-mono opacity-80">
              <div className="p-3 bg-slate-950 border border-indigo-900/50 rounded flex flex-col gap-1">
                <span className="text-indigo-400 font-bold uppercase text-[9px]">TTT Formulation (Dual)</span>
                <span>{'W_t = W_{t-1} + \\eta (z_t x_t^\\top)'}</span>
                <span className="text-slate-500 mt-1 italic">Uses static reference W_0 to compute grad.</span>
              </div>
              <div className="p-3 bg-slate-950 border border-emerald-900/50 rounded flex flex-col gap-1">
                <span className="text-emerald-400 font-bold uppercase text-[9px]">Linear Attention Formulation</span>
                <span>{'y_t = \\sum_{i=1}^t v_i (k_i^\\top q_t)'}</span>
                <span className="text-slate-500 mt-1 italic">Equivalent to matrix-vector associative memory.</span>
              </div>
           </div>
        </section>
      </main>
    </div>
  );
}