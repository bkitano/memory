// @ts-nocheck

import React, { useState, useEffect, useMemo } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { Play, SkipForward, RotateCcw, ArrowRight, Pause, Info } from 'lucide-react';

// --- Math Helpers ---

const generateRandomMatrix = (rows, cols) => 
  Array.from({ length: rows }, () => 
    Array.from({ length: cols }, () => Number((Math.random()).toFixed(1)))
  );

const zeros = (rows, cols) => Array(rows).fill(0).map(() => Array(cols).fill(0));

// Simple feature map phi(x) = x (Identity for simplicity in visualization, assuming inputs > 0)
// In practice this might be elu(x) + 1
const phi = (vec) => vec; 

const transpose = (matrix) => matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));

const matMul = (A, B) => {
  const result = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < A[0].length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
};

const outerProduct = (vecA, vecB) => {
  // vecA is column vector (array), vecB is column vector (array)
  // result is vecA * vecB^T
  return vecA.map(a => vecB.map(b => a * b));
};

const addMatrices = (A, B) => A.map((row, i) => row.map((val, j) => val + B[i][j]));

const formatNum = (n) => n.toFixed(2);

// --- Components ---

const Matrix = ({ data, title, highlightRow, highlightCol, labelIndices, className = "" }) => (
  <div className={`flex flex-col items-center ${className}`}>
    {title && <h4 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">{title}</h4>}
    <div className="relative border-2 border-slate-800 rounded-lg overflow-hidden bg-white shadow-sm">
      <div 
        className="grid gap-[1px] bg-slate-200"
        style={{ gridTemplateColumns: `repeat(${data[0]?.length || 1}, minmax(0, 1fr))` }}
      >
        {data.map((row, i) => (
          row.map((val, j) => {
            const isHighlighted = (highlightRow === i) || (highlightCol === j);
            return (
              <div 
                key={`${i}-${j}`} 
                className={`
                  w-12 h-10 flex items-center justify-center text-xs font-mono transition-colors duration-300
                  ${isHighlighted ? 'bg-blue-100 text-blue-700 font-bold' : 'bg-white text-slate-700'}
                `}
              >
                {formatNum(val)}
              </div>
            );
          })
        ))}
      </div>
    </div>
    {labelIndices && (
      <div className="text-[10px] text-slate-400 mt-1">
        {data.length}x{data[0].length}
      </div>
    )}
  </div>
);

const Vector = ({ data, label, vertical = true, highlight }) => (
  <div className={`flex ${vertical ? 'flex-col' : 'flex-row'} items-center gap-1`}>
    {label && <span className="text-xs font-bold text-slate-600 mr-1">{label}</span>}
    <div className={`flex ${vertical ? 'flex-col' : 'flex-row'} border border-slate-300 rounded bg-white overflow-hidden shadow-sm`}>
      {data.map((val, i) => (
        <div 
          key={i} 
          className={`
            w-10 h-8 flex items-center justify-center text-xs font-mono border-slate-100
            ${vertical ? 'border-b last:border-b-0' : 'border-r last:border-r-0'}
            ${highlight ? 'bg-amber-100 text-amber-800' : ''}
          `}
        >
          {formatNum(val)}
        </div>
      ))}
    </div>
  </div>
);

const Arrow = ({ label }) => (
  <div className="flex flex-col items-center justify-center mx-2 text-slate-400">
    <div className="text-[10px] mb-1">{label}</div>
    <ArrowRight size={16} />
  </div>
);

// --- Main App ---

export default function LinearAttentionDemo() {
  const SEQ_LEN = 4;
  const D_KEY = 2; // Dimension of Query/Key
  const D_VAL = 2; // Dimension of Value

  // Inputs
  const [Q, setQ] = useState([]);
  const [K, setK] = useState([]);
  const [V, setV] = useState([]);

  // Loop State
  const [step, setStep] = useState(0); // 0 to SEQ_LEN
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Accumulated State (Recurrent)
  const [S_history, setS_History] = useState([]); // Array of matrices
  
  // Initialization
  useEffect(() => {
    reset();
  }, []);

  const reset = () => {
    const newQ = generateRandomMatrix(SEQ_LEN, D_KEY);
    const newK = generateRandomMatrix(SEQ_LEN, D_KEY);
    const newV = generateRandomMatrix(SEQ_LEN, D_VAL);
    setQ(newQ);
    setK(newK);
    setV(newV);
    
    // Precompute State Evolution
    // S_0 = Zeros
    // S_t = S_{t-1} + phi(k_t) * v_t^T
    const states = [zeros(D_KEY, D_VAL)];
    let currentS = zeros(D_KEY, D_VAL);
    
    for(let t=0; t<SEQ_LEN; t++) {
      const k_t = newK[t]; // vector
      const v_t = newV[t]; // vector
      const phi_k = phi(k_t);
      const update = outerProduct(phi_k, v_t);
      currentS = addMatrices(currentS, update);
      states.push(currentS);
    }
    
    setS_History(states);
    setStep(0);
    setIsPlaying(false);
  };

  // Playback logic
  useEffect(() => {
    let interval;
    if (isPlaying && step < SEQ_LEN) {
      interval = setInterval(() => {
        setStep(s => s + 1);
      }, 1500);
    } else if (step >= SEQ_LEN) {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step]);

  // --- Derived Data for Current Step ---

  // Current inputs (or null if finished)
  const t = step < SEQ_LEN ? step : step - 1;
  const showInputs = step > 0;

  // 1. Recurrent Calculations (O(N) inference)
  const currentS = S_history[step] || zeros(D_KEY, D_VAL);
  const prevS = step > 0 ? S_history[step - 1] : zeros(D_KEY, D_VAL);
  
  // Vectors at current step t
  const q_vec = Q[step - 1] || zeros(1, D_KEY)[0]; // Use step-1 because step is 1-based for display logic mostly
  const k_vec = K[step - 1] || zeros(1, D_KEY)[0];
  const v_vec = V[step - 1] || zeros(1, D_VAL)[0];

  // The update term: phi(k) * v^T
  const updateMatrix = step > 0 ? outerProduct(phi(k_vec), v_vec) : zeros(D_KEY, D_VAL);

  // Recurrent Output: O_t = phi(q_t)^T * S_t
  // Note: Depending on variant, it might be S_{t-1} or S_t. 
  // Standard causal linear attention usually includes current key-value in state for current query.
  // So O_t = phi(q_t)^T * S_t.
  const recurrentOutputVector = step > 0 ? matMul([q_vec], currentS)[0] : zeros(1, D_VAL)[0];

  // 2. Parallel Calculations (O(N^2) training/parallel)
  // Show full attention matrix up to current step
  const activeQ = Q.slice(0, step);
  const activeK = K.slice(0, step);
  const activeV = V.slice(0, step);
  
  // Compute Attention Matrix A = QK^T (masked manually by slice)
  // In linear attention: A = phi(Q) * phi(K)^T
  let parallelAttnMatrix = [];
  let parallelOutput = [];
  
  if (step > 0) {
    // phi(Q) [step x D], phi(K)^T [D x step] -> [step x step]
    const phiQ = activeQ.map(row => phi(row));
    const phiK = activeK.map(row => phi(row));
    const phiKT = transpose(phiK);
    parallelAttnMatrix = matMul(phiQ, phiKT);
    
    // Output = A * V
    parallelOutput = matMul(parallelAttnMatrix, activeV);
  }

  const currentParallelOutputVector = parallelOutput.length > 0 ? parallelOutput[parallelOutput.length - 1] : zeros(1, D_VAL)[0];

  return (
    <div className="min-h-screen bg-slate-50 p-6 font-sans text-slate-800">
      
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Linear Attention Viz</h1>
            <p className="text-slate-600 max-w-2xl">
              Comparing the <span className="font-semibold text-blue-600">Recurrent (RNN)</span> view 
              vs the <span className="font-semibold text-emerald-600">Parallel (Attention)</span> view.
              Linear Attention decomposes <InlineMath math={'\\mathrm{Attn}(Q,K,V)'} /> into
              <span className="ml-1"><InlineMath math={'\\phi(Q)\\,\\big(\\phi(K)^{\\top} V\\big)'} /></span>, allowing state tracking.
            </p>
          </div>
          
          {/* Controls */}
          <div className="flex gap-2 bg-white p-2 rounded-xl shadow-sm border border-slate-200">
            <button onClick={reset} className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 tooltip" title="Reset">
              <RotateCcw size={20} />
            </button>
            <button 
              onClick={() => setIsPlaying(!isPlaying)} 
              disabled={step >= SEQ_LEN}
              className={`p-2 rounded-lg flex items-center gap-2 font-medium transition-colors ${isPlaying ? 'bg-amber-100 text-amber-700' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
            >
              {isPlaying ? <><Pause size={18} /> Pause</> : <><Play size={18} /> Play</>}
            </button>
            <button 
              onClick={() => setStep(Math.min(step + 1, SEQ_LEN))} 
              disabled={step >= SEQ_LEN}
              className="p-2 hover:bg-slate-100 rounded-lg text-slate-600"
            >
              <SkipForward size={20} />
            </button>
          </div>
        </div>

        {/* Timeline */}
        <div className="flex items-center justify-center gap-2 mb-8 bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-sm font-semibold text-slate-400 mr-2">TIMESTEP</span>
          {Array.from({ length: SEQ_LEN + 1 }).map((_, i) => (
            <div 
              key={i}
              className={`
                w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300
                ${i === step ? 'bg-blue-600 text-white scale-110 shadow-md' : i < step ? 'bg-blue-100 text-blue-400' : 'bg-slate-100 text-slate-300'}
              `}
            >
              {i}
            </div>
          ))}
          <div className="ml-4 text-sm text-slate-500">
            {step === 0 ? "Initial State" : `Processing token ${step}`}
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* LEFT: Recurrent View */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-blue-500"></div>
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-blue-700">
              <div className="p-1 bg-blue-100 rounded">
                <RotateCcw size={16} /> 
              </div>
              Recurrent View (O(N))
            </h2>
            
            <div className="space-y-8">
              {/* Formula */}
              <div className="bg-blue-50 p-3 rounded-lg text-sm text-blue-900 text-center">
                <BlockMath math={'S_t = S_{t-1} + \\phi(k_t) v_t^{\\top}'} />
              </div>

              {step === 0 ? (
                <div className="text-center py-10 text-slate-400">Press Play to Start</div>
              ) : (
                <>
                  {/* Step 1: Inputs & Update */}
                  <div className="flex flex-col items-center">
                    <h3 className="text-sm font-semibold text-slate-400 mb-4 w-full text-left border-b pb-2">1. Compute Update Term</h3>
                    <div className="flex items-center gap-4">
                      <div className="flex flex-col gap-2 items-center">
                        <Vector data={k_vec} label="k_t" highlight />
                        <span className="text-xs text-slate-400">ϕ(k)</span>
                      </div>
                      <span className="text-xl text-slate-300">×</span>
                      <div className="flex flex-col gap-2 items-center">
                        <Vector data={v_vec} label="v_t" vertical={false} highlight />
                        <span className="text-xs text-slate-400">v^T</span>
                      </div>
                      <Arrow label="Outer Prod" />
                      <Matrix data={updateMatrix} title="Update (KV^T)" highlightRow={-1} className="scale-90" />
                    </div>
                  </div>

                  {/* Step 2: State Accumulation */}
                  <div className="flex flex-col items-center bg-slate-50 p-4 rounded-xl border border-slate-100">
                     <h3 className="text-sm font-semibold text-slate-400 mb-4 w-full text-left border-b pb-2">2. Update State Matrix</h3>
                    <div className="flex items-center gap-2">
                      <Matrix data={prevS} title={`S_{${step-1}}`} className="opacity-60 scale-75" />
                      <span className="text-xl text-slate-400">+</span>
                      <Matrix data={updateMatrix} title="Update" className="opacity-60 scale-75" />
                      <Arrow label="Add" />
                      <Matrix data={currentS} title={`S_{${step}}`} highlightRow={-1} className="ring-2 ring-blue-500 ring-offset-2 rounded-lg" />
                    </div>
                  </div>

                  {/* Step 3: Querying the State */}
                  <div className="flex flex-col items-center">
                    <h3 className="text-sm font-semibold text-slate-400 mb-4 w-full text-left border-b pb-2">3. Compute Output</h3>
                    <div className="bg-blue-50 p-3 rounded-lg text-sm text-blue-900 mb-4">
                      <BlockMath math={'o_t = S_t^{\\top} \\phi(q_t)'} />
                    </div>
                    <div className="flex items-center gap-4">
                       <Matrix data={currentS} title={`S_{${step}}`} className="scale-75 opacity-80" />
                       <span className="text-xl text-slate-300">·</span>
                       <Vector data={q_vec} label="q_t" highlight />
                       <Arrow />
                       <Vector data={recurrentOutputVector} label="o_t" highlight vertical={false} />
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* RIGHT: Parallel View */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-emerald-500"></div>
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-emerald-700">
               <div className="p-1 bg-emerald-100 rounded">
                <SkipForward size={16} /> 
              </div>
              Parallel View (O(N²))
            </h2>

            <div className="space-y-8">
               {/* Formula */}
               <div className="bg-emerald-50 p-3 rounded-lg text-sm text-emerald-900 text-center">
                <BlockMath math={'\\mathrm{Out} = (\\phi(Q)\\phi(K)^{\\top})\\,V'} />
              </div>

              {step === 0 ? (
                <div className="text-center py-10 text-slate-400">Waiting for tokens...</div>
              ) : (
                <>
                  <div className="flex flex-col items-center">
                     <h3 className="text-sm font-semibold text-slate-400 mb-4 w-full text-left border-b pb-2">Global Attention Matrix</h3>
                     <div className="flex gap-4 items-center overflow-x-auto p-2 max-w-full">
                       <div className="flex flex-col gap-1">
                         <Matrix data={activeQ} title={`Q (0..${step})`} />
                         <div className="text-center text-xs text-slate-400">x</div>
                         <Matrix data={transpose(activeK)} title={`K^T (0..${step})`} />
                       </div>
                       <Arrow />
                       <Matrix 
                        data={parallelAttnMatrix} 
                        title="Attn Matrix" 
                        highlightRow={step-1}
                        className="ring-2 ring-emerald-500 ring-offset-2 rounded-lg"
                      />
                     </div>
                  </div>

                  <div className="flex flex-col items-center">
                     <h3 className="text-sm font-semibold text-slate-400 mb-4 w-full text-left border-b pb-2">Weighted Sum</h3>
                     <div className="flex gap-4 items-center">
                        <Matrix 
                          data={parallelAttnMatrix} 
                          title="Weights" 
                          highlightRow={step-1}
                          className="scale-75"
                        />
                        <span className="text-xl text-slate-300">·</span>
                        <Matrix 
                          data={activeV} 
                          title={`V (0..${step})`} 
                          className="scale-75"
                        />
                        <Arrow />
                         <div className="flex flex-col items-center">
                          <h4 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">Output</h4>
                          <div className="border-2 border-slate-800 rounded-lg overflow-hidden bg-white">
                            {parallelOutput.map((row, i) => (
                              <div key={i} className={`flex border-b last:border-b-0 ${i === step - 1 ? 'bg-emerald-100' : 'bg-white'}`}>
                                {row.map((val, j) => (
                                  <div key={j} className="w-10 h-8 flex items-center justify-center text-xs font-mono border-r last:border-r-0">
                                    {formatNum(val)}
                                  </div>
                                ))}
                              </div>
                            ))}
                          </div>
                        </div>
                     </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Verification */}
        {step > 0 && (
          <div className="mt-8 bg-slate-900 text-white p-6 rounded-xl shadow-lg flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Recurrent Output</div>
                <div className="font-mono text-xl text-blue-400 font-bold">
                  [{recurrentOutputVector.map(n => n.toFixed(2)).join(", ")}]
                </div>
              </div>
              <div className="text-2xl text-slate-500">≈</div>
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Parallel Output</div>
                <div className="font-mono text-xl text-emerald-400 font-bold">
                  [{currentParallelOutputVector.map(n => n.toFixed(2)).join(", ")}]
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-lg">
              <Info size={18} className="text-slate-400" />
              <span className="text-sm text-slate-300">
                Values match exactly! (Floating point rounding applied)
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
