import React, { useState, useEffect, useMemo } from 'react';
import { Play, Pause, SkipBack, ArrowRight, RefreshCw, Info, Calculator, Database, BrainCircuit } from 'lucide-react';

// --- Types & Constants ---
type Vector = number[];
type Matrix = number[][];

const DIM = 2; // We work in 2D for easy visualization
const SEQ_LEN = 3; // Length of the sequence

// Initial random data
const INITIAL_KEYS: Vector[] = [[0.8, 0.2], [0.3, 0.9], [0.6, 0.6]];
const INITIAL_VALUES: Vector[] = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
const INITIAL_QUERY: Vector = [0.8, 0.2];

// Kernel function phi(x) - using ReLU to keep it positive and simple for visual comprehension.
const phi = (v: Vector): Vector => v.map(x => Math.max(0, x));

// --- Math Helpers ---

// Outer product of two vectors: v * k^T
const outerProduct = (v: Vector, k: Vector): Matrix => {
  return v.map(vi => k.map(kj => vi * kj));
};

// Matrix addition
const addMatrix = (m1: Matrix, m2: Matrix): Matrix => {
  return m1.map((row, i) => row.map((val, j) => val + m2[i][j]));
};

// Matrix-Vector multiplication
const matVecMul = (m: Matrix, v: Vector): Vector => {
  return m.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
};

// Vector addition
const addVector = (v1: Vector, v2: Vector): Vector => {
  return v1.map((val, i) => val + v2[i]);
};

// Dot product
const dot = (v1: Vector, v2: Vector): number => {
  return v1.reduce((sum, val, i) => sum + val * v2[i], 0);
};

// Scale vector
const scaleVector = (v: Vector, s: number): Vector => {
  return v.map(val => val * s);
};

// Create zero matrix
const zeroMatrix = (): Matrix => Array(DIM).fill(0).map(() => Array(DIM).fill(0));
const zeroVector = (): Vector => Array(DIM).fill(0);

// --- Components ---

const HeatmapCell = ({ value, label }: { value: number; label?: string }) => {
  // Guard against non-number values to prevent React rendering errors
  if (typeof value !== 'number' || isNaN(value)) {
    return <div className="w-12 h-12 m-0.5 bg-gray-100" />;
  }

  // Color scale: Blue (neg) -> White (0) -> Red (pos)
  const intensity = Math.min(1, Math.abs(value) / 1.5);
  const bgColor = value >= 0 
    ? `rgba(59, 130, 246, ${intensity})` // Blue
    : `rgba(239, 68, 68, ${intensity})`; // Red
  
  const textColor = intensity > 0.5 ? 'white' : 'black';

  return (
    <div className="flex flex-col items-center justify-center w-12 h-12 m-0.5 rounded border border-slate-200 text-xs font-mono transition-colors duration-300"
         style={{ backgroundColor: bgColor, color: textColor }}>
      {value.toFixed(2)}
      {label && <span className="text-[8px] opacity-75">{label}</span>}
    </div>
  );
};

const MatrixVis = ({ matrix, title, label }: { matrix: Matrix; title?: string, label?: string }) => {
  if (!matrix || !Array.isArray(matrix)) return null;
  
  return (
    <div className="flex flex-col items-center mx-2">
      {title && <div className="text-xs font-bold text-slate-500 mb-1">{title}</div>}
      <div className="flex flex-col border-2 border-slate-800 rounded p-1 bg-white shadow-sm">
        {matrix.map((row, i) => (
          <div key={i} className="flex">
            {Array.isArray(row) && row.map((val, j) => (
              <HeatmapCell key={j} value={val} />
            ))}
          </div>
        ))}
      </div>
      {label && <div className="mt-1 text-xs font-mono text-blue-600">{label}</div>}
    </div>
  );
};

const VectorVis = ({ vector, title, vertical = true, editable = false, onChange }: { vector: Vector; title?: string; vertical?: boolean; editable?: boolean; onChange?: (v: Vector) => void }) => {
  // Guard against invalid vector data
  if (!vector || !Array.isArray(vector)) return null;

  const handleChange = (idx: number, val: string) => {
    if (!onChange) return;
    const num = parseFloat(val);
    if (isNaN(num)) return;
    const newVec = [...vector];
    newVec[idx] = num;
    onChange(newVec);
  };

  return (
    <div className="flex flex-col items-center mx-1">
      {title && <div className="text-xs font-bold text-slate-500 mb-1">{title}</div>}
      <div className={`flex ${vertical ? 'flex-col' : 'flex-row'} border border-slate-300 rounded p-1 bg-white`}>
        {vector.map((val, i) => (
          editable ? (
            <input
              key={i}
              type="number"
              step="0.1"
              className="w-12 h-12 m-0.5 text-center text-xs border rounded bg-slate-50 focus:ring-2 focus:ring-blue-400 outline-none"
              value={typeof val === 'number' ? val : 0}
              onChange={(e) => handleChange(i, e.target.value)}
            />
          ) : (
            <HeatmapCell key={i} value={val} />
          )
        ))}
      </div>
    </div>
  );
};

// --- Main Application ---

export default function LinearTransformerDemo() {
  // State
  const [keys, setKeys] = useState<Vector[]>(INITIAL_KEYS);
  const [values, setValues] = useState<Vector[]>(INITIAL_VALUES);
  const [query, setQuery] = useState<Vector>(INITIAL_QUERY);
  const [step, setStep] = useState<number>(0); // For FWP animation (0 to SEQ_LEN)
  const [isPlaying, setIsPlaying] = useState(false);

  // --- Calculations ---

  // 1. Transformer (Global) View Calculations
  const phiKeys = useMemo(() => keys.map(k => phi(k)), [keys]);
  const phiQuery = useMemo(() => phi(query), [query]);

  // Compute global matrix M (Numerator)
  const globalMatrix = useMemo(() => {
    let M = zeroMatrix();
    for (let i = 0; i < SEQ_LEN; i++) {
      const outer = outerProduct(values[i], phiKeys[i]); // V * K^T
      M = addMatrix(M, outer);
    }
    return M;
  }, [values, phiKeys]);

  // Compute global normalizer Z (Denominator)
  const globalZ = useMemo(() => {
    let z = zeroVector();
    for (let i = 0; i < SEQ_LEN; i++) {
      z = addVector(z, phiKeys[i]);
    }
    return z;
  }, [phiKeys]);

  // Global Output
  const globalOutputNumerator = matVecMul(globalMatrix, phiQuery);
  const globalOutputDenominator = dot(globalZ, phiQuery);
  const globalOutput = scaleVector(globalOutputNumerator, 1 / (globalOutputDenominator || 1));


  // 2. FWP (RNN) View Calculations (Step-by-step)
  // State at current 'step'
  const fwpState = useMemo(() => {
    let W = zeroMatrix(); // Fast Weights
    let z = zeroVector(); // Normalizer state
    
    // Process up to current step
    for (let i = 0; i < step; i++) {
      const update = outerProduct(values[i], phiKeys[i]);
      W = addMatrix(W, update); // Hebbian Update: W <- W + v * phi(k)^T
      z = addVector(z, phiKeys[i]); // Accumulate normalizer
    }
    return { W, z };
  }, [step, values, phiKeys]);

  const fwpOutputNumerator = matVecMul(fwpState.W, phiQuery);
  const fwpOutputDenominator = dot(fwpState.z, phiQuery);
  const fwpOutput = scaleVector(fwpOutputNumerator, 1 / (fwpOutputDenominator || 1));


  // --- Animation Loop ---
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setStep(s => {
          if (s >= SEQ_LEN) {
            setIsPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // --- Handlers ---
  const handleReset = () => {
    setKeys(INITIAL_KEYS.map(k => k.map(() => Math.random())));
    setValues(INITIAL_VALUES.map(v => v.map(() => Math.random())));
    setStep(0);
  };

  const handleUpdateVector = (setter: React.Dispatch<React.SetStateAction<Vector[]>>, idx: number, newVec: Vector) => {
    setter(prev => {
      const next = [...prev];
      next[idx] = newVec;
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-4 md:p-8">
      
      {/* Header */}
      <header className="max-w-6xl mx-auto mb-8">
        <h1 className="text-3xl font-bold text-slate-800 flex items-center gap-3">
          <BrainCircuit className="w-8 h-8 text-blue-600" />
          Linear Transformers = Fast Weight Programmers
        </h1>
        <p className="mt-2 text-slate-600 max-w-2xl">
          An interactive demonstration of the equivalence proved by <i>Schlag et al. (2021)</i>.
          Linear self-attention is mathematically identical to a Recurrent Neural Network (RNN) 
          that updates a matrix-valued hidden state (Fast Weights) using an outer-product rule.
        </p>
      </header>

      {/* Main Grid */}
      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* --- Input Section (Top) --- */}
        <div className="lg:col-span-12 bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="flex flex-wrap items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Database className="w-5 h-5 text-slate-500" />
              Sequence Data (Inputs)
            </h2>
            <div className="flex gap-2">
               <button onClick={handleReset} className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-100 hover:bg-slate-200 rounded-md transition-colors text-slate-700 font-medium">
                <RefreshCw className="w-4 h-4" /> Randomize
              </button>
            </div>
          </div>
          
          <div className="flex flex-wrap gap-8 items-start overflow-x-auto pb-2">
            {/* Input Sequence */}
            <div className="flex gap-4">
              {keys.map((k, i) => (
                <div key={i} className="flex flex-col items-center p-3 bg-slate-50 rounded-lg border border-slate-100 relative group">
                  <div className="absolute -top-2 left-1/2 -translate-x-1/2 bg-slate-200 px-2 rounded-full text-[10px] font-bold text-slate-600">t={i+1}</div>
                  <div className="flex gap-4">
                    <VectorVis 
                      vector={k} 
                      title={`Key ${i+1}`} 
                      editable 
                      onChange={(v) => handleUpdateVector(setKeys, i, v)} 
                    />
                    <VectorVis 
                      vector={values[i]} 
                      title={`Value ${i+1}`} 
                      editable 
                      onChange={(v) => handleUpdateVector(setValues, i, v)} 
                    />
                  </div>
                  {/* Outer product visual hint */}
                  <div className="mt-2 text-[10px] text-slate-400 font-mono">
                    Add: V{i+1} ⊗ ϕ(K{i+1})
                  </div>
                </div>
              ))}
            </div>

            {/* Query */}
            <div className="flex flex-col items-center p-3 bg-blue-50 rounded-lg border border-blue-100 ml-4">
              <div className="text-xs font-bold text-blue-600 mb-2">Query (at test time)</div>
              <VectorVis vector={query} title="Q" editable onChange={setQuery} />
            </div>
          </div>
        </div>

        {/* --- Left Column: Linear Transformer View --- */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex-1 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500"></div>
            <h2 className="text-lg font-bold text-slate-800 mb-2 flex items-center gap-2">
              <span className="bg-purple-100 text-purple-700 p-1 rounded">Parallel</span>
              Linear Transformer
            </h2>
            <p className="text-sm text-slate-500 mb-6">
              Computes the memory matrix <span className="font-mono font-bold">M</span> by summing all outer products at once.
            </p>

            <div className="flex flex-col items-center gap-6">
              {/* Formula */}
              <div className="bg-slate-50 p-3 rounded border border-slate-100 text-center">
                 <div className="font-mono text-sm text-slate-700">
                   M = <span className="text-purple-600">∑</span> (V<sub>i</sub> ⊗ ϕ(K<sub>i</sub>))
                 </div>
              </div>

              {/* The Matrix */}
              <div className="flex items-center gap-4 animate-in fade-in duration-700">
                <MatrixVis matrix={globalMatrix} title="Global Memory M" label="Sum of all V*K'" />
                <div className="text-2xl text-slate-300">×</div>
                <VectorVis vector={phiQuery} title="ϕ(Q)" />
                <div className="text-2xl text-slate-300">=</div>
                <VectorVis vector={globalOutputNumerator} title="Num." />
              </div>

              {/* Normalization part */}
              <div className="w-full border-t border-slate-100 pt-4 mt-2">
                 <div className="text-xs font-bold text-slate-400 mb-2">NORMALIZATION TERM</div>
                 <div className="flex items-center justify-center gap-4">
                    <div className="font-mono text-xs text-slate-500">Z = ∑ ϕ(K<sub>i</sub>)</div>
                    <ArrowRight className="w-4 h-4 text-slate-300" />
                    <VectorVis vector={globalZ} title="Sum ϕ(K)" vertical={false} />
                 </div>
              </div>
            </div>
          </div>
        </div>

        {/* --- Right Column: FWP (RNN) View --- */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex-1 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-cyan-500"></div>
            <div className="flex justify-between items-start mb-2">
              <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <span className="bg-blue-100 text-blue-700 p-1 rounded">Recurrent</span>
                Fast Weight Programmer
              </h2>
            </div>
            
            {/* Controls */}
            <div className="flex items-center gap-4 mb-6 bg-slate-50 p-2 rounded-lg border border-slate-100">
              <button 
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-2 rounded-full bg-blue-600 text-white hover:bg-blue-700 transition shadow-sm"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              
              <div className="flex items-center gap-2 flex-1">
                <span className="text-xs font-bold text-slate-500 uppercase">Time Step t={step}</span>
                <input 
                  type="range" 
                  min="0" 
                  max={SEQ_LEN} 
                  value={step} 
                  onChange={(e) => setStep(parseInt(e.target.value))}
                  className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>

              <button onClick={() => setStep(0)} className="p-2 text-slate-400 hover:text-slate-600">
                <SkipBack className="w-4 h-4" />
              </button>
            </div>

            <div className="flex flex-col items-center gap-6">
               {/* Formula */}
               <div className="bg-slate-50 p-3 rounded border border-slate-100 text-center">
                 <div className="font-mono text-sm text-slate-700">
                   W<sub>t</sub> = W<sub>t-1</sub> + (V<sub>t</sub> ⊗ ϕ(K<sub>t</sub>))
                 </div>
              </div>

              {/* The Matrix */}
              <div className="flex items-center gap-4">
                <MatrixVis matrix={fwpState.W} title={`Fast Weights W${step}`} label="Accumulated Memory" />
                <div className="text-2xl text-slate-300">×</div>
                <VectorVis vector={phiQuery} title="ϕ(Q)" />
                <div className="text-2xl text-slate-300">=</div>
                <VectorVis vector={fwpOutputNumerator} title="Num." />
              </div>

               {/* Normalization part */}
               <div className="w-full border-t border-slate-100 pt-4 mt-2">
                 <div className="text-xs font-bold text-slate-400 mb-2">NORMALIZATION STATE</div>
                 <div className="flex items-center justify-center gap-4">
                    <div className="font-mono text-xs text-slate-500">z<sub>t</sub> = z<sub>t-1</sub> + ϕ(k<sub>t</sub>)</div>
                    <ArrowRight className="w-4 h-4 text-slate-300" />
                    <VectorVis vector={fwpState.z} title={`State z${step}`} vertical={false} />
                 </div>
              </div>
            </div>
          </div>
        </div>

        {/* --- Comparison / Conclusion --- */}
        <div className="lg:col-span-12">
          <div className="bg-slate-900 text-slate-50 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Calculator className="w-5 h-5 text-green-400" />
              Equivalence Check
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              <div>
                <p className="text-slate-300 mb-4 text-sm leading-relaxed">
                  Notice that when the Recurrent view reaches step <strong>t={SEQ_LEN}</strong>, its internal matrix state <code className="bg-slate-800 px-1 rounded">W</code> and normalization vector <code className="bg-slate-800 px-1 rounded">z</code> become <strong>identical</strong> to the global Transformer sums.
                </p>
                <div className="flex gap-4 text-sm font-mono">
                  <div className={`px-3 py-1 rounded ${step === SEQ_LEN ? 'bg-green-500/20 text-green-300 border border-green-500/50' : 'bg-slate-800 text-slate-500'}`}>
                    Match Status: {step === SEQ_LEN ? "EXACT MATCH" : "WAITING FOR COMPLETION..."}
                  </div>
                </div>
              </div>

              <div className="flex justify-center gap-8">
                 <div className="flex flex-col items-center">
                    <span className="text-xs font-bold text-slate-400 mb-2">Transformer Output</span>
                    <VectorVis vector={globalOutput} vertical={false} />
                 </div>
                 <div className="flex items-center text-slate-600 font-bold">=</div>
                 <div className="flex flex-col items-center">
                    <span className="text-xs font-bold text-slate-400 mb-2">FWP Output (Current)</span>
                    <VectorVis vector={fwpOutput} vertical={false} />
                 </div>
              </div>
            </div>
          </div>
        </div>

        {/* --- Educational Theory --- */}
        <div className="lg:col-span-12 bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="flex items-start gap-4">
            <div className="bg-blue-100 p-2 rounded-full mt-1">
              <Info className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h3 className="font-bold text-slate-800">Why does this matter?</h3>
              <div className="prose prose-sm text-slate-600 mt-2 max-w-none">
                <p>
                  Standard Transformers have quadratic complexity because they compute attention scores for every pair of tokens. 
                  <strong>Linear Transformers</strong> reduce this to linear complexity by changing the order of multiplication. 
                </p>
                <p>
                  As shown above, this linear formulation is equivalent to an RNN (the Fast Weight Programmer). 
                  Instead of storing previous tokens in a list (like standard attention), the model "programs" a memory matrix 
                  by adding outer products of keys and values. This matrix acts as a limited-capacity short-term memory that can be queried later.
                </p>
                <ul className="list-disc pl-5 space-y-1 mt-2">
                  <li><strong>Standard Attention:</strong> Attention(Q, K, V) = softmax(QK^T)V</li>
                  <li><strong>Linear Attention (FWP):</strong> y_i = ϕ(q_i)^T * Sum(ϕ(k_j) ⊗ v_j)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}