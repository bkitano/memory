// @ts-nocheck
import React, { useState, useEffect, useCallback } from 'react';
import { Play, Pause, SkipForward, SkipBack, RefreshCw, Calculator, ArrowRight, Grid } from 'lucide-react';

const Card = ({ children, className = "" }) => (
  <div className={`bg-white rounded-lg shadow-sm border border-slate-200 ${className}`}>
    {children}
  </div>
);

const Button = ({ onClick, disabled, children, className = "", variant = "primary" }) => {
  const baseStyle = "px-4 py-2 rounded-md font-medium transition-colors flex items-center gap-2";
  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-700 disabled:bg-slate-300",
    secondary: "bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 disabled:bg-slate-100 disabled:text-slate-400",
    ghost: "text-slate-600 hover:bg-slate-100 disabled:text-slate-300"
  };
  return (
    <button 
      onClick={onClick} 
      disabled={disabled} 
      className={`${baseStyle} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

const MatrixDisplay = ({ data, title, highlight = false, color = "blue", size = "small" }) => {
  const maxVal = Math.max(...data.flat().map(Math.abs), 0.1); // Avoid div by zero
  
  return (
    <div className="flex flex-col items-center">
      <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wider">{title}</div>
      <div className={`grid gap-1 ${size === 'small' ? 'p-1' : 'p-2'} bg-slate-50 rounded-lg border border-slate-200 transition-all duration-300 ${highlight ? 'ring-2 ring-blue-400 shadow-md' : ''}`}
           style={{ gridTemplateColumns: `repeat(${data[0].length}, minmax(0, 1fr))` }}>
        {data.map((row, i) => (
          row.map((val, j) => {
            const intensity = Math.min(Math.abs(val) / 2, 1); // Normalize for color
            const bgColor = val >= 0 
              ? `rgba(59, 130, 246, ${intensity})` // Blue for positive
              : `rgba(239, 68, 68, ${intensity})`; // Red for negative
            
            return (
              <div 
                key={`${i}-${j}`}
                className={`flex items-center justify-center rounded font-mono text-sm
                  ${size === 'small' ? 'w-10 h-10' : 'w-16 h-16'}
                  transition-colors duration-300`}
                style={{ backgroundColor: bgColor }}
              >
                <span className={intensity > 0.5 ? 'text-white' : 'text-slate-900'}>
                  {val.toFixed(2)}
                </span>
              </div>
            );
          })
        ))}
      </div>
    </div>
  );
};

const VectorDisplay = ({ data, title, orientation = "vertical", color = "emerald" }) => {
  return (
    <div className="flex flex-col items-center mx-2">
      <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wider">{title}</div>
      <div className={`flex ${orientation === "vertical" ? "flex-col" : "flex-row"} gap-1 p-1 bg-slate-50 rounded border border-slate-200`}>
        {data.map((val, i) => (
          <div 
            key={i} 
            className="w-10 h-10 flex items-center justify-center bg-white border border-slate-100 rounded font-mono text-sm text-slate-700 shadow-sm"
          >
            {val.toFixed(1)}
          </div>
        ))}
      </div>
    </div>
  );
};

const MathEq = ({ children }) => (
  <span className="font-mono bg-slate-100 px-1 py-0.5 rounded text-slate-800 text-sm">
    {children}
  </span>
);

export default function FastWeightSSM() {
  // Configuration
  const SEQ_LEN = 5;
  const DIM = 2;
  
  // State
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [decay, setDecay] = useState(0.8);
  const [inputs, setInputs] = useState([]);
  const [history, setHistory] = useState([]);

  // Generate random inputs on mount
  const generateInputs = useCallback(() => {
    const newInputs = Array.from({ length: SEQ_LEN }, () => ({
      k: Array.from({ length: DIM }, () => (Math.random() * 2 - 1)),
      v: Array.from({ length: DIM }, () => (Math.random() * 2 - 1)),
      q: Array.from({ length: DIM }, () => (Math.random() * 2 - 1))
    }));
    setInputs(newInputs);
    setStep(0);
    setIsPlaying(false);
  }, []);

  useEffect(() => {
    generateInputs();
  }, [generateInputs]);

  // Compute full history based on current inputs and decay
  useEffect(() => {
    if (inputs.length === 0) return;

    const hist = [];
    // Initial State (S_0) is zero matrix
    let S = Array(DIM).fill().map(() => Array(DIM).fill(0));
    
    // Initial History Entry (Step 0 - Initial State)
    hist.push({
      step: 0,
      k: null, v: null, q: null,
      updateMatrix: Array(DIM).fill().map(() => Array(DIM).fill(0)),
      S_prev: JSON.parse(JSON.stringify(S)),
      S_curr: JSON.parse(JSON.stringify(S)),
      output: Array(DIM).fill(0),
      explanation: "Initial state S₀ is initialized to zero."
    });

    for (let t = 0; t < inputs.length; t++) {
      const { k, v, q } = inputs[t];
      
      // 1. Read: Output y = S_{t-1} * q
      // Note: Standard Linear Attention often does output = S_{t-1} * q + residual. 
      // We will focus on the pure Fast Weight part: y = S * q.
      const output = S.map(row => 
        row.reduce((sum, val, idx) => sum + val * q[idx], 0)
      );

      // 2. Update Term: V * K^T (Outer Product)
      const updateMatrix = v.map((vVal) => 
        k.map((kVal) => vVal * kVal)
      );

      // 3. State Update: S_t = decay * S_{t-1} + updateMatrix
      const S_prev = JSON.parse(JSON.stringify(S));
      S = S.map((row, i) => 
        row.map((val, j) => val * decay + updateMatrix[i][j])
      );

      hist.push({
        step: t + 1,
        k, v, q,
        updateMatrix,
        S_prev,
        S_curr: JSON.parse(JSON.stringify(S)),
        output,
        explanation: `Step ${t+1}: Read memory with Query, then write Key-Value to State.`
      });
    }
    setHistory(hist);
  }, [inputs, decay]);

  // Playback Loop
  useEffect(() => {
    let interval;
    if (isPlaying && step < SEQ_LEN) {
      interval = setInterval(() => {
        setStep(s => {
          if (s >= SEQ_LEN) {
            setIsPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, 1500);
    } else if (step >= SEQ_LEN) {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step]);

  const currentData = history[step] || history[0];

  // Helper to update input value manually
  const updateInput = (t, type, idx, val) => {
    const newInputs = [...inputs];
    newInputs[t][type][idx] = parseFloat(val);
    setInputs(newInputs);
  };

  if (!currentData) return <div className="p-8 flex justify-center text-slate-500">Initializing simulation...</div>;

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-6 bg-slate-50 min-h-screen font-sans">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-slate-800">Fast Weight Programmer <span className="text-slate-400 mx-2">vs</span> State Space Model</h1>
        <p className="text-slate-600 max-w-2xl mx-auto">
          Visualizing the mathematical equivalence. The "Fast Weight" matrix <MathEq>S</MathEq> created by outer products IS the hidden state <MathEq>h</MathEq> of a State Space Model.
        </p>
      </div>

      {/* Controls */}
      <Card className="p-4 flex flex-wrap gap-4 items-center justify-between bg-white sticky top-2 z-10 shadow-md">
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => setStep(0)} disabled={step === 0}>
            <SkipBack size={18} />
          </Button>
          <Button onClick={() => setIsPlaying(!isPlaying)} variant={isPlaying ? "secondary" : "primary"}>
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? "Pause" : "Simulate"}
          </Button>
          <Button variant="secondary" onClick={() => setStep(Math.min(step + 1, SEQ_LEN))} disabled={step === SEQ_LEN}>
            <SkipForward size={18} />
          </Button>
          <div className="ml-4 font-mono text-slate-600">
            Step: <span className="font-bold text-blue-600">{step}</span> / {SEQ_LEN}
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-slate-700">Decay (λ): {decay}</label>
            <input 
              type="range" 
              min="0" max="1" step="0.1" 
              value={decay} 
              onChange={(e) => setDecay(parseFloat(e.target.value))}
              className="w-32 accent-blue-600"
            />
          </div>
          <Button variant="ghost" onClick={generateInputs}>
            <RefreshCw size={18} className="mr-2" /> Reset
          </Button>
        </div>
      </Card>

      {/* Main Visualization Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Panel: Fast Weight View */}
        <Card className="p-6 space-y-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-2 bg-blue-100 text-blue-800 text-xs font-bold rounded-bl-lg uppercase">
            Fast Weight View (Programmer)
          </div>
          
          <div className="space-y-4">
            <h3 className="font-semibold text-lg text-slate-800 flex items-center gap-2">
              <Calculator size={20} className="text-blue-500" />
              Outer Product Update
            </h3>
            <p className="text-sm text-slate-500">
              The "Fast Weights" are updated by adding the outer product of the current key <MathEq>k_t</MathEq> and value <MathEq>v_t</MathEq>.
            </p>

            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <div className="text-center font-mono text-sm mb-4 text-slate-600">
                Update Rule: S<sub className="text-xs">t</sub> = λS<sub className="text-xs">t-1</sub> + (v<sub className="text-xs">t</sub> ⊗ k<sub className="text-xs">t</sub>)
              </div>
              
              <div className="flex items-center justify-center gap-4 flex-wrap">
                {step > 0 ? (
                  <>
                    <VectorDisplay data={currentData.v} title="Value (v)" orientation="vertical" />
                    <div className="text-xl text-slate-400">⊗</div>
                    <VectorDisplay data={currentData.k} title="Key (k)" orientation="horizontal" />
                    <div className="text-xl text-slate-400">=</div>
                    <MatrixDisplay data={currentData.updateMatrix} title="Write (Outer Prod)" />
                  </>
                ) : (
                  <div className="text-slate-400 italic">Waiting for input...</div>
                )}
              </div>
            </div>
          </div>
        </Card>

        {/* Right Panel: SSM View */}
        <Card className="p-6 space-y-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-2 bg-purple-100 text-purple-800 text-xs font-bold rounded-bl-lg uppercase">
            State Space Model View
          </div>

          <div className="space-y-4">
            <h3 className="font-semibold text-lg text-slate-800 flex items-center gap-2">
              <Grid size={20} className="text-purple-500" />
              Recurrent State Update
            </h3>
            <p className="text-sm text-slate-500">
              The exact same operation viewed as a linear recurrence. The matrix is the "Hidden State" <MathEq>h_t</MathEq>.
            </p>

            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <div className="text-center font-mono text-sm mb-4 text-slate-600">
                SSM Rule: h<sub className="text-xs">t</sub> = A h<sub className="text-xs">t-1</sub> + B u<sub className="text-xs">t</sub>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-500">State (h)</span>
                    <span className="font-mono text-purple-700">Matrix S</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-500">Input (u)</span>
                    <span className="font-mono text-purple-700">v ⊗ k</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-500">A (Dynamics)</span>
                    <span className="font-mono text-purple-700">λ · I</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-500">B (Input Map)</span>
                    <span className="font-mono text-purple-700">Identity</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Central State Visualization */}
      <Card className="p-8 border-t-4 border-t-blue-500">
        <div className="flex flex-col md:flex-row items-center justify-around gap-8">
          
          {/* Previous State */}
          <div className="flex flex-col items-center opacity-60">
            <MatrixDisplay data={currentData.S_prev} title={`State (t=${Math.max(0, step - 1)})`} size="small" />
            <div className="mt-2 text-xs font-mono text-slate-500">S<sub className="text-[10px]">{Math.max(0, step - 1)}</sub></div>
          </div>

          <div className="flex flex-col items-center">
            <ArrowRight className="text-slate-300 mb-2 rotate-90 md:rotate-0" size={32} />
            <div className="text-xs text-slate-400 font-mono">Decay λ={decay}</div>
            <div className="text-xs text-slate-400 font-mono">+ Update</div>
          </div>

          {/* Current State (The Hero) */}
          <div className="flex flex-col items-center scale-110">
            <MatrixDisplay 
              data={currentData.S_curr} 
              title={`Current State (t=${step})`} 
              size="large" 
              highlight={true} 
            />
            <div className="mt-4 px-4 py-2 bg-blue-50 text-blue-800 rounded-full text-sm font-medium shadow-sm">
              The Dual Representation
            </div>
          </div>

          <div className="flex flex-col items-center">
            <ArrowRight className="text-slate-300 mb-2 rotate-90 md:rotate-0" size={32} />
            <div className="text-xs text-slate-400 font-mono">Read w/ Query</div>
          </div>

          {/* Output */}
          <div className="flex flex-col items-center bg-green-50 p-4 rounded-xl border border-green-100">
            {step > 0 ? (
              <>
                <VectorDisplay data={currentData.q} title="Query (q)" color="blue" />
                <div className="my-2 text-slate-400 text-lg">↓</div>
                <VectorDisplay data={currentData.output} title="Output (y)" color="purple" />
              </>
            ) : (
               <div className="text-slate-400 text-sm italic w-24 text-center">No query yet</div>
            )}
          </div>

        </div>
      </Card>

      {/* Input Sequence Editor */}
      <Card className="p-6">
        <h3 className="font-semibold text-slate-800 mb-4">Input Sequence Editor</h3>
        <div className="overflow-x-auto pb-4">
          <div className="flex gap-4 min-w-max">
            {inputs.map((inp, t) => (
              <div key={t} className={`relative p-3 rounded-lg border transition-all ${t + 1 === step ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500' : 'border-slate-200 bg-white'}`}>
                <div className="absolute -top-3 left-3 px-1 bg-white text-xs font-bold text-slate-400">t={t+1}</div>
                <div className="space-y-3">
                  {/* Key Inputs */}
                  <div>
                    <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">Key</div>
                    <div className="flex gap-1">
                      {inp.k.map((val, idx) => (
                        <input
                          key={`k-${idx}`}
                          type="number"
                          step="0.1"
                          className="w-12 text-xs p-1 border rounded bg-slate-50 focus:ring-1 focus:ring-blue-500 outline-none"
                          value={val.toFixed(1)}
                          onChange={(e) => updateInput(t, 'k', idx, e.target.value)}
                        />
                      ))}
                    </div>
                  </div>
                  {/* Value Inputs */}
                  <div>
                    <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">Value</div>
                    <div className="flex gap-1">
                      {inp.v.map((val, idx) => (
                        <input
                          key={`v-${idx}`}
                          type="number"
                          step="0.1"
                          className="w-12 text-xs p-1 border rounded bg-slate-50 focus:ring-1 focus:ring-blue-500 outline-none"
                          value={val.toFixed(1)}
                          onChange={(e) => updateInput(t, 'v', idx, e.target.value)}
                        />
                      ))}
                    </div>
                  </div>
                  {/* Query Inputs */}
                  <div>
                    <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">Query</div>
                    <div className="flex gap-1">
                      {inp.q.map((val, idx) => (
                        <input
                          key={`q-${idx}`}
                          type="number"
                          step="0.1"
                          className="w-12 text-xs p-1 border rounded bg-slate-50 focus:ring-1 focus:ring-blue-500 outline-none"
                          value={val.toFixed(1)}
                          onChange={(e) => updateInput(t, 'q', idx, e.target.value)}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <div className="text-center text-xs text-slate-400 pb-8">
        Model: Linear Transformer / Fast Weight Programmer (Schmidhuber 1992, Katharopoulos et al. 2020)
      </div>
    </div>
  );
}