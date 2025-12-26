import React, { useState, useEffect, useMemo } from 'react';
import { RefreshCw, ArrowRight, BookOpen, Info, Calculator, CheckCircle, AlertCircle } from 'lucide-react';

/**
 * Utility functions for matrix math
 */
const generateRandomVector = (dim) => Array.from({ length: dim }, () => Math.random() * 2 - 1);
const generateRandomMatrix = (rows, cols) => Array.from({ length: rows }, () => generateRandomVector(cols));

const dotProduct = (v1, v2) => v1.reduce((acc, val, i) => acc + val * v2[i], 0);

const matMulVec = (matrix, vec) => {
  // Matrix (rows x cols) * Vec (cols x 1)
  return matrix.map(row => dotProduct(row, vec));
};

const vecMatMul = (vec, matrix) => {
  // Vec (1 x cols) * Matrix (cols x rows) -> Result (1 x rows)
  // Need to transpose matrix effectively here or just do dot products with columns
  const result = [];
  const cols = matrix[0].length;
  for (let i = 0; i < cols; i++) {
    const col = matrix.map(row => row[i]);
    result.push(dotProduct(vec, col));
  }
  return result;
};

const matMulMat = (matA, matB) => {
  // Simple A * B
  const result = [];
  const bCols = matB[0].length;
  for (let r = 0; r < matA.length; r++) {
    const row = [];
    for (let c = 0; c < bCols; c++) {
      const col = matB.map(m => m[c]);
      row.push(dotProduct(matA[r], col));
    }
    result.push(row);
  }
  return result;
};

const softmax = (vec) => {
  const max = Math.max(...vec);
  const exps = vec.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
};

const transpose = (matrix) => matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));

/**
 * Component: NumberCell
 * Renders a number with conditional coloring based on value
 */
const NumberCell = ({ value, label = null, highlight = false }) => {
  const intensity = Math.min(Math.abs(value), 1);
  // Blue for positive, Red for negative
  const color = value > 0 ? `rgba(59, 130, 246, ${intensity * 0.8})` : `rgba(239, 68, 68, ${intensity * 0.8})`;
  
  return (
    <div className={`flex flex-col items-center justify-center p-1 m-0.5 rounded text-xs font-mono transition-all duration-300 ${highlight ? 'ring-2 ring-yellow-400' : ''}`}
         style={{ backgroundColor: color, color: intensity > 0.5 ? 'white' : 'black', minWidth: '40px', height: '30px' }}>
      {value.toFixed(2)}
      {label && <span className="text-[0.6rem] opacity-70">{label}</span>}
    </div>
  );
};

const MatrixViz = ({ data, title, symbol, rows, cols }) => (
  <div className="flex flex-col items-center mx-1">
    <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-1 text-xs whitespace-nowrap">{title} <span className="text-gray-400 font-normal">({symbol})</span></h4>
    <div className="grid gap-0.5 border-2 border-gray-300 p-1 rounded bg-white shadow-sm" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {data.flat().map((val, idx) => (
        <NumberCell key={idx} value={val} />
      ))}
    </div>
    <span className="text-[10px] text-gray-500 mt-1 font-mono">{rows}×{cols}</span>
  </div>
);

const VectorViz = ({ data, title, symbol, orientation = 'vertical' }) => (
  <div className="flex flex-col items-center mx-1">
    <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-1 text-xs whitespace-nowrap">{title} <span className="text-gray-400 font-normal">({symbol})</span></h4>
    <div className={`flex ${orientation === 'vertical' ? 'flex-col' : 'flex-row'} gap-0.5 border-2 border-gray-300 p-1 rounded bg-white shadow-sm`}>
      {data.map((val, idx) => (
        <NumberCell key={idx} value={val} />
      ))}
    </div>
    <span className="text-[10px] text-gray-500 mt-1 font-mono">{orientation === 'vertical' ? `${data.length}×1` : `1×${data.length}`}</span>
  </div>
);

/**
 * Main Application Component
 */
export default function HopfieldTransformerTutorial() {
  // --- State ---
  const [dimension, setDimension] = useState(4); // d_k
  const [numPatterns, setNumPatterns] = useState(3); // N
  const [beta, setBeta] = useState(1.0); // Temperature / Scaling
  
  // Data State
  const [statePattern, setStatePattern] = useState([]); // xi / Query
  const [storedPatterns, setStoredPatterns] = useState([]); // X / Keys
  
  // --- Initialization ---
  useEffect(() => {
    resetData();
  }, [dimension, numPatterns]);

  const resetData = () => {
    setStatePattern(generateRandomVector(dimension));
    // X is (dimension x numPatterns) for Hopfield convention typically
    setStoredPatterns(generateRandomMatrix(dimension, numPatterns)); 
  };

  // --- Computations ---
  // 1. Hopfield Universe
  // Formula: xi_new = X * softmax(beta * X^T * xi)
  // X is (d x N). xi is (d x 1). X^T is (N x d).
  // X^T * xi -> (N x 1) similarity vector
  
  const hopfieldResults = useMemo(() => {
    if (statePattern.length === 0) return null;

    const X = storedPatterns; // d x N
    const xi = statePattern;  // d vector
    const XT = transpose(X);  // N x d
    
    // Step 1: Similarities (Energies)
    // dot product of each stored pattern with state
    const similarities = matMulVec(XT, xi); // vector of length N
    
    // Step 2: Softmax (Probabilities)
    const scaledSims = similarities.map(s => s * beta);
    const p = softmax(scaledSims); // vector of length N
    
    // Step 3: Retrieval (Update)
    // xi_new = X * p (Weighted sum of columns of X)
    const xi_new = matMulVec(X, p);
    
    return { similarities, p, xi_new };
  }, [statePattern, storedPatterns, beta]);

  // 2. Transformer Universe
  // Formula: Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d) ) V
  // Mapping: Q = xi^T (row), K = X^T (rows), V = X^T (rows)
  const transformerResults = useMemo(() => {
    if (statePattern.length === 0) return null;

    // Transpose to match Transformer "Row Vector" convention
    const Q = [statePattern]; // 1 x d
    const K = transpose(storedPatterns); // N x d (Keys are rows)
    const V = transpose(storedPatterns); // N x d (Values are rows)
    
    const scaleFactor = beta; // In paper beta corresponds to 1/sqrt(d_k)
    
    // Step 1: Scores = Q * K^T
    // Q is (1 x d), K^T is (d x N) -> Result (1 x N)
    // This is effectively dot product of Q with every K row.
    const KT = transpose(K); // d x N
    const scores = matMulMat(Q, KT)[0]; // vector length N
    
    // Step 2: Attention Weights
    const scaledScores = scores.map(s => s * scaleFactor);
    const attnWeights = softmax(scaledScores);
    
    // Step 3: Weighted Sum
    // Weights (1 x N) * V (N x d) -> (1 x d)
    const outputRow = vecMatMul(attnWeights, V);
    
    return { scores, attnWeights, outputRow };
  }, [statePattern, storedPatterns, beta]);

  if (!hopfieldResults || !transformerResults) return <div className="p-10 text-center">Loading calculations...</div>;

  // Comparison logic
  const diff = hopfieldResults.xi_new.reduce((acc, val, i) => acc + Math.abs(val - transformerResults.outputRow[i]), 0);
  const isEquivalent = diff < 1e-9;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans selection:bg-blue-100">
      {/* Header */}
      <header className="bg-slate-900 text-white p-6 shadow-lg border-b-4 border-blue-500">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-3">
                <BookOpen className="h-8 w-8 text-blue-400" />
                Hopfield Networks is All You Need
              </h1>
              <p className="mt-2 text-slate-300 max-w-2xl text-sm leading-relaxed">
                Interactive visualization of the equivalence between the <span className="text-blue-200 font-semibold">Continuous Dense Hopfield Update Rule</span> and <span className="text-blue-200 font-semibold">Transformer Self-Attention</span>.
              </p>
            </div>
            <div className="flex flex-col items-end gap-2 text-xs text-slate-400">
              <span className="bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">Ramsauer et al. (2020)</span>
              <span className="bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">arXiv:2008.02217v3</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Controls Sidebar */}
        <aside className="lg:col-span-3 space-y-6">
          <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200 sticky top-4">
            <h2 className="font-bold text-lg mb-5 flex items-center gap-2 border-b pb-3">
              <Calculator className="h-5 w-5 text-blue-600" /> Parameters
            </h2>
            
            <div className="space-y-6">
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">Dimension (<span className="font-serif italic">d</span>)</label>
                  <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded text-gray-600">{dimension}</span>
                </div>
                <input 
                  type="range" min="2" max="6" 
                  value={dimension} 
                  onChange={(e) => setDimension(parseInt(e.target.value))} 
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600" 
                />
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                   <label className="text-sm font-medium text-gray-700">Patterns (<span className="font-serif italic">N</span>)</label>
                   <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded text-gray-600">{numPatterns}</span>
                </div>
                <input 
                  type="range" min="2" max="5" 
                  value={numPatterns} 
                  onChange={(e) => setNumPatterns(parseInt(e.target.value))} 
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600" 
                />
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Beta / Scale (<span className="font-serif italic">β</span>)
                  </label>
                  <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded text-gray-600">{beta.toFixed(1)}</span>
                </div>
                <input 
                  type="range" 
                  min="0.1" 
                  max="5.0" 
                  step="0.1" 
                  value={beta} 
                  onChange={(e) => setBeta(parseFloat(e.target.value))} 
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600" 
                />
                <p className="text-[10px] text-gray-400 mt-2 leading-tight">
                  Controls the "sharpness" of the softmax. In attention, this is typically <span className="font-mono">1/√d</span>.
                </p>
              </div>

              <button 
                onClick={resetData}
                className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-white py-2.5 px-4 rounded-lg transition-colors text-sm font-medium shadow-sm active:scale-95 transform duration-100"
              >
                <RefreshCw className="h-4 w-4" /> Randomize Values
              </button>
            </div>

            <div className={`mt-6 p-4 rounded-lg border flex flex-col items-center text-center transition-colors duration-500 ${isEquivalent ? 'bg-green-50 border-green-200 text-green-800' : 'bg-red-50 border-red-200 text-red-800'}`}>
              <div className="flex items-center gap-2 font-bold mb-1">
                 {isEquivalent ? <CheckCircle className="h-5 w-5"/> : <AlertCircle className="h-5 w-5"/>}
                 <span>{isEquivalent ? "Equivalent" : "Mismatch"}</span>
              </div>
              <p className="text-xs opacity-80 mb-2">Error: <span className="font-mono">{diff.toExponential(2)}</span></p>
              <p className="text-[10px] leading-tight">
                The Hopfield update and Attention mechanism produced identical output vectors.
              </p>
            </div>
          </div>
        </aside>

        {/* Visualization Area */}
        <div className="lg:col-span-9 space-y-6">
          
          {/* Top: The Math Mapping */}
          <div className="bg-white p-4 rounded-lg shadow-sm border border-indigo-100 flex flex-col sm:flex-row justify-center items-center text-sm gap-4 sm:gap-12 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500"></div>
            
            <div className="text-center z-10">
              <div className="font-bold text-gray-900 text-lg mb-1">Hopfield</div>
              <div className="font-mono text-indigo-600 text-xs bg-indigo-50 px-2 py-1 rounded mb-1">State Pattern (ξ)</div>
              <div className="font-mono text-indigo-600 text-xs bg-indigo-50 px-2 py-1 rounded">Stored Patterns (X)</div>
            </div>
            
            <div className="flex flex-col items-center z-10 opacity-50">
               <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1">Transpose</span>
               <ArrowRight className="text-indigo-300 h-6 w-6" />
            </div>
            
            <div className="text-center z-10">
              <div className="font-bold text-gray-900 text-lg mb-1">Transformer</div>
              <div className="font-mono text-indigo-600 text-xs bg-indigo-50 px-2 py-1 rounded mb-1">Query (Q)</div>
              <div className="font-mono text-indigo-600 text-xs bg-indigo-50 px-2 py-1 rounded">Keys (K) & Values (V)</div>
            </div>
          </div>

          {/* Main Comparison Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            
            {/* Left: Hopfield Logic */}
            <div className="bg-white p-4 md:p-6 rounded-xl shadow-sm border border-gray-200 flex flex-col h-full">
              <div className="border-b pb-4 mb-6">
                <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <span className="w-2 h-6 bg-blue-500 rounded-full"></span>
                  Hopfield Universe
                </h2>
                <div className="mt-2 text-center bg-gray-50 py-2 rounded-md border border-gray-100">
                    <span className="font-serif italic text-lg text-gray-600">ξ<sup>new</sup> = X · softmax(β · X<sup>T</sup>ξ)</span>
                </div>
              </div>

              <div className="flex flex-col gap-6 items-center flex-grow justify-center">
                {/* Inputs */}
                <div className="flex gap-4 items-center justify-center bg-slate-50 p-4 rounded-lg w-full">
                   <div className="flex items-center gap-1">
                      <MatrixViz data={storedPatterns} title="Patterns X" symbol="d×N" rows={dimension} cols={numPatterns} />
                      <span className="text-xl text-gray-400 font-serif font-bold">·</span>
                      <VectorViz data={statePattern} title="State ξ" symbol="d×1" />
                   </div>
                </div>

                <div className="flex flex-col items-center">
                    <ArrowRight className="rotate-90 text-gray-300 mb-1" />
                    <span className="text-[10px] text-gray-400 font-medium">PROCESS</span>
                    <ArrowRight className="rotate-90 text-gray-300 mt-1" />
                </div>

                <div className="grid grid-cols-2 gap-4 w-full">
                    {/* Step 1: Similarity */}
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 flex flex-col items-center">
                    <h3 className="text-[10px] font-bold uppercase text-gray-500 mb-2 tracking-wider">1. Similarity (Energy)</h3>
                    <div className="flex justify-center items-center gap-2">
                        <span className="font-mono text-xs text-gray-400">X<sup>T</sup>ξ</span>
                        <ArrowRight className="w-3 h-3 text-gray-300"/>
                        <VectorViz data={hopfieldResults.similarities} title="Scores" symbol="N×1" orientation="horizontal" />
                    </div>
                    </div>

                    {/* Step 2: Softmax */}
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 flex flex-col items-center">
                    <h3 className="text-[10px] font-bold uppercase text-gray-500 mb-2 tracking-wider">2. Softmax</h3>
                    <div className="flex justify-center items-center gap-2">
                        <span className="font-mono text-xs text-gray-400">p</span>
                        <ArrowRight className="w-3 h-3 text-gray-300"/>
                        <VectorViz data={hopfieldResults.p} title="Weights" symbol="N×1" orientation="horizontal" />
                    </div>
                    </div>
                </div>

                <div className="flex flex-col items-center">
                    <ArrowRight className="rotate-90 text-gray-300" />
                </div>

                {/* Final Result */}
                <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg flex flex-col items-center w-full relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-blue-200"></div>
                  <h3 className="text-sm font-bold text-blue-900 mb-2">New State (ξ<sup>new</sup>)</h3>
                  <VectorViz data={hopfieldResults.xi_new} title="Update" symbol="d×1" />
                </div>
              </div>
            </div>

            {/* Right: Transformer Logic */}
            <div className="bg-white p-4 md:p-6 rounded-xl shadow-sm border border-gray-200 flex flex-col h-full">
              <div className="border-b pb-4 mb-6">
                <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <span className="w-2 h-6 bg-purple-500 rounded-full"></span>
                  Transformer Universe
                </h2>
                <div className="mt-2 text-center bg-gray-50 py-2 rounded-md border border-gray-100">
                    <span className="font-serif italic text-lg text-gray-600">Attn = softmax((QK<sup>T</sup>)/√d) · V</span>
                </div>
              </div>

              <div className="flex flex-col gap-6 items-center flex-grow justify-center">
                {/* Inputs */}
                <div className="flex gap-4 items-center justify-center bg-slate-50 p-4 rounded-lg w-full">
                    <div className="flex items-center gap-1">
                        <VectorViz data={statePattern} title="Query Q" symbol="1×d" orientation="horizontal" />
                        <span className="text-xl text-gray-400 font-serif font-bold">·</span>
                        <MatrixViz data={transpose(storedPatterns)} title="Keys K^T" symbol="d×N" rows={dimension} cols={numPatterns} />
                    </div>
                </div>

                <div className="flex flex-col items-center">
                    <ArrowRight className="rotate-90 text-gray-300 mb-1" />
                    <span className="text-[10px] text-gray-400 font-medium">PROCESS</span>
                    <ArrowRight className="rotate-90 text-gray-300 mt-1" />
                </div>

                <div className="grid grid-cols-2 gap-4 w-full">
                    {/* Step 1: Dot Product Attention */}
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 flex flex-col items-center">
                    <h3 className="text-[10px] font-bold uppercase text-gray-500 mb-2 tracking-wider">1. Scaled Dot-Product</h3>
                    <div className="flex justify-center items-center gap-2">
                        <span className="font-mono text-xs text-gray-400">QK<sup>T</sup></span>
                        <ArrowRight className="w-3 h-3 text-gray-300"/>
                        <VectorViz data={transformerResults.scores} title="Raw Attn" symbol="1×N" orientation="horizontal" />
                    </div>
                    </div>

                    {/* Step 2: Softmax */}
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 flex flex-col items-center">
                    <h3 className="text-[10px] font-bold uppercase text-gray-500 mb-2 tracking-wider">2. Softmax Weights</h3>
                    <div className="flex justify-center items-center gap-2">
                        <span className="font-mono text-xs text-gray-400">σ(·)</span>
                        <ArrowRight className="w-3 h-3 text-gray-300"/>
                        <VectorViz data={transformerResults.attnWeights} title="Weights" symbol="1×N" orientation="horizontal" />
                    </div>
                    </div>
                </div>

                <div className="flex flex-col items-center">
                    <ArrowRight className="rotate-90 text-gray-300" />
                </div>

                {/* Final Result */}
                <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg flex flex-col items-center w-full relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-purple-200"></div>
                  <h3 className="text-sm font-bold text-purple-900 mb-2">Attention Output</h3>
                  <VectorViz data={transformerResults.outputRow} title="Result" symbol="1×d" orientation="vertical" />
                </div>
              </div>
            </div>

          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 text-sm leading-relaxed text-gray-700">
            <h3 className="font-bold text-lg mb-3 flex items-center gap-2 text-gray-900">
                <Info className="h-5 w-5 text-blue-500"/> 
                The Core Insight
            </h3>
            <div className="space-y-3">
                <p>
                The <strong>Continuous Dense Hopfield Network</strong> generalizes classic binary associative memory to continuous values.
                By examining the math, we see that its update rule is not just <em>similar</em> to Transformer attention—it is <strong className="text-blue-700">identical</strong> up to a transpose operation.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
                    <div className="bg-slate-50 p-3 rounded border border-slate-100">
                        <strong className="block text-slate-800 mb-1">Energy Minimization</strong>
                        <p className="text-xs text-slate-600">The "Energy" of a Hopfield state measures how well it matches stored patterns. Minimizing this energy is exactly what the Attention mechanism does by selecting the most relevant Keys.</p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded border border-slate-100">
                        <strong className="block text-slate-800 mb-1">One-Shot Retrieval</strong>
                        <p className="text-xs text-slate-600">Unlike old Hopfield nets that needed many steps to converge, this modern version retrieves memories in a single update step—just like a Transformer layer processes input in one go.</p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded border border-slate-100">
                        <strong className="block text-slate-800 mb-1">Global vs Local</strong>
                        <p className="text-xs text-slate-600">Low $\beta$ (temperature) leads to global averaging (the network "sees" everything). High $\beta$ leads to sharp focus (the network "attends" to one specific memory).</p>
                    </div>
                </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}