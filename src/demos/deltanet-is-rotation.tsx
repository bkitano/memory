// @ts-nocheck
import React, { useState, useRef, useEffect } from 'react';
import { Play, RotateCcw, Info, Check, X } from 'lucide-react';

/**
 * Geometric Projection Visualization
 * * Demonstrates how L2-normalization of vector K turns the operation (I - KK^T)
 * into a geometric projection that "cleans" vector S of any component parallel to K.
 */

// --- Vector Math Helpers ---
const add = (v1, v2) => ({ x: v1.x + v2.x, y: v1.y + v2.y });
const sub = (v1, v2) => ({ x: v1.x - v2.x, y: v1.y - v2.y });
const scale = (v, s) => ({ x: v.x * s, y: v.y * s });
const dot = (v1, v2) => v1.x * v2.x + v1.y * v2.y;
const len = (v) => Math.sqrt(v.x * v.x + v.y * v.y);
const norm = (v) => {
  const l = len(v);
  return l === 0 ? { x: 0, y: 0 } : { x: v.x / l, y: v.y / l };
};

// --- Components ---

const VectorArrow = ({ start, end, color, label, isDashed = false, strokeWidth = 3, headSize = 10, opacity = 1 }) => {
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const lenVal = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));

  // Don't render tiny vectors
  if (lenVal < 5) return null;

  return (
    <g opacity={opacity} className="transition-all duration-300 ease-in-out">
      {/* Line */}
      <line
        x1={start.x}
        y1={start.y}
        x2={end.x}
        y2={end.y}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={isDashed ? "5,5" : "none"}
        markerEnd={`url(#arrowhead-${color.replace('#', '')})`}
      />
      
      {/* Label Background for readability */}
      {label && (
        <g transform={`translate(${(start.x + end.x) / 2}, ${(start.y + end.y) / 2})`}>
          <rect x="-12" y="-12" width="24" height="24" fill="rgba(255,255,255,0.7)" rx="4" />
          <text
            x="0"
            y="0"
            dy="5"
            textAnchor="middle"
            fill={color}
            fontWeight="bold"
            fontSize="14"
            className="select-none font-mono"
          >
            {label}
          </text>
        </g>
      )}
    </g>
  );
};

const DraggableHandle = ({ x, y, color, onDrag, label }) => {
  const [isDragging, setIsDragging] = useState(false);
  const handleRef = useRef(null);

  useEffect(() => {
    const handleGlobalMouseMove = (e) => {
      if (!isDragging) return;
      const svg = handleRef.current?.closest('svg');
      if (!svg) return;
      
      const rect = svg.getBoundingClientRect();
      // Calculate coordinates relative to SVG center (400, 300)
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;
      
      onDrag({ x: rawX, y: rawY });
    };

    const handleGlobalMouseUp = () => setIsDragging(false);

    if (isDragging) {
      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isDragging, onDrag]);

  return (
    <g 
      transform={`translate(${x}, ${y})`} 
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      onMouseDown={(e) => {
        e.stopPropagation();
        setIsDragging(true);
      }}
      ref={handleRef}
    >
      <circle r="12" fill={color} fillOpacity="0.2" stroke={color} strokeWidth="2" />
      <circle r="4" fill={color} />
      {label && (
        <text y="-20" textAnchor="middle" fill={color} fontWeight="bold" fontSize="12" className="select-none">
          {label}
        </text>
      )}
    </g>
  );
};

const Slider = ({ label, value, min, max, onChange, step = 0.01 }) => (
  <div className="flex flex-col gap-1 w-full">
    <div className="flex justify-between text-xs text-gray-600 font-mono">
      <span>{label}</span>
      <span>{value.toFixed(2)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
    />
  </div>
);

export default function GeometricProjection() {
  // Canvas State
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const center = { x: dimensions.width / 2, y: dimensions.height / 2 };
  const PIXELS_PER_UNIT = 100;

  // Logic State
  const [isNormalized, setIsNormalized] = useState(false);
  const [vectorK, setVectorK] = useState({ x: 100, y: -50 });
  const [vectorS, setVectorS] = useState({ x: 100, y: -150 });
  
  // Animation State for "Memory Cleanup" pulse
  const [pulse, setPulse] = useState(0);

  // --- Calculations ---

  // 1. Determine the effective K used in the formula
  // If normalized, K_eff = Unit Vector (length 1 pixel in this math space).
  // If not normalized, we must scale pixels to logical units to avoid explosion.
  // We treat 100 pixels as "Magnitude 1".
  // Derivation: Term = K(K.S). In pixels: (K_px/100) * ((K_px/100) . S_px) 
  const kLen = len(vectorK);
  const kNorm = norm(vectorK);
  
  // In the normalized case, we use the unit vector directly. 
  // In the unnormalized case, we scale the pixel vector by 1/PIXELS_PER_UNIT 
  // to get the "Logical K" that mimics the math behavior correctly on screen.
  const kEff = isNormalized ? kNorm : scale(vectorK, 1 / PIXELS_PER_UNIT);
  
  // 2. Calculate the term (K_eff * K_eff^T) * S
  // This is a vector in the direction of K_eff with magnitude dot(K_eff, S)
  // Let scalar alpha = dot(K_eff, S)
  const alpha = dot(kEff, vectorS);
  
  // The "Erased" component vector = alpha * K_eff
  // If normalized: This is the orthogonal projection of S onto K.
  // If not: This is the scaled interference component.
  const erasedComponent = scale(kEff, alpha);
  
  // 3. Calculate New State S_new = S - erasedComponent
  // S_new = (I - K_eff * K_eff^T) S
  const sNew = sub(vectorS, erasedComponent);

  // 4. Verification Check: Is S_new orthogonal to K?
  // Only true if normalized (or K was already unit length)
  const dotProductNew = dot(sNew, vectorK);
  const isOrthogonal = Math.abs(dotProductNew) < 0.1; // tolerance
  const kMagnitude = isNormalized ? 1.0 : (kLen / PIXELS_PER_UNIT).toFixed(2); 

  // --- Helpers for coordinate transformation ---
  // We treat (0,0) as center. Y is flipped for SVG (up is negative).
  const toSvg = (v) => ({ x: center.x + v.x, y: center.y + v.y });
  const fromSvg = (v) => ({ x: v.x - center.x, y: v.y - center.y });

  const handleDragK = (pos) => setVectorK(fromSvg(pos));
  const handleDragS = (pos) => setVectorS(fromSvg(pos));

  // Pulse effect when toggling normalization
  useEffect(() => {
    setPulse(1);
    const timer = setTimeout(() => setPulse(0), 500);
    return () => clearTimeout(timer);
  }, [isNormalized]);

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden font-sans border border-gray-200">
      
      {/* Header */}
      <div className="w-full p-6 border-b border-gray-100 bg-gray-50">
        <h1 className="text-2xl font-bold text-gray-800">Geometric Role of L2-Normalization in DeltaNet</h1>
        <p className="text-gray-600 mt-2 text-sm leading-relaxed">
          The operation <code className="bg-gray-200 px-1 rounded font-mono font-bold text-gray-800">S ← S(I - βkkᵀ) + βvkᵀ</code> updates the state <strong className="text-blue-600">S</strong> by removing components related to Key <strong className="text-red-500">K</strong>.
          <br/>
          Explore how <strong>normalizing K</strong> is critical for this to act as a clean <span className="italic text-green-600 font-semibold">geometric projection</span>.
        </p>
      </div>

      {/* Main Visualization Area */}
      <div className="relative w-full bg-slate-50 overflow-hidden cursor-crosshair">
        
        {/* Grid Background */}
        <svg 
          width={dimensions.width} 
          height={dimensions.height} 
          className="w-full h-full select-none"
          viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
        >
          {/* Definitions for arrowheads */}
          <defs>
            <marker id="arrowhead-3b82f6" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
            </marker>
            <marker id="arrowhead-ef4444" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
            </marker>
            <marker id="arrowhead-10b981" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
            </marker>
            <marker id="arrowhead-gray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
            </marker>
          </defs>

          {/* Grid Lines */}
          <g stroke="#e2e8f0" strokeWidth="1">
            {Array.from({ length: 20 }).map((_, i) => (
              <line key={`v${i}`} x1={i * 50} y1={0} x2={i * 50} y2={dimensions.height} />
            ))}
            {Array.from({ length: 20 }).map((_, i) => (
              <line key={`h${i}`} x1={0} y1={i * 50} x2={dimensions.width} y2={i * 50} />
            ))}
          </g>
          
          {/* Axes */}
          <line x1={center.x} y1={0} x2={center.x} y2={dimensions.height} stroke="#cbd5e1" strokeWidth="2" />
          <line x1={0} y1={center.y} x2={dimensions.width} y2={center.y} stroke="#cbd5e1" strokeWidth="2" />

          {/* --- The Vectors --- */}

          {/* 1. The Original Key Vector K (User input) */}
          <VectorArrow 
            start={center} 
            end={toSvg(vectorK)} 
            color="#ef4444" 
            label="K"
            opacity={isNormalized ? 0.3 : 1} 
          />
          
          {/* 1b. The Normalized Key Vector (Effective K) - Only distinct if normalized is active */}
          {isNormalized && (
             <VectorArrow 
               start={center} 
               end={toSvg(scale(kNorm, 100))} // Display unit vector as 100px 
               color="#ef4444" 
               label="K_norm"
               strokeWidth={4}
             />
          )}

          {/* 2. The State Vector S (User input) */}
          <VectorArrow start={center} end={toSvg(vectorS)} color="#3b82f6" label="S" />

          {/* 3. The Removed Component (Interference) */}
          {/* We draw a dashed line from S tip to S_new tip. This represents -(KK^T)S */}
          <line 
            x1={toSvg(vectorS).x} 
            y1={toSvg(vectorS).y} 
            x2={toSvg(sNew).x} 
            y2={toSvg(sNew).y} 
            stroke="#94a3b8" 
            strokeWidth="2" 
            strokeDasharray="5,5" 
          />
          <text 
             x={(toSvg(vectorS).x + toSvg(sNew).x) / 2 + 10} 
             y={(toSvg(vectorS).y + toSvg(sNew).y) / 2} 
             fill="#64748b" 
             fontSize="12"
             className="italic"
          >
            erased
          </text>

          {/* 4. The Result Vector S_new */}
          <VectorArrow 
            start={center} 
            end={toSvg(sNew)} 
            color="#10b981" 
            label="S_new" 
            strokeWidth={4}
          />

          {/* Projection Line Helper (Only makes sense if normalized) */}
          {isNormalized && (
             <line 
               x1={toSvg(sNew).x}
               y1={toSvg(sNew).y}
               x2={toSvg(add(sNew, scale(kNorm, 1000))).x} // Line extending perpendicular
               y2={toSvg(add(sNew, scale(kNorm, -1000))).y}
               stroke="#10b981"
               strokeOpacity="0.1"
               strokeWidth="1"
             />
          )}

          {/* Interaction Handles */}
          <DraggableHandle 
            x={toSvg(vectorK).x} 
            y={toSvg(vectorK).y} 
            color="#ef4444" 
            onDrag={handleDragK} 
            label={isNormalized ? "K (Direction)" : "K (Drag me)"}
          />
          <DraggableHandle 
            x={toSvg(vectorS).x} 
            y={toSvg(vectorS).y} 
            color="#3b82f6" 
            onDrag={handleDragS} 
            label="S"
          />

        </svg>

        {/* Legend / Status Overlay */}
        <div className="absolute top-4 left-4 bg-white/90 backdrop-blur p-4 rounded-lg shadow border border-gray-200 text-sm max-w-xs">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span className="font-semibold text-gray-700">State Vector (S)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span className="font-semibold text-gray-700">Key Vector (K)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="font-semibold text-gray-700">Cleaned State (S_new)</span>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex justify-between items-center mb-1">
              <span className="text-gray-500">Angle (S, S_new):</span>
              <span className="font-mono">{Math.acos(Math.max(-1, Math.min(1, dot(norm(vectorS), norm(sNew))))).toFixed(2)} rad</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Is Orthogonal?</span>
              <span className={`font-mono font-bold ${isOrthogonal ? 'text-green-600' : 'text-red-500'}`}>
                {isOrthogonal ? "YES (Clean)" : "NO (Distorted)"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="w-full bg-gray-100 p-6">
        <div className="flex flex-col md:flex-row gap-8 items-center justify-between">
          
          {/* Main Toggle */}
          <div className="flex-1 flex flex-col items-start gap-4">
             <div className="flex items-center gap-3">
               <button
                 onClick={() => setIsNormalized(!isNormalized)}
                 className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                   isNormalized ? 'bg-blue-600' : 'bg-gray-300'
                 }`}
               >
                 <span
                   className={`${
                     isNormalized ? 'translate-x-7' : 'translate-x-1'
                   } inline-block h-6 w-6 transform rounded-full bg-white transition-transform duration-200 ease-in-out`}
                 />
               </button>
               <span className="text-lg font-bold text-gray-800">
                 Apply L2-Normalization to K
               </span>
             </div>
             
             <div className={`text-sm px-4 py-3 rounded border transition-all duration-500 ${isNormalized ? 'bg-green-100 border-green-200 text-green-900' : 'bg-red-50 border-red-100 text-red-900'}`}>
               {isNormalized ? (
                 <div className="flex gap-2">
                   <Check size={20} className="shrink-0" />
                   <p>
                     <strong>Protection Active:</strong> K is treated as a unit vector. <br/>
                     <code className="font-mono text-xs">P = I - K_unit K_unitᵀ</code> is a projection matrix.
                     <br/>
                     Result: S_new is the orthogonal projection of S onto the plane perpendicular to K. Perfect cleanup.
                   </p>
                 </div>
               ) : (
                 <div className="flex gap-2">
                   <X size={20} className="shrink-0" />
                   <p>
                     <strong>Protection Inactive:</strong> K has magnitude {kMagnitude}. <br/>
                     <code className="font-mono text-xs">M = I - KKᵀ</code> is NOT a projection.
                     <br/>
                     Result: S_new is distorted. If |K| &gt; 1, S is flipped and scaled uncontrollably.
                   </p>
                 </div>
               )}
             </div>
          </div>

          {/* Math Panel */}
          <div className="flex-1 bg-white p-4 rounded shadow-sm border border-gray-200 font-mono text-sm overflow-x-auto">
            <h3 className="text-gray-500 font-sans font-bold mb-2 uppercase text-xs tracking-wider">Computation Log</h3>
            
            <div className="space-y-2">
               <div className="flex justify-between">
                 <span>||K|| (Norm):</span>
                 <span className={isNormalized ? "text-gray-400 line-through decoration-red-500" : "text-gray-800"}>
                   {(len(vectorK) / PIXELS_PER_UNIT).toFixed(3)}
                 </span>
                 {isNormalized && <span className="text-blue-600 font-bold ml-2">1.000 (Forced)</span>}
               </div>
               
               <div className="flex justify-between">
                 <span>Projection Scalar (KᵀS):</span>
                 {/* Visual scalar scaled back to logical units for display */}
                 <span className={Math.abs(dot(kEff, vectorS)/PIXELS_PER_UNIT) > 2 ? "text-red-500 font-bold" : "text-gray-800"}>
                   {(dot(kEff, vectorS) / PIXELS_PER_UNIT * (isNormalized ? PIXELS_PER_UNIT : 1)).toFixed(2)}
                 </span>
               </div>

               <div className="border-t border-gray-100 my-2"></div>
               
               <div className="text-xs text-gray-500 mb-1">Update Step:</div>
               <div className="bg-gray-50 p-2 rounded">
                 S_new = S - <span className="text-red-500">K</span>(<span className="text-red-500">K</span>ᵀ<span className="text-blue-600">S</span>)
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}