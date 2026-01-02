import { useState } from 'react'
import LinearTransformerDemo from './demos/fwp'
import HopfieldTransformerTutorial from './demos/modern-hopfield'
import FastWeightSSM from './demos/ssm'
import GeometricProjection from './demos/deltanet-is-rotation'
import LinearAttentionDemo from './demos/linear-attention'
import TTTDemo from './demos/ttt'

function App() {
  const [activeTab, setActiveTab] = useState<'fwp' | 'hopfield' | 'ssm' | 'deltanet' | 'linear-attention' | 'ttt'>('fwp')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-full mx-auto">
          <div className="flex">
            <button
              onClick={() => setActiveTab('fwp')}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'fwp'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Linear Transformers = Fast Weight Programmers
            </button>
            <button
              onClick={() => setActiveTab('hopfield')}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'hopfield'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Hopfield Networks is All You Need
            </button>
            <button
              onClick={() => setActiveTab('ssm')}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'ssm'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              State Space Models
            </button>
            <button
              onClick={() => setActiveTab('deltanet')}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'deltanet'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              DeltaNet
            </button>
            <button
              onClick={() => setActiveTab('linear-attention')}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'linear-attention'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Linear Attention
            </button>
            <button
              onClick={() => setActiveTab('ttt')}
              className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'ttt'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              TTT
            </button>
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="w-full">
        {activeTab === 'fwp' && <LinearTransformerDemo />}
        {activeTab === 'hopfield' && <HopfieldTransformerTutorial />}
        {activeTab === 'ssm' && <FastWeightSSM />}
        {activeTab === 'deltanet' && <GeometricProjection />}
        {activeTab === 'linear-attention' && <LinearAttentionDemo />}
        {activeTab === 'ttt' && <TTTDemo />}
      </div>
    </div>
  )
}

export default App
