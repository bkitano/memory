import { useState } from 'react'
import LinearTransformerDemo from './demos/fwp'
import HopfieldTransformerTutorial from './demos/modern-hopfield'

function App() {
  const [activeTab, setActiveTab] = useState<'fwp' | 'hopfield'>('fwp')

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
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="w-full">
        {activeTab === 'fwp' && <LinearTransformerDemo />}
        {activeTab === 'hopfield' && <HopfieldTransformerTutorial />}
      </div>
    </div>
  )
}

export default App
