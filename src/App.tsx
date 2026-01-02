import { useState } from 'react'
import LinearTransformerDemo from './demos/fwp'
import HopfieldTransformerTutorial from './demos/modern-hopfield'
import FastWeightSSM from './demos/ssm'
import GeometricProjection from './demos/deltanet-is-rotation'
import LinearAttentionDemo from './demos/linear-attention'
import TTTDemo from './demos/ttt'

type TabId = 'fwp' | 'hopfield' | 'ssm' | 'deltanet' | 'linear-attention' | 'ttt'

interface TabConfig {
  id: TabId
  label: string
  component: React.ComponentType
}

const tabs: TabConfig[] = [
  { id: 'fwp', label: 'Linear Transformers = Fast Weight Programmers', component: LinearTransformerDemo },
  { id: 'hopfield', label: 'Hopfield Networks is All You Need', component: HopfieldTransformerTutorial },
  { id: 'ssm', label: 'State Space Models', component: FastWeightSSM },
  // { id: 'deltanet', label: 'DeltaNet', component: GeometricProjection },
  { id: 'linear-attention', label: 'Linear Attention', component: LinearAttentionDemo },
  { id: 'ttt', label: 'TTT', component: TTTDemo },
]

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('fwp')
  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-full mx-auto">
          <div className="flex">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="w-full">
        {ActiveComponent && <ActiveComponent />}
      </div>
    </div>
  )
}

export default App
