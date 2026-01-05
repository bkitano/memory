import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import LinearTransformerDemo from './demos/fwp'
import HopfieldTransformerTutorial from './demos/modern-hopfield'
import FastWeightSSM from './demos/ssm'
import GeometricProjection from './demos/deltanet-is-rotation'
import LinearAttentionDemo from './demos/linear-attention'
import LinearAttentionUpdatesDemo from './demos/linear-attention-updates'
import TTTDemo from './demos/ttt'
import LandingPage from './components/LandingPage'
import TTTLinearRegressionDemo from './demos/ttt-linear-regression'
import TTTLinearRegression2DDemo from './demos/ttt-linear-regression-2d'

const routes = [
  { path: '/', label: 'Overview', element: <LandingPage /> },
  { path: '/fwp', label: 'Linear Transformers = Fast Weight Programmers', element: <LinearTransformerDemo /> },
  { path: '/hopfield', label: 'Hopfield Networks is All You Need', element: <HopfieldTransformerTutorial /> },
  { path: '/ssm', label: 'State Space Models', element: <FastWeightSSM /> },
  { path: '/deltanet', label: 'DeltaNet', element: <GeometricProjection /> },
  { path: '/linear-attention', label: 'Linear Attention', element: <LinearAttentionDemo /> },
  { path: '/linear-attention-updates', label: 'Linear Attention Updates', element: <LinearAttentionUpdatesDemo /> },
  { path: '/ttt', label: 'TTT', element: <TTTDemo /> },
  { path: '/ttt-linear-regression', label: 'TTT Linear Regression (1D)', element: <TTTLinearRegressionDemo /> },
  { path: '/ttt-linear-regression-2d', label: 'Delta-Net 2D TTT', element: <TTTLinearRegression2DDemo /> },
]

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-full mx-auto">
          <div className="flex">
            {routes.map((route) => (
              <NavLink
                key={route.path}
                to={route.path}
                end
                className={({ isActive }) =>
                  `flex-1 px-6 py-4 text-sm font-medium transition-colors ${isActive
                    ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`
                }
              >
                {route.label}
              </NavLink>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="w-full">
        <Routes>
          {routes.map((route) => (
            <Route key={route.path} path={route.path} element={route.element} />
          ))}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
