import { Routes, Route, Link, useLocation } from 'react-router-dom'
import {
    Upload,
    GitBranch,
    Cog,
    BarChart3,
    FlaskConical
} from 'lucide-react'

import UploadPage from './pages/UploadPage'
import SplitPage from './pages/SplitPage'
import TrainingPage from './pages/TrainingPage'
import EvaluationPage from './pages/EvaluationPage'
import ModelTesterPage from './pages/ModelTesterPage'

const navItems = [
    { path: '/', icon: Upload, label: 'Upload' },
    { path: '/split', icon: GitBranch, label: 'Split' },
    { path: '/training', icon: Cog, label: 'Train' },
    { path: '/evaluation', icon: BarChart3, label: 'Evaluate' },
    { path: '/test', icon: FlaskConical, label: 'Test Model' },
]

function App() {
    const location = useLocation()

    return (
        <div className="min-h-screen flex">
            {/* Sidebar */}
            <nav className="w-64 glass fixed h-full flex flex-col">
                {/* Logo */}
                <div className="p-6 border-b border-dark-700/50">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
                        Process Predictor
                    </h1>
                    <p className="text-xs text-dark-400 mt-1">Failure Prediction Demo</p>
                </div>

                {/* Navigation */}
                <div className="flex-1 py-6">
                    <ul className="space-y-1 px-3">
                        {navItems.map((item) => {
                            const isActive = location.pathname === item.path
                            const Icon = item.icon
                            return (
                                <li key={item.path}>
                                    <Link
                                        to={item.path}
                                        className={`
                      flex items-center gap-3 px-4 py-3 rounded-lg
                      transition-all duration-200
                      ${isActive
                                                ? 'bg-primary-600/20 text-primary-400 border-l-2 border-primary-400'
                                                : 'text-dark-400 hover:text-white hover:bg-dark-700/50'
                                            }
                    `}
                                    >
                                        <Icon size={18} />
                                        <span className="font-medium">{item.label}</span>
                                    </Link>
                                </li>
                            )
                        })}
                    </ul>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-dark-700/50 text-xs text-dark-500">
                    Process Failure Predictor v1.0
                </div>
            </nav>

            {/* Main content */}
            <main className="flex-1 ml-64 p-8 min-h-screen overflow-y-auto">
                <Routes>
                    <Route path="/" element={<UploadPage />} />
                    <Route path="/split" element={<SplitPage />} />
                    <Route path="/training" element={<TrainingPage />} />
                    <Route path="/evaluation" element={<EvaluationPage />} />
                    <Route path="/test" element={<ModelTesterPage />} />
                </Routes>
            </main>
        </div>
    )
}

export default App

