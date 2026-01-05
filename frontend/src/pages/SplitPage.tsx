import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { GitBranch, AlertTriangle, ArrowRight, Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, SplitResponse } from '../api/client'

export default function SplitPage() {
    const navigate = useNavigate()
    const [trainRatio, setTrainRatio] = useState(0.9)
    const [splitResult, setSplitResult] = useState<SplitResponse | null>(null)
    const logId = sessionStorage.getItem('logId')

    const splitMutation = useMutation({
        mutationFn: () => api.splitData(logId!, trainRatio),
        onSuccess: (data) => {
            setSplitResult(data)
        },
    })

    useEffect(() => {
        if (!logId) {
            navigate('/')
        }
    }, [logId, navigate])

    if (!logId) {
        return null
    }

    const trainPct = Math.round(trainRatio * 100)
    const testPct = 100 - trainPct

    return (
        <div className="max-w-4xl mx-auto animate-fade-in">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Temporal Data Split</h1>
                <p className="text-dark-400">
                    Split your data chronologically to prevent data leakage
                </p>
            </header>

            {/* Split Configuration */}
            {!splitResult && (
                <div className="glass-card mb-8">
                    <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                        <GitBranch className="w-5 h-5 text-primary-400" />
                        Configure Split
                    </h2>

                    {/* Ratio Slider */}
                    <div className="mb-8">
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-dark-400">Training Ratio</span>
                            <span className="font-medium">{trainPct}% / {testPct}%</span>
                        </div>
                        <input
                            type="range"
                            min="70"
                            max="95"
                            value={trainRatio * 100}
                            onChange={(e) => setTrainRatio(parseInt(e.target.value) / 100)}
                            className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-dark-500 mt-1">
                            <span>70%</span>
                            <span>95%</span>
                        </div>
                    </div>

                    {/* Visual Timeline */}
                    <div className="mb-8">
                        <p className="text-sm text-dark-400 mb-3">Timeline Preview</p>
                        <div className="h-12 rounded-lg overflow-hidden flex">
                            <div
                                className="bg-gradient-to-r from-primary-600 to-primary-500 flex items-center justify-center text-sm font-medium transition-all duration-300"
                                style={{ width: `${trainPct}%` }}
                            >
                                Training ({trainPct}%)
                            </div>
                            <div
                                className="bg-gradient-to-r from-purple-600 to-purple-500 flex items-center justify-center text-sm font-medium transition-all duration-300"
                                style={{ width: `${testPct}%` }}
                            >
                                Test ({testPct}%)
                            </div>
                        </div>
                        <div className="flex justify-between mt-2 text-xs text-dark-500">
                            <span>Past</span>
                            <span>↓ Cutoff Point</span>
                            <span>Future</span>
                        </div>
                    </div>

                    <button
                        className="btn-primary w-full justify-center"
                        onClick={() => splitMutation.mutate()}
                        disabled={splitMutation.isPending}
                    >
                        {splitMutation.isPending ? (
                            <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Splitting Data...
                            </>
                        ) : (
                            <>
                                <GitBranch className="w-5 h-5" />
                                Perform Temporal Split
                            </>
                        )}
                    </button>
                </div>
            )}

            {/* Split Results */}
            {splitResult && (
                <div className="space-y-6 animate-slide-up">
                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-4">
                        <div className="metric-card">
                            <div className="metric-value">{splitResult.train_traces.toLocaleString()}</div>
                            <div className="metric-label">Training Traces</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{splitResult.test_traces.toLocaleString()}</div>
                            <div className="metric-label">Test Traces</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value text-2xl">
                                {Math.round(splitResult.effective_ratio * 100)}%
                            </div>
                            <div className="metric-label">Effective Ratio</div>
                        </div>
                    </div>

                    {/* Excluded Traces Warning */}
                    {splitResult.excluded_traces > 0 && (
                        <div className="glass-card border-yellow-500/30">
                            <div className="flex items-start gap-3 text-yellow-400">
                                <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                                <div>
                                    <p className="font-medium">
                                        {splitResult.excluded_traces} traces excluded
                                    </p>
                                    <p className="text-sm opacity-80">
                                        These traces span the cutoff point and were excluded to prevent data leakage.
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Warnings */}
                    {splitResult.warnings.length > 0 && (
                        <div className="glass-card border-orange-500/30">
                            {splitResult.warnings.map((warning, i) => (
                                <div key={i} className="flex items-start gap-3 text-orange-400">
                                    <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                                    <p className="text-sm">{warning}</p>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Cutoff Time */}
                    <div className="glass-card">
                        <p className="text-sm text-dark-400 mb-1">Cutoff Timestamp</p>
                        <p className="font-mono text-lg">
                            {new Date(splitResult.cutoff_time).toLocaleString()}
                        </p>
                    </div>

                    {/* Continue Button */}
                    <div className="flex justify-end">
                        <button
                            className="btn-primary"
                            onClick={() => navigate('/training')}
                        >
                            Train Models
                            <ArrowRight className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
