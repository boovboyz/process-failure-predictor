import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { BarChart3, TrendingUp, Clock, AlertCircle, ArrowRight, Loader2 } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'
import { api, EvaluationResults } from '../api/client'

export default function EvaluationPage() {
    const navigate = useNavigate()
    const modelId = sessionStorage.getItem('modelId')

    const { data: evaluation, isLoading, error } = useQuery({
        queryKey: ['evaluation', modelId],
        queryFn: () => api.getEvaluation(modelId!),
        enabled: !!modelId,
    })

    useEffect(() => {
        if (!modelId) {
            navigate('/training')
        }
    }, [modelId, navigate])

    if (!modelId) return null

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-96">
                <Loader2 className="w-8 h-8 text-primary-400 animate-spin" />
            </div>
        )
    }

    if (error || !evaluation) {
        return (
            <div className="max-w-4xl mx-auto">
                <div className="glass-card border-red-500/30">
                    <div className="flex items-center gap-3 text-red-400">
                        <AlertCircle className="w-6 h-6" />
                        <p>Failed to load evaluation results</p>
                    </div>
                </div>
            </div>
        )
    }

    const earlyDetectionData = Object.entries(evaluation.early_detection).map(([pct, acc]) => ({
        prefix: pct,
        accuracy: Math.round(acc * 100),
    }))

    return (
        <div className="max-w-6xl mx-auto animate-fade-in">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Evaluation Results</h1>
                <p className="text-dark-400">
                    Model performance on held-out test set
                </p>
            </header>

            {/* Primary Metrics */}
            <div className="grid grid-cols-3 gap-4 mb-8">
                {/* Next Activity */}
                <div className="glass-card card-hover">
                    <div className="flex items-center gap-2 text-primary-400 mb-4">
                        <BarChart3 className="w-5 h-5" />
                        <span className="font-medium">Next Activity</span>
                    </div>
                    <div className="space-y-3">
                        <div>
                            <div className="metric-value">{Math.round(evaluation.next_activity_accuracy * 100)}%</div>
                            <div className="metric-label">Accuracy</div>
                        </div>
                        <div className="text-sm text-dark-400">
                            Top-3: {Math.round(evaluation.next_activity_top3_accuracy * 100)}%
                        </div>
                        <div className="text-sm text-dark-400">
                            Macro F1: {evaluation.next_activity_macro_f1.toFixed(3)}
                        </div>
                    </div>
                </div>

                {/* Outcome */}
                <div className="glass-card card-hover">
                    <div className="flex items-center gap-2 text-purple-400 mb-4">
                        <TrendingUp className="w-5 h-5" />
                        <span className="font-medium">Outcome</span>
                    </div>
                    <div className="space-y-3">
                        <div>
                            <div className="metric-value">{(evaluation.outcome_auc_roc * 100).toFixed(1)}%</div>
                            <div className="metric-label">AUC-ROC</div>
                        </div>
                        <div className="text-sm text-dark-400">
                            Precision: {(evaluation.outcome_precision * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-dark-400">
                            F1 Score: {evaluation.outcome_f1.toFixed(3)}
                        </div>
                    </div>
                </div>

                {/* Time */}
                <div className="glass-card card-hover">
                    <div className="flex items-center gap-2 text-green-400 mb-4">
                        <Clock className="w-5 h-5" />
                        <span className="font-medium">Time Prediction</span>
                    </div>
                    <div className="space-y-3">
                        <div>
                            <div className="metric-value">{evaluation.time_mae_hours.toFixed(1)}h</div>
                            <div className="metric-label">Mean Absolute Error</div>
                        </div>
                        <div className="text-sm text-dark-400">
                            RMSE: {evaluation.time_rmse_hours.toFixed(1)} hours
                        </div>
                        <div className="text-sm text-dark-400">
                            Coverage: {Math.round(evaluation.time_coverage * 100)}%
                        </div>
                    </div>
                </div>
            </div>

            {/* Calibration & Early Detection */}
            <div className="grid grid-cols-2 gap-6 mb-8">
                {/* ECE */}
                <div className="glass-card">
                    <h3 className="text-lg font-semibold mb-4">Calibration Quality</h3>
                    <div className="flex items-center gap-4">
                        <div>
                            <div className="text-4xl font-bold text-primary-400">
                                {(evaluation.expected_calibration_error * 100).toFixed(2)}%
                            </div>
                            <div className="text-sm text-dark-400">Expected Calibration Error</div>
                        </div>
                        <div className={`px-3 py-1.5 rounded-lg text-sm font-medium ${evaluation.expected_calibration_error < 0.05
                            ? 'bg-green-500/20 text-green-400'
                            : evaluation.expected_calibration_error < 0.1
                                ? 'bg-yellow-500/20 text-yellow-400'
                                : 'bg-red-500/20 text-red-400'
                            }`}>
                            {evaluation.expected_calibration_error < 0.05
                                ? 'Well Calibrated'
                                : evaluation.expected_calibration_error < 0.1
                                    ? 'Moderately Calibrated'
                                    : 'Needs Calibration'}
                        </div>
                    </div>
                </div>

                {/* Early Detection */}
                <div className="glass-card">
                    <h3 className="text-lg font-semibold mb-4">Early Detection</h3>
                    <ResponsiveContainer width="100%" height={120}>
                        <BarChart data={earlyDetectionData}>
                            <XAxis dataKey="prefix" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                            <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                            <Tooltip
                                contentStyle={{
                                    background: '#1e293b',
                                    border: '1px solid #334155',
                                    borderRadius: '8px'
                                }}
                            />
                            <Bar dataKey="accuracy" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-dark-400 mt-2 text-center">
                        Accuracy at different prefix completion percentages
                    </p>
                </div>
            </div>

            {/* Sample Predictions */}
            <div className="glass-card mb-8">
                <h3 className="text-lg font-semibold mb-4">Sample Predictions</h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="text-left text-dark-400 border-b border-dark-700">
                                <th className="pb-3 pr-4">Case</th>
                                <th className="pb-3 pr-4">Prefix</th>
                                <th className="pb-3 pr-4">Last Activity</th>
                                <th className="pb-3 pr-4">Predicted</th>
                                <th className="pb-3 pr-4">Probability</th>
                                <th className="pb-3 pr-4">Actual</th>
                                <th className="pb-3">Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {evaluation.sample_predictions.slice(0, 5).map((pred, i) => (
                                <tr key={i} className="border-b border-dark-700/50">
                                    <td className="py-3 pr-4 font-mono text-xs">{pred.case_id}</td>
                                    <td className="py-3 pr-4">{pred.prefix_length} events</td>
                                    <td className="py-3 pr-4">{pred.last_activity}</td>
                                    <td className="py-3 pr-4">{pred.predicted_next}</td>
                                    <td className="py-3 pr-4">{Math.round(pred.predicted_probability * 100)}%</td>
                                    <td className="py-3 pr-4">{pred.actual_next}</td>
                                    <td className="py-3">
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${pred.correct
                                            ? 'bg-green-500/20 text-green-400'
                                            : 'bg-red-500/20 text-red-400'
                                            }`}>
                                            {pred.correct ? '✓ Correct' : '✗ Wrong'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end">
                <button
                    className="btn-primary"
                    onClick={() => navigate('/test')}
                >
                    Test Model
                    <ArrowRight className="w-5 h-5" />
                </button>
            </div>
        </div>
    )
}
