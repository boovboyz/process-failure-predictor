import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Cog, CheckCircle, XCircle, Loader2, ArrowRight, Clock } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, TrainingResponse } from '../api/client'

export default function TrainingPage() {
    const navigate = useNavigate()
    const [trainingResult, setTrainingResult] = useState<TrainingResponse | null>(null)
    const logId = sessionStorage.getItem('logId')

    const trainMutation = useMutation({
        mutationFn: () => api.trainModels(logId!),
        onSuccess: (data) => {
            setTrainingResult(data)
            sessionStorage.setItem('modelId', data.model_id)
        },
    })

    useEffect(() => {
        if (!logId) {
            navigate('/')
        }
    }, [logId, navigate])

    if (!logId) return null

    const formatTime = (seconds: number) => {
        if (seconds < 60) return `${seconds.toFixed(1)}s`
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs.toFixed(0)}s`
    }

    return (
        <div className="max-w-4xl mx-auto animate-fade-in">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Model Training</h1>
                <p className="text-dark-400">
                    Train XGBoost models for activity, outcome, and time prediction
                </p>
            </header>

            {/* Training Button */}
            {!trainingResult && !trainMutation.isPending && (
                <div className="glass-card text-center py-12">
                    <div className="w-20 h-20 rounded-full bg-primary-500/20 flex items-center justify-center mx-auto mb-6">
                        <Cog className="w-10 h-10 text-primary-400" />
                    </div>
                    <h2 className="text-xl font-semibold mb-4">Ready to Train</h2>
                    <p className="text-dark-400 mb-8 max-w-md mx-auto">
                        This will train three XGBoost models: next activity prediction,
                        outcome prediction, and remaining time estimation.
                    </p>
                    <button
                        className="btn-primary"
                        onClick={() => trainMutation.mutate()}
                    >
                        <Cog className="w-5 h-5" />
                        Start Training
                    </button>
                </div>
            )}

            {/* Training Progress */}
            {trainMutation.isPending && (
                <div className="glass-card text-center py-12 animate-pulse-slow">
                    <Loader2 className="w-16 h-16 text-primary-400 animate-spin mx-auto mb-6" />
                    <h2 className="text-xl font-semibold mb-2">Training Models...</h2>
                    <p className="text-dark-400">
                        This may take a few minutes depending on your dataset size
                    </p>

                    <div className="mt-8 space-y-4 max-w-md mx-auto">
                        {['Next Activity Model', 'Outcome Model', 'Time Model', 'Calibration'].map((task) => (
                            <div key={task} className="flex items-center gap-3 text-left">
                                <div className="w-6 h-6 rounded-full bg-primary-500/30 flex items-center justify-center">
                                    <div className="w-2 h-2 rounded-full bg-primary-400 animate-pulse" />
                                </div>
                                <span className="text-dark-300">{task}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Training Error */}
            {trainMutation.isError && (
                <div className="glass-card border-red-500/30 animate-slide-up">
                    <div className="flex items-start gap-3 text-red-400">
                        <XCircle className="w-6 h-6 flex-shrink-0" />
                        <div>
                            <p className="font-semibold text-lg">Training Failed</p>
                            <p className="text-sm opacity-80 mt-1">
                                {trainMutation.error?.message || 'An error occurred during training'}
                            </p>
                            <button
                                className="btn-secondary mt-4"
                                onClick={() => trainMutation.mutate()}
                            >
                                Retry Training
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Training Results */}
            {trainingResult && (
                <div className="space-y-6 animate-slide-up">
                    {/* Success Header */}
                    <div className="glass-card border-green-500/30">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                                <CheckCircle className="w-6 h-6 text-green-400" />
                            </div>
                            <div>
                                <p className="font-semibold text-lg">Training Complete!</p>
                                <div className="flex items-center gap-2 text-dark-400 text-sm">
                                    <Clock className="w-4 h-4" />
                                    Completed in {formatTime(trainingResult.training_time_seconds)}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-4">
                        <div className="metric-card card-hover">
                            <div className="text-sm text-dark-400 mb-2">Next Activity</div>
                            <div className="metric-value">
                                {Math.round((trainingResult.metrics.next_activity?.validation_score || 0) * 100)}%
                            </div>
                            <div className="metric-label">Validation Accuracy</div>
                        </div>
                        <div className="metric-card card-hover">
                            <div className="text-sm text-dark-400 mb-2">Outcome</div>
                            <div className="metric-value">
                                {Math.round((trainingResult.metrics.outcome?.validation_score || 0) * 100)}%
                            </div>
                            <div className="metric-label">Validation Accuracy</div>
                        </div>
                        <div className="metric-card card-hover">
                            <div className="text-sm text-dark-400 mb-2">Time</div>
                            <div className="metric-value text-2xl">
                                {Math.round((trainingResult.metrics.time?.validation_score || 0) * 100)}%
                            </div>
                            <div className="metric-label">Coverage Score</div>
                        </div>
                    </div>

                    {/* Model ID */}
                    <div className="glass-card">
                        <p className="text-sm text-dark-400 mb-1">Model ID</p>
                        <p className="font-mono text-sm">{trainingResult.model_id}</p>
                    </div>

                    {/* Navigation */}
                    <div className="flex justify-end">
                        <button
                            className="btn-primary"
                            onClick={() => navigate('/evaluation')}
                        >
                            View Evaluation Results
                            <ArrowRight className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
