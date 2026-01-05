import { useEffect, useState, useRef } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
    Play, Pause, SkipForward, RotateCcw,
    ChevronDown, Check, X, Loader2, RefreshCw,
    Sparkles, Activity, Clock, Target, AlertTriangle,
    ChevronLeft, ChevronRight, Zap,
    FlaskConical
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { api, SimulatorStepResult } from '../api/client'

function ConfidenceBadge({ level }: { level: string }) {
    const colors = {
        HIGH: 'bg-green-500/20 text-green-400 border-green-500/30',
        MEDIUM: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
        LOW: 'bg-red-500/20 text-red-400 border-red-500/30',
    }
    return (
        <span className={`px-2 py-0.5 rounded border text-xs font-medium ${colors[level as keyof typeof colors] || colors.LOW}`}>
            {level}
        </span>
    )
}

export default function ModelTesterPage() {
    const navigate = useNavigate()
    const modelId = sessionStorage.getItem('modelId')

    const [selectedTrace, setSelectedTrace] = useState<string | null>(null)
    const [currentStep, setCurrentStep] = useState<SimulatorStepResult | null>(null)
    const [totalSteps, setTotalSteps] = useState(0)
    const [isPlaying, setIsPlaying] = useState(false)
    const [evolution, setEvolution] = useState<any[]>([])
    const autoPlayRef = useRef<NodeJS.Timeout | null>(null)

    // Fetch available test traces
    const { data: traces, isLoading: loadingTraces } = useQuery({
        queryKey: ['test-traces', modelId],
        queryFn: () => api.getSimulatorTraces(modelId!, 30),
        enabled: !!modelId,
    })

    // Load trace mutation
    const loadMutation = useMutation({
        mutationFn: (traceId: string) => api.loadSimulatorTrace(modelId!, traceId),
        onSuccess: (data) => {
            setTotalSteps(data.state.total_steps)
            setEvolution([])
            if (data.initial_prediction) {
                setCurrentStep(data.initial_prediction)
                // Add initial point to evolution
                setEvolution([{
                    step: 1,
                    nextConf: Math.round(data.initial_prediction.predicted_probability * 100),
                    outcomeConf: Math.round(data.initial_prediction.outcome_probability * 100),
                    correct: data.initial_prediction.next_correct ? 1 : 0,
                }])
            }
        },
    })

    // Step forward mutation
    const stepMutation = useMutation({
        mutationFn: () => api.simulatorStep(modelId!),
        onSuccess: (data) => {
            setCurrentStep(data.result)
            setEvolution(prev => [...prev, {
                step: data.result.step + 1,
                nextConf: Math.round(data.result.predicted_probability * 100),
                outcomeConf: Math.round(data.result.outcome_probability * 100),
                correct: data.result.next_correct ? 1 : 0,
            }])
        },
    })

    // Step backward mutation
    const stepBackMutation = useMutation({
        mutationFn: () => api.simulatorStepBack(modelId!),
        onSuccess: (data) => {
            if (data.result) {
                setCurrentStep(data.result)
            }
        },
    })

    // Reset mutation
    const resetMutation = useMutation({
        mutationFn: () => api.resetSimulator(modelId!),
        onSuccess: (data) => {
            setCurrentStep(data.initial_prediction)
            setEvolution(data.initial_prediction ? [{
                step: 1,
                nextConf: Math.round(data.initial_prediction.predicted_probability * 100),
                outcomeConf: Math.round(data.initial_prediction.outcome_probability * 100),
                correct: data.initial_prediction.next_correct ? 1 : 0,
            }] : [])
        },
    })

    // Jump to step mutation
    const jumpMutation = useMutation({
        mutationFn: (step: number) => api.simulatorJump(modelId!, step),
        onSuccess: (data) => {
            setCurrentStep(data.result)
        },
    })

    // Redirect if no model
    useEffect(() => {
        if (!modelId) {
            navigate('/training')
        }
    }, [modelId, navigate])

    // Load trace when selected
    useEffect(() => {
        if (selectedTrace) {
            loadMutation.mutate(selectedTrace)
        }
    }, [selectedTrace])

    // Auto-play logic
    useEffect(() => {
        if (isPlaying && currentStep && currentStep.step < totalSteps - 1) {
            autoPlayRef.current = setTimeout(() => {
                stepMutation.mutate()
            }, 1200)
        } else if (isPlaying) {
            setIsPlaying(false)
        }

        return () => {
            if (autoPlayRef.current) {
                clearTimeout(autoPlayRef.current)
            }
        }
    }, [isPlaying, currentStep?.step])

    if (!modelId) return null

    const handleLoadRandom = () => {
        if (traces && traces.length > 0) {
            const randomIdx = Math.floor(Math.random() * traces.length)
            setSelectedTrace(traces[randomIdx].trace_id)
        }
    }

    const isAtEnd = !!(currentStep && currentStep.step >= totalSteps - 1)
    const isAtStart = !currentStep || currentStep.step === 0
    const progressPct = currentStep ? ((currentStep.step + 1) / totalSteps) * 100 : 0

    return (
        <div className="max-w-7xl mx-auto animate-fade-in">
            {/* Header */}
            <header className="mb-8">
                <div className="flex items-center gap-3 mb-2">
                    <FlaskConical className="w-8 h-8 text-primary-400" />
                    <h1 className="text-3xl font-bold">Model Tester</h1>
                </div>
                <p className="text-dark-400">
                    Test your trained model by stepping through real cases event-by-event
                </p>
            </header>

            {/* Case Selector */}
            <div className="glass-card mb-6">
                <div className="flex gap-4">
                    <div className="flex-1">
                        <label className="block text-sm text-dark-400 mb-2">Select Test Case</label>
                        <div className="relative">
                            <select
                                value={selectedTrace || ''}
                                onChange={(e) => setSelectedTrace(e.target.value)}
                                className="w-full bg-dark-800 border border-dark-600 rounded-lg px-4 py-3 text-white appearance-none cursor-pointer focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                                disabled={loadingTraces}
                            >
                                <option value="">Choose a case to test...</option>
                                {traces?.map((t) => (
                                    <option key={t.trace_id} value={t.trace_id}>
                                        {t.case_id} — {t.event_count} events — {t.duration_hours.toFixed(1)}h
                                    </option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400 pointer-events-none" />
                        </div>
                    </div>
                    <div className="flex items-end gap-2">
                        <button
                            className="btn-secondary"
                            onClick={handleLoadRandom}
                            disabled={!traces || traces.length === 0 || loadMutation.isPending}
                        >
                            {loadMutation.isPending ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <RefreshCw className="w-4 h-4" />
                            )}
                            Random
                        </button>
                    </div>
                </div>
            </div>

            {/* Loading State */}
            {loadMutation.isPending && (
                <div className="glass-card text-center py-16">
                    <Loader2 className="w-12 h-12 text-primary-400 animate-spin mx-auto mb-4" />
                    <p className="text-dark-400">Loading case...</p>
                </div>
            )}

            {/* Main Content - When case is loaded */}
            {currentStep && !loadMutation.isPending && (
                <div className="space-y-6 animate-slide-up">

                    {/* Timeline & Controls */}
                    <div className="glass-card">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <Activity className="w-5 h-5 text-primary-400" />
                                <span className="font-semibold">Event Timeline</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-sm text-dark-400">Step</span>
                                <span className="text-lg font-bold text-primary-400">{currentStep.step + 1}</span>
                                <span className="text-sm text-dark-400">of {totalSteps}</span>
                            </div>
                        </div>

                        {/* Progress Bar */}
                        <div className="mb-4">
                            <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-300"
                                    style={{ width: `${progressPct}%` }}
                                />
                            </div>
                        </div>

                        {/* Event Steps */}
                        <div className="flex items-center gap-1 overflow-x-auto pb-2 mb-4">
                            {Array.from({ length: totalSteps }).map((_, i) => (
                                <button
                                    key={i}
                                    onClick={() => jumpMutation.mutate(i)}
                                    disabled={jumpMutation.isPending}
                                    className={`
                                        w-9 h-9 rounded-full flex items-center justify-center text-xs font-medium
                                        transition-all duration-200 flex-shrink-0
                                        ${i <= currentStep.step
                                            ? 'bg-primary-600 text-white'
                                            : 'bg-dark-700 text-dark-400 border border-dark-600 hover:border-primary-500'
                                        }
                                        ${i === currentStep.step ? 'ring-2 ring-primary-400 ring-offset-2 ring-offset-dark-900 scale-110' : ''}
                                    `}
                                >
                                    {i + 1}
                                </button>
                            ))}
                        </div>

                        {/* Playback Controls */}
                        <div className="flex justify-center gap-2">
                            <button
                                className="btn-secondary px-3"
                                onClick={() => resetMutation.mutate()}
                                disabled={resetMutation.isPending || isAtStart}
                                title="Reset"
                            >
                                <RotateCcw className="w-4 h-4" />
                            </button>
                            <button
                                className="btn-secondary px-3"
                                onClick={() => stepBackMutation.mutate()}
                                disabled={stepBackMutation.isPending || isAtStart}
                                title="Previous"
                            >
                                <ChevronLeft className="w-4 h-4" />
                            </button>
                            <button
                                className={`btn-primary w-28 justify-center ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-700' : ''}`}
                                onClick={() => setIsPlaying(!isPlaying)}
                                disabled={isAtEnd}
                            >
                                {isPlaying ? (
                                    <><Pause className="w-4 h-4" /> Pause</>
                                ) : (
                                    <><Play className="w-4 h-4" /> {isAtStart ? 'Play' : 'Resume'}</>
                                )}
                            </button>
                            <button
                                className="btn-secondary px-3"
                                onClick={() => stepMutation.mutate()}
                                disabled={isAtEnd || stepMutation.isPending}
                                title="Next"
                            >
                                {stepMutation.isPending ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <ChevronRight className="w-4 h-4" />
                                )}
                            </button>
                            <button
                                className="btn-secondary px-3"
                                onClick={() => jumpMutation.mutate(totalSteps - 1)}
                                disabled={isAtEnd || jumpMutation.isPending}
                                title="Jump to End"
                            >
                                <SkipForward className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {/* Current Event & Predictions Row */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                        {/* Current Event Card */}
                        <div className="glass-card border-primary-500/30">
                            <div className="flex items-center gap-2 text-primary-400 mb-4">
                                <Activity className="w-5 h-5" />
                                <span className="font-semibold">Current Event</span>
                            </div>
                            <div className="space-y-3">
                                <div>
                                    <p className="text-xs text-dark-400 mb-1">Activity</p>
                                    <p className="text-xl font-bold">{currentStep.current_activity}</p>
                                </div>
                                <div>
                                    <p className="text-xs text-dark-400 mb-1">Timestamp</p>
                                    <p className="font-mono text-sm">
                                        {currentStep.current_events[currentStep.step]?.timestamp
                                            ? new Date(currentStep.current_events[currentStep.step].timestamp).toLocaleString()
                                            : '—'}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-xs text-dark-400 mb-1">Resource</p>
                                    <p className="font-medium">
                                        {currentStep.current_events[currentStep.step]?.resource || 'Not specified'}
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* WHAT/WHEN Predictions */}
                        <div className="glass-card">
                            <div className="flex items-center gap-2 text-primary-400 mb-4">
                                <Zap className="w-5 h-5" />
                                <span className="font-semibold">WHAT & WHEN</span>
                            </div>
                            <div className="space-y-4">
                                {/* Next Activity */}
                                <div className="p-3 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs text-dark-400">Next Activity</span>
                                        <ConfidenceBadge level={currentStep.predicted_confidence} />
                                    </div>
                                    <p className="text-lg font-bold mb-1">{currentStep.predicted_next}</p>
                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-primary-500"
                                                style={{ width: `${currentStep.predicted_probability * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-xs font-medium text-primary-400">
                                            {Math.round(currentStep.predicted_probability * 100)}%
                                        </span>
                                    </div>
                                    {/* Alternatives */}
                                    {currentStep.alternatives.length > 0 && (
                                        <div className="mt-2 flex flex-wrap gap-1">
                                            {currentStep.alternatives.slice(0, 2).map((alt) => (
                                                <span key={alt.activity} className="text-xs bg-dark-700 px-2 py-0.5 rounded">
                                                    {alt.activity}: {Math.round(alt.probability * 100)}%
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* Remaining Time */}
                                <div className="p-3 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center gap-1 text-xs text-dark-400 mb-2">
                                        <Clock className="w-3 h-3" />
                                        Remaining Time
                                    </div>
                                    <p className="text-lg font-bold text-green-400">
                                        ~{currentStep.predicted_time_hours.toFixed(1)} hours
                                    </p>
                                    <p className="text-xs text-dark-400">
                                        Range: {currentStep.time_lower_hours.toFixed(1)} - {currentStep.time_upper_hours.toFixed(1)}h
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* HOW/WHY Predictions */}
                        <div className="glass-card">
                            <div className="flex items-center gap-2 text-purple-400 mb-4">
                                <Target className="w-5 h-5" />
                                <span className="font-semibold">HOW & WHY</span>
                            </div>
                            <div className="space-y-4">
                                {/* Outcome */}
                                <div className="p-3 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs text-dark-400">Predicted Outcome</span>
                                        <ConfidenceBadge level={currentStep.outcome_confidence} />
                                    </div>
                                    <p className="text-lg font-bold">{currentStep.predicted_outcome}</p>
                                    <p className="text-xs text-dark-400 mt-1">
                                        {Math.round(currentStep.outcome_probability * 100)}% probability
                                    </p>
                                </div>

                                {/* Risk Factors */}
                                {currentStep.risk_factors.length > 0 && (
                                    <div className="p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
                                        <div className="flex items-center gap-1 text-yellow-400 text-xs mb-2">
                                            <AlertTriangle className="w-3 h-3" />
                                            Risk Factors
                                        </div>
                                        <ul className="space-y-1">
                                            {currentStep.risk_factors.slice(0, 2).map((risk, i) => (
                                                <li key={i} className="text-xs text-dark-300 flex items-start gap-1">
                                                    <span className="text-yellow-500">•</span>
                                                    {risk}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* AI Recommendations */}
                    {currentStep.recommendations && currentStep.recommendations.length > 0 && (
                        <div className="glass-card border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-primary-900/20">
                            <div className="flex items-center gap-2 text-purple-400 mb-4">
                                <Sparkles className="w-5 h-5" />
                                <span className="font-semibold">AI Recommendations</span>
                                <span className="text-xs bg-purple-500/20 px-2 py-0.5 rounded-full">Powered by Claude</span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                {currentStep.recommendations.map((rec, i) => (
                                    <div
                                        key={i}
                                        className="flex items-start gap-3 p-3 bg-dark-800/50 rounded-lg border border-purple-500/20"
                                    >
                                        <div className="w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center flex-shrink-0 text-purple-400 text-sm font-bold">
                                            {i + 1}
                                        </div>
                                        <p className="text-sm text-dark-200">{rec}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Ground Truth Comparison */}
                    <div className="glass-card border-green-500/30">
                        <div className="flex items-center gap-2 text-green-400 mb-4">
                            <Check className="w-5 h-5" />
                            <span className="font-semibold">Ground Truth Comparison</span>
                        </div>
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-3 bg-dark-800/50 rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs text-dark-400">Actual Next Activity</span>
                                    {currentStep.next_correct !== null && (
                                        currentStep.next_correct ? (
                                            <span className="flex items-center gap-1 text-green-400 text-xs">
                                                <Check className="w-3 h-3" /> Correct
                                            </span>
                                        ) : (
                                            <span className="flex items-center gap-1 text-red-400 text-xs">
                                                <X className="w-3 h-3" /> Wrong
                                            </span>
                                        )
                                    )}
                                </div>
                                <p className="font-bold">
                                    {currentStep.actual_next || '— (End of trace)'}
                                </p>
                            </div>
                            <div className="p-3 bg-dark-800/50 rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs text-dark-400">Time Prediction Error</span>
                                </div>
                                <p className={`font-bold ${currentStep.time_error_hours !== null
                                    ? currentStep.time_error_hours < 1
                                        ? 'text-green-400'
                                        : currentStep.time_error_hours < 4
                                            ? 'text-yellow-400'
                                            : 'text-red-400'
                                    : 'text-dark-400'
                                    }`}>
                                    {currentStep.time_error_hours !== null
                                        ? `±${currentStep.time_error_hours.toFixed(1)} hours`
                                        : '—'}
                                </p>
                            </div>
                            <div className="p-3 bg-dark-800/50 rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs text-dark-400">Actual Outcome</span>
                                    {currentStep.outcome_correct !== null && (
                                        currentStep.outcome_correct ? (
                                            <span className="flex items-center gap-1 text-green-400 text-xs">
                                                <Check className="w-3 h-3" /> Correct
                                            </span>
                                        ) : (
                                            <span className="flex items-center gap-1 text-red-400 text-xs">
                                                <X className="w-3 h-3" /> Wrong
                                            </span>
                                        )
                                    )}
                                </div>
                                <p className="font-bold">
                                    {currentStep.actual_outcome || '—'}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Confidence Evolution Chart */}
                    {evolution.length > 1 && (
                        <div className="glass-card">
                            <h3 className="text-lg font-semibold mb-4">Confidence Evolution</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={evolution}>
                                    <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                    <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#1e293b',
                                            border: '1px solid #334155',
                                            borderRadius: '8px'
                                        }}
                                    />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="nextConf"
                                        stroke="#0ea5e9"
                                        name="Next Activity %"
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="outcomeConf"
                                        stroke="#a855f7"
                                        name="Outcome %"
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            )}

            {/* Empty State */}
            {!currentStep && !loadMutation.isPending && (
                <div className="glass-card text-center py-16">
                    <div className="w-20 h-20 rounded-full bg-primary-500/20 flex items-center justify-center mx-auto mb-6">
                        <FlaskConical className="w-10 h-10 text-primary-400" />
                    </div>
                    <h2 className="text-xl font-semibold mb-2">Select a Test Case</h2>
                    <p className="text-dark-400 max-w-md mx-auto mb-6">
                        Choose a case from the dropdown above or click "Random" to start testing
                        your model with step-by-step predictions.
                    </p>
                    <button
                        className="btn-primary"
                        onClick={handleLoadRandom}
                        disabled={!traces || traces.length === 0}
                    >
                        <RefreshCw className="w-4 h-4" />
                        Load Random Case
                    </button>
                </div>
            )}
        </div>
    )
}
