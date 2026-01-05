import { useEffect, useState, useRef } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
    Play, Pause, SkipForward, RotateCcw,
    ChevronDown, Check, X, Loader2, RefreshCw,
    Sparkles, Activity, Clock, Target, AlertTriangle,
    ChevronLeft, ChevronRight
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
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

export default function SimulatorPage() {
    const navigate = useNavigate()
    const modelId = sessionStorage.getItem('modelId')

    const [selectedTrace, setSelectedTrace] = useState<string | null>(null)
    const [currentStep, setCurrentStep] = useState<SimulatorStepResult | null>(null)
    const [totalSteps, setTotalSteps] = useState(0)
    const [isPlaying, setIsPlaying] = useState(false)
    const [evolution, setEvolution] = useState<any[]>([])
    const [caseLoaded, setCaseLoaded] = useState(false)
    const autoPlayRef = useRef<NodeJS.Timeout | null>(null)

    const { data: traces, isLoading: loadingTraces } = useQuery({
        queryKey: ['simulator-traces', modelId],
        queryFn: () => api.getSimulatorTraces(modelId!, 20),
        enabled: !!modelId,
    })

    const loadMutation = useMutation({
        mutationFn: (traceId: string) => api.loadSimulatorTrace(modelId!, traceId),
        onSuccess: (data) => {
            console.log('Load mutation success:', data)
            setTotalSteps(data.state.total_steps)
            setEvolution([])
            setCaseLoaded(true)
            // If initial_prediction is available, use it
            if (data.initial_prediction) {
                setCurrentStep(data.initial_prediction)
            } else {
                // Otherwise set null - user needs to step forward
                setCurrentStep(null)
            }
        },
        onError: (error) => {
            console.error('Load mutation error:', error)
            setCaseLoaded(false)
            alert(`Failed to load trace: ${error}`)
        },
    })

    const stepMutation = useMutation({
        mutationFn: () => api.simulatorStep(modelId!),
        onSuccess: (data) => {
            setCurrentStep(data.result)
            setEvolution(prev => [...prev, {
                step: data.result.step + 1,
                nextConf: Math.round(data.result.predicted_probability * 100),
                outcomeConf: Math.round(data.result.outcome_probability * 100),
                correct: data.result.next_correct ? 100 : 0,
            }])
        },
    })

    const stepBackMutation = useMutation({
        mutationFn: () => api.simulatorStepBack(modelId!),
        onSuccess: (data) => {
            if (data.result) {
                setCurrentStep(data.result)
            }
        },
    })

    const resetMutation = useMutation({
        mutationFn: () => api.resetSimulator(modelId!),
        onSuccess: (data) => {
            setCurrentStep(data.initial_prediction)
            setEvolution([])
        },
    })

    const jumpMutation = useMutation({
        mutationFn: (step: number) => api.simulatorJump(modelId!, step),
        onSuccess: (data) => {
            setCurrentStep(data.result)
        },
    })

    useEffect(() => {
        if (!modelId) {
            navigate('/training')
        }
    }, [modelId, navigate])

    useEffect(() => {
        if (selectedTrace) {
            loadMutation.mutate(selectedTrace)
        }
    }, [selectedTrace])

    // Auto-play
    useEffect(() => {
        if (isPlaying && currentStep && currentStep.step < totalSteps - 1) {
            autoPlayRef.current = setTimeout(() => {
                stepMutation.mutate()
            }, 1500)
        } else {
            setIsPlaying(false)
        }

        return () => {
            if (autoPlayRef.current) {
                clearTimeout(autoPlayRef.current)
            }
        }
    }, [isPlaying, currentStep?.step, stepMutation])

    if (!modelId) return null

    const handleLoadRandom = () => {
        if (traces && traces.length > 0) {
            const randomIdx = Math.floor(Math.random() * traces.length)
            setSelectedTrace(traces[randomIdx].trace_id)
        }
    }

    const isAtEnd = !!(currentStep && currentStep.step >= totalSteps - 1)
    const isAtStart = !currentStep || currentStep.step === 0

    return (
        <div className="max-w-7xl mx-auto animate-fade-in">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Event Simulator</h1>
                <p className="text-dark-400">
                    Step through a case event-by-event to see real-time predictions and AI recommendations
                </p>
            </header>

            {/* Case Selector */}
            <div className="glass-card mb-6">
                <div className="flex gap-4">
                    <div className="flex-1">
                        <label className="block text-sm text-dark-400 mb-2">Select a Case</label>
                        <div className="relative">
                            <select
                                value={selectedTrace || ''}
                                onChange={(e) => setSelectedTrace(e.target.value)}
                                className="w-full bg-dark-800 border border-dark-600 rounded-lg px-4 py-3 text-white appearance-none cursor-pointer focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                                disabled={loadingTraces}
                            >
                                <option value="">Choose a case to simulate...</option>
                                {traces?.map((t) => (
                                    <option key={t.trace_id} value={t.trace_id}>
                                        {t.case_id} — {t.event_count} events — {t.duration_hours.toFixed(1)}h — Final: {t.final_activity}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400 pointer-events-none" />
                        </div>
                    </div>
                    <div className="flex items-end">
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
                            Random Case
                        </button>
                    </div>
                </div>
            </div>

            {/* Main Simulator */}
            {currentStep && (
                <div className="space-y-6 animate-slide-up">

                    {/* Timeline Slider */}
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

                        {/* Slider */}
                        <div className="mb-4">
                            <input
                                type="range"
                                min="0"
                                max={totalSteps - 1}
                                value={currentStep.step}
                                onChange={(e) => jumpMutation.mutate(parseInt(e.target.value))}
                                className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                                disabled={jumpMutation.isPending}
                            />
                        </div>

                        {/* Event Steps Visualization */}
                        <div className="flex items-center gap-1 overflow-x-auto pb-2">
                            {Array.from({ length: totalSteps }).map((_, i) => (
                                <button
                                    key={i}
                                    onClick={() => jumpMutation.mutate(i)}
                                    disabled={jumpMutation.isPending}
                                    className={`
                                        w-10 h-10 rounded-full flex items-center justify-center text-xs font-medium
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
                    </div>

                    {/* Playback Controls */}
                    <div className="flex justify-center gap-3">
                        <button
                            className="btn-secondary px-4"
                            onClick={() => resetMutation.mutate()}
                            disabled={resetMutation.isPending || isAtStart}
                            title="Reset to start"
                        >
                            <RotateCcw className="w-5 h-5" />
                        </button>
                        <button
                            className="btn-secondary px-4"
                            onClick={() => stepBackMutation.mutate()}
                            disabled={stepBackMutation.isPending || isAtStart}
                            title="Previous event"
                        >
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        <button
                            className={`btn-primary w-36 justify-center ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-700' : ''}`}
                            onClick={() => setIsPlaying(!isPlaying)}
                            disabled={isAtEnd}
                        >
                            {isPlaying ? (
                                <>
                                    <Pause className="w-5 h-5" />
                                    Pause
                                </>
                            ) : (
                                <>
                                    <Play className="w-5 h-5" />
                                    {isAtStart ? 'Start' : 'Play'}
                                </>
                            )}
                        </button>
                        <button
                            className="btn-secondary px-4"
                            onClick={() => stepMutation.mutate()}
                            disabled={isAtEnd || stepMutation.isPending}
                            title="Next event"
                        >
                            {stepMutation.isPending ? (
                                <Loader2 className="w-5 h-5 animate-spin" />
                            ) : (
                                <ChevronRight className="w-5 h-5" />
                            )}
                        </button>
                        <button
                            className="btn-secondary px-4"
                            onClick={() => jumpMutation.mutate(totalSteps - 1)}
                            disabled={isAtEnd || jumpMutation.isPending}
                            title="Jump to end"
                        >
                            <SkipForward className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Current Event Details */}
                    <div className="glass-card border-primary-500/30">
                        <div className="flex items-center gap-2 text-primary-400 mb-4">
                            <Activity className="w-5 h-5" />
                            <span className="font-semibold">Current Event</span>
                            <span className="text-dark-400 text-sm">— Step {currentStep.step + 1}</span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                                <p className="text-sm text-dark-400 mb-1">Activity</p>
                                <p className="font-semibold text-lg">{currentStep.current_activity}</p>
                            </div>
                            <div>
                                <p className="text-sm text-dark-400 mb-1">Timestamp</p>
                                <p className="font-mono text-sm">
                                    {currentStep.current_events[currentStep.step]?.timestamp
                                        ? new Date(currentStep.current_events[currentStep.step].timestamp).toLocaleString()
                                        : '—'}
                                </p>
                            </div>
                            <div>
                                <p className="text-sm text-dark-400 mb-1">Resource</p>
                                <p className="font-medium">
                                    {currentStep.current_events[currentStep.step]?.resource || 'Not specified'}
                                </p>
                            </div>
                            <div>
                                <p className="text-sm text-dark-400 mb-1">Progress</p>
                                <div className="flex items-center gap-2">
                                    <div className="flex-1 h-2 bg-dark-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-300"
                                            style={{ width: `${((currentStep.step + 1) / totalSteps) * 100}%` }}
                                        />
                                    </div>
                                    <span className="text-sm font-medium">
                                        {Math.round(((currentStep.step + 1) / totalSteps) * 100)}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* LLM Recommendations - Prominent placement */}
                    {currentStep.recommendations && currentStep.recommendations.length > 0 && (
                        <div className="glass-card border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-primary-900/20">
                            <div className="flex items-center gap-2 text-purple-400 mb-4">
                                <Sparkles className="w-5 h-5" />
                                <span className="font-semibold">AI Recommendations</span>
                                <span className="text-xs bg-purple-500/20 px-2 py-0.5 rounded-full">Powered by Claude</span>
                            </div>
                            <div className="space-y-3">
                                {currentStep.recommendations.map((rec, i) => (
                                    <div
                                        key={i}
                                        className="flex items-start gap-3 p-3 bg-dark-800/50 rounded-lg border border-purple-500/20"
                                    >
                                        <div className="w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center flex-shrink-0 text-purple-400 text-sm font-bold">
                                            {i + 1}
                                        </div>
                                        <p className="text-dark-200">{rec}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Prediction vs Actual Cards */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Prediction Card */}
                        <div className="glass-card">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Target className="w-5 h-5 text-primary-400" />
                                <span className="text-primary-400">Predictions</span>
                            </h3>
                            <div className="space-y-4">
                                {/* Next Activity */}
                                <div className="p-4 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm text-dark-400">Next Activity</span>
                                        <ConfidenceBadge level={currentStep.predicted_confidence} />
                                    </div>
                                    <p className="text-xl font-bold mb-2">{currentStep.predicted_next}</p>
                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 h-2 bg-dark-700 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-primary-500 transition-all"
                                                style={{ width: `${currentStep.predicted_probability * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-sm font-medium text-primary-400">
                                            {Math.round(currentStep.predicted_probability * 100)}%
                                        </span>
                                    </div>
                                    {/* Alternatives */}
                                    {currentStep.alternatives.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-dark-700">
                                            <p className="text-xs text-dark-400 mb-2">Alternatives:</p>
                                            <div className="flex flex-wrap gap-2">
                                                {currentStep.alternatives.slice(0, 3).map((alt) => (
                                                    <span
                                                        key={alt.activity}
                                                        className="text-xs bg-dark-700 px-2 py-1 rounded"
                                                    >
                                                        {alt.activity}: {Math.round(alt.probability * 100)}%
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Remaining Time */}
                                <div className="p-4 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm text-dark-400 flex items-center gap-1">
                                            <Clock className="w-4 h-4" />
                                            Remaining Time
                                        </span>
                                    </div>
                                    <p className="text-xl font-bold text-green-400">
                                        ~{currentStep.predicted_time_hours.toFixed(1)} hours
                                    </p>
                                    <p className="text-xs text-dark-400 mt-1">
                                        Range: {currentStep.time_lower_hours.toFixed(1)} - {currentStep.time_upper_hours.toFixed(1)}h
                                    </p>
                                </div>

                                {/* Outcome */}
                                <div className="p-4 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm text-dark-400">Final Outcome</span>
                                        <ConfidenceBadge level={currentStep.outcome_confidence} />
                                    </div>
                                    <p className="text-xl font-bold">{currentStep.predicted_outcome}</p>
                                    <p className="text-sm text-dark-400 mt-1">
                                        {Math.round(currentStep.outcome_probability * 100)}% probability
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Ground Truth Card */}
                        <div className="glass-card">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Check className="w-5 h-5 text-green-400" />
                                <span className="text-green-400">Ground Truth</span>
                            </h3>
                            <div className="space-y-4">
                                {/* Actual Next Activity */}
                                <div className="p-4 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm text-dark-400">Actual Next</span>
                                        {currentStep.next_correct !== null && (
                                            currentStep.next_correct ? (
                                                <span className="flex items-center gap-1 text-green-400 text-sm">
                                                    <Check className="w-4 h-4" /> Correct
                                                </span>
                                            ) : (
                                                <span className="flex items-center gap-1 text-red-400 text-sm">
                                                    <X className="w-4 h-4" /> Wrong
                                                </span>
                                            )
                                        )}
                                    </div>
                                    <p className="text-xl font-bold">
                                        {currentStep.actual_next || '— (End of trace)'}
                                    </p>
                                </div>

                                {/* Time Error */}
                                <div className="p-4 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm text-dark-400">Time Prediction Error</span>
                                    </div>
                                    <p className={`text-xl font-bold ${currentStep.time_error_hours !== null
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

                                {/* Final Outcome */}
                                <div className="p-4 bg-dark-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm text-dark-400">Actual Outcome</span>
                                        {currentStep.outcome_correct !== null && (
                                            currentStep.outcome_correct ? (
                                                <span className="flex items-center gap-1 text-green-400 text-sm">
                                                    <Check className="w-4 h-4" /> Correct
                                                </span>
                                            ) : (
                                                <span className="flex items-center gap-1 text-red-400 text-sm">
                                                    <X className="w-4 h-4" /> Wrong
                                                </span>
                                            )
                                        )}
                                    </div>
                                    <p className="text-xl font-bold">
                                        {currentStep.actual_outcome || '—'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Risk Factors */}
                    {currentStep.risk_factors.length > 0 && (
                        <div className="glass-card border-yellow-500/30">
                            <div className="flex items-center gap-2 text-yellow-400 mb-4">
                                <AlertTriangle className="w-5 h-5" />
                                <span className="font-semibold">Risk Factors</span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {currentStep.risk_factors.map((risk, i) => (
                                    <div
                                        key={i}
                                        className="flex items-start gap-2 p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20"
                                    >
                                        <span className="text-yellow-500 mt-0.5">⚠</span>
                                        <span className="text-sm text-dark-300">{risk}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Evolution Chart */}
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
                                    <Line
                                        type="monotone"
                                        dataKey="nextConf"
                                        stroke="#0ea5e9"
                                        name="Next Activity"
                                        strokeWidth={2}
                                        dot={{ r: 4 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="outcomeConf"
                                        stroke="#a855f7"
                                        name="Outcome"
                                        strokeWidth={2}
                                        dot={{ r: 4 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                            <div className="flex justify-center gap-6 mt-4">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-primary-500" />
                                    <span className="text-sm text-dark-400">Next Activity Confidence</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-purple-500" />
                                    <span className="text-sm text-dark-400">Outcome Confidence</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Empty State - No case selected */}
            {!currentStep && !loadMutation.isPending && !caseLoaded && (
                <div className="glass-card text-center py-16">
                    <div className="w-20 h-20 rounded-full bg-primary-500/20 flex items-center justify-center mx-auto mb-6">
                        <Play className="w-10 h-10 text-primary-400" />
                    </div>
                    <h2 className="text-xl font-semibold mb-2">Select a Case to Simulate</h2>
                    <p className="text-dark-400 max-w-md mx-auto">
                        Choose a case from the dropdown above or click "Random Case" to start exploring
                        step-by-step predictions with AI-powered recommendations.
                    </p>
                </div>
            )}

            {/* Case Loaded but waiting for first prediction */}
            {!currentStep && !loadMutation.isPending && caseLoaded && (
                <div className="glass-card text-center py-16">
                    <div className="w-20 h-20 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-6">
                        <Activity className="w-10 h-10 text-green-400" />
                    </div>
                    <h2 className="text-xl font-semibold mb-2">Case Loaded: {totalSteps} Events</h2>
                    <p className="text-dark-400 max-w-md mx-auto mb-6">
                        Click the button below to generate the first prediction and start the simulation.
                    </p>
                    <button
                        className="btn-primary"
                        onClick={() => stepMutation.mutate()}
                        disabled={stepMutation.isPending}
                    >
                        {stepMutation.isPending ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <Play className="w-5 h-5" />
                        )}
                        Start Simulation
                    </button>
                </div>
            )}

            {/* Loading State */}
            {loadMutation.isPending && (
                <div className="glass-card text-center py-16">
                    <Loader2 className="w-12 h-12 text-primary-400 animate-spin mx-auto mb-4" />
                    <p className="text-dark-400">Loading case...</p>
                </div>
            )}
        </div>
    )
}
