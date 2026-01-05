import { useEffect, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Zap, Clock, Target, HelpCircle, AlertTriangle, ChevronDown, Loader2 } from 'lucide-react'
import { api, PredictionResult, Event } from '../api/client'

function ConfidenceBadge({ level }: { level: string }) {
    const colors = {
        HIGH: 'confidence-high',
        MEDIUM: 'confidence-medium',
        LOW: 'confidence-low',
    }
    return (
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[level as keyof typeof colors] || colors.LOW}`}>
            {level}
        </span>
    )
}

export default function PredictionPage() {
    const navigate = useNavigate()
    const modelId = sessionStorage.getItem('modelId')
    const logId = sessionStorage.getItem('logId')

    const [selectedCase, setSelectedCase] = useState<string | null>(null)
    const [prediction, setPrediction] = useState<PredictionResult | null>(null)
    const [events, setEvents] = useState<Event[]>([])
    const [prefixLength, setPrefixLength] = useState(3)

    const { data: testCases, isLoading: loadingCases } = useQuery({
        queryKey: ['test-cases', logId],
        queryFn: () => api.getTestCases(logId!, 15),
        enabled: !!logId,
    })

    const { data: caseEvents } = useQuery({
        queryKey: ['case-events', logId, selectedCase],
        queryFn: () => api.getCaseEvents(logId!, selectedCase!),
        enabled: !!logId && !!selectedCase,
    })

    const predictMutation = useMutation({
        mutationFn: (events: Event[]) => api.predict(modelId!, events),
        onSuccess: (data) => {
            setPrediction(data)
        },
    })

    useEffect(() => {
        if (!modelId) {
            navigate('/training')
        }
    }, [modelId, navigate])

    useEffect(() => {
        if (caseEvents) {
            setEvents(caseEvents.events)
            const prefix = caseEvents.events.slice(0, Math.min(prefixLength, caseEvents.events.length - 1))
            if (prefix.length > 0) {
                predictMutation.mutate(prefix)
            }
        }
    }, [caseEvents, prefixLength])

    if (!modelId) return null

    const handlePrefixChange = (newLength: number) => {
        if (events.length > 0) {
            const maxLen = events.length - 1
            const len = Math.max(1, Math.min(newLength, maxLen))
            setPrefixLength(len)
            predictMutation.mutate(events.slice(0, len))
        }
    }

    return (
        <div className="max-w-6xl mx-auto animate-fade-in">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Predictions</h1>
                <p className="text-dark-400">
                    Select a test case and adjust prefix length to see predictions
                </p>
            </header>

            {/* Case Selector */}
            <div className="glass-card mb-6">
                <label className="block text-sm text-dark-400 mb-2">Select Test Case</label>
                <div className="relative">
                    <select
                        value={selectedCase || ''}
                        onChange={(e) => setSelectedCase(e.target.value)}
                        className="w-full bg-dark-800 border border-dark-600 rounded-lg px-4 py-3 text-white appearance-none cursor-pointer"
                    >
                        <option value="">Choose a case...</option>
                        {testCases?.map((tc) => (
                            <option key={tc.case_id} value={tc.case_id}>
                                {tc.case_id} — {tc.event_count} events — {tc.final_outcome}
                            </option>
                        ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400 pointer-events-none" />
                </div>
            </div>

            {/* Main Content */}
            {selectedCase && events.length > 0 && (
                <div className="space-y-6">
                    {/* Trace Visualization */}
                    <div className="glass-card">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold">Current Trace</h3>
                            <div className="flex items-center gap-4">
                                <span className="text-sm text-dark-400">Prefix Length:</span>
                                <div className="flex items-center gap-2">
                                    <button
                                        className="w-8 h-8 rounded bg-dark-700 text-white flex items-center justify-center hover:bg-dark-600"
                                        onClick={() => handlePrefixChange(prefixLength - 1)}
                                        disabled={prefixLength <= 1}
                                    >
                                        -
                                    </button>
                                    <span className="w-8 text-center font-medium">{prefixLength}</span>
                                    <button
                                        className="w-8 h-8 rounded bg-dark-700 text-white flex items-center justify-center hover:bg-dark-600"
                                        onClick={() => handlePrefixChange(prefixLength + 1)}
                                        disabled={prefixLength >= events.length - 1}
                                    >
                                        +
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Activity Timeline */}
                        <div className="flex items-center gap-1 overflow-x-auto pb-2">
                            {events.map((event, i) => (
                                <div key={i} className="flex items-center gap-1">
                                    <div
                                        className={`
                      flex-shrink-0 px-3 py-2 rounded-lg text-xs font-medium
                      ${i < prefixLength
                                                ? 'bg-primary-600/30 text-primary-300 border border-primary-500/50'
                                                : i === prefixLength
                                                    ? 'bg-purple-600/30 text-purple-300 border-2 border-purple-500'
                                                    : 'bg-dark-700/50 text-dark-400 border border-dark-600'
                                            }
                    `}
                                    >
                                        {event.activity}
                                    </div>
                                    {i < events.length - 1 && (
                                        <span className={`text-dark-600 ${i < prefixLength - 1 ? 'text-primary-500' : ''}`}>→</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Prediction Cards */}
                    {predictMutation.isPending ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="w-8 h-8 text-primary-400 animate-spin" />
                        </div>
                    ) : prediction && (
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                            {/* WHAT */}
                            <div className="glass-card card-hover">
                                <div className="flex items-center gap-2 text-primary-400 mb-3">
                                    <Zap className="w-5 h-5" />
                                    <span className="font-medium">WHAT</span>
                                    <ConfidenceBadge level={prediction.next_activity_confidence} />
                                </div>
                                <p className="text-2xl font-bold mb-2">{prediction.next_activity}</p>
                                <p className="text-sm text-dark-400">
                                    {Math.round(prediction.next_activity_calibrated * 100)}% probability
                                </p>
                                <div className="mt-3 pt-3 border-t border-dark-700">
                                    <p className="text-xs text-dark-400 mb-1">Alternatives:</p>
                                    {prediction.alternatives.slice(0, 2).map((alt) => (
                                        <p key={alt.activity} className="text-xs text-dark-300">
                                            {alt.activity}: {Math.round(alt.probability * 100)}%
                                        </p>
                                    ))}
                                </div>
                            </div>

                            {/* WHEN */}
                            <div className="glass-card card-hover">
                                <div className="flex items-center gap-2 text-green-400 mb-3">
                                    <Clock className="w-5 h-5" />
                                    <span className="font-medium">WHEN</span>
                                    <ConfidenceBadge level={prediction.time_confidence} />
                                </div>
                                <p className="text-2xl font-bold mb-2">
                                    ~{prediction.remaining_time_hours.toFixed(1)}h
                                </p>
                                <p className="text-sm text-dark-400">
                                    Remaining time
                                </p>
                                <div className="mt-3 pt-3 border-t border-dark-700">
                                    <p className="text-xs text-dark-400">
                                        Range: {prediction.time_lower_bound_hours.toFixed(1)} - {prediction.time_upper_bound_hours.toFixed(1)} hours
                                    </p>
                                </div>
                            </div>

                            {/* HOW */}
                            <div className="glass-card card-hover">
                                <div className="flex items-center gap-2 text-purple-400 mb-3">
                                    <Target className="w-5 h-5" />
                                    <span className="font-medium">HOW</span>
                                    <ConfidenceBadge level={prediction.outcome_confidence} />
                                </div>
                                <p className="text-2xl font-bold mb-2">{prediction.predicted_outcome}</p>
                                <p className="text-sm text-dark-400">
                                    {Math.round(prediction.outcome_calibrated * 100)}% probability
                                </p>
                            </div>

                            {/* WHY */}
                            <div className="glass-card card-hover">
                                <div className="flex items-center gap-2 text-yellow-400 mb-3">
                                    <HelpCircle className="w-5 h-5" />
                                    <span className="font-medium">WHY</span>
                                </div>
                                <div className="space-y-2">
                                    {prediction.top_features.slice(0, 3).map((feat) => (
                                        <div key={feat.feature} className="flex justify-between text-sm">
                                            <span className="text-dark-300 truncate">{feat.feature}</span>
                                            <span className="text-dark-400">{(feat.importance * 100).toFixed(1)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Risk Factors */}
                    {prediction && prediction.risk_factors.length > 0 && (
                        <div className="glass-card border-yellow-500/30">
                            <div className="flex items-center gap-2 text-yellow-400 mb-3">
                                <AlertTriangle className="w-5 h-5" />
                                <span className="font-medium">Risk Factors</span>
                            </div>
                            <ul className="space-y-2">
                                {prediction.risk_factors.map((risk, i) => (
                                    <li key={i} className="text-sm text-dark-300 flex items-start gap-2">
                                        <span className="text-yellow-500">•</span>
                                        {risk}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* AI Recommendations */}
                    {prediction && prediction.recommendations && prediction.recommendations.length > 0 && (
                        <div className="glass-card border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-primary-900/20">
                            <div className="flex items-center gap-2 text-purple-400 mb-4">
                                <Zap className="w-5 h-5" />
                                <span className="font-semibold">AI Recommendations</span>
                                <span className="text-xs bg-purple-500/20 px-2 py-0.5 rounded-full">Powered by Claude</span>
                            </div>
                            <div className="space-y-3">
                                {prediction.recommendations.map((rec, i) => (
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

                    {/* Aggregate Confidence */}
                    {prediction && (
                        <div className="glass-card">
                            <div className="flex items-center justify-between">
                                <span className="text-dark-400">Overall Confidence</span>
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl font-bold text-primary-400">
                                        {Math.round(prediction.aggregate_confidence_score * 100)}%
                                    </span>
                                    <ConfidenceBadge level={prediction.aggregate_confidence_level} />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
