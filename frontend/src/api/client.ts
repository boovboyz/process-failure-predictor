/**
 * API client for communicating with the backend
 */

const API_BASE = '/api'

interface UploadResponse {
    log_id: string
    trace_count: number
    event_count: number
    unique_activities: number
    time_range_start: string
    time_range_end: string
    sample_activities: string[]
}

interface SplitResponse {
    train_traces: number
    test_traces: number
    excluded_traces: number
    cutoff_time: string
    effective_ratio: number
    warnings: string[]
}

interface TrainingResponse {
    model_id: string
    training_time_seconds: number
    metrics: Record<string, { loss: number, validation_score: number | null }>
}

interface EvaluationResults {
    next_activity_accuracy: number
    next_activity_top3_accuracy: number
    next_activity_macro_f1: number
    next_activity_confusion_matrix: number[][]
    next_activity_class_labels: string[]
    outcome_auc_roc: number
    outcome_precision: number
    outcome_recall: number
    outcome_f1: number
    time_mae_hours: number
    time_rmse_hours: number
    time_mape: number
    time_coverage: number
    early_detection: Record<string, number>
    expected_calibration_error: number
    sample_predictions: Array<{
        case_id: string
        prefix_length: number
        last_activity: string
        predicted_next: string
        predicted_probability: number
        actual_next: string
        correct: boolean
    }>
}

interface Event {
    activity: string
    timestamp: string
    resource?: string
}

interface PredictionResult {
    next_activity: string
    next_activity_probability: number
    next_activity_calibrated: number
    next_activity_confidence: string
    alternatives: Array<{ activity: string; probability: number }>
    remaining_time_hours: number
    time_lower_bound_hours: number
    time_upper_bound_hours: number
    time_confidence: string
    predicted_outcome: string
    outcome_probability: number
    outcome_calibrated: number
    outcome_confidence: string
    outcome_distribution: Record<string, number>
    top_features: Array<{ feature: string; importance: number }>
    risk_factors: string[]
    aggregate_confidence_score: number
    aggregate_confidence_level: string
    confidence_flags: string[]
    recommendations?: string[]
}

interface TraceSummary {
    trace_id: string
    case_id: string
    event_count: number
    final_activity: string
    duration_hours: number
}

interface SimulatorStepResult {
    step: number
    total_steps: number
    current_events: Array<{ activity: string; timestamp: string; resource?: string }>
    current_activity: string
    predicted_next: string
    predicted_probability: number
    predicted_confidence: string
    alternatives: Array<{ activity: string; probability: number }>
    predicted_time_hours: number
    time_lower_hours: number
    time_upper_hours: number
    predicted_outcome: string
    outcome_probability: number
    outcome_confidence: string
    risk_factors: string[]
    actual_next: string | null
    actual_time_hours: number | null
    actual_outcome: string | null
    next_correct: boolean | null
    time_error_hours: number | null
    outcome_correct: boolean | null
    recommendations: string[] | null
}

interface SimulatorState {
    trace_id: string
    current_step: number
    total_steps: number
    is_complete: boolean
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
}

export const api = {
    // Upload
    async uploadFile(file: File): Promise<UploadResponse> {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        })

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
            throw new Error(error.detail)
        }

        return response.json()
    },

    async getLogInfo(logId: string) {
        return fetchJson<any>(`${API_BASE}/logs/${logId}`)
    },

    // Split
    async splitData(logId: string, trainRatio = 0.9): Promise<SplitResponse> {
        return fetchJson<SplitResponse>(`${API_BASE}/split/${logId}?train_ratio=${trainRatio}`, {
            method: 'POST',
        })
    },

    async getLatestSplit(logId: string) {
        return fetchJson<any>(`${API_BASE}/split/${logId}/latest`)
    },

    // Training
    async trainModels(logId: string): Promise<TrainingResponse> {
        return fetchJson<TrainingResponse>(`${API_BASE}/train/${logId}`, {
            method: 'POST',
        })
    },

    async getModelStatus(modelId: string) {
        return fetchJson<{ model_id: string; status: string; progress?: number; error?: string }>(
            `${API_BASE}/models/${modelId}/status`
        )
    },

    // Evaluation
    async getEvaluation(modelId: string): Promise<EvaluationResults> {
        return fetchJson<EvaluationResults>(`${API_BASE}/evaluate/${modelId}`)
    },

    async getConfusionMatrix(modelId: string) {
        return fetchJson<{ matrix: number[][]; labels: string[]; total_predictions: number }>(
            `${API_BASE}/evaluate/${modelId}/confusion-matrix`
        )
    },

    async getEarlyDetection(modelId: string) {
        return fetchJson<{ results: Record<string, number> }>(`${API_BASE}/evaluate/${modelId}/early-detection`)
    },

    async getReliabilityDiagram(modelId: string) {
        return fetchJson<any>(`${API_BASE}/evaluate/${modelId}/reliability-diagram`)
    },

    // Prediction
    async predict(modelId: string, events: Event[]): Promise<PredictionResult> {
        return fetchJson<PredictionResult>(`${API_BASE}/predict/${modelId}`, {
            method: 'POST',
            body: JSON.stringify({ events }),
        })
    },

    async getTestCases(logId: string, limit = 10) {
        return fetchJson<Array<{ case_id: string; event_count: number; final_outcome: string; duration_hours: number }>>(
            `${API_BASE}/test-cases/${logId}?limit=${limit}`
        )
    },

    async getCaseEvents(logId: string, caseId: string) {
        return fetchJson<{ case_id: string; events: Event[] }>(
            `${API_BASE}/test-cases/${logId}/${caseId}/events`
        )
    },

    // Simulator
    async getSimulatorTraces(modelId: string, limit = 20): Promise<TraceSummary[]> {
        return fetchJson<TraceSummary[]>(`${API_BASE}/simulator/${modelId}/traces?limit=${limit}`)
    },

    async loadSimulatorTrace(modelId: string, traceId: string): Promise<{ state: SimulatorState; initial_prediction: SimulatorStepResult | null }> {
        return fetchJson<any>(`${API_BASE}/simulator/${modelId}/load`, {
            method: 'POST',
            body: JSON.stringify({ trace_id: traceId }),
        })
    },

    async simulatorStep(modelId: string): Promise<{ result: SimulatorStepResult; state: SimulatorState }> {
        return fetchJson<any>(`${API_BASE}/simulator/${modelId}/step`, {
            method: 'POST',
        })
    },

    async simulatorStepBack(modelId: string): Promise<{ result: SimulatorStepResult; state: SimulatorState }> {
        return fetchJson<any>(`${API_BASE}/simulator/${modelId}/step-back`, {
            method: 'POST',
        })
    },

    async simulatorJump(modelId: string, step: number): Promise<{ result: SimulatorStepResult; state: SimulatorState }> {
        return fetchJson<any>(`${API_BASE}/simulator/${modelId}/jump/${step}`, {
            method: 'POST',
        })
    },

    async getSimulatorEvolution(modelId: string) {
        return fetchJson<{ evolution: any[]; summary: any }>(`${API_BASE}/simulator/${modelId}/evolution`)
    },

    async resetSimulator(modelId: string) {
        return fetchJson<any>(`${API_BASE}/simulator/${modelId}/reset`, {
            method: 'POST',
        })
    },

    async getSimulatorState(modelId: string) {
        return fetchJson<{ loaded: boolean; state?: SimulatorState; current_prediction?: SimulatorStepResult }>(
            `${API_BASE}/simulator/${modelId}/state`
        )
    },
}

export type {
    UploadResponse,
    SplitResponse,
    TrainingResponse,
    EvaluationResults,
    PredictionResult,
    TraceSummary,
    SimulatorStepResult,
    SimulatorState,
    Event,
}
