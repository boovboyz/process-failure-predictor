import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Upload, FileCheck, AlertCircle, Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, UploadResponse } from '../api/client'

export default function UploadPage() {
    const navigate = useNavigate()
    const [dragActive, setDragActive] = useState(false)
    const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)

    const uploadMutation = useMutation({
        mutationFn: api.uploadFile,
        onSuccess: (data) => {
            setUploadResult(data)
            // Store in sessionStorage for other pages
            sessionStorage.setItem('logId', data.log_id)
        },
    })

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true)
        } else if (e.type === 'dragleave') {
            setDragActive(false)
        }
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0]
            if (file.name.endsWith('.xes')) {
                uploadMutation.mutate(file)
            }
        }
    }, [uploadMutation])

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            uploadMutation.mutate(e.target.files[0])
        }
    }, [uploadMutation])

    const formatDate = (dateStr: string) => {
        return new Date(dateStr).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    return (
        <div className="max-w-4xl mx-auto animate-fade-in">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Upload Event Log</h1>
                <p className="text-dark-400">
                    Upload your XES file to begin the prediction pipeline
                </p>
            </header>

            {/* Drop Zone */}
            <div
                className={`dropzone mb-8 ${dragActive ? 'active' : ''} ${uploadMutation.isPending ? 'opacity-50' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
            >
                <input
                    type="file"
                    id="file-input"
                    accept=".xes"
                    className="hidden"
                    onChange={handleFileInput}
                    disabled={uploadMutation.isPending}
                />

                {uploadMutation.isPending ? (
                    <div className="flex flex-col items-center gap-4">
                        <Loader2 className="w-12 h-12 text-primary-400 animate-spin" />
                        <p className="text-lg font-medium">Parsing XES file...</p>
                    </div>
                ) : (
                    <div className="flex flex-col items-center gap-4">
                        <div className="w-16 h-16 rounded-full bg-primary-500/20 flex items-center justify-center">
                            <Upload className="w-8 h-8 text-primary-400" />
                        </div>
                        <div>
                            <p className="text-lg font-medium mb-1">
                                Drag & drop your XES file here
                            </p>
                            <p className="text-sm text-dark-400">
                                or click to browse
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {/* Error */}
            {uploadMutation.isError && (
                <div className="glass-card mb-8 border-red-500/30 animate-slide-up">
                    <div className="flex items-start gap-3 text-red-400">
                        <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                        <div>
                            <p className="font-medium">Upload failed</p>
                            <p className="text-sm opacity-80">
                                {uploadMutation.error?.message || 'Unknown error occurred'}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Results */}
            {uploadResult && (
                <div className="space-y-6 animate-slide-up">
                    <div className="glass-card border-green-500/30">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                                <FileCheck className="w-5 h-5 text-green-400" />
                            </div>
                            <div>
                                <p className="font-semibold text-lg">File parsed successfully</p>
                                <p className="text-sm text-dark-400">Log ID: {uploadResult.log_id}</p>
                            </div>
                        </div>

                        {/* Stats Grid */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                            <div className="metric-card">
                                <div className="metric-value">{uploadResult.trace_count.toLocaleString()}</div>
                                <div className="metric-label">Traces</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-value">{uploadResult.event_count.toLocaleString()}</div>
                                <div className="metric-label">Events</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-value">{uploadResult.unique_activities}</div>
                                <div className="metric-label">Activities</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-value text-2xl">
                                    {Math.round((uploadResult.event_count / uploadResult.trace_count) * 10) / 10}
                                </div>
                                <div className="metric-label">Avg Events/Trace</div>
                            </div>
                        </div>

                        {/* Time Range */}
                        <div className="bg-dark-800/50 rounded-lg p-4 mb-6">
                            <p className="text-sm text-dark-400 mb-2">Time Range</p>
                            <p className="font-medium">
                                {formatDate(uploadResult.time_range_start)} — {formatDate(uploadResult.time_range_end)}
                            </p>
                        </div>

                        {/* Sample Activities */}
                        <div>
                            <p className="text-sm text-dark-400 mb-2">Sample Activities</p>
                            <div className="flex flex-wrap gap-2">
                                {uploadResult.sample_activities.map((activity) => (
                                    <span
                                        key={activity}
                                        className="px-3 py-1.5 bg-primary-600/20 text-primary-300 rounded-lg text-sm font-medium"
                                    >
                                        {activity}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Continue Button */}
                    <div className="flex justify-end">
                        <button
                            className="btn-primary"
                            onClick={() => navigate('/split')}
                        >
                            Continue to Data Split
                            <span className="ml-2">→</span>
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
