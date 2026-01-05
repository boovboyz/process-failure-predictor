# Process Failure Predictor

A domain-agnostic process failure prediction demo application that uses machine learning to predict process outcomes from XES event logs.

## Features

- **Upload** - Parse and validate XES event logs
- **Split** - Trace-aware temporal splitting (no data leakage)
- **Train** - XGBoost models for multi-dimensional predictions
- **Evaluate** - Comprehensive model metrics and calibration analysis
- **Test** - Step-through simulator with LLM-powered recommendations

### Prediction Capabilities

| Dimension | Description |
|-----------|-------------|
| **WHAT** | Next activity prediction |
| **WHEN** | Remaining time with confidence intervals |
| **HOW** | Process outcome prediction |
| **WHY** | Feature-based explanations |
| **WHAT TO DO** | LLM-generated recommendations (Claude) |

## Tech Stack

- **Backend**: FastAPI, Python, XGBoost, scikit-learn
- **Frontend**: React, TypeScript, Tailwind CSS, Vite
- **Database**: SQLite
- **LLM**: Claude API (optional)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm

### Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Copy environment file and add your API key
cp .env.example .env

# Start the server
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Access the Application

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

## Environment Variables

Create a `.env` file in the `backend` directory:

```env
ANTHROPIC_API_KEY=your_claude_api_key_here
```

The LLM integration is optional - the system works fully without it using rule-based recommendations.

## Sample Data

A sample XES file is included at `data/samples/sample_claims.xes` for testing.

## License

MIT
