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

---

## Quick Start (Local Development)

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

---

## Deploy to AWS EC2

### Prerequisites
- AWS Account
- SSH key pair for EC2

### Step 1: Launch EC2 Instance
1. Go to AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu 22.04 LTS** (t2.micro for free tier, t3.small recommended)
3. Configure Security Group:
   - **SSH (22)** - Your IP
   - **HTTP (80)** - Anywhere (0.0.0.0/0)
4. Create or select a key pair
5. Launch and note the **Public IP**

### Step 2: Deploy

```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Clone repository
git clone https://github.com/YOUR_USERNAME/process-failure-predictor.git
cd process-failure-predictor

# Run deployment script
chmod +x deploy.sh
./deploy.sh
```

### Step 3: Access Your Demo
Open in browser: `http://YOUR_EC2_PUBLIC_IP`

### Useful Commands on EC2
```bash
sudo docker compose logs -f    # View logs
sudo docker compose down       # Stop
sudo docker compose restart    # Restart
```

---

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
