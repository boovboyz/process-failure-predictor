# Process Failure Predictor Demo
## Streamlined Technical Architecture Document

**Version:** 2.0  
**Date:** December 21, 2025  
**Purpose:** Lean Technical Blueprint for Domain-Agnostic Process Failure Prediction Demo

---

## 1. Executive Summary

### 1.1 Objective

Build a demonstration application that showcases intelligent, domain-agnostic process failure prediction. Users upload process event data in XES format, the system performs temporal data splitting, applies feature engineering, trains prediction models, and delivers actionable predictions answering:

- **WHAT** will happen next in the process
- **HOW** will it end (outcome prediction)
- **WHEN** will it occur (temporal prediction)
- **WHY** will it happen (feature-based explanation)
- **WHAT TO DO** (LLM-generated recommendations)

### 1.2 Key Differentiators

| Capability | Traditional Systems | Our Demo System |
|------------|---------------------|-----------------|
| Domain Adaptation | Manual configuration | Automatic via feature engineering |
| Data Splitting | Random split | Trace-aware temporal split (no leakage) |
| Predictions | Single output | Multi-dimensional (WHAT/WHEN/HOW/WHY) |
| Explanations | Black box | Feature importance + natural language |
| Recommendations | Generic alerts | LLM-powered contextual guidance |

### 1.3 Architecture Philosophy

This document prioritizes **shipping a working demo in 8 weeks** over theoretical completeness. Key constraints:

- **Single model family** (XGBoost) — proven, fast, interpretable
- **20 core features** — expandable later, sufficient now
- **LLM as enhancement** — system works fully without it
- **Minimal infrastructure** — SQLite + filesystem, no distributed systems
- **Progressive complexity** — MVP first, polish second

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         REACT FRONTEND                                  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐   │  │
│  │  │  Upload  │ │   Data   │ │  Model   │ │    Prediction Dashboard  │   │  │
│  │  │  Module  │ │  Viewer  │ │  Status  │ │    (WHAT/WHEN/HOW/WHY)   │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ REST API
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                               BACKEND (FastAPI)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Endpoints: /upload, /split, /train, /predict, /explain                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       │                                       │
│         ┌─────────────────────────────┼─────────────────────────────┐        │
│         ▼                             ▼                             ▼        │
│  ┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐    │
│  │  Data Pipeline  │     │    ML Pipeline      │     │  LLM Service    │    │
│  ├─────────────────┤     ├─────────────────────┤     ├─────────────────┤    │
│  │ • XES Parser    │     │ • Feature Engineer  │     │ • Domain Hint   │    │
│  │ • Validator     │     │ • XGBoost Trainer   │     │ • Explain Gen   │    │
│  │ • Splitter      │     │ • Predictor         │     │ • Recommend Gen │    │
│  │ • Prefix Gen    │     │ • Evaluator         │     │ (Optional)      │    │
│  └─────────────────┘     └─────────────────────┘     └─────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                               DATA LAYER                                      │
│  ┌───────────────────────────┐  ┌───────────────────────────────────────┐   │
│  │  SQLite (metadata, runs)  │  │  Filesystem (XES files, models, logs) │   │
│  └───────────────────────────┘  └───────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary Model | XGBoost | Fast training, interpretable, handles mixed features well |
| Database | SQLite | Zero config, sufficient for demo scale |
| Model Storage | Filesystem (joblib) | Simple, no MLflow dependency |
| Task Queue | None (sync) | Demo scale doesn't need async workers |
| LLM Integration | Claude API (optional) | Single call at prediction time only |
| Frontend State | React Query | Handles loading/caching elegantly |

---

## 3. XES Data Processing Pipeline

### 3.1 XES Parser

The parser handles IEEE 1849-2016 XES format with streaming support for large files.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           XES PARSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │  XES File   │────▶│  SAX Parser │────▶│  Validation │                   │
│   │   Upload    │     │ (streaming) │     │   Checks    │                   │
│   └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                  │                           │
│                                                  ▼                           │
│                              ┌─────────────────────────────────┐            │
│                              │       OUTPUT STRUCTURES         │            │
│                              │                                 │            │
│                              │  EventLog:                      │            │
│                              │  • traces: List[Trace]          │            │
│                              │  • attributes: Dict             │            │
│                              │                                 │            │
│                              │  Trace:                         │            │
│                              │  • case_id: str                 │            │
│                              │  • events: List[Event]          │            │
│                              │  • start_time, end_time         │            │
│                              │                                 │            │
│                              │  Event:                         │            │
│                              │  • activity: str                │            │
│                              │  • timestamp: datetime          │            │
│                              │  • resource: Optional[str]      │            │
│                              │  • attributes: Dict             │            │
│                              └─────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Validation checks:**
- Required attributes present (concept:name, time:timestamp)
- Timestamps parseable and in chronological order within traces
- No empty traces
- Activity names non-empty

### 3.2 Implementation

```python
# xes_parser.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import xml.sax

@dataclass
class Event:
    activity: str
    timestamp: datetime
    resource: Optional[str] = None
    attributes: Dict = None

@dataclass  
class Trace:
    case_id: str
    events: List[Event]
    attributes: Dict = None
    
    @property
    def start_time(self) -> datetime:
        return self.events[0].timestamp
    
    @property
    def end_time(self) -> datetime:
        return self.events[-1].timestamp

@dataclass
class EventLog:
    traces: List[Trace]
    attributes: Dict = None

class XESHandler(xml.sax.ContentHandler):
    """SAX handler for streaming XES parsing"""
    # Implementation handles large files without loading into memory
```

---

## 4. Data Splitting Strategy

### 4.1 The Problem with Random Splitting

Random train/test splits cause **temporal data leakage** in process mining:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ❌ WRONG: Random Split                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Timeline: ──────────────────────────────────────────────────────▶          │
│                                                                              │
│  Trace A:   [═══════════════]           (Jan 1 - Jan 15)                    │
│  Trace B:        [════════════════]     (Jan 5 - Jan 25)                    │
│  Trace C:             [═══════════════] (Jan 10 - Jan 30)                   │
│  Trace D:   [════════]                  (Jan 1 - Jan 8)                     │
│                                                                              │
│  Random Split: Train = [A, C], Test = [B, D]                                │
│                                                                              │
│  ⚠️  Problem: Trace C (training) ends AFTER Trace B (test)                  │
│      Model learns from "future" information → inflated metrics              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Trace-Aware Temporal Splitting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ✅ CORRECT: Trace-Aware Temporal Split                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Timeline: ──────────────────────────────────────────────────────▶          │
│                          │                                                   │
│                          │ TEMPORAL CUTOFF                                   │
│                          │                                                   │
│  Trace A:   [════════]   │                  → TRAIN (ends before cutoff)    │
│  Trace B:   [════════════│══]               → EXCLUDE (spans cutoff)        │
│  Trace C:                │    [════════]    → TEST (starts after cutoff)    │
│  Trace D:   [════]       │                  → TRAIN (ends before cutoff)    │
│                          │                                                   │
│  Result: Clean temporal separation, no future information leakage           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Splitting Algorithm

```python
# splitter.py
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SplitResult:
    train_traces: List[Trace]
    test_traces: List[Trace]
    excluded_traces: List[Trace]
    cutoff_time: datetime
    
    @property
    def effective_train_ratio(self) -> float:
        total = len(self.train_traces) + len(self.test_traces)
        return len(self.train_traces) / total if total > 0 else 0

def temporal_split(log: EventLog, train_ratio: float = 0.9) -> SplitResult:
    """
    Trace-aware temporal split that prevents data leakage.
    
    Algorithm:
    1. Sort traces by END timestamp
    2. Find cutoff point at train_ratio
    3. Classify traces:
       - TRAIN: trace.end_time < cutoff_time
       - TEST: trace.start_time > cutoff_time  
       - EXCLUDE: trace spans the cutoff boundary
    """
    # Sort by completion time
    sorted_traces = sorted(log.traces, key=lambda t: t.end_time)
    
    # Find cutoff
    cutoff_idx = int(len(sorted_traces) * train_ratio)
    cutoff_time = sorted_traces[cutoff_idx].end_time
    
    train, test, excluded = [], [], []
    
    for trace in sorted_traces:
        if trace.end_time < cutoff_time:
            train.append(trace)
        elif trace.start_time > cutoff_time:
            test.append(trace)
        else:
            excluded.append(trace)
    
    return SplitResult(train, test, excluded, cutoff_time)
```

### 4.4 Handling Edge Cases

| Scenario | Handling |
|----------|----------|
| >15% traces excluded | Show warning in UI, suggest adjusting cutoff |
| <100 test traces | Show warning, model evaluation may be unreliable |
| Very short traces | Include in normal split (they rarely span cutoff) |
| Identical timestamps | Sort by case_id as tiebreaker |

---

## 5. Prefix Generation

For next-event prediction, we need to train on partial traces (prefixes):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREFIX GENERATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Complete Trace: [A] → [B] → [C] → [D] → [E]                                │
│                                                                              │
│  Generated Training Samples:                                                 │
│                                                                              │
│  Prefix: [A]                    →  Target: B                                │
│  Prefix: [A, B]                 →  Target: C                                │
│  Prefix: [A, B, C]              →  Target: D                                │
│  Prefix: [A, B, C, D]           →  Target: E                                │
│                                                                              │
│  CRITICAL RULES:                                                            │
│  • Training prefixes come ONLY from training traces                         │
│  • Test prefixes come ONLY from test traces                                 │
│  • No cross-contamination                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
# prefix_generator.py
@dataclass
class PrefixSample:
    case_id: str
    prefix_events: List[Event]
    target_activity: str           # For next-activity prediction
    target_outcome: Optional[str]  # For outcome prediction
    remaining_time: float          # For time prediction (seconds)

def generate_prefixes(traces: List[Trace], 
                      min_prefix_len: int = 1) -> List[PrefixSample]:
    """Generate all valid prefixes from traces."""
    samples = []
    
    for trace in traces:
        # Determine outcome (last activity or explicit attribute)
        outcome = trace.events[-1].activity
        
        for i in range(min_prefix_len, len(trace.events)):
            prefix = trace.events[:i]
            target = trace.events[i].activity
            remaining = (trace.end_time - prefix[-1].timestamp).total_seconds()
            
            samples.append(PrefixSample(
                case_id=trace.case_id,
                prefix_events=prefix,
                target_activity=target,
                target_outcome=outcome,
                remaining_time=remaining
            ))
    
    return samples
```

---

## 6. Feature Engineering

### 6.1 Core Feature Set (20 Features)

We use a focused set of domain-agnostic features that work across any process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE CATEGORIES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TEMPORAL FEATURES (5)                                                       │
│  ─────────────────────                                                       │
│  • elapsed_time_total      : Total time since trace start (normalized)      │
│  • elapsed_time_last_event : Time since previous event (normalized)         │
│  • hour_of_day             : Hour when last event occurred (0-23)           │
│  • day_of_week             : Day of week (0-6)                              │
│  • is_business_hours       : Boolean (9am-5pm weekdays)                     │
│                                                                              │
│  SEQUENCE FEATURES (8)                                                       │
│  ─────────────────────                                                       │
│  • trace_length            : Number of events so far                        │
│  • unique_activities       : Count of distinct activities in prefix         │
│  • last_activity_encoded   : One-hot or label encoding of last activity    │
│  • last_3_activities       : Encoded representation of last 3 activities   │
│  • activity_occurrence_*   : Count of each activity in prefix (top 5)      │
│                                                                              │
│  TRANSITION FEATURES (3)                                                     │
│  ───────────────────────                                                     │
│  • transition_probability  : P(next|current) from training data             │
│  • is_common_transition    : Boolean, transition seen >10 times in train   │
│  • loop_count              : How many repeated activities in prefix         │
│                                                                              │
│  STATISTICAL FEATURES (4)                                                    │
│  ─────────────────────────                                                   │
│  • trace_length_percentile : Where this prefix length falls in distribution│
│  • elapsed_time_percentile : Where elapsed time falls in distribution      │
│  • activity_rarity         : How rare is the last activity (1/frequency)   │
│  • avg_time_between_events : Mean inter-event time in this prefix          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Feature Engineering Implementation

```python
# feature_engineer.py
import numpy as np
from collections import Counter

class FeatureEngineer:
    def __init__(self):
        self.activity_encoder = {}       # activity -> int
        self.transition_probs = {}       # (from, to) -> probability
        self.time_stats = {}             # normalization statistics
        self.activity_frequencies = {}   # activity -> count
    
    def fit(self, train_traces: List[Trace]):
        """Learn statistics from training data."""
        # Build activity vocabulary
        all_activities = set()
        for trace in train_traces:
            for event in trace.events:
                all_activities.add(event.activity)
        self.activity_encoder = {a: i for i, a in enumerate(sorted(all_activities))}
        
        # Learn transition probabilities
        transition_counts = Counter()
        from_counts = Counter()
        for trace in train_traces:
            for i in range(len(trace.events) - 1):
                from_act = trace.events[i].activity
                to_act = trace.events[i+1].activity
                transition_counts[(from_act, to_act)] += 1
                from_counts[from_act] += 1
        
        self.transition_probs = {
            k: v / from_counts[k[0]] 
            for k, v in transition_counts.items()
        }
        
        # Compute time normalization stats
        all_durations = []
        for trace in train_traces:
            total_duration = (trace.end_time - trace.start_time).total_seconds()
            all_durations.append(total_duration)
        
        self.time_stats = {
            'mean_duration': np.mean(all_durations),
            'std_duration': np.std(all_durations) + 1e-6
        }
        
        # Activity frequencies
        activity_counts = Counter()
        for trace in train_traces:
            for event in trace.events:
                activity_counts[event.activity] += 1
        total = sum(activity_counts.values())
        self.activity_frequencies = {a: c/total for a, c in activity_counts.items()}
    
    def transform(self, prefix: PrefixSample) -> np.ndarray:
        """Extract features from a prefix."""
        events = prefix.prefix_events
        last_event = events[-1]
        
        features = []
        
        # Temporal features
        total_elapsed = (last_event.timestamp - events[0].timestamp).total_seconds()
        features.append(total_elapsed / self.time_stats['mean_duration'])  # normalized
        
        if len(events) > 1:
            last_gap = (last_event.timestamp - events[-2].timestamp).total_seconds()
        else:
            last_gap = 0
        features.append(last_gap / 3600)  # hours
        
        features.append(last_event.timestamp.hour)
        features.append(last_event.timestamp.weekday())
        features.append(int(9 <= last_event.timestamp.hour <= 17 
                           and last_event.timestamp.weekday() < 5))
        
        # Sequence features
        features.append(len(events))
        features.append(len(set(e.activity for e in events)))
        features.append(self.activity_encoder.get(last_event.activity, -1))
        
        # Last 3 activities (padded)
        last_3 = [e.activity for e in events[-3:]]
        while len(last_3) < 3:
            last_3.insert(0, '<PAD>')
        for act in last_3:
            features.append(self.activity_encoder.get(act, -1))
        
        # Activity occurrence counts (top 5 activities by frequency)
        activity_counts = Counter(e.activity for e in events)
        top_activities = sorted(self.activity_frequencies.keys(), 
                               key=lambda a: self.activity_frequencies[a], 
                               reverse=True)[:5]
        for act in top_activities:
            features.append(activity_counts.get(act, 0))
        
        # Transition features
        if len(events) > 1:
            last_transition = (events[-2].activity, last_event.activity)
            features.append(self.transition_probs.get(last_transition, 0))
            features.append(int(self.transition_probs.get(last_transition, 0) > 0))
        else:
            features.extend([0, 0])
        
        # Loop count
        features.append(len(events) - len(set(e.activity for e in events)))
        
        # Statistical features
        features.append(len(events) / 20)  # normalized trace length
        features.append(total_elapsed / self.time_stats['mean_duration'])
        features.append(1 / (self.activity_frequencies.get(last_event.activity, 0.01)))
        
        inter_event_times = []
        for i in range(1, len(events)):
            gap = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            inter_event_times.append(gap)
        features.append(np.mean(inter_event_times) if inter_event_times else 0)
        
        return np.array(features, dtype=np.float32)
```

### 6.3 Optional: LLM Domain Hints

If LLM integration is enabled, we make a **single API call at upload time** to get domain context:

```python
# llm_hints.py (optional enhancement)
async def get_domain_hints(activity_names: List[str], 
                           sample_traces: List[str]) -> dict:
    """
    Single LLM call to understand the process domain.
    Used for UI display and recommendation generation, NOT for features.
    """
    prompt = f"""
    Analyze this process based on activity names and sample traces.
    
    Activities: {activity_names[:20]}
    Sample trace: {sample_traces[0]}
    
    Return JSON:
    {{
        "domain": "detected domain (e.g., insurance, healthcare, order fulfillment)",
        "process_type": "type of process",
        "critical_activities": ["list of activities that seem important"],
        "potential_failure_indicators": ["patterns that might indicate problems"]
    }}
    """
    
    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)
```

---

## 7. Model Training Pipeline

### 7.1 Model Selection: XGBoost

We use XGBoost for all prediction tasks. Rationale:

| Consideration | XGBoost Advantage |
|---------------|-------------------|
| Training speed | Minutes, not hours |
| Interpretability | Native feature importance |
| Mixed features | Handles categorical + numerical |
| Small data | Works well with 10K-100K samples |
| Production | Easy serialization, fast inference |

### 7.2 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  Training Data  │                                                        │
│  │   (prefixes)    │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │    Feature      │────▶│   Validation    │────▶│    XGBoost      │       │
│  │   Engineering   │     │   Split (80/20) │     │    Training     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                          │                   │
│                          ┌───────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│          ┌───────────────────────────────────────────────┐                  │
│          │              TRAINED MODELS                    │                  │
│          │                                                │                  │
│          │  ┌─────────────────┐  ┌─────────────────┐     │                  │
│          │  │ Next Activity   │  │ Outcome         │     │                  │
│          │  │ (multi-class)   │  │ (binary/multi)  │     │                  │
│          │  └─────────────────┘  └─────────────────┘     │                  │
│          │                                                │                  │
│          │  ┌─────────────────┐                          │                  │
│          │  │ Remaining Time  │                          │                  │
│          │  │ (regression)    │                          │                  │
│          │  └─────────────────┘                          │                  │
│          │                                                │                  │
│          └───────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│                 ┌─────────────────┐                                         │
│                 │  Filesystem     │                                         │
│                 │  (models/*.pkl) │                                         │
│                 └─────────────────┘                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Model Implementation

```python
# trainer.py
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

class ProcessPredictor:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.next_activity_model = None
        self.outcome_model = None
        self.time_model = None
        self.label_encoder = {}  # activity -> label
    
    def train(self, train_traces: List[Trace], val_ratio: float = 0.2):
        """Train all prediction models."""
        # Fit feature engineer
        self.feature_engineer.fit(train_traces)
        
        # Generate prefixes
        prefixes = generate_prefixes(train_traces)
        
        # Build feature matrix
        X = np.array([self.feature_engineer.transform(p) for p in prefixes])
        
        # Prepare targets
        activities = list(set(p.target_activity for p in prefixes))
        self.label_encoder = {a: i for i, a in enumerate(activities)}
        
        y_activity = np.array([self.label_encoder[p.target_activity] for p in prefixes])
        y_outcome = np.array([self.label_encoder.get(p.target_outcome, 0) for p in prefixes])
        y_time = np.array([p.remaining_time for p in prefixes])
        
        # Train/val split (within training data)
        X_train, X_val, y_act_train, y_act_val = train_test_split(
            X, y_activity, test_size=val_ratio, random_state=42
        )
        
        # Train next activity model (multi-class)
        self.next_activity_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=len(activities),
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            random_state=42
        )
        self.next_activity_model.fit(
            X_train, y_act_train,
            eval_set=[(X_val, y_act_val)],
            verbose=False
        )
        
        # Train outcome model (using last prefix of each trace)
        # ... similar pattern
        
        # Train time regression model
        self.time_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        # ... fit on remaining time targets
        
        return self._get_training_metrics()
    
    def save(self, path: str):
        """Save all models and feature engineer."""
        joblib.dump({
            'feature_engineer': self.feature_engineer,
            'next_activity_model': self.next_activity_model,
            'outcome_model': self.outcome_model,
            'time_model': self.time_model,
            'label_encoder': self.label_encoder
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ProcessPredictor':
        """Load trained predictor."""
        data = joblib.load(path)
        predictor = cls()
        predictor.feature_engineer = data['feature_engineer']
        predictor.next_activity_model = data['next_activity_model']
        predictor.outcome_model = data['outcome_model']
        predictor.time_model = data['time_model']
        predictor.label_encoder = data['label_encoder']
        return predictor
```

### 7.4 Evaluation Metrics

| Task | Primary Metric | Secondary Metrics |
|------|----------------|-------------------|
| Next Activity | Accuracy | Top-3 Accuracy, Macro F1 |
| Outcome | AUC-ROC | F1-Score, Precision, Recall |
| Remaining Time | MAE | RMSE, MAPE |

---

## 8. Confidence Scoring & Calibration

### 8.1 The Calibration Problem

Raw model probabilities are often **poorly calibrated** — a model saying "80% confident" might only be correct 60% of the time. This erodes user trust and makes probability thresholds unreliable.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALIBRATION: THE PROBLEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  UNCALIBRATED MODEL                         CALIBRATED MODEL                 │
│  ──────────────────                         ────────────────                 │
│                                                                              │
│  Predicted   Actual                         Predicted   Actual               │
│  Confidence  Accuracy                       Confidence  Accuracy             │
│  ─────────── ────────                       ─────────── ────────             │
│     90%        72%    ← Overconfident          90%        89%   ✓           │
│     80%        61%    ← Overconfident          80%        81%   ✓           │
│     70%        58%    ← Overconfident          70%        69%   ✓           │
│     60%        54%    ← Slightly over          60%        61%   ✓           │
│     50%        51%    ← OK                     50%        50%   ✓           │
│                                                                              │
│  XGBoost is particularly prone to overconfidence on minority classes        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Calibration Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALIBRATION PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRAINING PHASE                                                              │
│  ──────────────                                                              │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  Training Data  │                                                        │
│  │     (80%)       │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐                                │
│  │  Train XGBoost  │────▶│  Raw Model      │                                │
│  │    Models       │     │  (uncalibrated) │                                │
│  └─────────────────┘     └────────┬────────┘                                │
│                                   │                                          │
│  CALIBRATION PHASE                │                                          │
│  ─────────────────                ▼                                          │
│                          ┌─────────────────┐                                │
│  ┌─────────────────┐     │  Get Raw Probs  │                                │
│  │ Validation Data │────▶│  on Validation  │                                │
│  │     (20%)       │     │      Set        │                                │
│  └─────────────────┘     └────────┬────────┘                                │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │ Fit Calibrator  │                                │
│                          │ (Isotonic Reg)  │                                │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │ Calibrated Model│                                │
│                          │  (trustworthy)  │                                │
│                          └─────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Calibration Implementation

```python
# calibration.py
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CalibrationResult:
    calibrated_probs: np.ndarray
    confidence_level: str  # HIGH, MEDIUM, LOW
    reliability_score: float  # How well-calibrated this prediction is

class ProbabilityCalibrator:
    """
    Calibrates model probabilities using isotonic regression.
    
    Why isotonic over Platt scaling?
    - Non-parametric: doesn't assume sigmoid shape
    - Better for XGBoost which has complex probability distributions
    - Monotonic: preserves ranking of predictions
    """
    
    def __init__(self):
        self.calibrators: Dict[int, IsotonicRegression] = {}  # per-class
        self.confidence_thresholds = {
            'HIGH': 0.75,
            'MEDIUM': 0.50,
            'LOW': 0.0
        }
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fit calibrators on validation set predictions.
        
        Args:
            y_true: True labels (n_samples,)
            y_prob: Predicted probabilities (n_samples, n_classes)
        """
        n_classes = y_prob.shape[1]
        
        for class_idx in range(n_classes):
            # Binary indicator for this class
            y_binary = (y_true == class_idx).astype(int)
            prob_class = y_prob[:, class_idx]
            
            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(prob_class, y_binary)
            self.calibrators[class_idx] = calibrator
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.
        
        Args:
            y_prob: Raw probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        calibrated = np.zeros_like(y_prob)
        
        for class_idx, calibrator in self.calibrators.items():
            calibrated[:, class_idx] = calibrator.predict(y_prob[:, class_idx])
        
        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-10)
        
        return calibrated
    
    def get_confidence_level(self, prob: float) -> str:
        """Classify probability into confidence level."""
        if prob >= self.confidence_thresholds['HIGH']:
            return 'HIGH'
        elif prob >= self.confidence_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def predict_with_confidence(self, y_prob: np.ndarray) -> List[CalibrationResult]:
        """Get calibrated predictions with confidence levels."""
        calibrated = self.calibrate(y_prob)
        results = []
        
        for i in range(len(calibrated)):
            max_prob = calibrated[i].max()
            results.append(CalibrationResult(
                calibrated_probs=calibrated[i],
                confidence_level=self.get_confidence_level(max_prob),
                reliability_score=max_prob
            ))
        
        return results


class ConfidenceMetrics:
    """Calculate calibration quality metrics."""
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, 
                                    y_prob: np.ndarray, 
                                    n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).
        
        Lower is better. <0.05 is well-calibrated.
        
        ECE = Σ (|bin_size| / n) * |accuracy(bin) - confidence(bin)|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        y_pred = y_prob.argmax(axis=1)
        confidences = y_prob.max(axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
        
        return ece
    
    @staticmethod
    def reliability_diagram_data(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  n_bins: int = 10) -> Dict:
        """Generate data for reliability diagram visualization."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        y_pred = y_prob.argmax(axis=1)
        confidences = y_prob.max(axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        bins = []
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                bins.append({
                    'bin_center': (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                    'avg_confidence': float(confidences[in_bin].mean()),
                    'avg_accuracy': float(accuracies[in_bin].mean()),
                    'count': int(in_bin.sum()),
                    'gap': float(abs(accuracies[in_bin].mean() - confidences[in_bin].mean()))
                })
        
        return {
            'bins': bins,
            'perfect_calibration': [{'x': i/10, 'y': i/10} for i in range(11)]
        }
```

### 8.4 Time Prediction Confidence Intervals

For remaining time prediction, we use **quantile regression** to get proper prediction intervals:

```python
# time_confidence.py
import xgboost as xgb
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TimePredictionWithCI:
    point_estimate: float      # Expected remaining time (hours)
    lower_bound: float         # 10th percentile
    upper_bound: float         # 90th percentile
    confidence_level: str      # HIGH, MEDIUM, LOW based on interval width
    interval_width_hours: float

class QuantileTimePredictor:
    """
    Predicts remaining time with confidence intervals using quantile regression.
    
    Instead of just predicting mean, we predict 10th, 50th, and 90th percentiles.
    This gives us honest uncertainty estimates.
    """
    
    def __init__(self):
        self.model_median = None   # 50th percentile
        self.model_lower = None    # 10th percentile  
        self.model_upper = None    # 90th percentile
        self.median_time = None    # For normalizing interval width
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train three quantile regression models."""
        
        base_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Median (50th percentile) - our point estimate
        self.model_median = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.5,
            **base_params
        )
        self.model_median.fit(X, y)
        
        # Lower bound (10th percentile)
        self.model_lower = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.1,
            **base_params
        )
        self.model_lower.fit(X, y)
        
        # Upper bound (90th percentile)
        self.model_upper = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.9,
            **base_params
        )
        self.model_upper.fit(X, y)
        
        # Store median for relative interval calculation
        self.median_time = np.median(y)
    
    def predict(self, X: np.ndarray) -> List[TimePredictionWithCI]:
        """Predict with confidence intervals."""
        
        median = self.model_median.predict(X)
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)
        
        # Ensure ordering (quantile crossing can happen)
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
        
        results = []
        for i in range(len(X)):
            interval_width = (upper[i] - lower[i]) / 3600  # Convert to hours
            
            # Confidence based on relative interval width
            relative_width = interval_width / (self.median_time / 3600 + 1)
            
            if relative_width < 0.5:
                confidence = 'HIGH'
            elif relative_width < 1.0:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            results.append(TimePredictionWithCI(
                point_estimate=median[i] / 3600,
                lower_bound=max(0, lower[i] / 3600),
                upper_bound=upper[i] / 3600,
                confidence_level=confidence,
                interval_width_hours=interval_width
            ))
        
        return results
    
    def coverage_score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate what fraction of true values fall within predicted intervals.
        Target: 80% (since we use 10th-90th percentile).
        """
        predictions = self.predict(X)
        
        covered = 0
        for i, pred in enumerate(predictions):
            true_hours = y_true[i] / 3600
            if pred.lower_bound <= true_hours <= pred.upper_bound:
                covered += 1
        
        return covered / len(predictions)
```

### 8.5 Aggregate Confidence Score

Combine confidence across WHAT/WHEN/HOW into a single score:

```python
# aggregate_confidence.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AggregateConfidence:
    overall_score: float           # 0-1, weighted combination
    overall_level: str             # HIGH, MEDIUM, LOW
    
    next_activity_confidence: float
    next_activity_level: str
    
    outcome_confidence: float
    outcome_level: str
    
    time_confidence: str           # Based on interval width
    
    flags: list                    # Specific concerns

def calculate_aggregate_confidence(
    next_activity_prob: float,
    outcome_prob: float,
    time_interval_width: float,
    median_process_time: float
) -> AggregateConfidence:
    """
    Calculate overall prediction confidence.
    
    Weights:
    - Next activity: 40% (most important for immediate action)
    - Outcome: 35% (important for prioritization)
    - Time: 25% (helpful but less critical)
    """
    
    # Normalize time confidence to 0-1 scale
    relative_interval = time_interval_width / (median_process_time + 1)
    time_score = max(0, 1 - relative_interval)  # Narrower = better
    
    # Weighted combination
    overall = (
        0.40 * next_activity_prob +
        0.35 * outcome_prob +
        0.25 * time_score
    )
    
    # Classify each component
    def classify(score: float) -> str:
        if score >= 0.75: return 'HIGH'
        if score >= 0.50: return 'MEDIUM'
        return 'LOW'
    
    # Identify specific concerns
    flags = []
    if next_activity_prob < 0.5:
        flags.append("Low confidence in next activity prediction")
    if outcome_prob < 0.5:
        flags.append("Outcome prediction is uncertain")
    if time_score < 0.5:
        flags.append("Wide time prediction interval")
    
    return AggregateConfidence(
        overall_score=overall,
        overall_level=classify(overall),
        next_activity_confidence=next_activity_prob,
        next_activity_level=classify(next_activity_prob),
        outcome_confidence=outcome_prob,
        outcome_level=classify(outcome_prob),
        time_confidence=classify(time_score),
        flags=flags
    )
```

### 8.6 Confidence-Aware UI Treatment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UI TREATMENT BY CONFIDENCE LEVEL                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH CONFIDENCE (≥75%)                                                      │
│  ──────────────────────                                                      │
│  ┌─────────────────────────────────────┐                                    │
│  │ ✓ Next Activity: Approve_Claim      │  • Solid border                    │
│  │   Confidence: 87%  ●●●●●●●●○○       │  • Green/blue accent               │
│  │                                     │  • Direct language                  │
│  │   "Will likely be approved"         │  • Prominent display               │
│  └─────────────────────────────────────┘                                    │
│                                                                              │
│  MEDIUM CONFIDENCE (50-74%)                                                  │
│  ──────────────────────────                                                  │
│  ┌─────────────────────────────────────┐                                    │
│  │ ~ Next Activity: Review_Documents   │  • Dashed border                   │
│  │   Confidence: 62%  ●●●●●●○○○○       │  • Yellow/amber accent             │
│  │                                     │  • Hedged language                  │
│  │   "Likely to require document       │  • Show alternatives               │
│  │    review, but other paths          │    prominently                     │
│  │    possible"                        │                                    │
│  │                                     │                                    │
│  │   Also possible:                    │                                    │
│  │   • Escalate (23%)                  │                                    │
│  │   • Request_Info (15%)              │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                              │
│  LOW CONFIDENCE (<50%)                                                       │
│  ─────────────────────                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │ ⚠ Next Activity: Uncertain          │  • Warning styling                 │
│  │   Confidence: 34%  ●●●○○○○○○○       │  • Gray/muted colors               │
│  │                                     │  • Explicit uncertainty            │
│  │   "Multiple paths equally likely,   │  • List all options                │
│  │    prediction unreliable"           │  • Suggest waiting for             │
│  │                                     │    more events                     │
│  │   Possibilities:                    │                                    │
│  │   • Review_Documents (34%)          │                                    │
│  │   • Escalate (28%)                  │                                    │
│  │   • Request_Info (22%)              │                                    │
│  │   • Cancel (16%)                    │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                              │
│  TIME PREDICTION CONFIDENCE                                                  │
│  ──────────────────────────                                                  │
│                                                                              │
│  HIGH: "Expected completion: 4.2 hours"                                      │
│        └─ Narrow interval (3.8 - 4.6 hrs)                                   │
│                                                                              │
│  MEDIUM: "Expected completion: 4-8 hours"                                    │
│          └─ Show range prominently                                          │
│                                                                              │
│  LOW: "Completion time highly variable: 2-15 hours"                          │
│       └─ Emphasize uncertainty, show full range                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.7 Confidence in API Response

```python
# Updated prediction response schema
class PredictionResult(BaseModel):
    # WHAT
    next_activity: str
    next_activity_probability: float      # Raw (for backward compat)
    next_activity_calibrated: float       # Calibrated probability
    next_activity_confidence: str         # HIGH, MEDIUM, LOW
    alternatives: List[Dict[str, Any]]
    
    # WHEN
    remaining_time_hours: float           # Point estimate
    time_lower_bound_hours: float         # 10th percentile
    time_upper_bound_hours: float         # 90th percentile
    time_confidence: str                  # Based on interval width
    
    # HOW
    predicted_outcome: str
    outcome_probability: float            # Raw
    outcome_calibrated: float             # Calibrated
    outcome_confidence: str               # HIGH, MEDIUM, LOW
    outcome_distribution: Dict[str, float]
    
    # WHY
    top_features: List[Dict[str, Any]]
    risk_factors: List[str]
    
    # OVERALL CONFIDENCE (NEW)
    aggregate_confidence_score: float     # 0-1 weighted score
    aggregate_confidence_level: str       # HIGH, MEDIUM, LOW
    confidence_flags: List[str]           # Specific concerns
    
    # ACTION
    recommendations: Optional[List[str]]
```

### 8.8 Calibration Metrics in Evaluation

Add to the evaluation dashboard:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALIBRATION METRICS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXPECTED CALIBRATION ERROR (ECE)                                            │
│  ────────────────────────────────                                            │
│                                                                              │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐            │
│  │  Next Activity   │ │    Outcome       │ │  Time Coverage   │            │
│  │  ECE: 0.042      │ │  ECE: 0.038      │ │  Coverage: 82%   │            │
│  │  ✓ Well-calibrated│ │  ✓ Well-calibrated│ │  ✓ Target: 80%   │            │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘            │
│                                                                              │
│  RELIABILITY DIAGRAM (Next Activity)                                         │
│  ───────────────────────────────────                                         │
│                                                                              │
│  Accuracy                                                                    │
│  100%│                              ╱                                       │
│   80%│                         ●  ╱                                         │
│   60%│                    ●  ╱                                              │
│   40%│              ●   ╱    ← Perfect calibration                          │
│   20%│        ●   ╱                                                         │
│    0%│───●──╱─────────────────────                                          │
│      0%  20%  40%  60%  80%  100%                                           │
│              Predicted Confidence                                            │
│                                                                              │
│  ● = Actual accuracy at each confidence level                               │
│  Dots on the diagonal = perfectly calibrated                                │
│                                                                              │
│  CONFIDENCE DISTRIBUTION                                                     │
│  ───────────────────────                                                     │
│                                                                              │
│  HIGH (≥75%):   ████████████████████░░░░░░  68% of predictions             │
│  MEDIUM (50-74%): ██████████░░░░░░░░░░░░░░  24% of predictions             │
│  LOW (<50%):    ████░░░░░░░░░░░░░░░░░░░░░░   8% of predictions             │
│                                                                              │
│  ✓ Majority of predictions are high-confidence                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Model Evaluation on Test Set

### 9.1 Evaluation Pipeline

After training, we evaluate models on the held-out 10% test set to get unbiased performance metrics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │   Test Traces   │  (10% held-out, never seen during training)            │
│  │   from Split    │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ Generate Test   │────▶│  Extract Test   │────▶│  Run Inference  │       │
│  │    Prefixes     │     │    Features     │     │  (All 3 Models) │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                          │                   │
│                                                          ▼                   │
│                          ┌───────────────────────────────────────────────┐  │
│                          │           EVALUATION RESULTS                   │  │
│                          │                                                │  │
│                          │  Next Activity Prediction:                     │  │
│                          │  ├─ Accuracy: 84.2%                           │  │
│                          │  ├─ Top-3 Accuracy: 95.1%                     │  │
│                          │  ├─ Macro F1: 0.812                           │  │
│                          │  └─ Confusion Matrix                          │  │
│                          │                                                │  │
│                          │  Outcome Prediction:                           │  │
│                          │  ├─ AUC-ROC: 0.891                            │  │
│                          │  ├─ Precision: 0.823                          │  │
│                          │  ├─ Recall: 0.867                             │  │
│                          │  └─ F1-Score: 0.844                           │  │
│                          │                                                │  │
│                          │  Remaining Time Prediction:                    │  │
│                          │  ├─ MAE: 2.3 hours                            │  │
│                          │  ├─ RMSE: 4.1 hours                           │  │
│                          │  └─ MAPE: 18.2%                               │  │
│                          │                                                │  │
│                          │  Early Detection Analysis:                     │  │
│                          │  ├─ Accuracy @ 25% prefix: 71.2%              │  │
│                          │  ├─ Accuracy @ 50% prefix: 82.4%              │  │
│                          │  └─ Accuracy @ 75% prefix: 89.1%              │  │
│                          │                                                │  │
│                          └───────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Evaluator Implementation

```python
# evaluator.py
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

@dataclass
class EvaluationResults:
    # Next Activity metrics
    next_activity_accuracy: float
    next_activity_top3_accuracy: float
    next_activity_macro_f1: float
    next_activity_confusion_matrix: List[List[int]]
    next_activity_class_labels: List[str]
    
    # Outcome metrics
    outcome_auc_roc: float
    outcome_precision: float
    outcome_recall: float
    outcome_f1: float
    
    # Time prediction metrics
    time_mae_hours: float
    time_rmse_hours: float
    time_mape: float
    
    # Early detection (accuracy at different prefix lengths)
    early_detection: Dict[str, float]  # {"25%": 0.71, "50%": 0.82, ...}
    
    # Sample predictions for UI display
    sample_predictions: List[Dict]

class ModelEvaluator:
    def __init__(self, predictor: ProcessPredictor):
        self.predictor = predictor
    
    def evaluate(self, test_traces: List[Trace]) -> EvaluationResults:
        """Run full evaluation on test set."""
        
        # Generate test prefixes
        test_prefixes = generate_prefixes(test_traces)
        
        # Extract features
        X_test = np.array([
            self.predictor.feature_engineer.transform(p) 
            for p in test_prefixes
        ])
        
        # True labels
        y_activity_true = [
            self.predictor.label_encoder[p.target_activity] 
            for p in test_prefixes
        ]
        y_outcome_true = [
            self.predictor.label_encoder.get(p.target_outcome, 0) 
            for p in test_prefixes
        ]
        y_time_true = [p.remaining_time / 3600 for p in test_prefixes]  # hours
        
        # ─────────────────────────────────────────────────────────
        # Next Activity Evaluation
        # ─────────────────────────────────────────────────────────
        activity_probs = self.predictor.next_activity_model.predict_proba(X_test)
        activity_preds = np.argmax(activity_probs, axis=1)
        
        # Top-3 accuracy
        top3_correct = sum(
            true in np.argsort(probs)[-3:] 
            for true, probs in zip(y_activity_true, activity_probs)
        )
        top3_accuracy = top3_correct / len(y_activity_true)
        
        # Confusion matrix
        label_decoder = {v: k for k, v in self.predictor.label_encoder.items()}
        unique_labels = sorted(set(y_activity_true))
        cm = confusion_matrix(y_activity_true, activity_preds, labels=unique_labels)
        
        # ─────────────────────────────────────────────────────────
        # Outcome Evaluation
        # ─────────────────────────────────────────────────────────
        outcome_probs = self.predictor.outcome_model.predict_proba(X_test)
        outcome_preds = np.argmax(outcome_probs, axis=1)
        
        # Use probability of positive class for AUC
        if outcome_probs.shape[1] == 2:
            auc = roc_auc_score(y_outcome_true, outcome_probs[:, 1])
        else:
            auc = roc_auc_score(y_outcome_true, outcome_probs, multi_class='ovr')
        
        # ─────────────────────────────────────────────────────────
        # Time Prediction Evaluation
        # ─────────────────────────────────────────────────────────
        time_preds = self.predictor.time_model.predict(X_test) / 3600  # hours
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((np.array(y_time_true) - time_preds) / 
                              (np.array(y_time_true) + 1e-6))) * 100
        
        # ─────────────────────────────────────────────────────────
        # Early Detection Analysis
        # ─────────────────────────────────────────────────────────
        early_detection = self._evaluate_early_detection(test_traces)
        
        # ─────────────────────────────────────────────────────────
        # Sample Predictions for UI
        # ─────────────────────────────────────────────────────────
        samples = self._get_sample_predictions(test_prefixes[:10], X_test[:10])
        
        return EvaluationResults(
            next_activity_accuracy=accuracy_score(y_activity_true, activity_preds),
            next_activity_top3_accuracy=top3_accuracy,
            next_activity_macro_f1=f1_score(y_activity_true, activity_preds, average='macro'),
            next_activity_confusion_matrix=cm.tolist(),
            next_activity_class_labels=[label_decoder[l] for l in unique_labels],
            outcome_auc_roc=auc,
            outcome_precision=precision_score(y_outcome_true, outcome_preds, average='weighted'),
            outcome_recall=recall_score(y_outcome_true, outcome_preds, average='weighted'),
            outcome_f1=f1_score(y_outcome_true, outcome_preds, average='weighted'),
            time_mae_hours=mean_absolute_error(y_time_true, time_preds),
            time_rmse_hours=np.sqrt(mean_squared_error(y_time_true, time_preds)),
            time_mape=mape,
            early_detection=early_detection,
            sample_predictions=samples
        )
    
    def _evaluate_early_detection(self, test_traces: List[Trace]) -> Dict[str, float]:
        """Evaluate prediction accuracy at different prefix completion percentages."""
        results = {}
        
        for pct in [25, 50, 75]:
            correct = 0
            total = 0
            
            for trace in test_traces:
                # Get prefix at this percentage of trace
                prefix_len = max(1, int(len(trace.events) * pct / 100))
                if prefix_len >= len(trace.events):
                    continue
                
                prefix_events = trace.events[:prefix_len]
                target = trace.events[prefix_len].activity
                
                # Create prefix sample and predict
                prefix = PrefixSample(
                    case_id=trace.case_id,
                    prefix_events=prefix_events,
                    target_activity=target,
                    target_outcome=None,
                    remaining_time=None
                )
                
                features = self.predictor.feature_engineer.transform(prefix)
                pred = self.predictor.next_activity_model.predict(features.reshape(1, -1))[0]
                
                if self.predictor.label_encoder.get(target) == pred:
                    correct += 1
                total += 1
            
            results[f"{pct}%"] = correct / total if total > 0 else 0
        
        return results
    
    def _get_sample_predictions(self, prefixes: List[PrefixSample], 
                                 X: np.ndarray) -> List[Dict]:
        """Get sample predictions for UI display."""
        samples = []
        label_decoder = {v: k for k, v in self.predictor.label_encoder.items()}
        
        for i, prefix in enumerate(prefixes):
            probs = self.predictor.next_activity_model.predict_proba(X[i:i+1])[0]
            pred_idx = np.argmax(probs)
            
            samples.append({
                "case_id": prefix.case_id,
                "prefix_length": len(prefix.prefix_events),
                "last_activity": prefix.prefix_events[-1].activity,
                "predicted_next": label_decoder[pred_idx],
                "predicted_probability": float(probs[pred_idx]),
                "actual_next": prefix.target_activity,
                "correct": label_decoder[pred_idx] == prefix.target_activity
            })
        
        return samples
```

### 9.3 Evaluation Dashboard UI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MODEL EVALUATION RESULTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Test Set: 1,238 traces │ 16,456 prediction samples                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  NEXT ACTIVITY PREDICTION                                               ││
│  │  ─────────────────────────                                              ││
│  │                                                                          ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                     ││
│  │  │   Accuracy   │ │  Top-3 Acc   │ │   Macro F1   │                     ││
│  │  │    84.2%     │ │    95.1%     │ │    0.812     │                     ││
│  │  │      ✓       │ │      ✓       │ │      ✓       │                     ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘                     ││
│  │                                                                          ││
│  │  Confusion Matrix:                Early Detection:                       ││
│  │  ┌─────────────────────────┐     ┌─────────────────────────────────┐   ││
│  │  │     A    B    C    D    │     │ Prefix   │ Accuracy             │   ││
│  │  │ A  142   12    3    1   │     │──────────┼──────────────────────│   ││
│  │  │ B   15  198    8    2   │     │   25%    │ ███████░░░░ 71.2%    │   ││
│  │  │ C    2    7  167    5   │     │   50%    │ █████████░░ 82.4%    │   ││
│  │  │ D    1    3    4  189   │     │   75%    │ ██████████░ 89.1%    │   ││
│  │  └─────────────────────────┘     └─────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  OUTCOME PREDICTION                  TIME PREDICTION                    ││
│  │  ──────────────────                  ───────────────                    ││
│  │                                                                          ││
│  │  ┌────────────┐ ┌────────────┐      ┌────────────┐ ┌────────────┐       ││
│  │  │  AUC-ROC   │ │  F1-Score  │      │    MAE     │ │    RMSE    │       ││
│  │  │   0.891    │ │   0.844    │      │  2.3 hrs   │ │  4.1 hrs   │       ││
│  │  │     ✓      │ │     ✓      │      │     ✓      │ │     ✓      │       ││
│  │  └────────────┘ └────────────┘      └────────────┘ └────────────┘       ││
│  │                                                                          ││
│  │  Precision: 0.823  │  Recall: 0.867  │  MAPE: 18.2%                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  SAMPLE PREDICTIONS                                                          │
│  ──────────────────                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Case          │ Prefix │ Predicted    │ Actual       │ Conf  │ Result  ││
│  │───────────────┼────────┼──────────────┼──────────────┼───────┼─────────││
│  │ CASE_12345    │ 4      │ Review_Docs  │ Review_Docs  │ 82%   │   ✓     ││
│  │ CASE_12346    │ 3      │ Approve      │ Approve      │ 76%   │   ✓     ││
│  │ CASE_12347    │ 5      │ Request_Info │ Escalate     │ 54%   │   ✗     ││
│  │ CASE_12348    │ 2      │ Initial_Rev  │ Initial_Rev  │ 91%   │   ✓     ││
│  │ CASE_12349    │ 6      │ Close_Case   │ Close_Case   │ 88%   │   ✓     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  [View Full Results]                      [Go to Simulator →]               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Real-Time Event Simulator

### 10.1 Simulator Concept

The simulator allows users to step through a trace event-by-event, watching how predictions evolve as more information becomes available:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REAL-TIME EVENT SIMULATOR                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PURPOSE:                                                                    │
│  • Demonstrate prediction evolution as events unfold                        │
│  • Show how confidence changes with more information                        │
│  • Visualize the "early detection" capability                               │
│  • Provide compelling demo experience                                       │
│                                                                              │
│  MODES:                                                                      │
│  1. Manual Step: User clicks "Next Event" to advance                        │
│  2. Auto-Play: Events advance automatically (configurable speed)            │
│  3. Jump to Position: Slider to jump to any point in trace                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Simulator Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SIMULATOR DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  Select Trace   │  (from test set or custom input)                       │
│  │  Case_12345     │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         TRACE TIMELINE                                   ││
│  │                                                                          ││
│  │  Events: [E1] → [E2] → [E3] → [E4] → [E5] → [E6] → [E7] → [E8]         ││
│  │            ▲                                                             ││
│  │            │                                                             ││
│  │      Current Position (step 1)                                          ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           │                                                                  │
│           │  On each step:                                                   │
│           │  1. Send prefix [E1...En] to prediction API                     │
│           │  2. Receive WHAT/WHEN/HOW/WHY predictions                       │
│           │  3. Compare with actual next event (ground truth)               │
│           │  4. Update UI with prediction + correctness                     │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      PREDICTION COMPARISON                               ││
│  │                                                                          ││
│  │  Step 3/8:  [Submit] → [Review] → [Request_Docs] → [?]                  ││
│  │                                                                          ││
│  │  ┌─────────────────────┐    ┌─────────────────────┐                     ││
│  │  │    PREDICTED        │    │      ACTUAL         │                     ││
│  │  │                     │    │                     │                     ││
│  │  │  Next: Receive_Docs │    │  Next: Receive_Docs │  ✓ CORRECT         ││
│  │  │  Conf: 78%          │    │                     │                     ││
│  │  │                     │    │                     │                     ││
│  │  │  Time: ~4.2 hrs     │    │  Time: 3.8 hrs      │  Error: 0.4 hrs    ││
│  │  │                     │    │                     │                     ││
│  │  │  Outcome: Approved  │    │  Final: Approved    │  ✓ CORRECT         ││
│  │  │  Conf: 72%          │    │                     │                     ││
│  │  └─────────────────────┘    └─────────────────────┘                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Simulator Implementation

```python
# simulator.py
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class SimulatorMode(Enum):
    MANUAL = "manual"
    AUTO_PLAY = "auto_play"

@dataclass
class SimulatorState:
    trace: Trace
    current_step: int  # 0-indexed, how many events we've "seen"
    total_steps: int
    prediction_history: List[Dict]  # Predictions at each step
    
    @property
    def current_prefix(self) -> List[Event]:
        return self.trace.events[:self.current_step + 1]
    
    @property
    def actual_next_event(self) -> Optional[Event]:
        if self.current_step + 1 < len(self.trace.events):
            return self.trace.events[self.current_step + 1]
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.trace.events) - 1

@dataclass 
class SimulatorStepResult:
    step: int
    total_steps: int
    
    # Current state
    current_events: List[Dict]  # Events seen so far
    current_activity: str
    
    # Prediction
    predicted_next: str
    predicted_probability: float
    alternatives: List[Dict]
    predicted_time_hours: float
    predicted_outcome: str
    outcome_probability: float
    risk_factors: List[str]
    
    # Ground truth (if available)
    actual_next: Optional[str]
    actual_time_hours: Optional[float]
    actual_outcome: Optional[str]
    
    # Comparison
    next_correct: Optional[bool]
    time_error_hours: Optional[float]
    outcome_correct: Optional[bool]
    
    # Recommendations
    recommendations: Optional[List[str]]

class EventSimulator:
    def __init__(self, predictor: ProcessPredictor, 
                 prediction_engine: PredictionEngine):
        self.predictor = predictor
        self.prediction_engine = prediction_engine
        self.state: Optional[SimulatorState] = None
    
    def load_trace(self, trace: Trace) -> SimulatorState:
        """Initialize simulator with a trace."""
        self.state = SimulatorState(
            trace=trace,
            current_step=0,
            total_steps=len(trace.events),
            prediction_history=[]
        )
        
        # Generate initial prediction
        self._generate_prediction()
        
        return self.state
    
    def step_forward(self) -> Optional[SimulatorStepResult]:
        """Advance to next event and generate new prediction."""
        if self.state is None or self.state.is_complete:
            return None
        
        self.state.current_step += 1
        return self._generate_prediction()
    
    def step_backward(self) -> Optional[SimulatorStepResult]:
        """Go back to previous event."""
        if self.state is None or self.state.current_step <= 0:
            return None
        
        self.state.current_step -= 1
        
        # Return cached prediction if available
        if self.state.current_step < len(self.state.prediction_history):
            return self.state.prediction_history[self.state.current_step]
        
        return self._generate_prediction()
    
    def jump_to_step(self, step: int) -> Optional[SimulatorStepResult]:
        """Jump to specific step in trace."""
        if self.state is None:
            return None
        
        step = max(0, min(step, self.state.total_steps - 1))
        self.state.current_step = step
        
        # Return cached prediction if available
        if step < len(self.state.prediction_history):
            return self.state.prediction_history[step]
        
        return self._generate_prediction()
    
    def _generate_prediction(self) -> SimulatorStepResult:
        """Generate prediction for current prefix."""
        prefix_events = self.state.current_prefix
        
        # Get prediction from engine
        prediction = self.prediction_engine.predict(prefix_events)
        
        # Get ground truth
        actual_next = self.state.actual_next_event
        actual_outcome = self.state.trace.events[-1].activity
        
        # Calculate actual remaining time
        if actual_next:
            actual_remaining = (
                self.state.trace.end_time - prefix_events[-1].timestamp
            ).total_seconds() / 3600
        else:
            actual_remaining = 0
        
        # Build result
        result = SimulatorStepResult(
            step=self.state.current_step,
            total_steps=self.state.total_steps,
            current_events=[
                {"activity": e.activity, "timestamp": e.timestamp.isoformat()}
                for e in prefix_events
            ],
            current_activity=prefix_events[-1].activity,
            predicted_next=prediction.next_activity,
            predicted_probability=prediction.next_activity_probability,
            alternatives=[
                {"activity": a, "probability": p} 
                for a, p in prediction.alternative_activities
            ],
            predicted_time_hours=prediction.expected_remaining_time_hours,
            predicted_outcome=prediction.predicted_outcome,
            outcome_probability=prediction.outcome_probability,
            risk_factors=prediction.risk_factors,
            actual_next=actual_next.activity if actual_next else None,
            actual_time_hours=actual_remaining if actual_next else None,
            actual_outcome=actual_outcome,
            next_correct=(
                prediction.next_activity == actual_next.activity 
                if actual_next else None
            ),
            time_error_hours=(
                abs(prediction.expected_remaining_time_hours - actual_remaining)
                if actual_next else None
            ),
            outcome_correct=prediction.predicted_outcome == actual_outcome,
            recommendations=prediction.recommendations
        )
        
        # Cache result
        if self.state.current_step >= len(self.state.prediction_history):
            self.state.prediction_history.append(result)
        else:
            self.state.prediction_history[self.state.current_step] = result
        
        return result
    
    def get_prediction_evolution(self) -> List[Dict]:
        """Get how predictions evolved over all steps (for charts)."""
        if self.state is None:
            return []
        
        evolution = []
        for i, result in enumerate(self.state.prediction_history):
            evolution.append({
                "step": i + 1,
                "prefix_length": i + 1,
                "completion_pct": (i + 1) / self.state.total_steps * 100,
                "next_activity_confidence": result.predicted_probability,
                "next_correct": result.next_correct,
                "outcome_confidence": result.outcome_probability,
                "outcome_correct": result.outcome_correct,
                "time_error": result.time_error_hours
            })
        
        return evolution
```

### 10.4 Simulator API Endpoints

```python
# api/simulator.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/simulator", tags=["simulator"])

@router.post("/{model_id}/load")
async def load_trace(model_id: str, request: LoadTraceRequest) -> SimulatorState:
    """
    Load a trace into the simulator.
    
    Request body:
    - trace_id: ID of trace from test set, OR
    - events: Custom list of events to simulate
    """
    pass

@router.post("/{model_id}/step")
async def step_forward(model_id: str) -> SimulatorStepResult:
    """Advance simulator by one event."""
    pass

@router.post("/{model_id}/step-back")
async def step_backward(model_id: str) -> SimulatorStepResult:
    """Go back one event."""
    pass

@router.post("/{model_id}/jump/{step}")
async def jump_to_step(model_id: str, step: int) -> SimulatorStepResult:
    """Jump to specific step."""
    pass

@router.get("/{model_id}/evolution")
async def get_evolution(model_id: str) -> List[Dict]:
    """Get prediction evolution data for charting."""
    pass

@router.get("/{model_id}/test-traces")
async def get_available_traces(model_id: str, limit: int = 20) -> List[TraceSummary]:
    """Get list of test traces available for simulation."""
    pass
```

### 10.5 Simulator UI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REAL-TIME EVENT SIMULATOR                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Select Trace: [▼ CASE_12345 - 8 events - Approved        ]  [Load Random]  │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  TRACE PROGRESS                                                    Step 3/8 │
│  ──────────────                                                              │
│                                                                              │
│  ●────●────●────○────○────○────○────○                                       │
│  │    │    │    │    │    │    │    │                                       │
│  E1   E2   E3   E4   E5   E6   E7   E8                                      │
│  ▼                                                                           │
│  Submit  Review  Request                                                     │
│                  Docs                                                        │
│                   ▲                                                          │
│              CURRENT                                                         │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  CURRENT STATE                              PREDICTION vs REALITY            │
│  ─────────────                              ─────────────────────            │
│                                                                              │
│  Events seen:                               ┌────────────────────────────┐  │
│  1. Submit_Claim     (09:30)                │      NEXT ACTIVITY         │  │
│  2. Initial_Review   (14:45)                │                            │  │
│  3. Request_Docs     (11:00) ← current      │  Predicted: Receive_Docs   │  │
│                                             │  Confidence: 78%           │  │
│  Time elapsed: 1.5 days                     │                            │  │
│                                             │  Actual: Receive_Docs  ✓   │  │
│                                             └────────────────────────────┘  │
│                                                                              │
│                                             ┌────────────────────────────┐  │
│  PREDICTION CONFIDENCE OVER TIME            │      REMAINING TIME        │  │
│  ───────────────────────────────            │                            │  │
│  100%│         ╭──────────                  │  Predicted: 4.2 hours      │  │
│   80%│    ╭────╯                            │  Actual: 3.8 hours         │  │
│   60%│ ───╯                                 │  Error: 0.4 hours  ✓       │  │
│   40%│                                      │                            │  │
│      └─────────────────────                 └────────────────────────────┘  │
│       E1  E2  E3  E4  E5  E6                                                │
│                                             ┌────────────────────────────┐  │
│                                             │      FINAL OUTCOME         │  │
│                                             │                            │  │
│                                             │  Predicted: Approved (72%) │  │
│                                             │  Actual: Approved  ✓       │  │
│                                             └────────────────────────────┘  │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  RISK FACTORS                               RECOMMENDATIONS                  │
│  ────────────                               ───────────────                  │
│  • Document request pending                 • Follow up on document request │
│  • Above average elapsed time               • Pre-stage for final review    │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  CONTROLS                                                                    │
│                                                                              │
│  [◀ Back]  [▶ Next Event]  [▶▶ Auto-Play]     Speed: [▼ 1 sec/event]       │
│                                                                              │
│  Progress: ═══════════●══════════════════════════════════════  37.5%        │
│            │                    │                              │             │
│          Start               Current                          End            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.6 Prediction Evolution Chart

As users step through the trace, we show how predictions evolved:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREDICTION EVOLUTION (after simulation)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  NEXT ACTIVITY CONFIDENCE                   OUTCOME CONFIDENCE               │
│  ────────────────────────                   ──────────────────               │
│                                                                              │
│  100%│                   ●───●              100%│            ●───●──●       │
│   80%│          ●───●───●                    80%│       ●───●               │
│   60%│     ●───●                             60%│  ●───●                    │
│   40%│●───●                                  40%│●                          │
│      └────────────────────                      └────────────────────        │
│       1   2   3   4   5   6                      1   2   3   4   5   6       │
│              Step                                       Step                 │
│                                                                              │
│  ● = Correct prediction   ○ = Wrong prediction                              │
│                                                                              │
│  KEY INSIGHT: Predictions stabilize after 50% of trace (step 4)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Prediction Engine

### 11.1 Unified Prediction Interface

```python
# predictor.py
@dataclass
class PredictionResult:
    # WHAT - Next activity
    next_activity: str
    next_activity_probability: float
    alternative_activities: List[Tuple[str, float]]  # top 3
    
    # WHEN - Timing
    expected_remaining_time_hours: float
    time_confidence_interval: Tuple[float, float]  # 80% CI
    
    # HOW - Outcome
    predicted_outcome: str
    outcome_probability: float
    outcome_distribution: Dict[str, float]
    
    # WHY - Explanation
    feature_importance: List[Tuple[str, float]]  # top 5 features
    risk_factors: List[str]
    
    # Recommendations (if LLM enabled)
    recommendations: Optional[List[str]] = None

class PredictionEngine:
    def __init__(self, predictor: ProcessPredictor):
        self.predictor = predictor
        self.llm_enabled = False
    
    def predict(self, case_events: List[Event]) -> PredictionResult:
        """Generate comprehensive prediction for a running case."""
        # Create prefix sample
        prefix = PrefixSample(
            case_id="runtime",
            prefix_events=case_events,
            target_activity=None,
            target_outcome=None,
            remaining_time=None
        )
        
        # Extract features
        features = self.predictor.feature_engineer.transform(prefix)
        X = features.reshape(1, -1)
        
        # WHAT - Next activity prediction
        activity_probs = self.predictor.next_activity_model.predict_proba(X)[0]
        label_decoder = {v: k for k, v in self.predictor.label_encoder.items()}
        
        sorted_indices = np.argsort(activity_probs)[::-1]
        next_activity = label_decoder[sorted_indices[0]]
        next_prob = activity_probs[sorted_indices[0]]
        alternatives = [
            (label_decoder[i], activity_probs[i]) 
            for i in sorted_indices[1:4]
        ]
        
        # WHEN - Time prediction
        predicted_time = self.predictor.time_model.predict(X)[0]
        # Simple confidence interval (±30% as placeholder)
        time_ci = (predicted_time * 0.7 / 3600, predicted_time * 1.3 / 3600)
        
        # HOW - Outcome prediction
        outcome_probs = self.predictor.outcome_model.predict_proba(X)[0]
        outcome_idx = np.argmax(outcome_probs)
        predicted_outcome = label_decoder.get(outcome_idx, "Unknown")
        
        # WHY - Feature importance (from XGBoost)
        feature_names = self._get_feature_names()
        importances = self.predictor.next_activity_model.feature_importances_
        top_features = sorted(
            zip(feature_names, importances), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Risk factors (simple heuristics)
        risk_factors = self._identify_risks(features, case_events)
        
        result = PredictionResult(
            next_activity=next_activity,
            next_activity_probability=float(next_prob),
            alternative_activities=alternatives,
            expected_remaining_time_hours=float(predicted_time / 3600),
            time_confidence_interval=time_ci,
            predicted_outcome=predicted_outcome,
            outcome_probability=float(outcome_probs[outcome_idx]),
            outcome_distribution={label_decoder.get(i, f"class_{i}"): float(p) 
                                 for i, p in enumerate(outcome_probs)},
            feature_importance=[(f, float(i)) for f, i in top_features],
            risk_factors=risk_factors
        )
        
        # Optional: LLM recommendations
        if self.llm_enabled:
            result.recommendations = await self._generate_recommendations(result, case_events)
        
        return result
    
    def _identify_risks(self, features: np.ndarray, events: List[Event]) -> List[str]:
        """Simple rule-based risk identification."""
        risks = []
        
        # Long elapsed time
        if features[0] > 2.0:  # 2x average duration
            risks.append("Case duration exceeds typical by 2x")
        
        # Many loops
        if features[-4] > 3:  # loop_count feature
            risks.append(f"Multiple activity repetitions detected ({int(features[-4])} loops)")
        
        # Low transition probability
        if features[-6] < 0.1:  # transition_probability
            risks.append("Unusual activity transition (rarely seen in training)")
        
        return risks
```

### 11.2 LLM Recommendation Generation (Optional)

```python
# recommendations.py
async def generate_recommendations(prediction: PredictionResult,
                                   events: List[Event],
                                   domain_hints: Optional[dict] = None) -> List[str]:
    """Generate actionable recommendations using LLM."""
    
    context = {
        "current_activity": events[-1].activity,
        "predicted_next": prediction.next_activity,
        "predicted_outcome": prediction.predicted_outcome,
        "outcome_probability": prediction.outcome_probability,
        "remaining_time_hours": prediction.expected_remaining_time_hours,
        "risk_factors": prediction.risk_factors,
        "top_features": prediction.feature_importance[:3]
    }
    
    if domain_hints:
        context["domain"] = domain_hints.get("domain", "unknown")
    
    prompt = f"""
Based on this process prediction, provide 2-3 specific, actionable recommendations.

Current state:
- Last activity: {context['current_activity']}
- Predicted next: {context['predicted_next']} ({prediction.next_activity_probability:.0%} confidence)
- Predicted outcome: {context['predicted_outcome']} ({context['outcome_probability']:.0%})
- Estimated remaining time: {context['remaining_time_hours']:.1f} hours
- Risk factors: {context['risk_factors']}

Generate recommendations as a JSON array of strings. Be specific and actionable.
Focus on what a process manager should do RIGHT NOW.
"""
    
    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)
```

---

## 12. API Design

### 12.1 Endpoints

```python
# main.py (FastAPI)
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Process Failure Predictor")

# ─────────────────────────────────────────────────────────
# DATA ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_xes(file: UploadFile) -> UploadResponse:
    """Upload and parse XES file."""
    # Returns: log_id, statistics, preview

@app.post("/api/split/{log_id}")
async def split_data(log_id: str, train_ratio: float = 0.9) -> SplitResponse:
    """Perform temporal split on uploaded log."""
    # Returns: train_count, test_count, excluded_count, cutoff_time

# ─────────────────────────────────────────────────────────
# TRAINING ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/api/train/{log_id}")
async def train_models(log_id: str) -> TrainingResponse:
    """Train prediction models on split data."""
    # Returns: model_id, metrics, training_time

@app.get("/api/models/{model_id}/status")
async def model_status(model_id: str) -> ModelStatus:
    """Get training status and metrics."""

# ─────────────────────────────────────────────────────────
# EVALUATION ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/api/evaluate/{model_id}")
async def evaluate_model(model_id: str) -> EvaluationResults:
    """Run evaluation on held-out test set."""
    # Returns: accuracy, AUC, MAE, confusion matrix, early detection

@app.get("/api/evaluate/{model_id}/confusion-matrix")
async def get_confusion_matrix(model_id: str) -> ConfusionMatrixResponse:
    """Get detailed confusion matrix for next activity prediction."""

@app.get("/api/evaluate/{model_id}/early-detection")
async def get_early_detection(model_id: str) -> EarlyDetectionResponse:
    """Get accuracy at different prefix completion percentages."""

# ─────────────────────────────────────────────────────────
# PREDICTION ENDPOINTS
# ─────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    events: List[dict]  # List of {activity, timestamp, ...}

@app.post("/api/predict/{model_id}")
async def predict(model_id: str, request: PredictionRequest) -> PredictionResult:
    """Generate prediction for a running case."""

@app.get("/api/test-cases/{log_id}")
async def get_test_cases(log_id: str, limit: int = 10) -> List[TestCase]:
    """Get sample test cases for demo predictions."""

# ─────────────────────────────────────────────────────────
# SIMULATOR ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/api/simulator/{model_id}/traces")
async def get_simulator_traces(model_id: str, limit: int = 20) -> List[TraceSummary]:
    """Get test traces available for simulation."""

@app.post("/api/simulator/{model_id}/load")
async def load_simulator_trace(model_id: str, trace_id: str) -> SimulatorState:
    """Load a trace into the simulator."""

@app.post("/api/simulator/{model_id}/step")
async def simulator_step_forward(model_id: str) -> SimulatorStepResult:
    """Advance simulator by one event."""

@app.post("/api/simulator/{model_id}/step-back")  
async def simulator_step_backward(model_id: str) -> SimulatorStepResult:
    """Go back one event in simulator."""

@app.post("/api/simulator/{model_id}/jump/{step}")
async def simulator_jump(model_id: str, step: int) -> SimulatorStepResult:
    """Jump to specific step in trace."""

@app.get("/api/simulator/{model_id}/evolution")
async def get_prediction_evolution(model_id: str) -> List[EvolutionPoint]:
    """Get how predictions evolved across all simulated steps."""

@app.post("/api/simulator/{model_id}/reset")
async def reset_simulator(model_id: str) -> SimulatorState:
    """Reset simulator to beginning of current trace."""
```

### 12.2 Response Schemas

```python
# schemas.py
class UploadResponse(BaseModel):
    log_id: str
    trace_count: int
    event_count: int
    unique_activities: int
    time_range: Tuple[datetime, datetime]
    sample_activities: List[str]

class SplitResponse(BaseModel):
    train_traces: int
    test_traces: int
    excluded_traces: int
    cutoff_time: datetime
    effective_ratio: float
    warnings: List[str]  # e.g., "15% traces excluded"

class TrainingResponse(BaseModel):
    model_id: str
    training_time_seconds: float
    metrics: Dict[str, Dict[str, float]]  # task -> metric -> value

class EvaluationResults(BaseModel):
    # Next Activity
    next_activity_accuracy: float
    next_activity_top3_accuracy: float
    next_activity_macro_f1: float
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    
    # Outcome
    outcome_auc_roc: float
    outcome_precision: float
    outcome_recall: float
    outcome_f1: float
    
    # Time
    time_mae_hours: float
    time_rmse_hours: float
    time_mape: float
    
    # Early Detection
    early_detection: Dict[str, float]  # {"25%": 0.71, "50%": 0.82, "75%": 0.89}
    
    # Samples
    sample_predictions: List[Dict]

class PredictionResult(BaseModel):
    # WHAT
    next_activity: str
    next_activity_probability: float
    alternatives: List[Dict[str, Any]]
    
    # WHEN
    remaining_time_hours: float
    time_confidence: Tuple[float, float]
    
    # HOW
    predicted_outcome: str
    outcome_probability: float
    outcome_distribution: Dict[str, float]
    
    # WHY
    top_features: List[Dict[str, Any]]
    risk_factors: List[str]
    
    # ACTION (optional)
    recommendations: Optional[List[str]]

class SimulatorState(BaseModel):
    trace_id: str
    total_steps: int
    current_step: int
    is_complete: bool

class SimulatorStepResult(BaseModel):
    step: int
    total_steps: int
    current_events: List[Dict]
    
    # Predictions
    predicted_next: str
    predicted_probability: float
    alternatives: List[Dict]
    predicted_time_hours: float
    predicted_outcome: str
    outcome_probability: float
    risk_factors: List[str]
    
    # Ground truth
    actual_next: Optional[str]
    actual_time_hours: Optional[float]
    
    # Comparison
    next_correct: Optional[bool]
    time_error_hours: Optional[float]
    outcome_correct: Optional[bool]
    
    recommendations: Optional[List[str]]

class EvolutionPoint(BaseModel):
    step: int
    completion_pct: float
    next_confidence: float
    next_correct: bool
    outcome_confidence: float
    outcome_correct: bool
    time_error: Optional[float]
```

---

## 13. Frontend Architecture

### 13.1 Page Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND PAGES                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PAGE 1: UPLOAD                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [Drag & Drop Zone]                                                    │ │
│  │                                                                        │ │
│  │  After upload:                                                         │ │
│  │  • Trace count, event count, unique activities                        │ │
│  │  • Time range                                                         │ │
│  │  • Sample trace preview                                               │ │
│  │                                                                        │ │
│  │  [Continue to Split →]                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PAGE 2: DATA SPLIT                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [Timeline visualization showing train/test split]                    │ │
│  │                                                                        │ │
│  │  Statistics:                                                          │ │
│  │  • Training: X traces (Y%)                                            │ │
│  │  • Test: X traces (Y%)                                                │ │
│  │  • Excluded: X traces (Y%) ⚠️ (if > 10%)                              │ │
│  │                                                                        │ │
│  │  [Adjust Ratio: ====●====]  [Train Models →]                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PAGE 3: TRAINING                                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [Progress bar: Training models...]                                   │ │
│  │                                                                        │ │
│  │  When complete: Shows basic training metrics                          │ │
│  │                                                                        │ │
│  │  [View Evaluation Results →]                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PAGE 4: EVALUATION (NEW)                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Test Set Results (evaluated on held-out 10% data)                    │ │
│  │                                                                        │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                      │ │
│  │  │ Next Act.   │ │  Outcome    │ │    Time     │                      │ │
│  │  │ Acc: 84.2%  │ │ AUC: 0.891  │ │ MAE: 2.3hrs │                      │ │
│  │  │ Top3: 95.1% │ │ F1: 0.844   │ │ RMSE: 4.1   │                      │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘                      │ │
│  │                                                                        │ │
│  │  [Confusion Matrix]  [Early Detection Chart]  [Sample Predictions]    │ │
│  │                                                                        │ │
│  │  [Try Predictions →]  [Open Simulator →]                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PAGE 5: PREDICTIONS                                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [Select test case: ▼ Case_12345 ]                                    │ │
│  │                                                                        │ │
│  │  Current trace:                                                       │ │
│  │  [Activity A] → [Activity B] → [Activity C] → [?]                    │ │
│  │                                                                        │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │ │
│  │  │    WHAT     │ │    WHEN     │ │     HOW     │ │     WHY     │     │ │
│  │  │ Activity D  │ │ ~4.2 hours  │ │ Approved    │ │ Top factors │     │ │
│  │  │   (78%)     │ │ (3.1-5.8)   │ │   (72%)     │ │             │     │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │ │
│  │                                                                        │ │
│  │  RECOMMENDATIONS (if LLM enabled)                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PAGE 6: SIMULATOR (NEW)                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  [Select Trace: ▼ CASE_12345 ]  [Load Random]                        │ │
│  │                                                                        │ │
│  │  TRACE PROGRESS                                           Step 3/8   │ │
│  │  ●────●────●────○────○────○────○────○                                │ │
│  │  E1   E2   E3   E4   E5   E6   E7   E8                               │ │
│  │                                                                        │ │
│  │  ┌────────────────────────┐  ┌────────────────────────┐              │ │
│  │  │     PREDICTED          │  │       ACTUAL           │              │ │
│  │  │ Next: Receive_Docs     │  │ Next: Receive_Docs  ✓  │              │ │
│  │  │ Time: 4.2 hrs          │  │ Time: 3.8 hrs          │              │ │
│  │  │ Outcome: Approved 72%  │  │ Final: Approved  ✓     │              │ │
│  │  └────────────────────────┘  └────────────────────────┘              │ │
│  │                                                                        │ │
│  │  [Confidence Evolution Chart]                                         │ │
│  │                                                                        │ │
│  │  [◀ Back]  [▶ Next Event]  [▶▶ Auto-Play]  Speed: [▼ 1s]            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 13.2 Component Structure

```
src/
├── components/
│   ├── upload/
│   │   ├── DropZone.tsx
│   │   └── LogStatistics.tsx
│   ├── split/
│   │   ├── TimelineViz.tsx
│   │   └── SplitStats.tsx
│   ├── training/
│   │   ├── ProgressBar.tsx
│   │   └── MetricsTable.tsx
│   ├── evaluation/                    # NEW
│   │   ├── MetricCards.tsx            # Accuracy, AUC, MAE displays
│   │   ├── ConfusionMatrix.tsx        # Interactive confusion matrix
│   │   ├── EarlyDetectionChart.tsx    # Accuracy vs prefix length
│   │   └── SamplePredictions.tsx      # Table of sample results
│   ├── prediction/
│   │   ├── CaseSelector.tsx
│   │   ├── TraceViewer.tsx
│   │   ├── PredictionCards.tsx
│   │   └── Recommendations.tsx
│   └── simulator/                     # NEW
│       ├── TraceSelector.tsx          # Dropdown to pick trace
│       ├── TraceProgress.tsx          # Visual step indicator
│       ├── PredictionComparison.tsx   # Side-by-side pred vs actual
│       ├── EvolutionChart.tsx         # Confidence over steps
│       └── SimulatorControls.tsx      # Play/pause/step buttons
├── pages/
│   ├── UploadPage.tsx
│   ├── SplitPage.tsx
│   ├── TrainingPage.tsx
│   ├── EvaluationPage.tsx             # NEW
│   ├── PredictionPage.tsx
│   └── SimulatorPage.tsx              # NEW
├── hooks/
│   ├── useUpload.ts
│   ├── useSplit.ts
│   ├── useTraining.ts
│   ├── useEvaluation.ts               # NEW
│   ├── usePrediction.ts
│   └── useSimulator.ts                # NEW
└── api/
    └── client.ts
```

---

## 14. Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | React + TypeScript | Industry standard, good ecosystem |
| UI Components | Tailwind + shadcn/ui | Fast development, consistent design |
| State/Data | React Query | Handles API state elegantly |
| Charts | Recharts | Simple, React-native |
| Backend | FastAPI (Python) | Fast, async, good typing |
| ML | XGBoost + scikit-learn | Proven, fast, interpretable |
| Database | SQLite | Zero config, sufficient for demo |
| Model Storage | Filesystem (joblib) | Simple, reliable |
| LLM (optional) | Claude API | Best reasoning for recommendations |
| Containerization | Docker Compose | Easy local development |

---

## 15. Implementation Roadmap

### 15.1 8-Week Sprint Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       IMPLEMENTATION ROADMAP (8 WEEKS)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WEEK 1-2: DATA PIPELINE                                                    │
│  ─────────────────────────                                                  │
│  □ XES parser with streaming support                                        │
│  □ Data validation and error handling                                       │
│  □ Temporal splitter with excluded trace handling                           │
│  □ Prefix generation                                                        │
│  □ Basic FastAPI endpoints (/upload, /split)                                │
│  □ Upload page UI with drag-drop                                            │
│  □ Split visualization page                                                 │
│                                                                              │
│  Deliverable: Upload XES → see statistics → perform split                   │
│                                                                              │
│  WEEK 3-4: MODEL TRAINING + EVALUATION                                      │
│  ─────────────────────────────────────                                      │
│  □ Implement 20 core features                                               │
│  □ Feature normalization pipeline                                           │
│  □ XGBoost training for all three tasks                                     │
│  □ Model serialization/loading                                              │
│  □ Test set evaluation pipeline                                             │
│  □ Confusion matrix generation                                              │
│  □ Early detection analysis (accuracy at 25/50/75% prefix)                  │
│  □ Training page UI with progress                                           │
│  □ Evaluation dashboard UI                                                  │
│                                                                              │
│  Deliverable: Complete training + evaluation on held-out test set           │
│                                                                              │
│  WEEK 5-6: PREDICTION ENGINE + SIMULATOR                                    │
│  ─────────────────────────────────────────                                  │
│  □ Unified prediction interface                                             │
│  □ Feature importance extraction                                            │
│  □ Risk factor identification                                               │
│  □ Real-time event simulator backend                                        │
│  □ Simulator state management (step forward/back/jump)                      │
│  □ Prediction vs ground truth comparison                                    │
│  □ Prediction evolution tracking                                            │
│  □ Prediction dashboard UI (WHAT/WHEN/HOW/WHY cards)                        │
│  □ Simulator UI with controls and charts                                    │
│                                                                              │
│  Deliverable: Full prediction flow + interactive event simulator            │
│                                                                              │
│  WEEK 7: LLM INTEGRATION + POLISH                                           │
│  ────────────────────────────────                                           │
│  □ Optional LLM domain analysis at upload                                   │
│  □ LLM recommendation generation at prediction                              │
│  □ Auto-play mode for simulator                                             │
│  □ Error handling and edge cases                                            │
│  □ Loading states and UX polish                                             │
│                                                                              │
│  Deliverable: LLM-enhanced recommendations + polished simulator             │
│                                                                              │
│  WEEK 8: DEMO PREP                                                          │
│  ─────────────────────                                                      │
│  □ Sample datasets (3 domains)                                              │
│  □ End-to-end testing                                                       │
│  □ Docker compose setup                                                     │
│  □ Demo script and documentation                                            │
│  □ Performance optimization                                                 │
│  □ Simulator demo scenarios                                                 │
│                                                                              │
│  Deliverable: Production-ready demo with compelling simulator walkthrough   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15.2 Scope Control

| Must Have (MVP) | Should Have | Nice to Have (Future) |
|-----------------|-------------|----------------------|
| XES upload + parsing | LLM domain hints | Multiple model comparison |
| Temporal split with viz | Risk factor identification | Hyperparameter tuning |
| 20 core features | Recommendations | Transformer model |
| XGBoost (3 tasks) | **Auto-play simulator** | Batch predictions |
| **Test set evaluation** | Docker deployment | Model export/import |
| **Evaluation dashboard** | Reliability diagrams | Real-time streaming |
| **Probability calibration** | | |
| **Confidence intervals** | | |
| Prediction dashboard | | |
| **Event simulator** | | |
| Feature importance | | |

---

## 16. Project Structure

```
process-failure-predictor/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── api/
│   │   │   ├── upload.py
│   │   │   ├── split.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py      # Evaluation endpoints
│   │   │   ├── predict.py
│   │   │   └── simulator.py     # Simulator endpoints
│   │   ├── core/
│   │   │   ├── xes_parser.py
│   │   │   ├── splitter.py
│   │   │   ├── prefix_generator.py
│   │   │   ├── feature_engineer.py
│   │   │   ├── trainer.py
│   │   │   ├── calibration.py   # Probability calibration
│   │   │   ├── time_confidence.py  # Quantile regression for time CI
│   │   │   ├── evaluator.py     # Test set evaluation
│   │   │   ├── predictor.py
│   │   │   └── simulator.py     # Event simulator logic
│   │   ├── models/
│   │   │   └── schemas.py
│   │   └── services/
│   │       └── llm_service.py   # Optional
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── upload/
│   │   │   ├── split/
│   │   │   ├── training/
│   │   │   ├── evaluation/
│   │   │   │   ├── MetricCards.tsx
│   │   │   │   ├── ConfusionMatrix.tsx
│   │   │   │   ├── ReliabilityDiagram.tsx  # Calibration visualization
│   │   │   │   └── EarlyDetectionChart.tsx
│   │   │   ├── prediction/
│   │   │   │   ├── ConfidenceBadge.tsx     # HIGH/MEDIUM/LOW indicator
│   │   │   │   └── ...
│   │   │   └── simulator/
│   │   ├── pages/
│   │   │   ├── UploadPage.tsx
│   │   │   ├── SplitPage.tsx
│   │   │   ├── TrainingPage.tsx
│   │   │   ├── EvaluationPage.tsx
│   │   │   ├── PredictionPage.tsx
│   │   │   └── SimulatorPage.tsx
│   │   ├── hooks/
│   │   └── api/
│   ├── package.json
│   └── Dockerfile
│
├── data/
│   └── samples/                 # Sample XES files
│
├── docker-compose.yml
└── README.md
```

---

## 17. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| XES parsing | <5s for 100K events | Streaming parser |
| Data split | <1s for 20K traces | In-memory operation |
| Feature engineering | <10s for 10K prefixes | Vectorized operations |
| Model training | <3 min total | XGBoost is fast |
| **Calibration fitting** | <5s | Isotonic regression on validation set |
| **Test set evaluation** | <30s for 1K test traces | Batch inference |
| **ECE calculation** | <1s | Simple binning operation |
| **Confusion matrix** | <1s | Cached after evaluation |
| Single prediction | <100ms | Cached model, numpy ops |
| **Calibrated prediction** | <110ms | +10ms for calibration lookup |
| **Simulator step** | <150ms | Single prediction + comparison |
| **Simulator load trace** | <200ms | Parse + initial prediction |
| LLM recommendation | <3s | Single API call |

---

## 18. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM API unavailable | System works fully without LLM; recommendations are optional |
| Large file upload fails | Streaming parser, chunked upload, clear error messages |
| Poor model performance | Show confidence scores, allow retraining with different split |
| Too many excluded traces | UI warning at >10%, suggest adjusting split ratio |
| Slow training | Progress indicator, reasonable timeouts |
| **Simulator state lost** | Store state in session, allow reset to beginning |
| **Evaluation takes too long** | Cache results after first computation |
| **Poor calibration (high ECE)** | Show warning, fall back to raw probabilities with disclaimer |
| **Overconfident predictions** | Calibration catches this; UI shows confidence level badges |
| **Time interval too wide** | Flag as LOW confidence, show full range to user |

---

*Document Version 2.0 | December 2025*  
*Optimized for 8-week delivery with focused scope*  
*Includes: Probability calibration + Test set evaluation + Real-time event simulator*
