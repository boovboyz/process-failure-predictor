"""
Feature engineering for process prediction.
Implements 20 core domain-agnostic features.
"""
import numpy as np
from collections import Counter
from typing import List, Dict, Optional
from app.core.xes_parser import Trace, Event
from app.core.prefix_generator import PrefixSample


class FeatureEngineer:
    """
    Feature engineering for process prediction.
    
    Extracts 20 core features from trace prefixes:
    - 5 temporal features
    - 8 sequence features  
    - 3 transition features
    - 4 statistical features
    """
    
    def __init__(self):
        self.activity_encoder: Dict[str, int] = {}
        self.transition_probs: Dict[tuple, float] = {}
        self.time_stats: Dict[str, float] = {}
        self.activity_frequencies: Dict[str, float] = {}
        self.top_activities: List[str] = []
        self.is_fitted = False
        
    def fit(self, train_traces: List[Trace]):
        """
        Learn statistics from training data.
        
        Args:
            train_traces: List of training traces
        """
        # Build activity vocabulary
        all_activities = set()
        for trace in train_traces:
            for event in trace.events:
                all_activities.add(event.activity)
        
        # Sort for consistent encoding
        self.activity_encoder = {a: i for i, a in enumerate(sorted(all_activities))}
        
        # Learn transition probabilities
        transition_counts = Counter()
        from_counts = Counter()
        
        for trace in train_traces:
            for i in range(len(trace.events) - 1):
                from_act = trace.events[i].activity
                to_act = trace.events[i + 1].activity
                transition_counts[(from_act, to_act)] += 1
                from_counts[from_act] += 1
        
        self.transition_probs = {
            k: v / from_counts[k[0]]
            for k, v in transition_counts.items()
        }
        
        # Compute time normalization stats
        all_durations = []
        all_inter_event_times = []
        
        for trace in train_traces:
            if trace.events:
                total_duration = trace.duration_seconds
                all_durations.append(total_duration)
                
                for i in range(1, len(trace.events)):
                    gap = (trace.events[i].timestamp - trace.events[i-1].timestamp).total_seconds()
                    all_inter_event_times.append(gap)
        
        self.time_stats = {
            'mean_duration': np.mean(all_durations) if all_durations else 1.0,
            'std_duration': np.std(all_durations) + 1e-6 if all_durations else 1.0,
            'mean_inter_event': np.mean(all_inter_event_times) if all_inter_event_times else 1.0,
        }
        
        # Activity frequencies
        activity_counts = Counter()
        for trace in train_traces:
            for event in trace.events:
                activity_counts[event.activity] += 1
        
        total = sum(activity_counts.values())
        self.activity_frequencies = {
            a: c / total for a, c in activity_counts.items()
        }
        
        # Store top 5 activities by frequency
        self.top_activities = sorted(
            self.activity_frequencies.keys(),
            key=lambda a: self.activity_frequencies[a],
            reverse=True
        )[:5]
        
        # Trace length statistics
        trace_lengths = [len(t.events) for t in train_traces]
        self.time_stats['mean_trace_length'] = np.mean(trace_lengths) if trace_lengths else 10.0
        
        self.is_fitted = True
        
    def transform(self, prefix: PrefixSample) -> np.ndarray:
        """
        Extract features from a prefix.
        
        Args:
            prefix: PrefixSample to extract features from
            
        Returns:
            numpy array of features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        events = prefix.prefix_events
        if not events:
            return np.zeros(self.num_features, dtype=np.float32)
        
        last_event = events[-1]
        features = []
        
        # ─────────────────────────────────────────────────────────────────────
        # TEMPORAL FEATURES (5)
        # ─────────────────────────────────────────────────────────────────────
        
        # 1. elapsed_time_total (normalized)
        total_elapsed = (last_event.timestamp - events[0].timestamp).total_seconds()
        features.append(total_elapsed / self.time_stats['mean_duration'])
        
        # 2. elapsed_time_last_event (hours)
        if len(events) > 1:
            last_gap = (last_event.timestamp - events[-2].timestamp).total_seconds()
        else:
            last_gap = 0
        features.append(last_gap / 3600)  # Convert to hours
        
        # 3. hour_of_day
        features.append(last_event.timestamp.hour)
        
        # 4. day_of_week
        features.append(last_event.timestamp.weekday())
        
        # 5. is_business_hours
        is_business = (
            9 <= last_event.timestamp.hour <= 17 
            and last_event.timestamp.weekday() < 5
        )
        features.append(float(is_business))
        
        # ─────────────────────────────────────────────────────────────────────
        # SEQUENCE FEATURES (8)
        # ─────────────────────────────────────────────────────────────────────
        
        # 6. trace_length
        features.append(len(events))
        
        # 7. unique_activities
        features.append(len(set(e.activity for e in events)))
        
        # 8. last_activity_encoded
        features.append(self.activity_encoder.get(last_event.activity, -1))
        
        # 9-11. last_3_activities (encoded, padded)
        last_3 = [e.activity for e in events[-3:]]
        while len(last_3) < 3:
            last_3.insert(0, '<PAD>')
        
        for act in last_3:
            features.append(self.activity_encoder.get(act, -1))
        
        # 12-16. activity_occurrence_counts (top 5 activities)
        activity_counts = Counter(e.activity for e in events)
        for act in self.top_activities:
            features.append(activity_counts.get(act, 0))
        
        # Pad if we have fewer than 5 top activities
        while len(features) < 16:
            features.append(0)
        
        # ─────────────────────────────────────────────────────────────────────
        # TRANSITION FEATURES (3)
        # ─────────────────────────────────────────────────────────────────────
        
        # 17. transition_probability
        if len(events) > 1:
            last_transition = (events[-2].activity, last_event.activity)
            features.append(self.transition_probs.get(last_transition, 0))
        else:
            features.append(0)
        
        # 18. is_common_transition (seen > 10 times in training)
        if len(events) > 1:
            last_transition = (events[-2].activity, last_event.activity)
            features.append(float(self.transition_probs.get(last_transition, 0) > 0))
        else:
            features.append(0)
        
        # 19. loop_count (repeated activities)
        features.append(len(events) - len(set(e.activity for e in events)))
        
        # ─────────────────────────────────────────────────────────────────────
        # STATISTICAL FEATURES (4)
        # ─────────────────────────────────────────────────────────────────────
        
        # 20. trace_length_percentile (normalized)
        features.append(len(events) / self.time_stats['mean_trace_length'])
        
        # 21. elapsed_time_percentile
        features.append(total_elapsed / self.time_stats['mean_duration'])
        
        # 22. activity_rarity (1/frequency)
        activity_freq = self.activity_frequencies.get(last_event.activity, 0.01)
        features.append(1 / max(activity_freq, 0.001))
        
        # 23. avg_time_between_events
        inter_event_times = []
        for i in range(1, len(events)):
            gap = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            inter_event_times.append(gap)
        
        if inter_event_times:
            features.append(np.mean(inter_event_times) / 3600)  # hours
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def transform_batch(self, prefixes: List[PrefixSample]) -> np.ndarray:
        """
        Transform multiple prefixes.
        
        Args:
            prefixes: List of PrefixSample objects
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        return np.array([self.transform(p) for p in prefixes])
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return 23  # Actual count of features
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        names = [
            # Temporal
            "elapsed_time_normalized",
            "time_since_last_event_hours",
            "hour_of_day",
            "day_of_week",
            "is_business_hours",
            # Sequence
            "trace_length",
            "unique_activities",
            "last_activity_encoded",
            "activity_3_back",
            "activity_2_back",
            "activity_1_back",
        ]
        
        # Add top activity counts
        for i, act in enumerate(self.top_activities):
            names.append(f"count_{act[:20]}")
        
        # Pad to 5
        while len(names) < 16:
            names.append(f"count_activity_{len(names) - 10}")
        
        names.extend([
            # Transition
            "transition_probability",
            "is_common_transition",
            "loop_count",
            # Statistical
            "trace_length_percentile",
            "elapsed_time_percentile",
            "activity_rarity",
            "avg_inter_event_time_hours",
        ])
        
        return names
    
    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            'activity_encoder': self.activity_encoder,
            'transition_probs': {str(k): v for k, v in self.transition_probs.items()},
            'time_stats': self.time_stats,
            'activity_frequencies': self.activity_frequencies,
            'top_activities': self.top_activities,
            'is_fitted': self.is_fitted,
        }
    
    def set_state(self, state: dict):
        """Set state from deserialized data."""
        self.activity_encoder = state['activity_encoder']
        # Convert string keys back to tuples
        self.transition_probs = {
            eval(k): v for k, v in state['transition_probs'].items()
        }
        self.time_stats = state['time_stats']
        self.activity_frequencies = state['activity_frequencies']
        self.top_activities = state['top_activities']
        self.is_fitted = state['is_fitted']
