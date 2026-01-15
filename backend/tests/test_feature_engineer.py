"""
Unit tests for feature engineering module.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from app.core.xes_parser import Event, Trace
from app.core.prefix_generator import PrefixSample
from app.core.feature_engineer import FeatureEngineer, UNK_TOKEN, PAD_TOKEN


def create_sample_traces():
    """Create sample traces for testing."""
    traces = []
    base_time = datetime(2024, 1, 1, 9, 0)
    
    for i in range(10):
        events = [
            Event(activity="Start", timestamp=base_time + timedelta(hours=i*24)),
            Event(activity="Process", timestamp=base_time + timedelta(hours=i*24 + 1)),
            Event(activity="Review", timestamp=base_time + timedelta(hours=i*24 + 2)),
            Event(activity="End", timestamp=base_time + timedelta(hours=i*24 + 3)),
        ]
        traces.append(Trace(case_id=f"case_{i}", events=events))
    
    return traces


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_fit_creates_encoder(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        assert fe.is_fitted
        assert "Start" in fe.activity_encoder
        assert "Process" in fe.activity_encoder
        assert "Review" in fe.activity_encoder
        assert "End" in fe.activity_encoder
    
    def test_special_tokens_in_encoder(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        # Special tokens should be present
        assert PAD_TOKEN in fe.activity_encoder
        assert UNK_TOKEN in fe.activity_encoder
        
        # PAD should be 0, UNK should be 1
        assert fe.activity_encoder[PAD_TOKEN] == 0
        assert fe.activity_encoder[UNK_TOKEN] == 1
    
    def test_transform_returns_correct_shape(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        prefix = PrefixSample(
            case_id="test",
            prefix_events=traces[0].events[:2],
            target_activity="Review",
            target_outcome="Success",
            remaining_time=3600.0
        )
        
        features = fe.transform(prefix)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == fe.num_features
    
    def test_unknown_activity_handling(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        # Create prefix with unknown activity
        unknown_event = Event(
            activity="UNKNOWN_ACTIVITY_XYZ",
            timestamp=datetime.now()
        )
        prefix = PrefixSample(
            case_id="test",
            prefix_events=[unknown_event],
            target_activity="End",
            target_outcome="Success",
            remaining_time=3600.0
        )
        
        # Should not raise, should use UNK token
        features = fe.transform(prefix)
        
        # Unknown activity should be tracked
        assert "UNKNOWN_ACTIVITY_XYZ" in fe.unknown_activities
        
        # Feature for last_activity should be UNK token encoding (1)
        assert features[7] == fe.activity_encoder[UNK_TOKEN]
    
    def test_transform_batch(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        prefixes = [
            PrefixSample(
                case_id="test1",
                prefix_events=traces[0].events[:2],
                target_activity="Review",
                target_outcome="Success",
                remaining_time=3600.0
            ),
            PrefixSample(
                case_id="test2",
                prefix_events=traces[1].events[:3],
                target_activity="End",
                target_outcome="Success",
                remaining_time=1800.0
            ),
        ]
        
        features = fe.transform_batch(prefixes)
        
        assert features.shape == (2, fe.num_features)
    
    def test_serialization(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        # Add some unknown activities
        unknown_event = Event(activity="Unknown", timestamp=datetime.now())
        prefix = PrefixSample(
            case_id="test",
            prefix_events=[unknown_event],
            target_activity="End",
            target_outcome="Success",
            remaining_time=0.0
        )
        fe.transform(prefix)
        
        # Serialize
        state = fe.get_state()
        
        # Deserialize into new instance
        fe2 = FeatureEngineer()
        fe2.set_state(state)
        
        assert fe2.is_fitted
        assert fe2.activity_encoder == fe.activity_encoder
        assert fe2.unknown_activities == fe.unknown_activities
    
    def test_feature_names(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        names = fe.feature_names
        
        assert len(names) == fe.num_features
        assert "elapsed_time_normalized" in names
        assert "trace_length" in names


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""
    
    def test_business_hours_feature(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        # Business hours event (9am on Monday)
        business_event = Event(
            activity="Start",
            timestamp=datetime(2024, 1, 1, 10, 0)  # Monday 10am
        )
        prefix = PrefixSample(
            case_id="test",
            prefix_events=[business_event],
            target_activity="End",
            target_outcome="Success",
            remaining_time=3600.0
        )
        
        features = fe.transform(prefix)
        
        # is_business_hours is feature index 4
        assert features[4] == 1.0
    
    def test_non_business_hours_feature(self):
        traces = create_sample_traces()
        fe = FeatureEngineer()
        fe.fit(traces)
        
        # Non-business hours event (2am on Monday)
        night_event = Event(
            activity="Start",
            timestamp=datetime(2024, 1, 1, 2, 0)  # Monday 2am
        )
        prefix = PrefixSample(
            case_id="test",
            prefix_events=[night_event],
            target_activity="End",
            target_outcome="Success",
            remaining_time=3600.0
        )
        
        features = fe.transform(prefix)
        
        # is_business_hours is feature index 4
        assert features[4] == 0.0
