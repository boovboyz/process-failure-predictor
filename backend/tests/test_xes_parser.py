"""
Unit tests for XES parser module.
"""
import pytest
from datetime import datetime, timedelta
from app.core.xes_parser import (
    Event, Trace, EventLog, 
    parse_xes_string, validate_log, validate_and_clean
)


# Sample XES content for testing
SIMPLE_XES = """<?xml version="1.0" encoding="UTF-8"?>
<log>
    <trace>
        <string key="concept:name" value="case_1"/>
        <event>
            <string key="concept:name" value="Start"/>
            <date key="time:timestamp" value="2024-01-01T09:00:00.000+00:00"/>
        </event>
        <event>
            <string key="concept:name" value="Process"/>
            <date key="time:timestamp" value="2024-01-01T10:00:00.000+00:00"/>
        </event>
        <event>
            <string key="concept:name" value="End"/>
            <date key="time:timestamp" value="2024-01-01T11:00:00.000+00:00"/>
        </event>
    </trace>
</log>
"""

EMPTY_TRACE_XES = """<?xml version="1.0" encoding="UTF-8"?>
<log>
    <trace>
        <string key="concept:name" value="empty_case"/>
    </trace>
</log>
"""


class TestEvent:
    """Tests for Event dataclass."""
    
    def test_event_creation(self):
        event = Event(
            activity="Test Activity",
            timestamp=datetime.now(),
            resource="User1"
        )
        assert event.activity == "Test Activity"
        assert event.resource == "User1"
    
    def test_event_without_resource(self):
        event = Event(activity="Test", timestamp=datetime.now())
        assert event.resource is None
        assert event.attributes == {}


class TestTrace:
    """Tests for Trace dataclass."""
    
    def test_trace_timestamps(self):
        events = [
            Event(activity="A", timestamp=datetime(2024, 1, 1, 9, 0)),
            Event(activity="B", timestamp=datetime(2024, 1, 1, 10, 0)),
            Event(activity="C", timestamp=datetime(2024, 1, 1, 11, 0)),
        ]
        trace = Trace(case_id="test_case", events=events)
        
        assert trace.start_time == datetime(2024, 1, 1, 9, 0)
        assert trace.end_time == datetime(2024, 1, 1, 11, 0)
        assert trace.duration_seconds == 7200  # 2 hours
    
    def test_empty_trace_raises(self):
        trace = Trace(case_id="empty", events=[])
        with pytest.raises(ValueError):
            _ = trace.start_time


class TestXESParsing:
    """Tests for XES parsing functions."""
    
    def test_parse_simple_xes(self):
        log = parse_xes_string(SIMPLE_XES)
        
        assert len(log.traces) == 1
        assert log.traces[0].case_id == "case_1"
        assert len(log.traces[0].events) == 3
        assert log.traces[0].events[0].activity == "Start"
        assert log.traces[0].events[1].activity == "Process"
        assert log.traces[0].events[2].activity == "End"
    
    def test_parse_empty_trace(self):
        log = parse_xes_string(EMPTY_TRACE_XES)
        # Empty traces should be filtered out during parsing
        assert len(log.traces) == 0
    
    def test_invalid_xes(self):
        # Truly malformed XML that will cause SAX parser to fail
        with pytest.raises(ValueError):
            parse_xes_string("<log><trace><unclosed>")


class TestValidation:
    """Tests for validation functions."""
    
    def test_validate_empty_log(self):
        log = EventLog(traces=[])
        issues = validate_log(log)
        assert any("no traces" in issue.lower() for issue in issues)
    
    def test_validate_log_with_issues(self):
        # Create trace with missing activity
        events = [
            Event(activity="", timestamp=datetime.now())  # Missing activity
        ]
        trace = Trace(case_id="", events=events)  # Missing case_id
        log = EventLog(traces=[trace])
        
        issues = validate_log(log)
        assert len(issues) >= 2  # At least case_id and activity warnings
    
    def test_validate_and_clean(self):
        events = [
            Event(activity="B", timestamp=datetime(2024, 1, 1, 10, 0)),
            Event(activity="A", timestamp=datetime(2024, 1, 1, 9, 0)),  # Out of order
        ]
        trace = Trace(case_id="test", events=events)
        log = EventLog(traces=[trace])
        
        cleaned, warnings = validate_and_clean(log)
        
        # Events should be sorted by timestamp
        assert cleaned.traces[0].events[0].activity == "A"
        assert cleaned.traces[0].events[1].activity == "B"


class TestEventLog:
    """Tests for EventLog class."""
    
    def test_all_activities(self):
        events1 = [Event(activity="A", timestamp=datetime.now())]
        events2 = [Event(activity="B", timestamp=datetime.now())]
        log = EventLog(traces=[
            Trace(case_id="1", events=events1),
            Trace(case_id="2", events=events2),
        ])
        
        activities = log.all_activities
        assert "A" in activities
        assert "B" in activities
    
    def test_total_events(self):
        events1 = [
            Event(activity="A", timestamp=datetime.now()),
            Event(activity="B", timestamp=datetime.now()),
        ]
        events2 = [Event(activity="C", timestamp=datetime.now())]
        log = EventLog(traces=[
            Trace(case_id="1", events=events1),
            Trace(case_id="2", events=events2),
        ])
        
        assert log.total_events == 3
