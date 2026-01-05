"""
XES Parser for IEEE 1849-2016 XES format.
Supports streaming parsing for large files.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import xml.sax
from xml.sax.handler import ContentHandler
from pathlib import Path


@dataclass
class Event:
    """Single event in a process trace."""
    activity: str
    timestamp: datetime
    resource: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A single process trace (case)."""
    case_id: str
    events: List[Event] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def start_time(self) -> datetime:
        """Get the start time of the trace."""
        if not self.events:
            raise ValueError("Trace has no events")
        return self.events[0].timestamp
    
    @property
    def end_time(self) -> datetime:
        """Get the end time of the trace."""
        if not self.events:
            raise ValueError("Trace has no events")
        return self.events[-1].timestamp
    
    @property
    def duration_seconds(self) -> float:
        """Get the duration of the trace in seconds."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class EventLog:
    """Complete event log."""
    traces: List[Trace] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_activities(self) -> List[str]:
        """Get all unique activity names."""
        activities = set()
        for trace in self.traces:
            for event in trace.events:
                activities.add(event.activity)
        return sorted(activities)
    
    @property
    def total_events(self) -> int:
        """Get total number of events."""
        return sum(len(trace.events) for trace in self.traces)
    
    @property
    def time_range(self) -> tuple:
        """Get the time range of the log."""
        if not self.traces:
            return None, None
        
        min_time = min(t.start_time for t in self.traces if t.events)
        max_time = max(t.end_time for t in self.traces if t.events)
        return min_time, max_time


class XESHandler(ContentHandler):
    """SAX handler for streaming XES parsing."""
    
    def __init__(self):
        super().__init__()
        self.log = EventLog()
        self.current_trace: Optional[Trace] = None
        self.current_event: Optional[Event] = None
        self.in_trace = False
        self.in_event = False
        self.current_key: Optional[str] = None
        
    def startElement(self, name: str, attrs):
        if name == "trace":
            self.in_trace = True
            self.current_trace = Trace(case_id="", events=[], attributes={})
            
        elif name == "event":
            self.in_event = True
            self.current_event = Event(
                activity="",
                timestamp=datetime.now(),
                attributes={}
            )
            
        elif name == "string":
            key = attrs.get("key", "")
            value = attrs.get("value", "")
            self._handle_string(key, value)
            
        elif name == "date":
            key = attrs.get("key", "")
            value = attrs.get("value", "")
            self._handle_date(key, value)
            
        elif name == "int":
            key = attrs.get("key", "")
            value = attrs.get("value", "")
            self._handle_int(key, value)
            
        elif name == "float":
            key = attrs.get("key", "")
            value = attrs.get("value", "")
            self._handle_float(key, value)
            
        elif name == "boolean":
            key = attrs.get("key", "")
            value = attrs.get("value", "")
            self._handle_boolean(key, value)
    
    def endElement(self, name: str):
        if name == "trace":
            if self.current_trace and self.current_trace.events:
                # Sort events by timestamp
                self.current_trace.events.sort(key=lambda e: e.timestamp)
                self.log.traces.append(self.current_trace)
            self.current_trace = None
            self.in_trace = False
            
        elif name == "event":
            if self.current_event and self.current_trace:
                self.current_trace.events.append(self.current_event)
            self.current_event = None
            self.in_event = False
    
    def _handle_string(self, key: str, value: str):
        if self.in_event and self.current_event:
            if key == "concept:name":
                self.current_event.activity = value
            elif key == "org:resource":
                self.current_event.resource = value
            else:
                self.current_event.attributes[key] = value
                
        elif self.in_trace and self.current_trace:
            if key == "concept:name":
                self.current_trace.case_id = value
            else:
                self.current_trace.attributes[key] = value
                
        else:
            self.log.attributes[key] = value
    
    def _handle_date(self, key: str, value: str):
        try:
            # Handle various datetime formats
            dt = self._parse_datetime(value)
            
            if self.in_event and self.current_event:
                if key == "time:timestamp":
                    self.current_event.timestamp = dt
                else:
                    self.current_event.attributes[key] = dt
                    
            elif self.in_trace and self.current_trace:
                self.current_trace.attributes[key] = dt
                
            else:
                self.log.attributes[key] = dt
                
        except (ValueError, TypeError):
            pass  # Skip invalid dates
    
    def _handle_int(self, key: str, value: str):
        try:
            int_val = int(value)
            if self.in_event and self.current_event:
                self.current_event.attributes[key] = int_val
            elif self.in_trace and self.current_trace:
                self.current_trace.attributes[key] = int_val
            else:
                self.log.attributes[key] = int_val
        except ValueError:
            pass
    
    def _handle_float(self, key: str, value: str):
        try:
            float_val = float(value)
            if self.in_event and self.current_event:
                self.current_event.attributes[key] = float_val
            elif self.in_trace and self.current_trace:
                self.current_trace.attributes[key] = float_val
            else:
                self.log.attributes[key] = float_val
        except ValueError:
            pass
    
    def _handle_boolean(self, key: str, value: str):
        bool_val = value.lower() in ("true", "1", "yes")
        if self.in_event and self.current_event:
            self.current_event.attributes[key] = bool_val
        elif self.in_trace and self.current_trace:
            self.current_trace.attributes[key] = bool_val
        else:
            self.log.attributes[key] = bool_val
    
    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse various datetime formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        
        # Handle timezone offset format variations
        clean_value = value.replace("+00:00", "+0000").replace("-00:00", "-0000")
        if clean_value.endswith("Z"):
            clean_value = clean_value[:-1] + "+0000"
        
        for fmt in formats:
            try:
                return datetime.strptime(clean_value, fmt)
            except ValueError:
                continue
        
        # Last resort: try fromisoformat
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Cannot parse datetime: {value}")


def parse_xes(file_path: str) -> EventLog:
    """
    Parse a XES file and return an EventLog.
    
    Args:
        file_path: Path to the XES file
        
    Returns:
        Parsed EventLog
        
    Raises:
        ValueError: If the file is invalid or cannot be parsed
    """
    handler = XESHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    
    try:
        parser.parse(file_path)
    except xml.sax.SAXException as e:
        raise ValueError(f"Invalid XES file: {e}")
    
    return handler.log


def parse_xes_string(content: str) -> EventLog:
    """
    Parse XES content from a string.
    
    Args:
        content: XES content as string
        
    Returns:
        Parsed EventLog
    """
    from io import StringIO
    
    handler = XESHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    
    try:
        parser.parse(StringIO(content))
    except xml.sax.SAXException as e:
        raise ValueError(f"Invalid XES content: {e}")
    
    return handler.log


def validate_log(log: EventLog) -> List[str]:
    """
    Validate an event log and return any warnings/errors.
    
    Args:
        log: EventLog to validate
        
    Returns:
        List of warning/error messages
    """
    issues = []
    
    if not log.traces:
        issues.append("Event log has no traces")
        return issues
    
    for i, trace in enumerate(log.traces):
        if not trace.case_id:
            issues.append(f"Trace {i} has no case ID")
        
        if not trace.events:
            issues.append(f"Trace '{trace.case_id}' has no events")
            continue
        
        for j, event in enumerate(trace.events):
            if not event.activity:
                issues.append(f"Event {j} in trace '{trace.case_id}' has no activity name")
    
    return issues
