"""
Prefix generation for training next-event prediction models.
"""
from dataclasses import dataclass
from typing import List, Optional
from app.core.xes_parser import Event, Trace


@dataclass
class PrefixSample:
    """A training sample generated from a trace prefix."""
    case_id: str
    prefix_events: List[Event]
    target_activity: str           # For next-activity prediction
    target_outcome: Optional[str]  # Final activity of the trace (for outcome prediction)
    remaining_time: float          # Seconds until trace completion (for time prediction)
    
    @property
    def prefix_length(self) -> int:
        """Get the length of the prefix."""
        return len(self.prefix_events)
    
    @property
    def last_activity(self) -> str:
        """Get the last activity in the prefix."""
        return self.prefix_events[-1].activity if self.prefix_events else ""


def generate_prefixes(traces: List[Trace], 
                      min_prefix_len: int = 1,
                      max_prefix_len: Optional[int] = None) -> List[PrefixSample]:
    """
    Generate all valid prefixes from a list of traces.
    
    For each trace, generates prefixes of length 1 to n-1 (where n is trace length).
    Each prefix has a target which is the next activity.
    
    Args:
        traces: List of traces to generate prefixes from
        min_prefix_len: Minimum prefix length (default 1)
        max_prefix_len: Maximum prefix length (None for no limit)
        
    Returns:
        List of PrefixSample objects
    """
    samples = []
    
    for trace in traces:
        if len(trace.events) < 2:
            # Need at least 2 events to create a prefix with a target
            continue
        
        # Determine outcome (last activity of the trace)
        outcome = trace.events[-1].activity
        
        # Calculate max length for this trace
        trace_max = len(trace.events) - 1  # Leave at least 1 event for target
        if max_prefix_len is not None:
            trace_max = min(trace_max, max_prefix_len)
        
        # Generate prefixes
        for i in range(min_prefix_len, trace_max + 1):
            prefix_events = trace.events[:i]
            target_event = trace.events[i]
            
            # Calculate remaining time from end of prefix to end of trace
            remaining_time = (trace.end_time - prefix_events[-1].timestamp).total_seconds()
            
            samples.append(PrefixSample(
                case_id=trace.case_id,
                prefix_events=prefix_events,
                target_activity=target_event.activity,
                target_outcome=outcome,
                remaining_time=remaining_time
            ))
    
    return samples


def generate_prefixes_at_percentage(traces: List[Trace], 
                                    percentages: List[int] = [25, 50, 75]) -> dict:
    """
    Generate prefixes at specific completion percentages.
    Useful for early detection analysis.
    
    Args:
        traces: List of traces
        percentages: List of percentages (e.g., [25, 50, 75])
        
    Returns:
        Dict mapping percentage to list of PrefixSample objects
    """
    result = {pct: [] for pct in percentages}
    
    for trace in traces:
        if len(trace.events) < 2:
            continue
        
        outcome = trace.events[-1].activity
        
        for pct in percentages:
            prefix_len = max(1, int(len(trace.events) * pct / 100))
            
            # Ensure we have a target
            if prefix_len >= len(trace.events):
                continue
            
            prefix_events = trace.events[:prefix_len]
            target_event = trace.events[prefix_len]
            remaining_time = (trace.end_time - prefix_events[-1].timestamp).total_seconds()
            
            result[pct].append(PrefixSample(
                case_id=trace.case_id,
                prefix_events=prefix_events,
                target_activity=target_event.activity,
                target_outcome=outcome,
                remaining_time=remaining_time
            ))
    
    return result


def get_prefix_statistics(samples: List[PrefixSample]) -> dict:
    """
    Get statistics about generated prefixes.
    
    Args:
        samples: List of PrefixSample objects
        
    Returns:
        Dictionary with statistics
    """
    if not samples:
        return {
            "total_samples": 0,
            "unique_cases": 0,
            "avg_prefix_length": 0,
            "min_prefix_length": 0,
            "max_prefix_length": 0,
            "unique_targets": 0,
            "unique_outcomes": 0,
        }
    
    lengths = [s.prefix_length for s in samples]
    
    return {
        "total_samples": len(samples),
        "unique_cases": len(set(s.case_id for s in samples)),
        "avg_prefix_length": sum(lengths) / len(lengths),
        "min_prefix_length": min(lengths),
        "max_prefix_length": max(lengths),
        "unique_targets": len(set(s.target_activity for s in samples)),
        "unique_outcomes": len(set(s.target_outcome for s in samples if s.target_outcome)),
    }
