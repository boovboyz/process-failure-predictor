"""
Trace-aware temporal data splitting.
Prevents data leakage by ensuring no test traces overlap with training traces.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
from app.core.xes_parser import Trace, EventLog


@dataclass
class SplitResult:
    """Result of temporal split."""
    train_traces: List[Trace]
    test_traces: List[Trace]
    excluded_traces: List[Trace]
    cutoff_time: datetime
    
    @property
    def effective_train_ratio(self) -> float:
        """Calculate the effective training ratio (excluding excluded traces)."""
        total = len(self.train_traces) + len(self.test_traces)
        return len(self.train_traces) / total if total > 0 else 0
    
    @property
    def exclusion_rate(self) -> float:
        """Calculate what percentage of traces were excluded."""
        total = len(self.train_traces) + len(self.test_traces) + len(self.excluded_traces)
        return len(self.excluded_traces) / total if total > 0 else 0
    
    def get_warnings(self) -> List[str]:
        """Get warnings about the split."""
        warnings = []
        
        if self.exclusion_rate > 0.15:
            warnings.append(
                f"High exclusion rate: {self.exclusion_rate:.1%} of traces were excluded. "
                "Consider adjusting the split ratio."
            )
        
        if len(self.test_traces) < 100:
            warnings.append(
                f"Small test set: only {len(self.test_traces)} traces. "
                "Model evaluation may be unreliable."
            )
        
        if self.effective_train_ratio < 0.7:
            warnings.append(
                f"Low effective training ratio: {self.effective_train_ratio:.1%}. "
                "Model may have insufficient training data."
            )
        
        return warnings


def temporal_split(log: EventLog, train_ratio: float = 0.9) -> SplitResult:
    """
    Perform trace-aware temporal split on an event log.
    
    This prevents data leakage by:
    1. Sorting traces by END timestamp
    2. Finding a cutoff point
    3. Classifying traces:
       - TRAIN: trace ends before cutoff
       - TEST: trace starts after cutoff
       - EXCLUDE: trace spans the cutoff (to prevent leakage)
    
    Args:
        log: EventLog to split
        train_ratio: Target ratio of training traces (0.0 to 1.0)
        
    Returns:
        SplitResult with train, test, and excluded traces
        
    Raises:
        ValueError: If log is empty or train_ratio is invalid
    """
    if not log.traces:
        raise ValueError("Cannot split empty event log")
    
    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.1 and 0.95")
    
    # Filter out empty traces
    valid_traces = [t for t in log.traces if t.events]
    
    if not valid_traces:
        raise ValueError("No valid traces with events found")
    
    # Sort by completion time (end_time)
    sorted_traces = sorted(valid_traces, key=lambda t: t.end_time)
    
    # Find cutoff index
    cutoff_idx = int(len(sorted_traces) * train_ratio)
    cutoff_idx = max(1, min(cutoff_idx, len(sorted_traces) - 1))
    
    # Get cutoff time
    cutoff_time = sorted_traces[cutoff_idx].end_time
    
    # Classify traces
    train_traces = []
    test_traces = []
    excluded_traces = []
    
    for trace in sorted_traces:
        if trace.end_time < cutoff_time:
            # Trace completed before cutoff -> training
            train_traces.append(trace)
        elif trace.start_time > cutoff_time:
            # Trace started after cutoff -> test
            test_traces.append(trace)
        else:
            # Trace spans the cutoff -> exclude
            excluded_traces.append(trace)
    
    return SplitResult(
        train_traces=train_traces,
        test_traces=test_traces,
        excluded_traces=excluded_traces,
        cutoff_time=cutoff_time
    )


def adaptive_split(log: EventLog, 
                   target_train_ratio: float = 0.9,
                   max_exclusion_rate: float = 0.15) -> Tuple[SplitResult, float]:
    """
    Perform adaptive temporal split that adjusts ratio to minimize exclusions.
    
    Args:
        log: EventLog to split
        target_train_ratio: Initial target training ratio
        max_exclusion_rate: Maximum acceptable exclusion rate
        
    Returns:
        Tuple of (SplitResult, actual_ratio_used)
    """
    # Try initial split
    result = temporal_split(log, target_train_ratio)
    
    if result.exclusion_rate <= max_exclusion_rate:
        return result, target_train_ratio
    
    # Try adjusting ratio to reduce exclusions
    best_result = result
    best_ratio = target_train_ratio
    
    for ratio in [0.85, 0.8, 0.75, 0.7]:
        if ratio >= target_train_ratio:
            continue
            
        result = temporal_split(log, ratio)
        
        if result.exclusion_rate < best_result.exclusion_rate:
            best_result = result
            best_ratio = ratio
            
        if result.exclusion_rate <= max_exclusion_rate:
            break
    
    return best_result, best_ratio
