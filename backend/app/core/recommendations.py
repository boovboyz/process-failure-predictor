"""
LLM-powered recommendation generation using Claude API.
"""
import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class RecommendationContext:
    """Context for generating recommendations."""
    current_activity: str
    predicted_next: str
    next_probability: float
    predicted_outcome: str
    outcome_probability: float
    remaining_time_hours: float
    risk_factors: List[str]
    prefix_length: int
    total_activities_seen: int


class RecommendationEngine:
    """
    Generates contextual recommendations using Claude API.
    
    Provides actionable suggestions based on prediction results.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = None
        self.enabled = False
        
        if self.api_key and HAS_ANTHROPIC:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.enabled = True
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
    
    def generate_recommendations(
        self,
        context: RecommendationContext,
        domain_hint: Optional[str] = None
    ) -> List[str]:
        """
        Generate actionable recommendations based on prediction context.
        
        Args:
            context: Current prediction context
            domain_hint: Optional domain description (e.g., "loan approval process")
            
        Returns:
            List of 2-3 actionable recommendations
        """
        if not self.enabled:
            return self._generate_fallback_recommendations(context)
        
        try:
            return self._call_claude(context, domain_hint)
        except Exception as e:
            print(f"LLM recommendation failed: {e}")
            return self._generate_fallback_recommendations(context)
    
    def _call_claude(
        self, 
        context: RecommendationContext,
        domain_hint: Optional[str] = None
    ) -> List[str]:
        """Call Claude API to generate recommendations."""
        
        # Build context description
        risk_text = "\n".join(f"  - {r}" for r in context.risk_factors) if context.risk_factors else "  None identified"
        
        domain_context = f"\nProcess Domain: {domain_hint}" if domain_hint else ""
        
        prompt = f"""You are an AI assistant helping process analysts optimize business processes. Based on the current prediction state, provide 2-3 specific, actionable recommendations.

Current Process State:
- Current activity: {context.current_activity}
- Events completed: {context.prefix_length}
- Predicted next activity: {context.predicted_next} ({context.next_probability:.0%} confidence)
- Predicted final outcome: {context.predicted_outcome} ({context.outcome_probability:.0%} probability)
- Estimated remaining time: {context.remaining_time_hours:.1f} hours{domain_context}

Risk Factors:
{risk_text}

Provide exactly 2-3 SHORT, actionable recommendations. Each should be:
- Specific to this prediction (not generic advice)
- Actionable by a process operator right now
- One sentence each, max 15 words

Respond with ONLY a JSON array of strings, no other text. Example:
["Prioritize document review to avoid delay", "Escalate to senior staff if no response in 2 hours"]"""

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Parse JSON response
        try:
            # Handle case where response might have markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            recommendations = json.loads(response_text)
            if isinstance(recommendations, list):
                return [str(r) for r in recommendations[:3]]
        except json.JSONDecodeError:
            # Try to extract recommendations from text
            lines = [l.strip() for l in response_text.split("\n") if l.strip()]
            return lines[:3]
        
        return self._generate_fallback_recommendations(context)
    
    def _generate_fallback_recommendations(
        self, 
        context: RecommendationContext
    ) -> List[str]:
        """Generate rule-based recommendations when LLM is unavailable."""
        recommendations = []
        
        # Low confidence next activity
        if context.next_probability < 0.5:
            recommendations.append(
                f"Low prediction confidence ({context.next_probability:.0%}) - verify process state before proceeding"
            )
        
        # High-risk factors
        if context.risk_factors:
            recommendations.append(
                f"Review identified risks: {context.risk_factors[0]}"
            )
        
        # Long remaining time
        if context.remaining_time_hours > 24:
            recommendations.append(
                f"Extended timeline expected ({context.remaining_time_hours:.0f}h) - consider expediting if urgent"
            )
        
        # Negative outcome predicted
        outcome_lower = context.predicted_outcome.lower()
        if any(neg in outcome_lower for neg in ['reject', 'fail', 'cancel', 'abort']):
            if context.outcome_probability > 0.6:
                recommendations.append(
                    f"High probability of {context.predicted_outcome} - review case for intervention opportunity"
                )
        
        # Default if no specific recommendations
        if not recommendations:
            recommendations.append(
                f"Process tracking normally - predicted {context.predicted_next} next"
            )
        
        return recommendations[:3]


# Global instance
_recommendation_engine: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Get or create the global recommendation engine."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine


def generate_recommendations(
    current_activity: str,
    predicted_next: str,
    next_probability: float,
    predicted_outcome: str,
    outcome_probability: float,
    remaining_time_hours: float,
    risk_factors: List[str],
    prefix_length: int = 1,
    domain_hint: Optional[str] = None
) -> List[str]:
    """
    Convenience function to generate recommendations.
    
    Returns list of 2-3 actionable recommendations.
    """
    engine = get_recommendation_engine()
    
    context = RecommendationContext(
        current_activity=current_activity,
        predicted_next=predicted_next,
        next_probability=next_probability,
        predicted_outcome=predicted_outcome,
        outcome_probability=outcome_probability,
        remaining_time_hours=remaining_time_hours,
        risk_factors=risk_factors,
        prefix_length=prefix_length,
        total_activities_seen=prefix_length
    )
    
    return engine.generate_recommendations(context, domain_hint)
