"""This module contains logfire metrics.

NB: Metric names must be <= 63 characters, lower cased, and snake cased!
"""

# Third Party Library
import logfire

# Counters.
litellm_price_counter = logfire.metric_counter(
    "litellm_price",
    description="Total price of all Litellm API calls.",
    unit="USD",
)

validator_call_failure_counter = logfire.metric_counter(
    "validator_failures",
    description="Number of times an LLM validator call fails.",
    unit="1",
)

# Histograms.
ekyc_agent_hist = logfire.metric_histogram(
    "duration_of_ekyc_agent_calls",
    description="Duration of calls to the EKYC agent.",
    unit="s",
)
