"""This module contains logfire metrics.

NB: Metric names must be <= 63 characters, lower cased, and snake cased!
"""

# Third Party Library
import logfire

# Counters.
assistants_counter = logfire.metric_counter(
    "assistant_calls",
    description="Number of times all assistants are called.",
    unit="1",
)

litellm_price_counter = logfire.metric_counter(
    "litellm_price",
    description="Total price of all Litellm API calls.",
    unit="USD",
)

student_counter = logfire.metric_counter(
    "student_calls",
    description="Number of times the student is called.",
    unit="1",
)

validator_call_failure_counter = logfire.metric_counter(
    "validator_failures",
    description="Number of times an LLM validator call fails.",
    unit="1",
)

# Histograms.
chatur_agent_hist = logfire.metric_histogram(
    "duration_of_chatur_agent_calls",
    description="Duration of calls to Chatur agent.",
    unit="s",
)

login_agent_hist = logfire.metric_histogram(
    "duration_of_login_agent_calls",
    description="Duration of calls to the login agent.",
    unit="s",
)

profile_completion_agent_hist = logfire.metric_histogram(
    "duration_of_profile_completion_agent_calls",
    description="Duration of calls to the profile completion agent.",
    unit="s",
)

register_student_agent_hist = logfire.metric_histogram(
    "duration_of_register_student_agent_calls",
    description="Duration of calls to the register student agent.",
    unit="s",
)
