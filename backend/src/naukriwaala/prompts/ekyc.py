"""This module contains the prompts used for EKYC."""

# Standard Library
from textwrap import dedent

# Third Party Library
from dotmap import DotMap

# Package Library
from naukriwaala.prompts.base import BasePrompts


class EKYCPrompts(BasePrompts):
    """EKYC prompts."""

    system_messages = DotMap(
        {
            **BasePrompts.system_messages,
            "ekyc_agent": dedent("""You are a helpful assistant."""),
        }
    )
    prompts = DotMap(
        {
            **BasePrompts.prompts,
            "ekyc_agent": dedent("""Please help me."""),
        }
    )
