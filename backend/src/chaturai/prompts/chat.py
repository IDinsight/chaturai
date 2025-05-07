"""This module contains the prompts used for chat."""

# Third Party Library
from dotmap import DotMap

# Package Library
from chaturai.prompts.base import BasePrompts


class ChatPrompts(BasePrompts):
    """Chat prompts."""

    system_messages = DotMap(
        {
            **BasePrompts.system_messages,
        }
    )
    prompts = DotMap(
        {
            **BasePrompts.prompts,
        }
    )
