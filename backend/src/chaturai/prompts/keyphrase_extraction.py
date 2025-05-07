"""This module contains the prompts for keyphrase extraction."""

# Standard Library
from textwrap import dedent

# Third Party Library
from dotmap import DotMap

# Package Library
from chaturai.prompts.base import BasePrompts


class KeyphraseExtractionPrompts(BasePrompts):
    """Keyphrase extraction prompts."""

    system_messages = DotMap(
        {
            **BasePrompts.system_messages,
            **{
                "extract_relevant_keyphrases": dedent(
                    """You are an expert in extracting relevant keyphrases and keywords
from medical queries. You will be provided with a query from a medical professional.

Your task is to extract all relevant keywords that pertain to observations, signs,
and/or symptoms observed by the medical professional. Do not include any irrelevant
keywords or phrases. Pay attention to the context and meaning of the query to ensure
that you capture only the most pertinent keyphrases.
                    """
                ),
            },
        }
    )
    prompts = DotMap(
        {
            **BasePrompts.prompts,
            **{
                "extract_relevant_keyphrases": dedent(
                    """You are given the following query from a medical professional:
[DOCUMENT]

Based on the query, extract all relevant keywords that pertain to observations, signs,
and/or symptoms observed by the medical professional.
Use the following format separated by commas:
<keywords>
                    """
                ),
            },
        }
    )
