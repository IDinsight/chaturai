"""This module contains the base class for all prompts."""

# Standard Library
from textwrap import dedent

# Third Party Library
from dotmap import DotMap


class BasePrompts:
    """Base prompts. This class mainly serves to contain generally useful prompts and
    type hints for all prompt classes. If a prompt is NOT listed here, then `mypy` will
    catch it.

    PLEASE LIST THINGS IN ALPHABETICAL ORDER HERE!
    """

    _react_answer = "Reasonable Answer:"
    system_messages = DotMap(
        {
            "default": "You are a helpful assistant.",
        }
    )
    prompts = DotMap(
        {
            "agent_scratchpad": "\n\nYour previous findings are as follows (but I haven't seen any of it! I only see what you return as the Final Answer):\n{agent_scratchpad}\nThought: ",
            "cot": "Let's think step by step.",
            "error_correction": dedent(
                """Your last message resulted in the following errors:

⚠️ **Error during response validation**

{error_info_str}

Please correct your response and try again.
                """
            ),
            "goal": "Your goal is to do exactly as you are told.",
            "summarize_chat_history": "Summarize the following conversation to be used as a prompt for continuing the conversation later:\n\n{conversation}",
            "yes_no": "\n\nYour response MUST be either 'Yes' or 'No' and nothing else.",
            "yes_no_with_explanation": "\n\nYour response MUST be either 'Yes' or 'No', followed by an explanation for your reason.",
            "yes_no_with_explanation_and_examples": "\n\nYour response MUST be either 'Yes' or 'No', followed by an explanation for your reason AND a list of the most relevant and descriptive examples to go along with your response.",
            "zero_shot_react": dedent(
                """Answer the following questions as best you can. You have access to the following list of tools:

                Tools List:\n\n{tools_list}

                ALWAYS use the following format:

                Question: The original input Question you must answer
                Thought: You should always think about the steps involved in answering the original input Question
                Action: The Action to take, this can involve using a provided data source and/or using a provided tool
                Action Input: The input(s) to the Action, there may not always be Action Inputs
                Observation: The result of the Action, your Observations should be complete
                ... (this Thought/Action/Action Input/Observation sequence can repeat many times until you have a Reasonable Answer to the original input Question)
                Thought: Based on all the previous information, I now have a Reasonable Answer
                Reasonable Answer: A reasonable answer to the original input Question

                Begin! Reminder to always use the exact characters `Reasonable Answer:` when responding.
                """
            ),
        }
    )
