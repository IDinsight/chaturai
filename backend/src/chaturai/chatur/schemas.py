"""This module contains Pydantic models for chatur."""

# Future Library
from __future__ import annotations

# Standard Library
import re

from enum import Enum
from typing import Any, Literal, Optional

# Third Party Library
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    ValidationInfo,
    field_validator,
)


class NextChatAction(str, Enum):
    """Enum for the next action to take in the chat flow. For now, this will be used to
    determine the branches in Turn.io
    """

    GO_TO_AAQ = "GO_TO_AAQ"  # Pass query to AAQ
    GO_TO_HELPDESK = "GO_TO_HELPDESK"  # Pass query to Helpdesk
    GO_TO_MENU = "GO_TO_MENU"  # Go to menu
    REQUEST_OTP = "REQUEST_OTP"  # Request OTP from the user
    REQUEST_USER_QUERY = "REQUEST_USER_QUERY"  # Request user query


# Queries.
class BaseQuery(BaseModel):
    """Pydantic model for base queries, serving as a discriminator."""

    email: EmailStr
    otp: Optional[str] = Field(None, description="6-digit OTP sent to the student")
    user_query: Optional[str] = Field(None, description="The student's message")
    user_id: str

    model_config = ConfigDict(from_attributes=True)

    def model_post_init(self, context: Any) -> None:
        """Validate the OTP/user query.

        Raises
        ------
        ValueError
            If neither OTP nor user query is provided.
            If the OTP is not exactly 6 digits.
        """

        if not (self.otp or self.user_query):
            raise ValueError("Either OTP or user_query must be provided")
        if self.otp and not (self.otp.isdigit() and len(self.otp) == 6):
            raise ValueError(f"OTP must be exactly 6 digits: {self.otp}")


class LoginStudentQuery(BaseQuery):
    """Pydantic model for validating student login queries."""


class ProfileCompletionQuery(BaseQuery):
    """Pydantic model for validating profile completion queries."""

    father_name: str
    mother_name: str


class RegisterStudentQuery(BaseQuery):
    """Pydantic model for validating student registration queries."""

    is_iti_student: bool = False
    is_new_student: bool
    mobile_number: str = Field(
        ..., description="10-digit Indian mobile number; accepts optional +91 prefix"
    )
    roll_number: Optional[str] = Field(
        None,
        description="At least 13-digit roll number; required if is_iti_student=True",
    )

    @field_validator("mobile_number")
    @classmethod
    def validate_mobile_number(cls, v: str) -> str:
        """Validate the mobile number as follows:

        1. Remove leading plus sign if present.
        2. If it starts with country code '91' and is longer than 10 digits, strip it off.
        3. Ensure the number is exactly 10 digits.
        4. Ensure the number starts with 6, 7, 8, or 9.

        Parameters
        ----------
        v
            The mobile number to validate.

        Returns
        -------
        str
            The validated mobile number.

        Raises
        ------
        ValueError
            If the mobile number is not valid.
        """

        # 1.
        if v.startswith("+"):
            v = v[1:]

        # 2.
        if v.startswith("91") and len(v) > 10:
            v = v[2:]

        # 3.
        if not (v.isdigit() and len(v) == 10):
            raise ValueError(
                f"Mobile number must be exactly 10 digits after removing any '+91' "
                f"prefix: {v}"
            )

        # 4.
        if not re.match(r"^[6-9]\d{9}$", v):
            raise ValueError(f"Indian mobile number must start with 6, 7, 8, or 9: {v}")

        return v

    @field_validator("roll_number", mode="after")
    @classmethod
    def validate_roll_and_iti(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Validate roll number and ITI student flag.

        Parameters
        ----------
        v
            The roll number.
        info
            Additional information about the field being validated.

        Returns
        -------
        Optional[str]
            The validated roll number.

        Raises
        ------
        ValueError
            If the roll number is not valid or if it does not match the ITI student
            flag.
        """

        iti = info.data["is_iti_student"]

        if v is not None and len(v) != 13:
            raise ValueError(f"Roll number must be exactly 13 digits: {v}")
        if iti and v is None:
            raise ValueError("roll_number is required when is_iti_student=True")
        if not iti and v is not None:
            raise ValueError("roll_number must be omitted when is_iti_student=False")

        return v


ChaturQueryUnion = ProfileCompletionQuery | RegisterStudentQuery | LoginStudentQuery


# Graph run results.
class ChaturFlowResults(BaseModel):
    """Pydantic model for validating chatur flow results."""

    explanation_for_student_input: str
    last_assistant_call: str | None
    last_graph_run_results: Optional[Any] = None
    next_chat_action: NextChatAction = NextChatAction.REQUEST_USER_QUERY
    require_student_input: bool
    summary_for_student: str
    user_id: str

    model_config = ConfigDict(from_attributes=True)


class PageResults(BaseModel):
    """Pydantic model for validating page results."""

    next_chat_action: NextChatAction = NextChatAction.REQUEST_USER_QUERY
    session_id: int | str
    summary_of_page_results: str

    model_config = ConfigDict(from_attributes=True)


class LoginStudentResults(PageResults):
    """Pydantic model for validating login student results."""


class ProfileCompletionResults(PageResults):
    """Pydantic model for validating profile completion results."""


class RegisterStudentResults(PageResults):
    """Pydantic model for validating register student results."""


# Adding these for now. Might remove later.
class RegistrationCompleteResults(PageResults):
    """Pydantic model for validating register activation link send results."""

    activation_link_expiry: str
    naps_id: str


class SubmitButtonResponse(BaseModel):
    """Pydantic model for validating submit button responses."""

    api_response: Optional[dict[str, Any]]
    is_error: bool
    is_success: bool
    message: str
    source: Literal["api", "toast", "timeout"]

    model_config = ConfigDict(from_attributes=True)
