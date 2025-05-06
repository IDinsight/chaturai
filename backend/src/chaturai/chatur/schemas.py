"""This module contains Pydantic models for chatur."""

# Standard Library
import re

from typing import Annotated, Any, Literal, Optional

# Third Party Library
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    ValidationInfo,
    field_validator,
)


# Queries.
class BaseQuery(BaseModel):
    """Pydantic model for base queries, serving as a discriminator."""

    type: str
    user_query: str
    user_id: str

    model_config = ConfigDict(from_attributes=True)


class LoginStudentQuery(BaseQuery):
    """Pydantic model for validating student login queries."""

    email: EmailStr
    type: Literal["login"]


class RegisterStudentQuery(BaseQuery):
    """Pydantic model for validating student registration queries."""

    email: EmailStr
    is_iti_student: bool = False
    is_new_student: bool
    mobile_number: str = Field(
        ..., description="10-digit Indian mobile number; accepts optional +91 prefix"
    )
    roll_number: Optional[str] = Field(
        None,
        description="At least 13-digit roll number; required if is_iti_student=True",
    )
    type: Literal["registration"]

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


class RegisterStudentOTPQuery(BaseQuery):
    """Pydantic model for validating student registration OTP queries."""

    otp: str = Field(
        ..., description="6-digit OTP sent to the registered mobile number"
    )
    type: Literal["registration_otp"]

    @field_validator("otp")
    @classmethod
    def validate_otp(cls, v: str) -> str:
        """Validate the OTP.

        Parameters
        ----------
        v
            The OTP to validate.

        Returns
        -------
        str
            The validated OTP.

        Raises
        ------
        ValueError
            If the OTP is not exactly 6 digits.
        """

        if not (v.isdigit() and len(v) == 6):
            raise ValueError(f"OTP must be exactly 6 digits: {v}")

        return v


ChaturQueryUnion = Annotated[
    RegisterStudentQuery | LoginStudentQuery | RegisterStudentOTPQuery,
   
    Field(discriminator="type"),,
]


# Graph run results.
class ChaturFlowResults(BaseModel):
    """Pydantic model for validating chatur flow results."""

    explanation_for_student_input: str
    last_assistant_call: str | None
    last_graph_run_results: Optional[Any] = None
    require_student_input: bool
    summary_for_student: str
    user_id: str

    model_config = ConfigDict(from_attributes=True)


class PageResults(BaseModel):
    """Pydantic model for validating page results."""

    summary_of_page_results: str

    model_config = ConfigDict(from_attributes=True)


class LoginStudentResults(PageResults):
    """Pydantic model for validating login student results."""


class RegisterStudentResults(PageResults):
    """Pydantic model for validating register student results."""
