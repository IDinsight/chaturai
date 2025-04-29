"""This module contains Pydantic models for E-KYC."""

# Standard Library
import re

from typing import Any, Optional

# Third Party Library
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    ValidationInfo,
    field_validator,
)


class EKYCQuery(BaseModel):
    """Pydantic model for validating E-KYC queries."""

    email: EmailStr
    is_iti_student: bool = False
    is_new_student: bool
    mobile_number: str = Field(
        ..., description="10-digit Indian mobile number; accepts optional +91 prefix"
    )
    roll_number: Optional[int] = Field(
        None,
        description="At least 13-digit roll number; required if is_iti_student=True",
    )
    user_id: str

    model_config = ConfigDict(from_attributes=True)

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
        cls, v: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        """Validate roll number and ITI student flag.

        Parameters
        ----------
        v
            The roll number.
        info
            Additional information about the field being validated.

        Returns
        -------
        Optional[int]
            The validated roll number.

        Raises
        ------
        ValueError
            If the roll number is not valid or if it does not match the ITI student
            flag.
        """

        iti = info.data["is_iti_student"]

        if v is not None and v < 10**12:
            raise ValueError(f"Roll number must be at least 13 digits: {v}")
        if iti and v is None:
            raise ValueError("roll_number is required when is_iti_student=True")
        if not iti and v is not None:
            raise ValueError("roll_number must be omitted when is_iti_student=False")

        return v


class EKYCResults(EKYCQuery):
    """Pydantic model for validating E-KYC results."""

    last_graph_run_results: Optional[list[Any]] = None
