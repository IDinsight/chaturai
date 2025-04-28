from pydantic import BaseModel, EmailStr, Field, field_validator


class InitRegistrationRequest(BaseModel):
    """
    Request model for registration
    """

    email: EmailStr = Field(..., description="Email of the user")
    phone_number: str = Field(..., description="Phone number of the user")

    @field_validator("phone_number")
    def validate_phone_number(cls, v):
        """
        Validate phone number
        """
        if len(v) != 10 or not v.isdigit():
            raise ValueError("Phone number must be 10 digits long")
        return v


class OTPRequest(BaseModel):
    """
    Request model for OTP
    """

    otp: str = Field(..., description="OTP sent to the user")

    @field_validator("otp")
    def validate_otp(cls, v):
        """
        Validate OTP
        """
        if len(v) != 6 or not v.isdigit():
            raise ValueError("OTP must be 6 digits long")
        return v
