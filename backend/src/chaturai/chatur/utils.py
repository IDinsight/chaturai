"""This module contains utilities for chatur."""

# Standard Library
import asyncio
import random

from typing import Any

# Third Party Library
import cv2
import numpy as np
import pytesseract

from cv2.typing import MatLike
from loguru import logger
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

# Package Library
from chaturai.chatur.schemas import SubmitButtonResponse


async def select_login_radio(*, page: Page) -> None:
    """Click the “Login as a candidate” radio button on `apprenticeshipIndia.gov.in`.

    The process is as follows:

    1. Get by accessible role + name.
    2. Get by label text.
    3. Get by direct CSS selector.

    Parameters
    ----------
    page
        A Playwright Page already at /candidate-login.

    Raises
    ------
    RuntimeError
        If the radio button is not checked.
    """

    # 1.
    try:
        await page.get_by_role("radio", name="Login as a candidate").check()
        return
    except PlaywrightError:
        pass

    # 2.
    try:
        await page.get_by_label("Login as a candidate").check()
        return
    except PlaywrightError:
        pass

    # 3.
    try:
        await page.check("input#disyesCan")
        return
    except PlaywrightError as e:
        raise RuntimeError(
            f"Failed to select the ‘Login as a candidate’ radio: {e}"
        ) from e


async def select_register_radio(*, page: Page, timeout: int = 10_000) -> None:
    """Click the “Register as a candidate” radio button on `apprenticeshipIndia.gov.in`
    and waits for the register form page to render.

    The process is as follows:

    1. Click the label so we don't race re-renders.
    2. Wait for the unique register-form field to render.
    3. Sanity-check that the radio button ended up checked.

    Parameters
    ----------
    page
        A Playwright Page already at /candidate-login.
    timeout
        Maximum time (in ms) to wait for the register form to appear.

    Raises
    ------
    RuntimeError
        If the register form does not appear within the timeout or if the radio button
        is not checked.
    """
    # 1.
    await page.locator("label:has-text('Register as a candidate')").click(force=True)

    # 2.
    try:
        await page.locator("input[placeholder='Enter your mobile number']").wait_for(
            state="visible", timeout=timeout
        )
    except PlaywrightTimeoutError as e:
        raise RuntimeError(
            "Timed out waiting for the register form after clicking the radio."
        ) from e

    # 3.
    if not await page.locator("input#disnoCan").is_checked():
        raise RuntimeError("Radio ‘Register as a candidate’ was not checked.")


async def solve_and_submit_captcha_with_retries(
    page: Page,
    api_url: str,
    button_name: str,
    timeout: int = 5000,
    max_retries: int = 3,
) -> SubmitButtonResponse:
    """Solve the CAPTCHA and submit the form with retries."""
    try_count = 0
    wait_ms = random.randint(1500, 3000)
    while try_count < max_retries:
        await page.wait_for_timeout(wait_ms)

        await solve_and_fill_captcha(page=page)

        # TODO: remove this line
        logger.debug(
            f"Remaining retries: {max_retries-try_count}. Press Enter to continue.. "
        )
        await asyncio.get_event_loop().run_in_executor(None, input)

        response = await submit_and_capture_api_response(
            page=page,
            api_url=api_url,
            button_name=button_name,
            timeout=timeout,
        )

        if "captcha" in response.message.lower():
            try_count += 1
            continue
        elif response.is_error:
            raise RuntimeError(f"Error: {response.message}")
        return response


async def solve_and_fill_captcha(page: Page) -> None:
    """Solve the CAPTCHA and fill it in the form."""
    canvas = page.locator("canvas.captcha-canvas")
    await canvas.wait_for(state="visible")
    await canvas.scroll_into_view_if_needed()
    captcha_bytes: bytes = await canvas.screenshot()

    captcha_text = await solve_captcha(captcha_bytes)

    await page.fill('input[placeholder="Enter CAPTCHA"]', captcha_text)


async def solve_captcha(image_buffer: bytes) -> str:
    """Extract captcha text from image."""
    preprocessed_img = preprocess_captcha_image(image_buffer)
    captcha_text = pytesseract.image_to_string(preprocessed_img)
    return captcha_text.strip()


def preprocess_captcha_image(image_buffer: bytes) -> MatLike:
    """
    Preprocess the captcha image to improve OCR accuracy.

    NB: For debugging use

        import opencv_jupyter_ui as jcv2
        jcv2.imshow("label", img, colorspace="gray")

    """
    img_array = np.asarray(bytearray(image_buffer), dtype="uint8")

    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_contrast = cv2.convertScaleAbs(img_gray, alpha=2, beta=0)

    threshold_value = 127
    _, img_thrshold = cv2.threshold(
        img_contrast, threshold_value, 255, cv2.THRESH_BINARY
    )
    return img_thrshold


async def submit_and_capture_api_response(
    page: Page, button_name: str, api_url: str, timeout: int = 5000
) -> SubmitButtonResponse:
    """Click a button and wait for the page to navigate."""

    async def check_for_toast():
        """Wait for a toast message to appear and parse the response"""
        try:
            # Wait for any toast message to appear
            toast_message = await page.wait_for_selector(
                "#toast-container .toast-message", state="attached", timeout=timeout
            )

            # Check if it's an error or success toast
            is_error = await page.locator("#toast-container .toast-error").is_visible()
            is_success = await page.locator(
                "#toast-container .toast-success"
            ).is_visible()

            # Get the message text
            message_text = await toast_message.text_content()
            message_text = message_text.strip()

            return SubmitButtonResponse(
                source="toast",
                message=message_text,
                is_error=is_error,
                is_success=is_success,
                api_response=None,
            )
        except PlaywrightTimeoutError:
            # No toast appeared within the timeout
            return None

    # Function to wait for API response
    async def wait_for_api():
        """Wait for API response and parse the response"""
        try:
            async with page.expect_response(
                lambda response: api_url in response.url, timeout=timeout
            ) as response_info:
                pass  # We'll click the button outside this context

            response = await response_info.value

            json_response = await response.json()

            if "errors" in json_response:
                # json_response["errors"] is keyed by fields with error message values.
                message_text = " ".join(json_response["errors"].values())
                is_error = True
                is_success = False
            elif "message" in json_response:
                message_text = json_response["message"]
                if "status" in json_response:
                    is_success = json_response["status"] == "success"
                    is_error = json_response["status"] == "error"
            else:
                message_text = ""
                is_success = response.ok
                is_error = not response.ok

            return SubmitButtonResponse(
                source="api",
                message=message_text,
                is_error=is_error,
                is_success=is_success,
                api_response=json_response,
            )
        except PlaywrightTimeoutError:
            return None

    # Click the button
    await page.get_by_role("button", name=button_name, exact=True).click()

    # Start both waiters
    toast_task = asyncio.create_task(check_for_toast())
    api_task = asyncio.create_task(wait_for_api())

    # Wait for either the toast or API response to happen first
    done, pending = await asyncio.wait(
        [toast_task, api_task],
        return_when=asyncio.FIRST_COMPLETED,
        timeout=timeout / 1000,  # Convert to seconds
    )

    # Cancel any pending tasks
    for task in pending:
        task.cancel()

    # If we got a toast message first, check if it's a captcha error
    if toast_task in done:
        toast_result = await toast_task
        if toast_result:
            # If it's a captcha error, the API call won't happen
            if "Captcha" in toast_result.message:
                return toast_result

    # If we got an API response
    if api_task in done:
        api_result = await api_task
        if api_result:
            return api_result

    # If we made it here, check if either task completed
    for task in done:
        try:
            result = await task
            if result:
                return result
        except Exception:
            pass

    # If nothing happened, return a timeout error
    return SubmitButtonResponse(
        source="timeout",
        message="No toast or API response detected",
        is_error=True,
        is_success=False,
        api_response=None,
    )


# TODO: update return values and type after implementing browser session management
async def submit_register_otp(page: Page, otp: str) -> Any:
    """Enter the OTP in the OTP input field.

    Parameters
    ----------
    page
        The Playwright page object.
    otp
        The OTP to enter.
    """
    otp_field = page.locator("input[placeholder='Enter 6 Digit OTP']")
    await otp_field.click()
    await otp_field.fill(otp)
    await otp_field.scroll_into_view_if_needed()

    response_json = await submit_and_capture_api_response(
        page=page,
        api_url="https://api.apprenticeshipindia.gov.in/auth/register-otp",
        button_name="Submit",
    )

    response = await response_json
    if "status" in response and response["status"] == "success":
        data = response.get("data")
        candidate_info = data.get("candidate")
        naps_id = candidate_info["code"]
        activation_link_expiry = candidate_info["activation_link_expiry_date"]

        return naps_id, activation_link_expiry

    assert "errors" in response
    error_messages = response["errors"].values()
    message = " ".join(error_messages)
    raise RuntimeError(f"Error: {message}")
