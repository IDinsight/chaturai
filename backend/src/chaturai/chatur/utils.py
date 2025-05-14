"""This module contains utilities for chatur."""

# Standard Library
import asyncio
import random

from contextlib import suppress

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


async def fill_email(*, email: str, page: Page, url: str) -> None:
    """Fill the email field in the form.

    Parameters
    ----------
    email
        The email to fill in the field.
    page
        The Playwright page object.
    url
        The URL to navigate to.
    """

    await page.goto(url, wait_until="domcontentloaded")
    await select_login_radio(page=page)
    await page.fill("input[placeholder='Enter Your Email ID']", email)


async def fill_otp(*, otp: int | str, page: Page) -> None:
    """Fill the OTP field in the form.

    Parameters
    ----------
    otp
        The OTP to fill in the field.
    page
        The Playwright page object.
    """

    otp_field_selector = "input[placeholder='Enter 6 Digit OTP']"
    await page.wait_for_selector(otp_field_selector, state="visible")
    await page.fill(otp_field_selector, str(otp))


async def fill_roll_number(*, page: Page, roll_number: str, url: str) -> None:
    """Fill the roll number field in the form.

    Parameters
    ----------
    page
        The Playwright page object.
    roll_number
        The roll number to fill in the field.
    url
        The URL to navigate to.
    """

    await page.goto(url, wait_until="domcontentloaded")
    await select_register_radio(page=page)
    await page.locator("label:has-text('ITI Student')").click(force=True)
    roll_selector = "input[placeholder*='Roll']"
    await page.wait_for_selector(roll_selector, state="visible")
    await page.fill(roll_selector, roll_number)


def preprocess_captcha_image(*, image_buffer: bytes) -> MatLike:
    """Preprocess the captcha image to improve OCR accuracy.

    NB: For debugging use

        import opencv_jupyter_ui as jcv2
        jcv2.imshow("label", img, colorspace="gray")

    Parameters
    ----------
    image_buffer
        The image buffer containing the captcha image.

    Returns
    -------
    MatLike
        The preprocessed image ready for OCR.
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


async def solve_and_fill_captcha(*, page: Page) -> None:
    """Solve the CAPTCHA and fill it in the form.

    Parameters
    ----------
    page
        Playwright page object.
    """

    canvas = page.locator("canvas.captcha-canvas")
    await canvas.wait_for(state="visible")
    await canvas.scroll_into_view_if_needed()
    captcha_bytes: bytes = await canvas.screenshot()

    captcha_text = await solve_captcha(image_buffer=captcha_bytes)

    await page.fill('input[placeholder="Enter CAPTCHA"]', captcha_text)


async def solve_and_submit_captcha_with_retries(
    *,
    api_url: str,
    button_name: str,
    max_retries: int = 3,
    page: Page,
    timeout: int = 5000,
) -> SubmitButtonResponse | None:
    """Solve the CAPTCHA and submit the form with retries.

    Returns
    -------
    api_url
        The URL of the API to wait for.
    button_name
        The name of the button to click.
    max_retries
        Maximum number of retries for CAPTCHA solving.
    page
        Playwright page object.
    timeout
        The maximum time to wait for the API response (in milliseconds).

    Returns
    -------
    SubmitButtonResponse | None
        The response from the API or toast message.

    Raises
    ------
    RuntimeError
        If the API response indicates an error.
    """

    for attempt in range(max_retries):
        await page.wait_for_timeout(random.randint(1500, 3000))
        await solve_and_fill_captcha(page=page)

        # TODO: Remove this line
        logger.debug(
            f"Remaining retries: {max_retries - attempt}. Press Enter to continue.. "
        )
        await asyncio.get_event_loop().run_in_executor(None, input)

        response = await submit_and_capture_api_response(
            api_url=api_url, button_name=button_name, page=page, timeout=timeout
        )

        if "captcha" in response.message.lower():
            continue
        if response.is_error:
            raise RuntimeError(f"Error: {response.message}")
        return response
    return None


async def solve_captcha(*, image_buffer: bytes) -> str:
    """Extract captcha text from image.

    Parameters
    ----------
    image_buffer
        The image buffer containing the captcha image.

    Returns
    -------
    str
        The extracted captcha text.
    """

    preprocessed_img = preprocess_captcha_image(image_buffer=image_buffer)
    captcha_text = pytesseract.image_to_string(preprocessed_img)
    return captcha_text.strip()


async def submit_and_capture_api_response(
    *, api_url: str, button_name: str, page: Page, timeout: int = 5000
) -> SubmitButtonResponse:
    """Click a button and wait for the page to navigate.

    Parameters
    ----------
    api_url
        The URL of the API to wait for.
    button_name
        The name of the button to click.
    page
        The Playwright page object.
    timeout
        The maximum time to wait for the API response (in milliseconds).

    Returns
    -------
    SubmitButtonResponse
        The response from the API or toast message.
    """

    async def _wait_api() -> SubmitButtonResponse | None:
        """Wait for API response and parse the response.

        Returns
        -------
        SubmitButtonResponse | None
            The response from the API or None if no response was received.
        """

        with suppress(PlaywrightTimeoutError):
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
                is_success = json_response.get("status", "success") == "success"
                is_error = json_response.get("status", "success") == "error"
            else:
                message_text = ""
                is_success = response.ok
                is_error = not response.ok

            return SubmitButtonResponse(
                api_response=json_response,
                is_error=is_error,
                is_success=is_success,
                message=message_text,
                source="api",
            )
        return None

    async def _wait_toast() -> SubmitButtonResponse | None:
        """Wait for a toast message to appear and parse the response.

        Returns
        -------
        SubmitButtonResponse | None
            The response from the toast message or None if no toast appeared.
        """

        with suppress(PlaywrightTimeoutError):
            # Wait for any toast message to appear.
            toast_message = await page.wait_for_selector(
                "#toast-container .toast-message", state="attached", timeout=timeout
            )
            # Check if it's an error or success toast.
            is_error = await page.locator("#toast-container .toast-error").is_visible()
            is_success = await page.locator(
                "#toast-container .toast-success"
            ).is_visible()

            # Get the message text.
            assert toast_message is not None
            message_text = await toast_message.text_content()
            assert message_text is not None
            message_text = message_text.strip()

            return SubmitButtonResponse(
                api_response=None,
                is_error=is_error,
                is_success=is_success,
                message=message_text,
                source="toast",
            )
        return None

    # Click the button.
    await page.get_by_role("button", name=button_name, exact=True).click()

    # Start both waiters.
    toast_task = asyncio.create_task(_wait_toast())
    api_task = asyncio.create_task(_wait_api())

    # Wait for either the toast or API response to happen first.
    done, pending = await asyncio.wait(
        [toast_task, api_task],
        return_when=asyncio.FIRST_COMPLETED,
        timeout=timeout / 1000,  # Convert to seconds
    )

    # Cancel any pending tasks.
    for task in pending:
        task.cancel()

    # If we got a toast message first, check if it's a captcha error.
    if toast_task in done:
        toast_result = await toast_task
        if toast_result:
            # If it's a captcha error, the API call won't happen.
            if "Captcha" in toast_result.message:
                return toast_result

    # If we got an API response.
    if api_task in done:
        api_result = await api_task
        if api_result:
            return api_result

    # If we made it here, check if either task completed.
    for task in done:
        try:
            result = await task
            if result:
                return result
        except Exception:  # pylint: disable=W0718
            pass

    # If nothing happened, return a timeout error.
    return SubmitButtonResponse(
        api_response=None,
        is_error=True,
        is_success=False,
        message="No toast or API response detected.",
        source="timeout",
    )
