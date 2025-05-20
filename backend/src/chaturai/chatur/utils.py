"""This module contains utilities for chatur."""

# Standard Library
import asyncio
import json
import random
import re

from contextlib import suppress
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

# Third Party Library
import cv2
import numpy as np
import pytesseract

from cv2.typing import MatLike
from loguru import logger
from playwright.async_api import Browser
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from redis import asyncio as aioredis

# Package Library
from chaturai.chatur.schemas import ChaturFlowResults, SubmitButtonResponse
from chaturai.config import Settings
from chaturai.graphs.utils import save_browser_state
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.schemas import ValidatorCall
from chaturai.utils.browser import BrowserSessionStore
from chaturai.utils.litellm_ import get_acompletion

LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
TEXT_GENERATION_BEDROCK = Settings.TEXT_GENERATION_BEDROCK


def check_student_translation_response(content: str) -> None:
    """Assert that the generated response from the LLM is correct.

    Parameters
    ----------
    content
        The generated response from the LLM.
    """

    generated_response = json.loads(content)
    assert (
        "requires_translation" in generated_response
    ), "`requires_translation` key not found."
    assert isinstance(generated_response["requires_translation"], bool), (
        f"requires_translation` must be a boolean.\n"
        f"Got: {type(generated_response['requires_translation'])}"
    )
    if generated_response["requires_translation"]:
        assert generated_response[
            "translated_text"
        ], "`translated_text` must not be empty if `requires_translation` is `true`."


def check_chatur_agent_translation_response(content: str) -> None:
    """Assert that the generated response from the LLM is correct.

    Parameters
    ----------
    content
        The generated response from the LLM.
    """

    generated_response = json.loads(content)
    assert "translated_text" in generated_response, "`translated_text` key not found."
    assert generated_response[
        "translated_text"
    ], "`translated_text` key must not be empty."


def extract_otp(message: str) -> str:
    """Look for a 6-digit number in the message, not preceded or followed by
    another digit.

    Parameters
    ----------
    message
        The message containing the OTP.

    Returns
    -------
    str
        The extracted OTP.
    """

    match = re.search(r"(?<!\d)\d{6}(?!\d)", message)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"OTP not found in the message: {message}")


async def fill_login_email(*, email: str, page: Page, url: str) -> None:
    """Fill the email field in the login form.

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


async def fill_registration_form(
    *,
    email: str,
    mobile_number: str,
    page: Page,
    url: str,
) -> None:
    """Fill the registration form with email and mobile number.

    Parameters
    ----------
    email
        The email to fill in the field.
    mobile_number
        The mobile number to fill in the field.
    page
        The Playwright page object.
    url
        The URL to navigate to.
    """

    await page.goto(url, wait_until="domcontentloaded")
    await select_register_radio(page=page)
    await page.fill("input[placeholder='Enter your mobile number']", mobile_number)
    await page.fill("input[placeholder='Enter Your Email ID']", email)
    await page.fill("input[placeholder='Confirm Your Email ID']", email)


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


async def persist_browser_and_page(
    *,
    browser: Browser,
    browser_session_store: BrowserSessionStore,
    cache_browser_state: bool = False,
    overwrite_browser_session: bool = False,
    page: Page,
    redis_client: Optional[aioredis.Redis] = None,
    reset_ttl: bool = False,
    session_id: int | str,
) -> None:
    """Persist the browser and page session.

    Parameters
    ----------
    browser
        The Playwright `Browser` that owns *page*.
    browser_session_store
        The `BrowserSessionStore` instance to store the session.
    cache_browser_state
        Specifies whether to save the browser state in Redis.
    overwrite_browser_session
        If True, an existing browser session with the same `session_id` will be
        **closed** and replaced.
    page
        The current tab whose DOM you want to preserve.
    redis_client
        The Redis client.
    reset_ttl
        Specifies whether to reset the TTL of the browser session.
    session_id
        A unique ID that identifies the session inside this Python process.
    """

    if cache_browser_state:
        assert isinstance(redis_client, aioredis.Redis)
        await save_browser_state(
            page=page, redis_client=redis_client, session_id=session_id
        )

    await browser_session_store.create(
        browser=browser,
        overwrite=overwrite_browser_session,
        page=page,
        session_id=session_id,
    )
    browser_session_saved = await browser_session_store.get(session_id=session_id)
    assert browser_session_saved, (
        f"Browser session not saved in RAM for session ID: " f"{session_id}"
    )
    if reset_ttl:
        await browser_session_store.reset_ttl(session_id=session_id)


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
        await page.wait_for_timeout(random.randint(1000, 2000))
        await solve_and_fill_captcha(page=page)

        if not Settings.PLAYWRIGHT_HEADLESS:
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
                # json_response["errors"] is keyed by fields with error messages list values.
                message_text = " ".join(*zip(*json_response["errors"].values()))
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

            # Check for error toasts
            error_locator = page.locator("#toast-container .toast-error")
            is_error_present = await error_locator.first.is_visible()

            if is_error_present:
                is_error = True  # Set main error flag
            else:
                is_error = False

            # Check for success toast
            success_locator = page.locator("#toast-container .toast-success")
            is_success_present = await success_locator.is_visible()

            if is_success_present:
                is_success = True  # Set main success flag
            else:
                is_success = False

            # Get the message text across all toasts.
            assert toast_message is not None
            message_text = await toast_message.text_content()
            assert message_text is not None
            message_text = message_text.strip()

            # Close current toast messages
            all_toasts = await page.locator("#toast-container .toast-message").all()
            for toast_handle in all_toasts:
                if await toast_handle.is_visible():
                    await toast_handle.click(timeout=100)

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


def translation_sandwich(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator to handle translation for the chatur agent.

    Parameters
    ----------
    func
        The function to be decorated.

    Returns
    -------
    Callable[..., Awaitable[Any]]
        The decorated function.
    """

    @wraps(func)
    async def wrapper(**kwargs: Any) -> ChaturFlowResults:
        """Wrapper function to translate before and after calling the chatur agent.

        Parameters
        ----------
        kwargs
            Additional keyword arguments.

        Returns
        -------
        ChaturFlowResults
            The response from calling the chatur agent.
        """

        chatur_query = kwargs["chatur_query"]

        # Forward translation (Hindi -> English) if required.
        if chatur_query.user_query:
            response = await get_acompletion(
                model=LITELLM_MODEL_CHAT,
                system_msg=ChaturPrompts.system_messages["translate_student_message"],
                text_generation_params=TEXT_GENERATION_BEDROCK,
                user_msg=ChaturPrompts.prompts["translate_student_message"].format(
                    student_message=chatur_query.user_query
                ),
                validator_call=ValidatorCall(
                    num_retries=3, validator_module=check_student_translation_response
                ),
            )
            translation_dict = json.loads(response)
            if translation_dict["requires_translation"]:
                chatur_query.user_query_translated = translation_dict["translated_text"]
            else:
                chatur_query.user_query_translated = chatur_query.user_query
        else:
            translation_dict = {"requires_translation": False, "translated_text": None}
            chatur_query.user_query_translated = chatur_query.user_query

        # Call chatur agent.
        chatur_agent_response = await func(**kwargs)

        # Back translation (English -> Hindi) if required.
        if translation_dict["requires_translation"]:
            response = await get_acompletion(
                model=LITELLM_MODEL_CHAT,
                system_msg=ChaturPrompts.system_messages[
                    "translate_chatur_agent_message"
                ],
                text_generation_params=TEXT_GENERATION_BEDROCK,
                user_msg=ChaturPrompts.prompts["translate_chatur_agent_message"].format(
                    summary_for_student=chatur_agent_response.summary_for_student
                ),
                validator_call=ValidatorCall(
                    num_retries=3,
                    validator_module=check_chatur_agent_translation_response,
                ),
            )
            translation_dict = json.loads(response)
            translated_text = translation_dict["translated_text"]
            chatur_agent_response.summary_for_student_translated = translated_text

        return chatur_agent_response

    return wrapper
