"""This module contains utilities for chatur."""

# Third Party Library
import cv2
import pytesseract

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError


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


async def solve_and_fill_captcha(page: Page):
    """Solve the CAPTCHA and fill it in the form."""
    canvas = page.locator("canvas.captcha-canvas")
    await canvas.wait_for(state="visible")
    captcha_bytes: bytes = await canvas.screenshot()

    captcha_text = await solve_captcha(captcha_bytes)

    await page.fill('input[placeholder="Enter CAPTCHA"]', captcha_text)


async def solve_captcha(image_buffer: bytes) -> str:
    """Extract captcha text from image."""
    preprocessed_img = preprocess_captcha_image(image_buffer)
    captcha_text = pytesseract.image_to_string(preprocessed_img)
    return captcha_text.strip()


def preprocess_captcha_image(image_buffer: bytes):
    """
    Preprocess the captcha image to improve OCR accuracy.

    NB: For debugging use

        import opencv_jupyter_ui as jcv2
        jcv2.imshow("label", img, colorspace="gray")

    """
    img = cv2.imdecode(image_buffer)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_contrast = cv2.convertScaleAbs(img_gray, alpha=2, beta=0)

    threshold_value = 127
    _, img_thrshold = cv2.threshold(
        img_contrast, threshold_value, 255, cv2.THRESH_BINARY
    )
    return img_thrshold


async def submit_and_capture_api_response(
    page: Page, button_name: str, api_url: str
) -> dict:
    """Click a button and wait for the page to navigate."""
    async with page.expect_response(
        lambda response: api_url in response.url
    ) as response_info:
        await page.get_by_role("button", name=button_name, exact=True).click()

    response = await response_info.value
    return response.json()
