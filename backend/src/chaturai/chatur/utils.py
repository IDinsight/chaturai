"""This module contains utilities for chatur."""

# Third Party Library
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
