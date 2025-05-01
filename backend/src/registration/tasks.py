from playwright.async_api import async_playwright
from src.registration.schema import InitRegistrationRequest, CompleteRegistrationRequest

import pytesseract
import cv2
from loguru import logger
import asyncio
from typing import Dict, Any


active_sessions: dict[str, dict[str, Any]] = {}


async def initiate_registration(request: InitRegistrationRequest) -> Dict[str, Any]:
    """
    Initiate registration process by filling in the form and handling captcha.
    Keeps the browser session open and stores it for later OTP submission.
    """
    # Generate a unique session ID for this registration attempt
    session_id = f"{request.phone_number}_{request.email.split('@')[0]}"

    # Check if there's already an active session
    if session_id in active_sessions:
        # Close the existing session before creating a new one
        try:
            old_browser = active_sessions[session_id]["browser"]
            await old_browser.close()
            logger.info(f"Closed existing session for {session_id}")
        except Exception as e:
            logger.error(f"Error closing existing session: {str(e)}")

        del active_sessions[session_id]

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Store references to all objects we need to keep alive
        active_sessions[session_id] = {
            "playwright": playwright,
            "browser": browser,
            "context": context,
            "page": page,
            "timestamp": asyncio.get_event_loop().time(),
        }

        # Navigate to registration page
        await page.goto("https://www.apprenticeshipindia.gov.in/candidate-login")
        logger.debug("Navigated to the registration page")

        # Select Register as Candidate
        await page.locator('label[for="disnoCan"]').click(force=True)
        assert await page.locator("#disnoCan").is_checked()
        assert await page.locator("#disnoCan").input_value() == "false"
        # TODO: optionally accept ITI Student checkbox

        # Fill in the form
        await page.get_by_placeholder("Enter your mobile number").fill(
            request.phone_number
        )
        await page.locator('input[formcontrolname="email"]').fill(request.email)
        await page.get_by_placeholder("Confirm Your Email ID").fill(request.email)

        logger.debug("Filled in the registration form")

        # Try up to 3 times to solve the captcha
        max_attempts = 3
        # TODO: check for max attempts and send to users, failure captchas?
        for attempt in range(max_attempts):
            captcha_text = await extract_captcha_from_canvas(page)
            logger.debug(f"Extracted CAPTCHA text: {captcha_text}")

            if captcha_text:
                await page.get_by_placeholder("Enter CAPTCHA").fill(captcha_text)

                # Submit and wait for OTP prompt
                await submit_form_and_capture_toast(page)

                # Check if we successfully moved to OTP page
                try:
                    # Wait for OTP input field to appear with the specific placeholder
                    await page.wait_for_selector(
                        'input[placeholder="Enter 6 Digit OTP"]',
                        state="visible",
                        timeout=5000,
                    )
                    logger.info("Successfully reached OTP page")

                    # Return success but keep session alive
                    return {
                        "status": "awaiting_otp",
                        "message": "Registration initiated. OTP has been sent to the phone number.",
                        "session_id": session_id,
                    }

                except Exception as e:
                    logger.warning(f"Failed to reach OTP page: {str(e)}")

                    # Check if there's an error message about incorrect captcha
                    error_selector = ".error-message"  # Adjust if needed
                    error_element = await page.query_selector(error_selector)

                    if error_element:
                        error_message = await error_element.inner_text()
                        if "captcha" in error_message.lower():
                            logger.warning(f"Captcha error: {error_message}")

                            # Try to refresh the captcha using the provided selector
                            try:
                                await page.click("div.captcha-button.reload")
                                await page.wait_for_timeout(500)
                            except Exception as refresh_error:
                                logger.warning(
                                    f"Failed to refresh captcha: {str(refresh_error)}"
                                )
                        else:
                            # Some other error occurred
                            logger.error(f"Registration error: {error_message}")
                            await cleanup_session(session_id)
                            return {"status": "error", "message": error_message}

        # If we get here, all captcha attempts failed
        await cleanup_session(session_id)
        return {
            "status": "error",
            "message": "Failed to solve captcha after multiple attempts",
        }

    except Exception as e:
        logger.error(f"Registration initiation error: {str(e)}")
        # Clean up if needed
        if session_id in active_sessions:
            await cleanup_session(session_id)
        return {"status": "error", "message": f"Registration process failed: {str(e)}"}


async def complete_registration_with_otp(
    request: CompleteRegistrationRequest,
) -> Dict[str, Any]:
    """
    Complete the registration process using the provided OTP.
    Uses the existing browser session.
    """
    session_id = request.session_id

    # Check if session exists
    if session_id not in active_sessions:
        return {
            "status": "error",
            "message": "Session expired or not found. Please restart the registration process.",
        }

    try:
        # Get the page from active session
        page = active_sessions[session_id]["page"]

        # Enter OTP in the field with the specific placeholder
        await page.fill('input[placeholder="Enter 6 Digit OTP"]', request.otp)

        # Submit OTP by clicking the Submit button
        await page.get_by_role("button", name="Submit").click()

        # Wait for registration confirmation
        try:
            # Look for success message or redirect to dashboard
            await page.wait_for_selector(
                ".success-message", state="visible", timeout=5000
            )
            success_message = await page.inner_text(".success-message")

            # Extract registration number if available
            registration_number = None
            reg_elem = await page.query_selector(".registration-number")
            if reg_elem:
                registration_number = await reg_elem.inner_text()

            # Clean up the session now that we're done
            await cleanup_session(session_id)

            return {
                "status": "success",
                "message": success_message,
                "registration_number": registration_number,
            }
        except Exception:
            # Check for error message
            error_element = await page.query_selector(".error-message")
            error_message = (
                await error_element.inner_text() if error_element else "Unknown error"
            )

            # Only clean up if this was a permanent failure
            if "invalid" in error_message.lower() or "expired" in error_message.lower():
                await cleanup_session(session_id)

            return {"status": "error", "message": error_message}

    except Exception as e:
        logger.error(f"OTP verification error: {str(e)}")
        await cleanup_session(session_id)
        return {"status": "error", "message": f"OTP verification failed: {str(e)}"}


async def resend_otp(session_id: str) -> Dict[str, Any]:
    """
    Resend OTP for an existing registration session.
    """
    # Check if session exists
    if session_id not in active_sessions:
        return {
            "status": "error",
            "message": "Session expired or not found. Please restart the registration process.",
        }

    try:
        # Get the page from active session
        page = active_sessions[session_id]["page"]

        # Click the Resend OTP link
        await page.get_by_role("link", name="Resend OTP").click()

        # Wait for confirmation that OTP was sent (adjust selector if needed)
        try:
            # You might need to adjust this part based on how the page behaves when OTP is resent
            # It could show a message or simply refresh the page
            await page.wait_for_timeout(2000)  # Wait a bit to see if there's feedback

            # Check for any success message
            # TODO: check resend OTP behaviour
            success_element = await page.query_selector(".success-notification")
            if success_element:
                success_message = await success_element.inner_text()
                return {"status": "success", "message": success_message}

            # If no specific message but no errors either, assume success
            return {"status": "success", "message": "OTP has been resent."}

        except Exception as e:
            logger.warning(f"Issue when resending OTP: {str(e)}")

            # Check for any error message
            # TODO: check for error message that appears in <div id="toast-container"></div>
            error_element = await page.query_selector(".error-message")
            if error_element:
                error_message = await error_element.inner_text()
                return {"status": "error", "message": error_message}

            return {"status": "error", "message": f"Failed to resend OTP: {str(e)}"}

    except Exception as e:
        logger.error(f"Resend OTP error: {str(e)}")
        return {"status": "error", "message": f"Failed to resend OTP: {str(e)}"}


async def cleanup_session(session_id: str):
    """Clean up a session and its resources."""
    if session_id in active_sessions:
        try:
            await active_sessions[session_id]["browser"].close()
            await active_sessions[session_id]["playwright"].stop()
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        finally:
            del active_sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")


# TODO: Periodicaly clean up stale sessions
# Periodically clean up stale sessions (could be run in a background task)
# async def cleanup_stale_sessions(max_age_seconds: int = 300):
#     """Clean up sessions older than the specified age."""
#     current_time = asyncio.get_event_loop().time()
#     stale_sessions = []

#     for session_id, session_data in active_sessions.items():
#         if current_time - session_data["timestamp"] > max_age_seconds:
#             stale_sessions.append(session_id)

#     for session_id in stale_sessions:
#         logger.info(f"Cleaning up stale session {session_id}")
#         await cleanup_session(session_id)


async def extract_captcha_from_canvas(page):
    """Extract CAPTCHA from canvas element."""
    await page.wait_for_selector("canvas.captcha-canvas", state="visible")
    await page.wait_for_timeout(500)
    canvas_element = await page.query_selector("canvas.captcha-canvas")
    if canvas_element:
        # Take screenshot of just the canvas element
        screenshot_buffer = await canvas_element.screenshot()

        # Save to a temporary file
        with open("temp_captcha.png", "wb") as f:
            f.write(screenshot_buffer)

        # Process the captcha image
        captcha_text = process_captcha("temp_captcha.png")
        return captcha_text


def preprocess_captcha_image(image_path):
    """
    Preprocess the captcha image to improve OCR accuracy.
    For debugging use

        import opencv_jupyter_ui as jcv2
        jcv2.imshow("label", img, colorspace="gray")

    """
    img = cv2.imread(image_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_contrast = cv2.convertScaleAbs(img_gray, alpha=2, beta=0)

    threshold_value = 127
    _, img_thrshold = cv2.threshold(
        img_contrast, threshold_value, 255, cv2.THRESH_BINARY
    )
    return img_thrshold


def process_captcha(image_path):
    """Read CAPTCHA text"""
    preprocessed_img = preprocess_captcha_image(image_path)
    captcha_text = pytesseract.image_to_string(preprocessed_img)
    return captcha_text.strip()


async def submit_form_and_capture_toast(page):
    """Submit the form and capture the toast message and status."""
    try:
        # Create a promise that will resolve when the toast appears
        async with page.expect_response(
            lambda response: "https://api.apprenticeshipindia.gov.in/auth/register-otp"
            in response.url
        ) as response_info:
            await page.get_by_role("button", name="Register", exact=True).click()

        # You can check the API response if needed
        response = response_info.value

        # Wait for toast message to appear
        toast_message = page.wait_for_selector(
            "#toast-container .toast-message", state="attached", timeout=10000
        )

        # Check if it's an error or success
        is_error = page.locator("#toast-container .toast-error").is_visible()
        is_success = page.locator("#toast-container .toast-success").is_visible()

        # Get the message text
        message_text = toast_message.text_content().strip()

        return {
            "message": message_text,
            "is_error": is_error,
            "is_success": is_success,
            "response": response,
        }
    except Exception as e:
        return {"error": str(e), "is_error": True, "is_success": False}
