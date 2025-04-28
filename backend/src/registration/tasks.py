from playwright.async_api import async_playwright
from .schema import InitRegistrationRequest

import pytesseract
import cv2
from loguru import logger


async def initiate_registration(request: InitRegistrationRequest):
    """
    Initiate registration process by filling in the form and handling captcha.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://www.apprenticeshipindia.gov.in/candidate-login")
        logger.debug("Navigated to the registration page")

        # Fill in the form
        await page.get_by_placeholder("Enter your mobile number").fill(
            request.phone_number
        )
        await page.get_by_placeholder("Enter Your Email ID").fill(request.email)
        await page.get_by_placeholder("Confirm Your Email ID").fill(request.email)
        logger.debug("Filled in the registration form")

        captcha_text = await extract_captcha_from_canvas(page)
        logger.info(f"Extracted CAPTCHA text: {captcha_text}")
        if captcha_text:
            await page.get_by_placeholder("Enter CAPTCHA").fill(captcha_text)

        # Submit and wait for OTP prompt
        await page.get_by_role("button", name="Register").click()
        logger.debug("Clicked on Register button")

        # Additional logic for OTP handling
        await browser.close()


async def extract_captcha_from_canvas(page):
    """Extract CAPTCHA from canvas element."""
    await page.wait_for_selector("canvas#captcha-canvas", state="visible")
    await page.wait_for_timeout(500)
    canvas_element = await page.query_selector("canvas#captcha-canvas")
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


if __name__ == "__main__":
    request = InitRegistrationRequest(email="abc@gmail.com", phone_number="1234567890")
    import asyncio

    asyncio.run(initiate_registration(request))
