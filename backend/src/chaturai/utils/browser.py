"""This module contains utilities for handling browser sessions and automation.

This module primarily focuses on **in-memory** management of **live** Playwright
browser sessions.

Why This Is Needed
------------------

A FastAPI endpoint (or the functions that it calls) may need to “pause” automation
while it awaits for the user to supply extra data (e.g. an OTP).  Because a Playwright
`Page` object cannot be pickled/serialized, we keep the tab—and its JavaScript state—in
RAM.

The `BrowserSessionStore` class is a TTL‑aware registry that lets you:

1, Create a **browser** session based on a (unique) provided session ID.
2. Retrieve the same browser session on a later HTTP call.
3. Automatically clean up zombie sessions after TTL seconds.

The store is/should be safe for concurrent async use **inside one Python process**; it
is *not* multi‑process/multi‑machine safe. You’ll need sticky sessions or a dedicated
browser service to scale horizontally.
"""

# Standard Library
import asyncio
import time

# Third Party Library
from loguru import logger
from playwright.async_api import Browser, Page


class BrowserSession:
    """Wraps a Playwright page + browser and tracks its age.

    Attributes
    ----------
    browser
        The Playwright `Browser` instance that owns `page`.
    created_at
        UNIX timestamp (float) when the session was registered—used for TTL.
    page
        The open `Page` positioned at whatever DOM state you paused in.
    """

    __slots__ = ("browser", "created_at", "page")

    def __init__(self, *, browser: Browser, page: Page) -> None:
        """

        Parameters
        ----------
        browser
            A launched Playwright `Browser` (Chromium/WebKit/Firefox).
        page
            The `Page` object you want to keep alive between requests.
        """

        self.browser = browser
        self.created_at = time.time()
        self.page = page

    async def close(self) -> None:
        """Close the browser session.

        Always call this once the session is finished; otherwise the browser keeps
        consuming CPU/RAM in the background.
        """

        await self.browser.close()


class BrowserSessionStore:
    """Async‑safe registry mapping session IDs to browser sessions.

    Each public method grabs an `asyncio.Lock` so concurrent FastAPI handlers cannot
    corrupt the internal state.

    NB:
    1. **Single‑process only** – every Uvicorn worker gets its own
        `BrowserSessionStore`.
    2. A background task should call a `sweep` function periodically.
    """

    _data: dict[str, BrowserSession]
    _lock: asyncio.Lock
    _ttl: int

    def __init__(self, *, ttl: int = 600) -> None:
        """

        Parameters
        ----------
        ttl
            Time‑to‑live in **seconds** for *idle* sessions. Once the stored
            `BrowserSession.created_at` is older than *ttl*, the session is considered
            stale and will be closed on the next access or sweep.
        """

        self._data = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl

    async def _remove_browser_session(self, *, session_id: str) -> None:
        """Remove `session_id` from the registry and close its browser (best‑effort).

        Parameters
        ----------
        session_id
            The ID of the session to remove.
        """

        browser_session = self._data.pop(session_id, None)
        if browser_session:
            await browser_session.close()

    async def create(
        self,
        *,
        browser: Browser,
        overwrite: bool = False,
        page: Page,
        session_id: int | str,
    ) -> bool:
        """Register a new live session.

        Parameters
        ----------
        browser
            The Playwright `Browser` that owns *page*.
        overwrite
            If True, an existing browser session with the same `session_id` will be
            **closed** and replaced. Defaults to False to enforce uniqueness.
        page
            The current tab whose DOM you want to preserve.
        session_id
            A unique ID that identifies the session inside this Python process.

        Returns
        -------
        bool
            True, If the browser session was stored successfully.
            False, If the browser session with the same session ID already exists and
            `overwrite` is False (nothing was modified).

        Side Effects
        ------------
        * Closes and replaces the existing session if `overwrite` is True.
        * Always executes under `self._lock` for thread‑safety.
        """

        session_id_str = str(session_id)

        async with self._lock:
            if not overwrite and session_id_str in self._data:
                return False

        if session_id_str in self._data:
            logger.warning(
                f"Overwriting existing browser session with ID: {session_id_str}!"
            )
            await self._remove_browser_session(session_id=session_id_str)

        self._data[session_id_str] = BrowserSession(browser=browser, page=page)

        return True

    async def delete(self, *, session_id: int | str) -> None:
        """Explicitly close and remove `session_id`.

        NB: If `session_id` does not exist, the call is a no‑op.

        Parameters
        ----------
        session_id
            The session ID to destroy.
        """

        session_id_str = str(session_id)

        async with self._lock:
            await self._remove_browser_session(session_id=session_id_str)

    async def get(self, *, session_id: int | str) -> BrowserSession | None:
        """Retrieve a session if it exists *and* has not expired.

        Parameters
        ----------
        session_id
            The session ID associated with the `BrowserSession` you want to retrieve.

        Returns
        -------
        BrowserSession | None
            The live session, or `None` if the session ID is unknown **or** the session
             exceeded the TTL.

        Side Effects
        ------------
        Expired sessions are *immediately* closed and removed.
        """

        session_id_str = str(session_id)

        async with self._lock:
            browser_session = self._data.get(session_id_str, None)
            if browser_session is not None and (
                time.time() - browser_session.created_at < self._ttl
            ):
                return browser_session

            # Expired -> clean up now.
            if browser_session is not None:
                await self._remove_browser_session(session_id=session_id_str)
            return None

    async def sweep(self) -> None:
        """Purge **all** sessions whose age exceeds the TTL."""

        async with self._lock:
            for session_id in list(self._data):
                if time.time() - self._data[session_id].created_at > self._ttl:
                    await self._remove_browser_session(session_id=session_id)
