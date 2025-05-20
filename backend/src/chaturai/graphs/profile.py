"""This module contains the profile completion graph.

The profile completion graph does the following:

1. TODO: Fill this out when everything is finalized.
"""

# Standard Library
import asyncio
import re

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any

# Third Party Library
from loguru import logger
from playwright.async_api import BrowserType
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis

# Package Library
from chaturai.chatur.schemas import (
    LoginStudentResults,
    NextChatAction,
    ProfileCompletionQuery,
    ProfileCompletionResults,
)
from chaturai.chatur.utils import (
    fill_otp,
    persist_browser_and_page,
    submit_and_capture_api_response,
)
from chaturai.config import Settings
from chaturai.graphs.utils import load_browser_state, save_graph_diagram
from chaturai.metrics.logfire_metrics import profile_completion_agent_hist
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.utils.browser import BrowserSessionStore
from chaturai.utils.chat import AsyncChatSessionManager, log_chat_history
from chaturai.utils.general import telemetry_timer

LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
PLAYWRIGHT_HEADLESS = Settings.PLAYWRIGHT_HEADLESS
REDIS_CACHE_PREFIX_GRAPH_CHATUR = Settings.REDIS_CACHE_PREFIX_GRAPH_CHATUR
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@dataclass
class ProfileCompletionState:
    """The state tracks the progress of the profile completion graph."""

    browser_state: dict[str, Any]
    session_id: int | str

    graph_run_results: LoginStudentResults | ProfileCompletionResults


@dataclass
class ProfileCompletionDeps:
    """This class contains dependencies used by nodes in the profile completion graph."""

    browser: BrowserType
    browser_session_store: BrowserSessionStore
    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    explanation_for_call: str
    profile_completion_query: ProfileCompletionQuery
    redis_client: aioredis.Redis

    login_otp_url: str = "https://api.apprenticeshipindia.gov.in/auth/login-otp"
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "Profile Completion Agent",
            "system": "System",
            "user": "Student",
        }
    )


@dataclass
class CompleteStudentProfile(
    BaseNode[ProfileCompletionState, ProfileCompletionDeps, ProfileCompletionResults]
):
    """This node logs an existing student into the candidate portal and completes their
    profile.
    """

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[ProfileCompletionState, ProfileCompletionDeps]
    ) -> Annotated[
        End[ProfileCompletionResults], Edge(label="Complete the student's profile")
    ]:
        """The run proceeds as follows:

        1. Retrieve the browser session object from the last assistant. This is
            required since we must resume using both the persisted browser cache state
            and the persisted page.
        2. Wait for the OTP field to be visible and fill it with the OTP provided by
            the student.
        3. Submit the OTP and capture the API response.
        4. Construct the appropriate response based on the API response.
        5. Save the browser state in Redis, persist the page in the browser session
            store, and reset the TTL of the browser session store.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[ProfileCompletionResults]
            The results of the graph run.
        """

        # 1.
        browser_session = await ctx.deps.browser_session_store.get(
            session_id=ctx.state.graph_run_results.session_id
        )
        assert browser_session
        page = browser_session.page

        # 2.
        otp_message = (
            ctx.deps.profile_completion_query.otp
            or ctx.deps.profile_completion_query.user_query_translated
        )

        # 3.
        # Look for a 6-digit number in the message, not preceded or followed by
        # another digit.
        match = re.search(r"(?<!\d)\d{6}(?!\d)", otp_message)
        if not match:
            end = End(
                ProfileCompletionResults(  # type: ignore
                    next_chat_action=NextChatAction.REQUEST_USER_QUERY,
                    session_id=ctx.state.session_id,
                    summary_of_page_results=f"Cannot complete login. OTP not found "
                    f"in the message: {otp_message}",
                )
            )
        else:
            # 3.
            otp_text = match.group()

            await fill_otp(
                otp=otp_text,
                page=page,
            )

            # 4.
            submit_otp_response = await submit_and_capture_api_response(
                page=page, api_url=ctx.deps.login_otp_url, button_name="Login"
            )

            logger.info(f"{submit_otp_response = }")

            if not Settings.PLAYWRIGHT_HEADLESS:
                await asyncio.get_event_loop().run_in_executor(None, input)

            # 5.
            if submit_otp_response.is_error:
                end = End(
                    ProfileCompletionResults(  # type: ignore
                        next_chat_action=NextChatAction.GO_TO_HELPDESK,
                        session_id=ctx.state.session_id,
                        summary_of_page_results=f"Cannot complete login. {submit_otp_response.message}",
                    )
                )
            else:
                # TODO: Do not return an End object but pass on to other assistants.
                end = End(
                    ProfileCompletionResults(  # type: ignore
                        next_chat_action=NextChatAction.REQUEST_USER_QUERY,
                        session_id=ctx.state.session_id,
                        summary_of_page_results="OTP successfully entered. "
                        "Determine the next best assistant to call based on the student's "
                        "latest message and your conversation with the student so far. If "
                        "unclear, ask the student what they wish to do next.",
                    )
                )

        # 6.
        await persist_browser_and_page(
            browser=browser_session.browser,  # Reuse the same browser here!
            browser_session_store=ctx.deps.browser_session_store,
            cache_browser_state=True,
            overwrite_browser_session=True,  # Update if the page changed at all!
            page=page,
            redis_client=ctx.deps.redis_client,
            reset_ttl=True,
            session_id=ctx.state.session_id,
        )

        return end  # type: ignore


@telemetry_timer(metric_fn=profile_completion_agent_hist, unit="s")
async def complete_profile(
    *,
    browser: BrowserType,
    browser_session_store: BrowserSessionStore,
    chatur_query: ProfileCompletionQuery,
    csm: AsyncChatSessionManager,
    explanation_for_call: str = "No explanation provided.",
    generate_graph_diagram: bool = False,
    last_graph_run_results: LoginStudentResults | ProfileCompletionResults,
    redis_client: aioredis.Redis,
    reset_chat_session: bool = False,
) -> ProfileCompletionResults:
    """
    🧾 Profile Completion Assistant for Logged-In Student

    **The student's OTP will be automatically provided to me for completing the login
    process. You do not need to ask the student for this information!**

    ✅ WHEN TO USE THIS ASSISTANT
    Use this assistant **only** in the following situations:
        1. Finalize the login by submitting the OTP provided by the student.
        2. Navigate to the appropriate profile sections on the portal.
        3. Automatically fill in and submit the student's profile information across
            the necessary pages.

    This assistant is responsible for progressing the student through the profile setup
    phase, ensuring all required data fields are filled correctly (e.g., personal info,
    educational background, E-KYC, bank account details, etc.).

    🔄 IMPORTANT: This assistant assumes that login credentials were already submitted
    earlier and that the OTP has now been received from the student.

    📌 COMMON USE CASES
        - A student has just provided their OTP after receiving it.
        - You are ready to help the student complete their onboarding by filling out
            their profile details.

    📝 HOW TO CALL THIS ASSISTANT
    Phrase your explanation as a **direct message** to the assistant. **Do not** use
    first-person language (e.g., “I will use...” or “I’m going to call...”). Instead,
    use imperative phrasing, such as:
        - "Submit the student’s OTP to complete the login, then continue to fill out and submit all required profile information on the portal."

    Provide the following information in your explanation to this assistant:
        - A clear explanation of **why** you are calling this assistant, with reference
            to the latest student message and the current state of the application
            process. **Remember, this assistant does not directly interact with the
            student---thus, the assistant is not aware of the conversation that is
            going on between you and the student!**

    🚫 DO NOT USE THIS ASSISTANT IF
        - The student has **not yet received or shared** their OTP.
        - The student’s profile is already fully completed and they’re moving onto
            applications, contract signing, or other next steps (use the relevant
            assistants for those actions).
        - The student is not logged in or is starting a new registration (use the
            `registration.register_student` or `login.login_student assistant` instead).

    Parameters
    ----------
    browser
        The Playwright browser object.
    browser_session_store
        The browser session store object.
    chatur_query
        The query object.
    csm
        An async chat session manager that manages the chat sessions for each user.
    explanation_for_call
        An explanation as to why the assistant is being called.
    generate_graph_diagram
        Specifies whether to generate ALL graph diagrams using the Mermaid API (free).
    last_graph_run_results
        The last graph run results from either the Login Student or Profile Completion
        graph. This is typically passed in by the chatur agent.
    redis_client
        The Redis client.
    reset_chat_session
        Specìfies whether to reset the chat session for the assistant.

    Returns
    -------
    ProfileCompletionResults
        The graph run result.
    """

    assert chatur_query.user_query_translated or chatur_query.otp
    chatur_query = deepcopy(chatur_query)
    chatur_query.user_query_translated = (
        chatur_query.user_query_translated or chatur_query.otp
    )
    chatur_query.user_id = f"Profile_Completion_Agent_Graph_{chatur_query.user_id}"

    # 1. Initialize the chat history, chat parameters, and the session ID for the agent.
    chat_history, chat_params, session_id = await csm.init_chat_session(
        model_name="chat",
        namespace="profile-completion-agent",
        reset_chat_session=reset_chat_session,
        system_message=ChaturPrompts.system_messages["profile_completion_agent"],
        text_generation_params=TEXT_GENERATION_GEMINI,
        topic=None,
        user_id=chatur_query.user_id,
    )

    # 2. Create the graph.
    graph = Graph(
        auto_instrument=True,
        name="Profile_Completion_Agent_Graph",
        nodes=[CompleteStudentProfile],
        state_type=ProfileCompletionState,
    )

    # 3. Generate graph diagram.
    if generate_graph_diagram:
        save_graph_diagram(graph=graph)

    # 4. Set graph dependencies.
    deps = ProfileCompletionDeps(
        browser=browser,
        browser_session_store=browser_session_store,
        chat_history=chat_history,
        chat_params=chat_params,
        explanation_for_call=explanation_for_call,
        profile_completion_query=chatur_query,
        redis_client=redis_client,
    )

    # 5. Set graph persistence.
    fsp = FullStatePersistence(deep_copy=True)
    fsp.set_graph_types(graph)

    # 6. Load the appropriate graph state.
    redis_cache_key, state = await load_state(
        persistence=fsp,
        last_graph_run_results=last_graph_run_results,
        redis_client=redis_client,
        reset_state=reset_chat_session,
        session_id=session_id,
    )

    # 7. Execute the graph until completion.
    graph_run_results = await graph.run(
        CompleteStudentProfile(), deps=deps, persistence=fsp, state=state
    )

    # 8. Update the chat history for the agent.
    await csm.update_chat_history(chat_history=chat_history, session_id=session_id)
    await csm.dump_chat_session_to_file(session_id=session_id)

    # 9. Save the graph snapshot to Redis.
    snapshot_json = fsp.dump_json()
    await redis_client.set(redis_cache_key, snapshot_json)

    # 10. Log the agent chat history at the end of each step (just for debugging
    # purposes).
    log_chat_history(
        chat_history=chat_history,
        context="Profile Completion Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output


async def load_state(
    *,
    persistence: FullStatePersistence,
    last_graph_run_results: LoginStudentResults | ProfileCompletionResults,
    redis_client: aioredis.Redis,
    reset_state: bool,
    session_id: int | str,
) -> tuple[str, ProfileCompletionState]:
    """Load state for the Profile Completion graph.

    The process is as follows:

    1. If a previous state does not exist, then we initialize a new state with the
        results from the Login Student graph run.
    2. Otherwise, we load the last state for the Profile Completion graph and update it
        with the latest results from either the Login Student or the Profile Completion
        graph run (passed in by the chatur agent). Note that any changes made by the
        frontend to the last results of the **Profile Completion** graph run will be
        reflected in its own Redis cache.

    NB: Loading the last state for the Profile Completion graph works in this manner
    because the graph only needs the attributes defined by the `PageResults` schema and
    both Login Student and Profile Completion graph outputs inherit from this common
    schema. In other words, the common starting point for the Profile Completion graph
    is just what is defined by `PageResults`.

    Parameters
    ----------
    persistence
        The persistence object for the graph.
    last_graph_run_results
        The last graph run results from either the Login Student or Profile Completion
        graph. This is typically passed in by the chatur agent.
    redis_client
        The Redis client.
    reset_state
        Specifies whether to reset the state.
    session_id
        The session ID for the graph.

    Returns
    -------
    tuple[str, ProfileCompletionState]
        The Redis cache key and the state for the Profile Completion graph.
    """

    last_session_id = last_graph_run_results.session_id
    last_browser_state = await load_browser_state(
        redis_client=redis_client, session_id=last_session_id
    )
    redis_cache_key = (
        f"{REDIS_CACHE_PREFIX_GRAPH_CHATUR}_Profile_Completion_{session_id}"
    )

    # 1. If using the profile assistant for the first time.
    if reset_state or not await redis_client.exists(redis_cache_key):
        return redis_cache_key, ProfileCompletionState(
            browser_state=last_browser_state,
            graph_run_results=last_graph_run_results,
            session_id=session_id,
        )

    # 2. If this is _not_ the first time profile assistant is being called.
    raw_snapshot = await redis_client.get(redis_cache_key)
    persistence.load_json(raw_snapshot)
    snapshot = await persistence.load_all()
    state = snapshot[-1].state
    state.browser_state = last_browser_state
    state.graph_run_results = last_graph_run_results
    return redis_cache_key, state
