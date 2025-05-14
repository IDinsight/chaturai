"""This module contains the student login graph.

The student login graph does the following:

1. Navigates to the login page of the candidate portal.
2. Fills in the login form with the email.
3. Solves the captcha and submits the form.
4. Also clicks the Login button to trigger an OTP email and text message to the student.
5. Persists both the browser and page states for the next assistant.
"""

# Standard Library
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any

# Third Party Library
from playwright.async_api import BrowserType
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis

# Package Library
from chaturai.chatur.schemas import (
    LoginStudentQuery,
    LoginStudentResults,
    NextChatAction,
)
from chaturai.chatur.utils import (
    fill_email,
    persist_browser_and_page,
    solve_and_submit_captcha_with_retries,
)
from chaturai.config import Settings
from chaturai.graphs.utils import load_browser_state, save_graph_diagram
from chaturai.metrics.logfire_metrics import login_agent_hist
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.utils.browser import BrowserSessionStore
from chaturai.utils.chat import AsyncChatSessionManager, log_chat_history
from chaturai.utils.general import telemetry_timer

LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
PLAYWRIGHT_HEADLESS = Settings.PLAYWRIGHT_HEADLESS
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@dataclass
class LoginStudentState:
    """The state tracks the progress of the student login graph."""

    browser_state: dict[str, Any]
    session_id: int | str


@dataclass
class LoginStudentDeps:
    """This class contains dependencies used by nodes in the student login graph."""

    browser: BrowserType
    browser_session_store: BrowserSessionStore
    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    explanation_for_call: str
    login_student_query: LoginStudentQuery
    redis_client: aioredis.Redis

    login_get_otp_url: str = "https://api.apprenticeshipindia.gov.in/auth/login-get-otp"
    login_url: str = "https://www.apprenticeshipindia.gov.in/candidate-login"
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "Login Student Agent",
            "system": "System",
            "user": "Student",
        }
    )


@dataclass
class LoginExistingStudent(
    BaseNode[LoginStudentState, LoginStudentDeps, LoginStudentResults]
):
    """This node logs an existing student into the candidate portal."""

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[LoginStudentState, LoginStudentDeps]
    ) -> Annotated[
        End[LoginStudentResults],
        Edge(
            label="Submit student's email for an OTP activation email on the candidate "
            "portal"
        ),
    ]:
        """The run proceeds as follows:

        1. Launch a new browser page.
        2. Navigate to the login URL, select the login radio button, and fill in the
            login form with the email.
        3. Solve the captcha and submit the form.
        4. Save the browser state in Redis and persist the page in the browser session
            store.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[LoginStudentResults]
            The results of the graph run.
        """

        # 1.
        browser = await ctx.deps.browser.launch(headless=PLAYWRIGHT_HEADLESS)
        context = await browser.new_context(storage_state=ctx.state.browser_state)  # type: ignore
        page = await context.new_page()

        # 2.
        await fill_email(
            email=str(ctx.deps.login_student_query.email),
            page=page,
            url=ctx.deps.login_url,
        )

        # 3.
        try:
            _ = await solve_and_submit_captcha_with_retries(
                api_url=ctx.deps.login_get_otp_url, button_name="Submit", page=page
            )
            message = (
                "Initiated login. Please ask the student for a 6-digit OTP which "
                "should be sent to the registered mobile number and the email. Remind "
                "the student that the OTP is valid for 10 minutes."
            )
            next_action = NextChatAction.REQUEST_OTP

        except RuntimeError as e:
            message = f"Cannot initiate login. {str(e)}"
            next_action = NextChatAction.GO_TO_HELPDESK

        end = End(
            LoginStudentResults(  # type: ignore
                next_chat_action=next_action,
                session_id=ctx.state.session_id,
                summary_of_page_results=message,
            )
        )

        # 4.
        await persist_browser_and_page(
            browser=browser,
            browser_session_store=ctx.deps.browser_session_store,
            cache_browser_state=True,
            overwrite_browser_session=False,
            page=page,
            redis_client=ctx.deps.redis_client,
            session_id=ctx.state.session_id,
        )

        return end  # type: ignore


@telemetry_timer(metric_fn=login_agent_hist, unit="s")
async def login_student(
    *,
    browser: BrowserType,
    browser_session_store: BrowserSessionStore,
    chatur_query: LoginStudentQuery,
    csm: AsyncChatSessionManager,
    explanation_for_call: str = "No explanation provided.",
    generate_graph_diagram: bool = False,
    redis_client: aioredis.Redis,
    reset_chat_session: bool = False,
) -> LoginStudentResults:
    """
    🔐 Login Initiation Assistant for Existing Student

    **The student's email and password will be automatically provided to me for logging
    in. You do not need to ask the student for this information!**

    ✅ WHEN TO USE THIS ASSISTANT
    Use this assistant **only** in the following situations:
        1. To **initiate the login process** for a student who is **already
            registered** on the Indian Government Apprenticeship Portal.

    This assistant performs the initial login attempt using the provided credentials.
    If the credentials are valid, it will prompt the system to send a One-Time Password
    (OTP) to the student's registered email and mobile number.

    🔄 IMPORTANT: This assistant **does not complete the full login process.**
    It stops after triggering OTP delivery. A separate assistant must be used to
    proceed once the student receives and shares their OTP.

    📌 COMMON USE CASES
        - A student needs to log in to complete their apprenticeship profile, but has
            not yet received the OTP.

     **After calling this assistant, you should use another assistant appropriate
     for the next step**, such as completing the profile, searching and applying
     for apprenticeships, or reviewing and signing contracts.

    📝 HOW TO CALL THIS ASSISTANT
    Phrase your explanation as a **direct message** to the assistant. **Do not** use
    first-person language (e.g., “I will use...” or “I’m going to call...”). Instead,
    use imperative phrasing, such as:
        - "Start the login process so the student receives an OTP on their registered contact details."

    Provide the following information in your explanation to this assistant:
        - A clear explanation of **why** you are calling this assistant, with reference
            to the latest student message and the current state of the application
            process. **Remember, this assistant does not directly interact with the
            student---thus, the assistant is not aware of the conversation that is
            going on between you and the student!**

    🚫 DO NOT USE THIS ASSISTANT IF
        - The student is **not yet registered** or is trying to create a new account.
        - The student has **already received and shared the OTP** (use another
            assistant to continue the candidate process instead).
        - The student is already logged in and working on later steps like profile
            completion, document upload, or application tasks (use other assistants for
            those purposes).

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
    redis_client
        The Redis client.
    reset_chat_session
        Specìfies whether to reset the chat session for the assistant.

    Returns
    -------
    LoginStudentResults
        The graph run result.
    """

    assert chatur_query.user_query
    chatur_query = deepcopy(chatur_query)
    chatur_query.user_id = f"Login_Student_Agent_Graph_{chatur_query.user_id}"

    # 1. Initialize the chat history, chat parameters, and the session ID for the agent.
    chat_history, chat_params, session_id = await csm.init_chat_session(
        model_name="chat",
        namespace="login-student-agent",
        reset_chat_session=reset_chat_session,
        system_message=ChaturPrompts.system_messages["login_student_agent"],
        text_generation_params=TEXT_GENERATION_GEMINI,
        topic=None,
        user_id=chatur_query.user_id,
    )

    # 2. Create the graph.
    graph = Graph(
        auto_instrument=True,
        name="Login_Student_Agent_Graph",
        nodes=[LoginExistingStudent],
        state_type=LoginStudentState,
    )

    # 3. Generate graph diagram.
    if generate_graph_diagram:
        save_graph_diagram(graph=graph)

    # 4. Set graph dependencies.
    deps = LoginStudentDeps(
        browser=browser,
        browser_session_store=browser_session_store,
        chat_history=chat_history,
        chat_params=chat_params,
        explanation_for_call=explanation_for_call,
        login_student_query=chatur_query,
        redis_client=redis_client,
    )

    # 5. Set graph persistence.
    fsp = FullStatePersistence(deep_copy=True)
    fsp.set_graph_types(graph)

    # 6. Execute the graph until completion.
    browser_state = await load_browser_state(
        redis_client=redis_client, session_id=session_id
    )
    graph_run_results = await graph.run(
        LoginExistingStudent(),
        deps=deps,
        persistence=fsp,
        state=LoginStudentState(browser_state=browser_state, session_id=session_id),
    )

    # 7. Update the chat history for the agent.
    await csm.update_chat_history(chat_history=chat_history, session_id=session_id)
    await csm.dump_chat_session_to_file(session_id=session_id)

    # 8. Log the agent chat history at the end of each step (just for debugging
    # purposes).
    log_chat_history(
        chat_history=chat_history,
        context="Login Student Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output
