"""This module contains the student login graph.

The student login graph does the following:

1. Navigate to the `Login as a candidate` page and auto-fill the required fields.
2. Solve the text-based CAPTCHA.
3. XXX
4. Submit the login form.
5. Return the graph run results containing (among other things) the persisted **page**
    state for the next assistant.

NB: This graph does **not** handle the entire Naukri process (e.g., E-KYC, bank account
setup, profile completion, etc.). It simply logs an existing student into the portal
and forwards the persisted page to the next assistant.
"""

# Standard Library
import asyncio

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any

# Third Party Library
from loguru import logger
from playwright.async_api import async_playwright
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis

# Package Library
from naukriwaala.config import Settings
from naukriwaala.graphs.utils import (
    load_browser_state,
    save_browser_state,
    save_graph_diagram,
)
from naukriwaala.metrics.logfire_metrics import login_agent_hist
from naukriwaala.naukri.schemas import (
    LoginStudentQuery,
    LoginStudentResults,
    RegisterStudentResults,
)
from naukriwaala.naukri.utils import select_login_radio  # , solve_captcha
from naukriwaala.prompts.naukri import NaukriPrompts
from naukriwaala.utils.chat import AsyncChatSessionManager, log_chat_history
from naukriwaala.utils.general import telemetry_timer

# Globals.
LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
REDIS_CACHE_PREFIX_GRAPH_STUDENT_LOGIN = Settings.REDIS_CACHE_PREFIX_GRAPH_STUDENT_LOGIN
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@dataclass
class LoginStudentState:
    """The state tracks the progress of the student login graph."""

    browser_state: dict[str, Any]
    session_id: int | str


@dataclass
class LoginStudentDeps:
    """This class contains dependencies used by nodes in the student login graph."""

    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    explanation_for_call: str
    login_student_query: LoginStudentQuery
    redis_client: aioredis.Redis

    login_url: str = "https://www.apprenticeshipindia.gov.in/candidate-login"
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "Login Stduent Agent",
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
        Edge(label="Login an existing student into the candidate portal"),
    ]:
        """The run proceeds as follows:

        1. Load the correct browser state.
        2. Navigate to the page shell.
        3. Switch into Login mode.
        4. Possibly resend an activation link.
        5. Fill in the email field.
        6. Capture the CAPTCHA canvas exactly.
        7. Solve the CAPTCHA text.
        8. Fill out the CAPTCHA text.
        9. XXX
        X. Save the browser state in Redis and close the browser.

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

        async with async_playwright() as pw:
            # 1.
            browser = await pw.chromium.launch(headless=False)
            context = await browser.new_context(storage_state=ctx.state.browser_state)  # type: ignore
            page = await context.new_page()

            # 2.
            await page.goto(ctx.deps.login_url, wait_until="domcontentloaded")

            # 3.
            await select_login_radio(page=page)

            # 4. TODO

            # 5. TODO

            # 6.
            canvas = page.locator("canvas.captcha-canvas")
            await canvas.wait_for(state="visible")
            captcha_bytes: bytes = await canvas.screenshot()

            # 7.
            captcha_text = await solve_captcha(captcha_bytes=captcha_bytes)

            # 8.
            await page.fill('input[placeholder="Enter CAPTCHA"]', captcha_text)

            # 9.

            # Pause to verify in the browser.
            await asyncio.get_event_loop().run_in_executor(None, input)

            # X.
            await save_browser_state(page=page, redis_client=ctx.deps.redis_client)

        return End(  # type: ignore
            LoginStudentResults(  # type: ignore
                summary_of_page_results="Login successful. Please continue with apprenticeship process.",
            )
        )


@telemetry_timer(metric_fn=login_agent_hist, unit="s")
async def login_student(
    *,
    csm: AsyncChatSessionManager,
    explanation_for_call: str = "No explanation provided.",
    generate_graph_diagram: bool = False,
    last_graph_run_results: RegisterStudentResults,
    naukri_query: LoginStudentQuery,
    redis_client: aioredis.Redis,
    reset_chat_session: bool = False,
) -> LoginStudentResults:
    """
    🔍 Login Existing Student Assistant

    **The student's email and password will be automatically provided to me for logging
    in. You do not need to ask the student for this information!**

    ✅ WHEN TO USE THIS ASSISTANT
    Use this assistant **only** in the following situations:
        1. An **existing student** needs to log into the Indian government
            apprenticeship portal to continue with their application process. This is
            typically the case when the student has already completed the initial
            account registration and now wants to:
            - Fill out or update their apprenticeship profile,
            - Submit necessary documents, or
            - Apply for available apprenticeship opportunities.
        2. A student has previously started an application process but did **not
            complete** it, and now they want to resume their session by logging in.

     This assistant is used **only to facilitate a login for an already registered
     student**. It ensures that the student is authenticated on the portal so they can
     proceed to the next step in their apprenticeship journey. **After calling this
     assistant, you should use another assistant appropriate for the next step**, such
     as completing profile details, uploading documents, or searching and applying for
     apprenticeships.

    📝 HOW TO CALL THIS ASSISTANT
    Phrase your explanation as a **direct message** to the assistant. **Do not** use
    first-person language (e.g., “I will use...” or “I’m going to call...”). Instead,
    use imperative phrasing, such as:
        - "Log in the student so they can continue the application process."
        - "Authenticate the existing student to resume their application workflow."
        - "Initiate login for a registered student returning to complete their apprenticeship application."

    Provide the following information in your explanation to this assistant:
        - A clear explanation of **why** you are calling this assistant, with reference
            to the latest student message and the current state of th application
            process. **Remember, this assistant does not directly interact with the
            student---thus, the assistant is not aware of the conversation that is
            going on between you and the student!**

    🚫 DO NOT USE THIS ASSISTANT IF
        - The student is **not yet registered** or is asking to create a new account.
        - The student is already logged in and is trying to proceed with profile
            completion, document upload, or application steps (use other assistants for
            those actions).

    Parameters
    ----------
    csm
        An async chat session manager that manages the chat sessions for each user.
    explanation_for_call
        An explanation as to why the assistant is being called.
    generate_graph_diagram
        Specifies whether to generate ALL graph diagrams using the Mermaid API (free).
    last_graph_run_results
        The last graph run results from either the Register Student graph. This is
        typically passed in by the Naukri agent.
    naukri_query
        The query object.
    redis_client
        The Redis client.
    reset_chat_session
        Specìfies whether to reset the chat session for the assistant.

    Returns
    -------
    LoginStudentResults
        The graph run result.
    """

    logger.info(f"{explanation_for_call} for {naukri_query = }")
    logger.info(f"{last_graph_run_results = }")
    logger.info(
        "Press Enter to continue but note that it will fail if solve_captcha is not "
        "implemented!"
    )
    input()
    naukri_query = deepcopy(naukri_query)
    naukri_query.user_id = f"Login_Student_Agent_Graph_{naukri_query.user_id}"

    # 1. Initialize the chat history, chat parameters, and the session ID for the agent.
    chat_history, chat_params, session_id = await csm.init_chat_session(
        model_name="chat",
        namespace="login-student-agent",
        reset_chat_session=reset_chat_session,
        system_message=NaukriPrompts.system_messages["login_student_agent"],
        text_generation_params=TEXT_GENERATION_GEMINI,
        topic=None,
        user_id=naukri_query.user_id,
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
        chat_history=chat_history,
        chat_params=chat_params,
        explanation_for_call=explanation_for_call,
        login_student_query=naukri_query,
        redis_client=redis_client,
    )

    # 5. Set graph persistence.
    fsp = FullStatePersistence(deep_copy=True)
    fsp.set_graph_types(graph)

    # 6. Execute the graph until completion.
    browser_state = await load_browser_state(redis_client=redis_client)
    graph_run_results = await graph.run(
        LoginExistingStudent(),
        deps=deps,
        persistence=fsp,
        state=LoginStudentState(browser_state=browser_state, session_id=session_id),
    )

    # 7. Update the chat history.
    await csm.update_chat_history(chat_history=chat_history, session_id=session_id)
    await csm.dump_chat_session_to_file(session_id=session_id)

    log_chat_history(
        chat_history=chat_history,
        context="Login Student Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output
