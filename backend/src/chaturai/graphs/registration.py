"""This module contains the student registration graph.

The student registration graph does the following:

1. TODO: Fill this out when everything is finalized.
"""

# Standard Library
import asyncio

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any

# Third Party Library
from loguru import logger
from playwright.async_api import Browser, BrowserType, Page
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis

# Package Library
from chaturai.chatur.schemas import (
    NextChatAction,
    RegisterStudentQuery,
    RegisterStudentResults,
    RegistrationCompleteResults,
)
from chaturai.chatur.utils import (
    fill_otp,
    fill_roll_number,
    select_register_radio,
    solve_and_submit_captcha_with_retries,
    submit_and_capture_api_response,
)
from chaturai.config import Settings
from chaturai.graphs.utils import save_graph_diagram
from chaturai.metrics.logfire_metrics import register_student_agent_hist
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.utils.browser import BrowserSessionStore
from chaturai.utils.chat import AsyncChatSessionManager, log_chat_history
from chaturai.utils.general import telemetry_timer

LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
PLAYWRIGHT_HEADLESS = Settings.PLAYWRIGHT_HEADLESS
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@dataclass
class RegisterStudentState:
    """The state tracks the progress of the student registration graph."""

    session_id: int | str


@dataclass
class RegisterStudentDeps:
    """This class contains dependencies used by nodes in the student registration
    graph.
    """

    browser: BrowserType
    browser_session_store: BrowserSessionStore
    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    explanation_for_call: str
    redis_client: aioredis.Redis
    register_student_query: RegisterStudentQuery

    candidate_details_url: str = (
        "https://api.apprenticeshipindia.gov.in/auth/get-candidate-details-from-ncvt"
    )
    login_url: str = "https://www.apprenticeshipindia.gov.in/candidate-login"
    register_get_otp_url: str = (
        "https://api.apprenticeshipindia.gov.in/auth/register-get-otp"
    )
    register_otp_url: str = "https://api.apprenticeshipindia.gov.in/auth/register-otp"
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "Register Student Agent",
            "system": "System",
            "user": "Student",
        }
    )


@dataclass
class RegisterNewStudent(
    BaseNode[
        RegisterStudentState,
        RegisterStudentDeps,
        RegisterStudentResults | RegistrationCompleteResults,
    ]
):
    """This node registers a new student."""

    docstring_notes = True

    @staticmethod
    async def get_iti_student_details(
        *, ctx: GraphRunContext[RegisterStudentState, RegisterStudentDeps]
    ) -> tuple[Browser, Page, dict[str, Any]]:
        """Get ITI student details.

        The process is as follows:

        1. Launch a new browser page.
        2. Navigate to login URL, select the register radio button, and fill in the
            roll number for the student.
        3. Submit the form and capture the API response.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        tuple[Browser, Page, dict[str, Any]]
            The browser, page, and API response.
        """

        assert ctx.deps.register_student_query.is_iti_student

        # 1.
        browser = await ctx.deps.browser.launch(headless=PLAYWRIGHT_HEADLESS)
        page = await browser.new_page()

        # 2.
        await fill_roll_number(
            page=page,
            roll_number=ctx.deps.register_student_query.roll_number,
            url=ctx.deps.login_url,
        )

        # 3.
        response_json = await submit_and_capture_api_response(
            api_url=ctx.deps.candidate_details_url,
            button_name="Find Details",
            page=page,
        )
        response = await response_json

        return browser, page, response

    async def run(
        self, ctx: GraphRunContext[RegisterStudentState, RegisterStudentDeps]
    ) -> Annotated[
        End[RegisterStudentResults | RegistrationCompleteResults],
        Edge(label="Register a new student on the candidate portal"),
    ]:
        """The run proceeds as follows:

        1. If the student is an ITI student, get the ITI student details first.
            Otherwise, launch a new browser page.
        2. Load the preview page from the browser session store if it exists.
            2.1. If the page exists, fill in the OTP field.
            2.2. Submit the form and capture the API response.
            2.3. If the response is successful, extract the NAPS ID and activation link
                expiry date from the response. Otherwise, extract the error messages
                from the response.
            2.4. Persist the page in the browser session store.
            2.5. Reset the TTL of the browser session store.
        3. Navigate to the login URL, select the register radio button, and fill in the
            registration form.
        4. Solve the captcha and submit the form.
        5. Persist the page in the browser session store.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[RegisterStudentResults] | RegistrationCompleteResults]
            The results of the graph run or the next node in the graph run.
        """

        # 1.
        if ctx.deps.register_student_query.is_iti_student:
            browser, page, iti_student_details = await self.get_iti_student_details(
                ctx=ctx
            )
            logger.info(f"{iti_student_details = }")
            await asyncio.get_event_loop().run_in_executor(None, input)
        else:
            browser = await ctx.deps.browser.launch(headless=PLAYWRIGHT_HEADLESS)
            page = await browser.new_page()

        # 2.
        browser_session = await ctx.deps.browser_session_store.get(
            session_id=ctx.state.session_id
        )

        if browser_session:
            page = browser_session.page

            # 2.1.
            await fill_otp(
                otp=ctx.deps.register_student_query.otp
                or ctx.deps.register_student_query.user_query,
                page=page,
            )

            # 2.2.
            response_json = await submit_and_capture_api_response(
                api_url=ctx.deps.register_otp_url, button_name="Submit", page=page
            )
            response = await response_json

            # 2.3.
            if "errors" in response:
                error_messages = " ".join(response["errors"].values())
                end = End(  # type: ignore
                    RegisterStudentResults(  # type: ignore
                        next_chat_action=NextChatAction.GO_TO_HELPDESK,
                        session_id=ctx.state.session_id,
                        summary_of_page_results=f"Error in registration: {error_messages}",
                    )
                )
            else:
                assert response["status"] == "success"
                candidate_info = response.get("data", {}).get("candidate", {})
                naps_id = candidate_info["code"]
                activation_link_expiry = candidate_info["activation_link_expiry_date"]
                end = End(
                    RegistrationCompleteResults(  # type: ignore
                        activation_link_expiry=activation_link_expiry,
                        naps_id=naps_id,
                        session_id=ctx.state.session_id,
                        summary_of_page_results="Registration completed successfully. "
                        f"Your NAPS ID is {naps_id}. "
                        f"Please prompt the student to check their email for the "
                        f"activation link. Remind them that they must click on the "
                        f"link before {activation_link_expiry} to be able to log in, "
                        f"and complete their candidate profile in order to start "
                        f"applying for apprenticeships.",
                    )
                )

            # 2.4.
            await ctx.deps.browser_session_store.create(
                browser=browser_session.browser,  # Reuse the same browser here!
                overwrite=True,  # Update if the page changed at all!
                page=page,
                session_id=ctx.state.session_id,
            )
            browser_session_saved = await ctx.deps.browser_session_store.get(
                session_id=ctx.state.session_id
            )
            assert browser_session_saved, (
                f"Browser session not saved in RAM for session ID: "
                f"{ctx.state.session_id}"
            )

            # 2.5.
            await ctx.deps.browser_session_store.reset_ttl(
                session_id=ctx.state.session_id
            )
            return end  # type: ignore

        # 3.
        await page.goto(ctx.deps.login_url, wait_until="domcontentloaded")
        await select_register_radio(page=page)
        await page.fill(
            "input[placeholder='Enter your mobile number']",
            ctx.deps.register_student_query.mobile_number,
        )
        await page.fill(
            "input[placeholder='Enter Your Email ID']",
            str(ctx.deps.register_student_query.email),
        )
        await page.fill(
            "input[placeholder='Confirm Your Email ID']",
            str(ctx.deps.register_student_query.email),
        )

        # 4.
        try:
            _ = await solve_and_submit_captcha_with_retries(
                api_url=ctx.deps.register_get_otp_url, button_name="Register", page=page
            )
            message = (
                "Initiated account creation successfully. "
                "Please request OTP from the student. It should be sent to the "
                "mobile number or the email address."
            )
            next_action = NextChatAction.REQUEST_OTP
        except RuntimeError as e:
            message = f"Could not initiate registration. {str(e)}"
            next_action = NextChatAction.GO_TO_HELPDESK

        end = End(  # type: ignore
            RegisterStudentResults(  # type: ignore
                next_chat_action=next_action,
                session_id=ctx.state.session_id,
                summary_of_page_results=message,
            )
        )

        # 5.
        await ctx.deps.browser_session_store.create(
            browser=browser,
            overwrite=False,
            page=page,
            session_id=ctx.state.session_id,
        )
        browser_session_saved = await ctx.deps.browser_session_store.get(
            session_id=ctx.state.session_id
        )
        assert browser_session_saved, (
            f"Browser session not saved in RAM for session ID: "
            f"{ctx.state.session_id}"
        )

        return end  # type: ignore


@telemetry_timer(metric_fn=register_student_agent_hist, unit="s")
async def register_student(
    *,
    browser: BrowserType,
    browser_session_store: BrowserSessionStore,
    chatur_query: RegisterStudentQuery,
    csm: AsyncChatSessionManager,
    explanation_for_call: str = "No explanation provided.",
    generate_graph_diagram: bool = False,
    redis_client: aioredis.Redis,
    reset_chat_session: bool = False,
) -> RegisterStudentResults:
    """
    🔍 Account Creation For New Student Assistant

    **The student's registration information will be automatically provided to me for
    the registration process. You do not need to ask the student for this information!**

    ✅ WHEN TO USE THIS ASSISTANT
    Use this assistant **only** in the following situations:
        1. A **new student** needs a new account to be created on the Indian
            government apprenticeship portal---the goal is to create a **new user
            account**.
        2. An existing student indicates they wish to **register afresh and create a
            new account**.

    This assistant focuses **exclusively on registering students by creating accounts
    on the official portal.** **After calling this assistant, be sure to inform the
    student of the outcome**--whether the account creation was successful, if any
    required fields were missing, if additional documents are needed, or if there were
    system errors--before proceeding with further application steps.

    📝 HOW TO CALL THIS ASSISTANT
    Phrase your explanation as a **direct message** to the assistant. **Do not** use
    first-person language (e.g., “I will use...” or “I’m going to call...”). Instead,
    use imperative phrasing, such as:
        - "Create an account for a new student based on the following details..."
        - "Initiate the account creation process because..."

    Provide the following information in your explanation to this assistant:
        - A clear explanation of **why** you are calling this assistant, with reference
            to the latest student message and the current state of the application
            process. **Remember, this assistant does not directly interact with the
            student---thus, the assistant is not aware of the conversation that is
            going on between you and the student!**

    🚫 DO NOT USE THIS ASSISTANT IF
        - You are trying to **continue the registration process** for a student who
            already initiated the registration process.
        - You are trying to **log in a student who already has an account**.
        - You are trying to **continue the application or document submission** process
            for a student with an existing account.

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
    RegisterStudentResults
        The graph run result.
    """

    assert chatur_query.user_query or chatur_query.otp
    chatur_query = deepcopy(chatur_query)
    chatur_query.user_query = chatur_query.user_query or chatur_query.otp
    chatur_query.user_id = f"Register_Student_Agent_Graph_{chatur_query.user_id}"

    # 1. Initialize the chat history, chat parameters, and the session ID for the agent.
    chat_history, chat_params, session_id = await csm.init_chat_session(
        model_name="chat",
        namespace="register-student-agent",
        reset_chat_session=reset_chat_session,
        system_message=ChaturPrompts.system_messages["register_student_agent"],
        text_generation_params=TEXT_GENERATION_GEMINI,
        topic=None,
        user_id=chatur_query.user_id,
    )

    # 2. Create the graph.
    graph = Graph(
        auto_instrument=True,
        name="Register_Student_Agent_Graph",
        nodes=[RegisterNewStudent],
        state_type=RegisterStudentState,
    )

    # 3. Generate graph diagram.
    if generate_graph_diagram:
        save_graph_diagram(graph=graph)

    # 4. Set graph dependencies.
    deps = RegisterStudentDeps(
        browser=browser,
        browser_session_store=browser_session_store,
        chat_history=chat_history,
        chat_params=chat_params,
        explanation_for_call=explanation_for_call,
        redis_client=redis_client,
        register_student_query=chatur_query,
    )

    # 5. Set graph persistence.
    fsp = FullStatePersistence(deep_copy=True)
    fsp.set_graph_types(graph)

    # 6. Execute the graph until completion.
    graph_run_results = await graph.run(
        RegisterNewStudent(),
        deps=deps,
        persistence=fsp,
        state=RegisterStudentState(session_id=session_id),
    )

    # 7. Update the chat history.
    await csm.update_chat_history(chat_history=chat_history, session_id=session_id)
    await csm.dump_chat_session_to_file(session_id=session_id)

    # 8. Log the agent chat history at the end of each step (just for debugging
    # purposes).
    log_chat_history(
        chat_history=chat_history,
        context="Register Student Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output
