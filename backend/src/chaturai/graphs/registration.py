"""This module contains the student registration graph.

The student registration graph does the following:

1. Navigate to the `Register as a candidate` page and auto-fill the required fields.
2. If the student is an ITI student, then it will fill out the roll number:
    2.1 Find additional details regarding the student.
    2.2 XXX # TODO: Test with ITI roll number.
3. Solve the text-based CAPTCHA.
4. Submit the registration form.
5. Return the graph run results containing (among other things) the persisted **page**
    state for the next assistant.

NB: This graph does **not** handle the entire Chatur process (e.g., E-KYC, bank account
setup, profile completion, etc.). It simply registers a new student and forwards the
persisted page to the next assistant.
"""

# Standard Library

# Standard Library
import asyncio

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any

# Third Party Library
from playwright.async_api import async_playwright
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis

# Package Library
from chaturai.chatur.schemas import RegisterStudentQuery, RegisterStudentResults
from chaturai.chatur.utils import (
    select_register_radio,
    solve_and_fill_captcha,
    submit_and_capture_api_response,
)
from chaturai.config import Settings
from chaturai.graphs.utils import save_browser_state, save_graph_diagram
from chaturai.metrics.logfire_metrics import register_student_agent_hist
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.utils.chat import AsyncChatSessionManager, log_chat_history
from chaturai.utils.general import telemetry_timer

LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
REDIS_CACHE_PREFIX_BROWSER_STATE = Settings.REDIS_CACHE_PREFIX_BROWSER_STATE
REDIS_CACHE_PREFIX_GRAPH_STUDENT_REGISTRATION = (
    Settings.REDIS_CACHE_PREFIX_GRAPH_STUDENT_REGISTRATION
)
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

    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    explanation_for_call: str
    redis_client: aioredis.Redis
    register_student_query: RegisterStudentQuery

    login_url: str = "https://www.apprenticeshipindia.gov.in/candidate-login"
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "Register Student Agent",
            "system": "System",
            "user": "Student",
        }
    )


@dataclass
class GetITIStudentDetails(
    BaseNode[RegisterStudentState, RegisterStudentDeps, RegisterStudentResults]
):
    """This node obtains details on ITI students."""

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[RegisterStudentState, RegisterStudentDeps]
    ) -> Annotated[
        End[RegisterStudentResults], Edge(label="Obtain details on ITI students")
    ]:
        """The run proceeds as follows:

        1. Navigate to the page shell.
        2. Switch into Register mode.
        3. Fill out the roll number for the ITI student.
        4. XXX
        X. Save the browser state in Redis and close the browser.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[RegisterStudentResults]
            The results of the graph run.
        """

        assert ctx.deps.register_student_query.is_iti_student

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            page = await browser.new_page()

            # 1.
            await page.goto(ctx.deps.login_url, wait_until="domcontentloaded")

            # 2.
            await select_register_radio(page=page)

            # 3.
            await page.locator("label:has-text('ITI Student')").click(force=True)
            roll_selector = "input[placeholder*='Roll']"
            await page.wait_for_selector(roll_selector, state="visible")
            await page.fill(
                roll_selector, str(ctx.deps.register_student_query.roll_number)
            )

            # 4.
            response_json = await submit_and_capture_api_response(
                page=page,
                api_url="https://api.apprenticeshipindia.gov.in/auth/get-candidate-details-from-ncvt",
                button_name="Find Details",
            )
            response = await response_json

            if "errors" in response:
                end = End(
                    RegisterStudentResults(  # type: ignore
                        summary_of_page_results=f"Error in registration: {' '.join(response['errors'].values())}"
                    )
                )
            else:
                assert response["status"] == "success"
                end = End(  # type: ignore
                    RegisterStudentResults(  # type: ignore
                        summary_of_page_results="ITI student details obtained successfully. Shall I continue with the next step in the apprenticeship process for you?",
                    )
                )

            # Pause to verify in the browser.
            # await asyncio.get_event_loop().run_in_executor(None, input)

            # X.
            await save_browser_state(page=page, redis_client=ctx.deps.redis_client)

        return end


@dataclass
class RegisterNewStudent(
    BaseNode[
        RegisterStudentState,
        RegisterStudentDeps,
        RegisterStudentResults | GetITIStudentDetails,
    ]
):
    """This node registers a new student."""

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[RegisterStudentState, RegisterStudentDeps]
    ) -> End[RegisterStudentResults] | GetITIStudentDetails:
        """The run proceeds as follows:

        1. If we are registering an ITI student, then we proceed to the next node.
        2. Navigate to the page shell.
        3. Switch into Register mode.
        4. Fill out the rest of the register form.
        5. Solve CAPTCHA.
        6. Save the browser state in Redis and close the browser.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[RegisterStudentResults] | GetITIStudentDetails
            The results of the graph run or the next node in the graph run.
        """

        # 1.
        if ctx.deps.register_student_query.is_iti_student:
            return GetITIStudentDetails()

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)  # TODO: set to True
            page = await browser.new_page()

            # 2.
            await page.goto(ctx.deps.login_url, wait_until="domcontentloaded")

            # 3.
            await select_register_radio(page=page)

            # 4.
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

            # 5.
            await solve_and_fill_captcha(page=page)

            # TODO: Remove this
            # Pause to verify in the browser. Can remove this later.
            await asyncio.get_event_loop().run_in_executor(None, input)

            # 6.
            response_json = await submit_and_capture_api_response(
                page=page,
                api_url="https://api.apprenticeshipindia.gov.in/auth/register-get-otp",
                button_name="Register",
            )

            response = await response_json

            if "errors" in response:
                end = End(
                    RegisterStudentResults(  # type: ignore
                        summary_of_page_results=f"Error in registration: {' '.join(response['errors'].values())}"
                    )
                )
                # TODO: handle error cases better
            else:
                assert response["status"] == "success"
                end = End(  # type: ignore
                    RegisterStudentResults(  # type: ignore
                        summary_of_page_results="Initiated account creation successfully. "
                        "Please request OTP from the student. It should be sent to the "
                        "mobile number or the email address."
                    )
                )

            # Pause to verify in the browser. Can remove this later.
            # await asyncio.get_event_loop().run_in_executor(None, input)

            # X.
            await save_browser_state(page=page, redis_client=ctx.deps.redis_client)

        return end


@telemetry_timer(metric_fn=register_student_agent_hist, unit="s")
async def register_student(
    *,
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

    chatur_query = deepcopy(chatur_query)
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
        nodes=[RegisterNewStudent, GetITIStudentDetails],
        state_type=RegisterStudentState,
    )

    # 3. Generate graph diagram.
    if generate_graph_diagram:
        save_graph_diagram(graph=graph)

    # 4. Set graph dependencies.
    deps = RegisterStudentDeps(
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

    log_chat_history(
        chat_history=chat_history,
        context="Register Student Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output
