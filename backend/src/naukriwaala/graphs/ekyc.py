"""This module contains the E-KYC graph."""

# Standard Library
import asyncio
import json

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

# Third Party Library
from playwright.async_api import async_playwright
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

# Package Library
from naukriwaala.config import Settings
from naukriwaala.ekyc.schemas import EKYCQuery, EKYCResults
from naukriwaala.ekyc.utils import (
    select_login_radio,
    select_register_radio,
    solve_captcha,
)
from naukriwaala.graphs.utils import (
    load_graph_run_results,
    save_graph_diagram,
    save_graph_run_results,
)
from naukriwaala.metrics.logfire_metrics import ekyc_agent_hist
from naukriwaala.prompts.ekyc import EKYCPrompts
from naukriwaala.utils.chat import AsyncChatSessionManager, log_chat_history
from naukriwaala.utils.general import telemetry_timer

# Globals.
LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
REDIS_CACHE_PREFIX_EKYC_GRAPH = Settings.REDIS_CACHE_PREFIX_EKYC_GRAPH
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@dataclass
class EKYCState:
    """The state tracks the progress of the DA graph."""

    session_id: int | str

    ekyc_queries: list[EKYCQuery] = field(default_factory=list)
    last_graph_run_results: list | None = None
    last_graph_run_results_cache_key: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization processes."""

        # 1.
        self.last_graph_run_results_cache_key = f"{REDIS_CACHE_PREFIX_EKYC_GRAPH}_EKYC_Agent_last_graph_run_results_{self.session_id}"


@dataclass
class EKYCDeps:
    """This class contains dependencies used by nodes in the DA graph."""

    asession: AsyncSession
    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    csm: AsyncChatSessionManager
    ekyc_query: EKYCQuery
    embedding_model: Any
    redis_client: aioredis.Redis

    inner_thought_template: str = """-- MY OWN INNER THOUGHT START --

{inner_thought}

-- MY OWN INNER THOUGHT END --
        """
    generate_graph_diagrams: bool = False
    login_url: str = "https://www.apprenticeshipindia.gov.in/candidate-login"
    reset_chat_session: bool = False
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "EKYC Agent",
            "system": "System",
            "user": "Student",
        }
    )


@dataclass
class LoginExistingStudent(BaseNode[EKYCState, EKYCDeps, EKYCResults]):
    """This node logs an existing student into the candidate portal and continues the
    registration process.
    """

    docstring_notes = True

    async def run(self, ctx: GraphRunContext[EKYCState, EKYCDeps]) -> Annotated[
        End[EKYCResults],
        Edge(label="Login an existing student and continue the registration process"),
    ]:
        """The run proceeds as follows:

        1. Navigate to the page shell.
        2. Switch into Login mode.
        3. Capture the CAPTCHA canvas exactly.
        4. Solve the CAPTCHA text.
        5. Fill out the CAPTCHA text.
        6. XXX
        X. Cache the last graph run results in Redis for the frontend.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[EKYCResults]
            The results of the graph run.
        """

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            page = await browser.new_page()

            # 1.
            await page.goto(ctx.deps.login_url, wait_until="domcontentloaded")

            # 2.
            await select_login_radio(page=page)

            # 3.
            canvas = page.locator("canvas.captcha-canvas")
            await canvas.wait_for(state="visible")
            captcha_bytes: bytes = await canvas.screenshot()

            # 4.
            captcha_text = await solve_captcha(captcha_bytes=captcha_bytes)

            # 5.
            await page.fill('input[placeholder="Enter CAPTCHA"]', captcha_text)

            # 6.

            # Pause to verify in the browser.
            await asyncio.get_event_loop().run_in_executor(None, input)

            await browser.close()

        return End(  # type: ignore
            EKYCResults(  # type: ignore
                **ctx.deps.ekyc_query.model_dump(),
                last_graph_run_results=ctx.state.last_graph_run_results,
            )
        )


@dataclass
class GetITIStudentDetails(BaseNode[EKYCState, EKYCDeps, EKYCResults]):
    """This node obtains details on ITI students."""

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[EKYCState, EKYCDeps]
    ) -> Annotated[End[EKYCResults], Edge(label="Obtain details on ITI students")]:
        """The run proceeds as follows:

        1. Navigate to the page shell.
        2. Switch into Register mode.
        3. Fill out the roll number for the ITI student.
        4. XXX
        X. Cache the last graph run results in Redis for the frontend.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[EKYCResults]
            The results of the graph run.
        """

        assert ctx.deps.ekyc_query.is_iti_student

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
            await page.fill(roll_selector, str(ctx.deps.ekyc_query.roll_number))

            # 4.

            # Pause to verify in the browser.
            await asyncio.get_event_loop().run_in_executor(None, input)

            await browser.close()

        return End(  # type: ignore
            EKYCResults(  # type: ignore
                **ctx.deps.ekyc_query.model_dump(),
                last_graph_run_results=ctx.state.last_graph_run_results,
            )
        )


@dataclass
class RegisterNewStudent(
    BaseNode[EKYCState, EKYCDeps, EKYCResults | GetITIStudentDetails]
):
    """This node registers a new student."""

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[EKYCState, EKYCDeps]
    ) -> End[EKYCResults] | GetITIStudentDetails:
        """The run proceeds as follows:

        1. Navigate to the page shell.
        2. Switch into Register mode.
        3. Fill out the rest of the register form.
        4. XXX
        X. Cache the last graph run results in Redis for the frontend.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[EKYCResults] | GetITIStudentDetails
            The results of the graph run or the next node in the graph run.
        """

        if ctx.deps.ekyc_query.is_iti_student:
            return GetITIStudentDetails()

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            page = await browser.new_page()

            # 1.
            await page.goto(ctx.deps.login_url, wait_until="domcontentloaded")

            # 2.
            await select_register_radio(page=page)

            # 3.
            await page.fill(
                "input[placeholder='Enter your mobile number']",
                ctx.deps.ekyc_query.mobile_number,
            )
            await page.fill(
                "input[placeholder='Enter Your Email ID']",
                str(ctx.deps.ekyc_query.email),
            )
            await page.fill(
                "input[placeholder='Confirm Your Email ID']",
                str(ctx.deps.ekyc_query.email),
            )

            # 4.

            # Pause to verify in the browser.
            await asyncio.get_event_loop().run_in_executor(None, input)

            await browser.close()

        # X.
        await save_graph_run_results(
            graph_run_results=ctx.state.last_graph_run_results,
            redis_cache_key=ctx.state.last_graph_run_results_cache_key,
            redis_client=ctx.deps.redis_client,
        )

        return End(  # type: ignore
            EKYCResults(  # type: ignore
                **ctx.deps.ekyc_query.model_dump(),
                last_graph_run_results=ctx.state.last_graph_run_results,
            )
        )


@dataclass
class SelectLoginOrRegister(
    BaseNode[EKYCState, EKYCDeps, RegisterNewStudent | LoginExistingStudent]
):
    """This node determines whether the student can login as a candidate or needs to
    first register as a candidate on the candidate login page.
    """

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[EKYCState, EKYCDeps]
    ) -> RegisterNewStudent | LoginExistingStudent:
        """The run proceeds as follows:

        1. Determine whether the student can login as a candidate or needs to first
            register as a candidate.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        RegisterNewStudent | LoginExistingStudent
            The next node in the graph run.
        """

        if ctx.deps.ekyc_query.is_new_student:
            return RegisterNewStudent()
        return LoginExistingStudent()


@telemetry_timer(metric_fn=ekyc_agent_hist, unit="s")
async def ekyc_graph(
    *,
    asession: AsyncSession,
    csm: AsyncChatSessionManager,
    ekyc_query: EKYCQuery,
    embedding_model: Any,
    generate_graph_diagrams: bool = False,
    redis_client: aioredis.Redis,
    reset_chat_session: bool = False,
) -> EKYCResults:
    """Help a student with the Electronic-Know Your Customer process.

    The process is as follows:

    1. Initialize the chat history, chat parameters, and the session ID for the
        E-KYC agent.
    2. Create the E-KYC Agent graph.
    3. Generate the graph diagram (optional).
    4. Set graph dependencies.
    5. Set graph persistence.
    6. Load the appropriate graph state.
    7. Execute the graph until completion.
    8. Update the chat history for the E-KYC agent.
    9. Save the graph snapshot to Redis.
    10. Log the E-KYC agent chat history at the end of each step.

    Parameters
    ----------
    asession
        A Neo4j async session.
    csm
        An async chat session manager that manages the chat sessions for each user.
    ekyc_query
        The E-KYC query object.
    embedding_model
        The embedding model to use for the graph.
    generate_graph_diagrams
        Specifies whether to generate ALL graph diagrams using the Mermaid API (free).
    redis_client
        The Redis client.
    reset_chat_session
        Specìfies whether to reset the chat session for the agent.

    Returns
    -------
    EKYCResults
        The graph run result.
    """

    ekyc_query = deepcopy(ekyc_query)
    ekyc_query.user_id = f"EKYC_Agent_Graph_{ekyc_query.user_id}"

    # 1.
    chat_history, chat_params, session_id = await csm.init_chat_session(
        model_name="chat",
        namespace="ekyc-agent",
        reset_chat_session=reset_chat_session,
        system_message=EKYCPrompts.system_messages["ekyc_agent"],
        text_generation_params=TEXT_GENERATION_GEMINI,
        topic=None,
        user_id=ekyc_query.user_id,
    )

    # 2.
    graph = Graph(
        auto_instrument=True,
        name="EKYC_Agent_Graph",
        nodes=[
            SelectLoginOrRegister,
            RegisterNewStudent,
            GetITIStudentDetails,
            LoginExistingStudent,
        ],
        state_type=EKYCState,
    )

    # 3.
    if generate_graph_diagrams:
        save_graph_diagram(graph=graph)

    # 4.
    deps = EKYCDeps(
        asession=asession,
        chat_history=chat_history,
        chat_params=chat_params,
        csm=csm,
        ekyc_query=ekyc_query,
        embedding_model=embedding_model,
        generate_graph_diagrams=generate_graph_diagrams,
        redis_client=redis_client,
        reset_chat_session=reset_chat_session,
    )

    # 5.
    fsp = FullStatePersistence(deep_copy=True)
    fsp.set_graph_types(graph)

    # 6.
    redis_cache_key, state = await load_state(
        ekyc_query=ekyc_query,
        persistence=fsp,
        redis_client=redis_client,
        reset_state=reset_chat_session,
        session_id=session_id,
    )

    # 7.
    graph_run_results = await graph.run(
        SelectLoginOrRegister(), deps=deps, persistence=fsp, state=state
    )

    # 8.
    await csm.update_chat_history(chat_history=chat_history, session_id=session_id)
    await csm.dump_chat_session_to_file(session_id=session_id)

    # 9.
    snapshot_json = fsp.dump_json()
    await redis_client.set(redis_cache_key, snapshot_json)

    # 10.
    log_chat_history(
        chat_history=chat_history,
        context="EKYC Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output


async def load_state(
    *,
    ekyc_query: EKYCQuery,
    persistence: FullStatePersistence,
    redis_client: aioredis.Redis,
    reset_state: bool,
    session_id: int | str,
) -> tuple[str, EKYCState]:
    """Load state for the E-KYC agent graph.

    The process is as follows:

    1. If a previous state does not exist, then we initialize a new state with the EKYC
        query. Otherwise,
    2. We append the latest EKYC query to the list.
    3. We pull any changes to graph run results made by the frontend from Redis. Note
        that we only pull changes for the last graph run results since the frontend
        should never see more than one set of results at a time.

    Parameters
    ----------
    ekyc_query
        The EKYC query object.
    persistence
        The persistence object for the graph.
    redis_client
        The Redis client.
    reset_state
        Specifies whether to reset the state.
    session_id
        The session ID for the graph.

    Returns
    -------
    tuple[str, EKYCState]
        The Redis cache key and the state for the EKYC agent graph.
    """

    redis_cache_key = f"{REDIS_CACHE_PREFIX_EKYC_GRAPH}_EKYC_Agent_{session_id}"

    # 1.
    if reset_state or not await redis_client.exists(redis_cache_key):
        return redis_cache_key, EKYCState(
            session_id=session_id, ekyc_queries=[ekyc_query]
        )

    raw_snapshot = await redis_client.get(redis_cache_key)
    persistence.load_json(raw_snapshot)
    snapshot = await persistence.load_all()
    state = snapshot[-1].state

    # 2.
    state.ekyc_queries.append(ekyc_query)

    # 3.
    last_graph_run_results = await redis_client.get(
        state.last_graph_run_results_cache_key
    )
    state.last_graph_run_results = load_graph_run_results(
        graph_run_results=json.loads(last_graph_run_results), model_class=EKYCResults
    )

    return redis_cache_key, state
