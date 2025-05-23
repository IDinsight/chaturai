"""This module contains the Chatur graph."""

# Standard Library
import json

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

# Third Party Library
from playwright.async_api import BrowserType
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import FullStatePersistence
from redis import asyncio as aioredis

# Package Library
from chaturai.chatur.schemas import (
    BaseQuery,
    ChaturFlowResults,
    LoginStudentResults,
    NextChatAction,
    ProfileCompletionResults,
    RegisterStudentResults,
)
from chaturai.config import Settings
from chaturai.graphs.login import login_student
from chaturai.graphs.profile import complete_profile
from chaturai.graphs.registration import register_student
from chaturai.graphs.utils import (
    create_graph_mappings,
    load_graph_run_results,
    save_graph_diagram,
    save_graph_run_results,
)
from chaturai.metrics.logfire_metrics import (
    assistants_counter,
    chatur_agent_hist,
    student_counter,
)
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.schemas import ValidatorCall
from chaturai.utils.browser import BrowserSessionStore
from chaturai.utils.chat import (
    AsyncChatSessionManager,
    append_message_content_to_chat_history,
    get_chat_response,
    log_chat_history,
    prettify_chat_history,
)
from chaturai.utils.general import telemetry_timer
from chaturai.utils.litellm_ import get_acompletion

AGENTS_CHATUR_AGENT = Settings.AGENTS_CHATUR_AGENT
AGENTS_LOGIN_STUDENT = Settings.AGENTS_LOGIN_STUDENT
AGENTS_PROFILE_COMPLETION = Settings.AGENTS_PROFILE_COMPLETION
AGENTS_REGISTER_STUDENT = Settings.AGENTS_REGISTER_STUDENT
GRAPHS_CHATUR_AGENT = Settings.GRAPHS_CHATUR_AGENT
GRAPHS_LOGIN_STUDENT = Settings.GRAPHS_LOGIN_STUDENT
GRAPHS_PROFILE_COMPLETION = Settings.GRAPHS_PROFILE_COMPLETION
GRAPHS_REGISTER_STUDENT = Settings.GRAPHS_REGISTER_STUDENT
LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
REDIS_CACHE_PREFIX_CHATUR_AGENT = Settings.REDIS_CACHE_PREFIX_CHATUR_AGENT
REDIS_CACHE_PREFIX_LOGIN_STUDENT = Settings.REDIS_CACHE_PREFIX_LOGIN_STUDENT
REDIS_CACHE_PREFIX_PROFILE_COMPLETION = Settings.REDIS_CACHE_PREFIX_PROFILE_COMPLETION
REDIS_CACHE_PREFIX_REGISTER_STUDENT = Settings.REDIS_CACHE_PREFIX_REGISTER_STUDENT
TEXT_GENERATION_BEDROCK = Settings.TEXT_GENERATION_BEDROCK
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@dataclass
class ChaturState:
    """The state tracks the progress of the Chatur graph."""

    last_assistant_call: str | None = None
    last_graph_run_results: Any = None
    next_chat_action: NextChatAction = NextChatAction.REQUEST_USER_QUERY


@dataclass
class ChaturDeps:
    """This class contains dependencies used by nodes in the Chatur graph."""

    browser: BrowserType
    browser_session_store: BrowserSessionStore
    chat_history: list[dict[str, str | None]]
    chat_params: dict[str, Any]
    chatur_query: BaseQuery
    csm: AsyncChatSessionManager
    redis_client: aioredis.Redis
    session_id: int | str

    inner_thought_template: str = """-- MY OWN INNER THOUGHT START --

{inner_thought}

-- MY OWN INNER THOUGHT END --
        """
    generate_graph_diagrams: bool = False
    last_graph_run_results_cache_key: str | None = None
    reset_chat_session: bool = False
    role_labels: dict[str, str] = field(
        default_factory=lambda: {
            "assistant": "Chatur Agent",
            "system": "System",
            "tool": "Support Assistant",
            "user": "Student",
        }
    )

    def __post_init__(self) -> None:
        """Post-initialization processes."""

        self.last_graph_run_results_cache_key = f"{REDIS_CACHE_PREFIX_CHATUR_AGENT}_last_graph_run_results_{self.session_id}"


@dataclass
class SelectStudentOrAssistant(BaseNode[ChaturState, ChaturDeps, ChaturFlowResults]):
    """This node selects either an assistant tool graph or terminates graph execution
    in order to get physician input.
    """

    student_inner_thoughts: str
    student_intent: Literal["proceed", "revert"]

    _default_explanation: str = "No explanation available."
    _default_first_assistant: str = "registration.register_student"
    docstring_notes = True
    summary_of_last_assistant_call: str | None = None

    def _update_system_message(
        self, *, ctx: GraphRunContext[ChaturState, ChaturDeps]
    ) -> None:
        """Update the system message for the Chatur agent with the correct set of
        assistant names and descriptions based on the student's intent. The available
        assistants depend on both the student's intent and the adjacency lists of the
        Chatur agent graph.

        NB: The assumption here is that the very first message in the Chatur agent chat
        history corresponds to the system message!

        NB: If there is no last assistant call, then we assume that the only assistant
        available is the default assistant.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.
        """

        graph_mapping = Settings._INTERNAL_GRAPH_MAPPING or create_graph_mappings()
        key = "adj_list" if self.student_intent == "proceed" else "adj_list_reverse"
        if ctx.state.last_assistant_call is None:
            valid_assistants = graph_mapping[self._default_first_assistant][key]
        else:
            valid_assistants = graph_mapping[ctx.state.last_assistant_call][key]
        assistant_names_and_descriptions = [
            graph_mapping[assistant]["description"] for assistant in valid_assistants
        ]
        assistant_names_and_descriptions_str = "\n---\n\n".join(
            assistant_names_and_descriptions
        )
        ctx.deps.chat_history[0]["content"] = ChaturPrompts.system_messages[
            "chatur_agent"
        ].format(assistant_names_and_descriptions=assistant_names_and_descriptions_str)

    async def call_next_assistant(
        self,
        *,
        assistant_name: str,
        ctx: GraphRunContext[ChaturState, ChaturDeps],
        explanation_for_assistant_call: str,
    ) -> None:
        """Call the next assistant in the graph and get its graph run results. For each
        assistant call, the results from its graph run are summarized by the Chatur
        agent and maintained in its chat history.

        Parameters
        ----------
        assistant_name
            The name of the assistant to call.
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.
        explanation_for_assistant_call
            An explanation as to why the assistant is being called.

        Raises
        ------
        ValueError
            If the assistant name is invalid.
        """

        match assistant_name:
            case "registration.register_student":
                graph_run_results = await register_student(
                    browser=ctx.deps.browser,
                    browser_session_store=ctx.deps.browser_session_store,
                    chatur_query=ctx.deps.chatur_query,
                    csm=ctx.deps.csm,
                    generate_graph_diagram=ctx.deps.generate_graph_diagrams,
                    redis_client=ctx.deps.redis_client,
                    reset_chat_session=ctx.deps.reset_chat_session,
                )
            case "login.login_student":
                graph_run_results = await login_student(
                    browser=ctx.deps.browser,
                    browser_session_store=ctx.deps.browser_session_store,
                    chatur_query=ctx.deps.chatur_query,
                    csm=ctx.deps.csm,
                    generate_graph_diagram=ctx.deps.generate_graph_diagrams,
                    redis_client=ctx.deps.redis_client,
                    reset_chat_session=ctx.deps.reset_chat_session,
                )
            case "profile.complete_profile":
                graph_run_results = await complete_profile(
                    browser_session_store=ctx.deps.browser_session_store,
                    chatur_query=ctx.deps.chatur_query,
                    csm=ctx.deps.csm,
                    generate_graph_diagram=ctx.deps.generate_graph_diagrams,
                    last_graph_run_results=ctx.state.last_graph_run_results,
                    redis_client=ctx.deps.redis_client,
                    reset_chat_session=ctx.deps.reset_chat_session,
                )
            case _:
                raise ValueError(f"Invalid assistant name: {assistant_name}")

        assistants_counter.add(
            1,
            {
                "explanation_for_assistant_call": explanation_for_assistant_call,
                "name": assistant_name,
            },
        )
        ctx.state.last_assistant_call = assistant_name
        ctx.state.last_graph_run_results = graph_run_results

        content = await get_acompletion(
            model=LITELLM_MODEL_CHAT,
            system_msg=ChaturPrompts.system_messages["summarize_assistant_response"],
            text_generation_params=ctx.deps.chat_params["text_generation_params"],
            user_msg=ChaturPrompts.prompts["summarize_assistant_response"].format(
                assistant_call_results=ctx.state.last_graph_run_results,
                explanation_for_assistant_call=explanation_for_assistant_call,
                student_message=ctx.deps.chatur_query.user_query_translated,
            ),
        )
        self.summary_of_last_assistant_call = content

    async def determine_next_step(
        self, *, ctx: GraphRunContext[ChaturState, ChaturDeps]
    ) -> tuple[Optional[str], str, str, bool]:
        """Determine the next step in the Chatur process. This would either be an
        assistant call or a request for student input.

        NB: In a given conversation turn between the **student and the Chatur agent**,
        the student's query is only used as the message for the first time that this
        method is called. Within the same conversation **turn**, subsequent calls to
        this method, would mean that one or more assistant calls have been made and,
        now, the Chatur agent needs to make the decision on the next step based on not
        only the student's last query but also the assistant call results.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        tuple[Optional[str], str, str, bool]
            A tuple containing the name of the next assistant to call, the explanation
            for calling the assistant, the explanation for calling the student, and a
            boolean indicating whether student input is required.
        """

        if self.summary_of_last_assistant_call is None:
            message = ChaturPrompts.prompts["chatur_agent"].format(
                student_inner_thoughts=self.student_inner_thoughts,
                student_message=ctx.deps.chatur_query.user_query_translated
                or ctx.deps.chatur_query.otp,
            )
        else:
            message = (
                f"Here is the summary of the results from the last assistant call that "
                f"I made for you:\n\n{self.summary_of_last_assistant_call}"
            )

        content = await get_chat_response(
            chat_history=ctx.deps.chat_history,
            chat_params=ctx.deps.chat_params,
            litellm_model=LITELLM_MODEL_CHAT,
            message=message,
            remove_json_strs=True,
            role="user",
            session_id=ctx.deps.session_id,
            validator_call=ValidatorCall(num_retries=3, validator_module=json.loads),
        )
        json_response = json.loads(content)

        log_chat_history(
            chat_history=ctx.deps.chat_history,
            context="Chatur Agent: after determining next step",
            session_id=ctx.deps.session_id,
        )

        await ctx.deps.csm.update_chat_history(
            chat_history=ctx.deps.chat_history, session_id=ctx.deps.session_id
        )
        await ctx.deps.csm.dump_chat_session_to_file(session_id=ctx.deps.session_id)

        next_step = json_response["next_step"]
        assistant_name = next_step.get("assistant_name", None)
        explanation_for_assistant_call = next_step.get(
            "explanation_for_assistant_call", self._default_explanation
        )
        explanation_for_student = next_step.get(
            "explanation_for_student", self._default_explanation
        )
        require_student_input = next_step.get("require_student_input", True)

        return (
            assistant_name,
            explanation_for_assistant_call,
            explanation_for_student,
            require_student_input,
        )

    @staticmethod
    async def generate_self_summary(
        *, ctx: GraphRunContext[ChaturState, ChaturDeps]
    ) -> None:
        """Generate a summary of the Chatur process so far, **for the Chatur agent.**
        This is an internal summarization meant to help keep track of the conversation
        for the Chatur agent and is not presented to the student.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.
        """

        chat_history_str = prettify_chat_history(
            chat_history=ctx.deps.chat_history, role_labels=ctx.deps.role_labels
        )
        content = await get_acompletion(
            model=LITELLM_MODEL_CHAT,
            system_msg=ChaturPrompts.system_messages["summarize_chatur_process"],
            text_generation_params=ctx.deps.chat_params["text_generation_params"],
            user_msg=ChaturPrompts.prompts["summarize_chatur_process"].format(
                conversation_history=chat_history_str,
            ),
        )
        append_message_content_to_chat_history(
            chat_history=ctx.deps.chat_history,
            message_content=ctx.deps.inner_thought_template.format(
                inner_thought=content
            ),
            model=ctx.deps.chat_params["model"],
            model_context_length=ctx.deps.chat_params["max_input_tokens"],
            name=ctx.deps.session_id,
            role="assistant",
            total_tokens_for_next_generation=ctx.deps.chat_params["max_output_tokens"],
        )

        log_chat_history(
            chat_history=ctx.deps.chat_history,
            context="Chatur Agent: after process summary",
            session_id=ctx.deps.session_id,
        )

    async def run(self, ctx: GraphRunContext[ChaturState, ChaturDeps]) -> Annotated[
        End[ChaturFlowResults],
        Edge(label="Select either an assistant call or ask for student input"),
    ]:
        """The run proceeds as follows:

        1. Update the Chatur agent system prompt with the correct set of assistant
            names and descriptions based on the student's intent. The available
            assistants depends on both the student's intent and the adjacency lists of
            the Chatur agent graph.
        2. Ask the Chatur agent to determine the next step in the Chatur process. This
            would either be an assistant call or a request for student input.
            2a. In order to avoid infinite loops, we always assume that student input
                is required if the key is not in the LLM JSON response. Furthermore, if
                both student input is required and an assistant call are specified,
                then we default back to requiring student input.
        3. If student input is required, then we stop the Chatur process.
        4. Otherwise, we call the next assistant in the graph and get its graph run
            results.
            4a. For each assistant call, the results from its graph run are summarized
                by the Chatur agent and maintained in its chat history.
            4b. Update the Chatur agent system prompt with the correct set of assistant
                names and descriptions for the next step in the Chatur process. Note
                that the student's intent is always set to "proceed" since
                `require_student_input` must be `False` at this point.
        5. Cache the last graph run results in Redis for the frontend.
        6. A summary of the Chatur process so far is generated by the Chatur agent,
            **for** the Chatur agent.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        End[ChaturFlowResults]
            The results of the graph run.
        """

        # 1.
        self._update_system_message(ctx=ctx)

        while True:
            # 2, 2a.
            (
                assistant_name,
                explanation_for_assistant_call,
                explanation_for_student,
                require_student_input,
            ) = await self.determine_next_step(ctx=ctx)

            # 3.
            if (
                require_student_input
                or not isinstance(assistant_name, str)
                or assistant_name not in Settings._INTERNAL_GRAPH_MAPPING
            ):
                student_counter.add(
                    1, {"explanation_for_student": explanation_for_student}
                )
                break

            # 4, 4a.
            await self.call_next_assistant(
                assistant_name=assistant_name,
                ctx=ctx,
                explanation_for_assistant_call=explanation_for_assistant_call,
            )

            # 4b.
            self.student_intent = "proceed"
            self._update_system_message(ctx=ctx)

        # 5.
        await save_graph_run_results(
            graph_run_results=ctx.state.last_graph_run_results,
            redis_cache_key=ctx.deps.last_graph_run_results_cache_key,
            redis_client=ctx.deps.redis_client,
        )

        # 6.
        await self.generate_self_summary(ctx=ctx)

        explanation_for_student_input = (
            explanation_for_student or self._default_explanation
        )
        if ctx.state.last_graph_run_results is not None:
            next_chat_action = ctx.state.last_graph_run_results.next_chat_action
        else:
            next_chat_action = NextChatAction.REQUEST_USER_QUERY

        summary_for_student = (
            self.summary_of_last_assistant_call or explanation_for_student_input
        )
        return End(  # type: ignore
            ChaturFlowResults(  # type: ignore
                explanation_for_student_input=explanation_for_student_input,
                last_assistant_call=ctx.state.last_assistant_call,
                last_graph_run_results=ctx.state.last_graph_run_results,
                require_student_input=True,  # Always True for now
                session_id=ctx.deps.session_id,
                summary_for_student=summary_for_student,
                summary_for_student_translated=summary_for_student,
                user_id=ctx.deps.chatur_query.user_id,
                next_chat_action=next_chat_action,
            )
        )


@dataclass
class DetermineStudentIntent(BaseNode[ChaturState, ChaturDeps, dict[str, Any]]):
    """This node determines the student's intent at each step in the Chatur process.
    The student can either choose to proceed with the process or to revert back to a
    previous step in the process.
    """

    docstring_notes = True

    async def run(self, ctx: GraphRunContext[ChaturState, ChaturDeps]) -> Annotated[
        SelectStudentOrAssistant,
        Edge(
            label="Determine the student's intent at the current step in the Chatur "
            "process"
        ),
    ]:
        """The run proceeds as follows:

        1. Determine the student's intent before proceeding with the process.

        Parameters
        ----------
        ctx
            The context for the graph run, which contains the state and dependencies
            for the graph run.

        Returns
        -------
        SelectStudentOrAssistant
            The next node in the graph run.
        """

        chat_history = ctx.deps.chat_history
        if len(chat_history) == 1:
            return SelectStudentOrAssistant(
                student_inner_thoughts="No inner thoughts available.",
                student_intent="proceed",
            )

        chat_history = deepcopy(chat_history[1:])  # Don't copy system message
        chat_history.append(
            {
                "content": ctx.deps.chatur_query.user_query_translated,
                "name": str(ctx.deps.session_id),
                "role": "user",
            }
        )
        chat_history_str = prettify_chat_history(
            chat_history=chat_history, role_labels=ctx.deps.role_labels
        )
        system_msg = ChaturPrompts.system_messages["determine_student_intent"]
        user_msg = ChaturPrompts.prompts["determine_student_intent"].format(
            conversation_history=chat_history_str
        )
        content = await get_acompletion(
            model=LITELLM_MODEL_CHAT,
            remove_json_strs=True,
            system_msg=system_msg,
            text_generation_params=ctx.deps.chat_params["text_generation_params"],
            user_msg=user_msg,
            validator_call=ValidatorCall(num_retries=3, validator_module=json.loads),
        )
        json_response = json.loads(content)

        return SelectStudentOrAssistant(
            student_inner_thoughts=json_response["reason"],
            student_intent=json_response["student_intent"],
        )


@telemetry_timer(metric_fn=chatur_agent_hist, unit="s")
async def chatur(
    *,
    browser: BrowserType,
    browser_session_store: BrowserSessionStore,
    chatur_query: BaseQuery,
    csm: AsyncChatSessionManager,
    generate_graph_diagrams: bool = False,
    redis_client: aioredis.Redis,
    reset_chat_and_graph_state: bool = False,
) -> ChaturFlowResults:
    """Help a student onboard and apply to apprenticeships.

    The process is as follows:

    1. Reset the chat session and graph state for all assistants if specified. This has
        to be done by the main chatur agent graph since this graph is the primary entry
        point for the other graphs. Otherwise, the other graphs cannot reset their chat
        sessions and graph state for subsequent calls.
    2. Initialize the chat history, chat parameters, and the session ID for the agent.
    3. Create the agent graph.
    4. Generate the graph diagram (optional).
    5. Set graph dependencies.
    6. Set graph persistence.
    7. Load the appropriate graph state.
    8. Execute the graph until completion.
    9. Update the chat history for the agent.
    10. Save the graph snapshot to Redis.
    11. Log the agent chat history at the end of each step (just for debugging
        purposes).

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
    generate_graph_diagrams
        Specifies whether to generate ALL graph diagrams using the Mermaid API (free).
    redis_client
        The Redis client.
    reset_chat_and_graph_state
        Specifies whether to reset the chat session and the graph state for the user.
        This can be used to clear the chat history and graph state, effectively
        starting a completely new session. This is useful for testing or debugging
        purposes.

    Returns
    -------
    ChaturFlowResults
        The graph run result.
    """

    chatur_query = deepcopy(chatur_query)
    chatur_query.user_id = f"{GRAPHS_CHATUR_AGENT}_{chatur_query.user_id}"

    # 1.
    await reset_assistants(
        chatur_query_user_id=chatur_query.user_id,
        csm=csm,
        redis_client=redis_client,
        reset_chat_and_graph_state=reset_chat_and_graph_state,
    )

    # 2.
    chat_namespace = AGENTS_CHATUR_AGENT
    chat_history, chat_params, session_id = await csm.init_chat_session(
        model_name="chat",
        namespace=chat_namespace,
        reset_chat_session=reset_chat_and_graph_state,
        system_message=ChaturPrompts.system_messages["chatur_agent"],
        text_generation_params=TEXT_GENERATION_BEDROCK,
        topic=None,
        user_id=chatur_query.user_id,
    )

    # 3.
    graph = Graph(
        auto_instrument=True,
        name=GRAPHS_CHATUR_AGENT,
        nodes=[DetermineStudentIntent, SelectStudentOrAssistant],
        state_type=ChaturState,
    )

    # 4.
    if generate_graph_diagrams:
        save_graph_diagram(graph=graph)

    # 5.
    deps = ChaturDeps(
        browser=browser,
        browser_session_store=browser_session_store,
        chat_history=chat_history,
        chat_params=chat_params,
        chatur_query=chatur_query,
        csm=csm,
        generate_graph_diagrams=generate_graph_diagrams,
        redis_client=redis_client,
        reset_chat_session=reset_chat_and_graph_state,
        session_id=session_id,
    )

    # 6.
    fsp = FullStatePersistence(deep_copy=True)
    fsp.set_graph_types(graph)

    # 7.
    redis_cache_key, state = await load_state(
        last_graph_run_results_cache_key=deps.last_graph_run_results_cache_key,
        persistence=fsp,
        redis_client=redis_client,
        reset_state=reset_chat_and_graph_state,
        session_id=session_id,
    )

    # 8.
    graph_run_results = await graph.run(
        DetermineStudentIntent(), deps=deps, persistence=fsp, state=state
    )

    # 9.
    await csm.update_chat_history(chat_history=chat_history, session_id=session_id)
    await csm.dump_chat_session_to_file(session_id=session_id)

    # 10.
    snapshot_json = fsp.dump_json()
    await redis_client.set(redis_cache_key, snapshot_json)

    # 11.
    log_chat_history(
        chat_history=chat_history,
        context="Chatur Agent: END",
        session_id=session_id,
    )

    return graph_run_results.output


async def load_state(
    *,
    last_graph_run_results_cache_key: str | None,
    persistence: FullStatePersistence,
    redis_client: aioredis.Redis,
    reset_state: bool,
    session_id: int | str,
) -> tuple[str, ChaturState]:
    """Load state for the agent graph.

    The process is as follows:

    1. Check if the Redis cache key for the agent graph exists. If it does, then
        preload the previous state.
    2. If we are resetting state or the Redis cache key for the agent graph does not
        exist, then we initialize a new state with the query object and delete previous
        caches.

    Otherwise,

    3. We pull any changes to graph run results made by the frontend from Redis. Note
        that we only pull changes for the last graph run results since the frontend
        should never see more than one set of results at a time.

    NB: The Chatur agent graph is responsible for passing updated **dependencies** from
    the frontend to other graphs. However, each child graph is responsible for loading
    its own **state**. This is because the Chatur agent graph calls the other graphs
    and, thus, maintains responsibility for passing correct dependencies. On the other
    hand, each child graph executes in isolation (aside from the dependencies that are
    passed to it) and, thus, each child graph ensures that they can execute faithfully.

    Parameters
    ----------
    last_graph_run_results_cache_key
        The Redis cache key for the last graph run results.
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
    tuple[str, ChaturState]
        The Redis cache key and the state for the agent graph.

    Raises
    ------
    ValueError
        If the last assistant call is unknown.
    """

    redis_cache_key = f"{REDIS_CACHE_PREFIX_CHATUR_AGENT}_{session_id}"
    state = None

    # 1.
    agent_cache_exists = await redis_client.exists(redis_cache_key)
    if not reset_state and agent_cache_exists:
        raw_snapshot = await redis_client.get(redis_cache_key)
        persistence.load_json(raw_snapshot)
        snapshot = await persistence.load_all()
        state = snapshot[-1].state

    # 2.
    if reset_state or not agent_cache_exists:
        if state is not None:
            assert last_graph_run_results_cache_key
            await redis_client.delete(redis_cache_key)
            await redis_client.delete(state.last_graph_run_results_cache_key)
        return redis_cache_key, ChaturState()

    assert last_graph_run_results_cache_key
    assert state is not None

    # 3.
    last_graph_run_results = await redis_client.get(last_graph_run_results_cache_key)
    last_graph_run_results = json.loads(last_graph_run_results)
    last_assistant_call = state.last_assistant_call
    match last_assistant_call:
        case None:
            model_class = None
        case "registration.register_student":
            model_class = RegisterStudentResults
        case "login.login_student":
            model_class = LoginStudentResults
        case "profile.complete_profile":
            model_class = ProfileCompletionResults
        case _:
            raise ValueError(f"Unknown last assistant call: {last_assistant_call}")

    if model_class is not None:
        state.last_graph_run_results = load_graph_run_results(
            graph_run_results=last_graph_run_results, model_class=model_class
        )

    return redis_cache_key, state


async def reset_assistants(
    *,
    chatur_query_user_id: str,
    csm: AsyncChatSessionManager,
    redis_client: aioredis.Redis,
    reset_chat_and_graph_state: bool,
) -> None:
    """Reset the chat session and graph state for all assistants.

    Parameters
    ----------
    chatur_query_user_id
        The user ID for the chatur query.
    csm
        An async chat session manager that manages the chat sessions for each user.
    redis_client
        The Redis client.
    reset_chat_and_graph_state
        Specifies whether to reset the chat session and the graph state for the user.
         This can be used to clear the chat history and graph state, effectively
         starting a completely new session. This is useful for testing or debugging
         purposes.
    """

    if not reset_chat_and_graph_state:
        return

    for namespace, user_id_prefix, redis_cache_key_prefix in [
        (
            AGENTS_REGISTER_STUDENT,
            GRAPHS_REGISTER_STUDENT,
            REDIS_CACHE_PREFIX_REGISTER_STUDENT,
        ),
        (AGENTS_LOGIN_STUDENT, GRAPHS_LOGIN_STUDENT, REDIS_CACHE_PREFIX_LOGIN_STUDENT),
        (
            AGENTS_PROFILE_COMPLETION,
            GRAPHS_PROFILE_COMPLETION,
            REDIS_CACHE_PREFIX_PROFILE_COMPLETION,
        ),
    ]:
        chat_session_exists, session_id = await csm.check_if_chat_session_exists(
            namespace=namespace,
            signed=True,
            user_id=f"{user_id_prefix}_{chatur_query_user_id}",
        )
        if chat_session_exists:
            await redis_client.delete(csm.get_redis_key(session_id=session_id))
            redis_cache_key = f"{redis_cache_key_prefix}_{session_id}"
            await redis_client.delete(redis_cache_key)
