from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    AbortMsg,
    ResetSchedulerMetricsMsg,
    ShutdownMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs
from .health import BackendSupervisor

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False
    deadline_ms: float | None = None
    ttft_deadline_ms: float | None = None
    e2e_deadline_ms: float | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendRequestState:
    completed: bool = False
    cancel_sent: bool = False
    sse_stop_sent: bool = False
    terminal_enqueued: bool = False

    def claim_sse_stop(self) -> bool:
        if self.sse_stop_sent:
            return False
        self.sse_stop_sent = True
        return True


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    request_states: Dict[int, FrontendRequestState] = field(default_factory=dict)
    supervisor: BackendSupervisor | None = None
    listener_task: asyncio.Task | None = None
    health_task: asyncio.Task | None = None

    def readiness(self) -> dict:
        if self.supervisor is None:
            return {
                "ready": False,
                "accepting_requests": False,
                "fatal_error": "backend_supervisor_unavailable",
            }
        return self.supervisor.snapshot().as_dict()

    def require_ready(self) -> None:
        snapshot = self.readiness()
        if not snapshot["ready"]:
            raise HTTPException(status_code=503, detail=snapshot)

    def new_user(self) -> int:
        self.require_ready()
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.request_states[uid] = FrontendRequestState()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    async def monitor_health(self):
        while True:
            await asyncio.sleep(0.1)
            snapshot = self.readiness()
            if snapshot["fatal_error"] is None and snapshot["ready"]:
                continue
            if snapshot["fatal_error"] is None:
                continue
            error = str(snapshot["fatal_error"])
            for uid, state in tuple(self.request_states.items()):
                if state.completed or uid not in self.ack_map:
                    continue
                if state.terminal_enqueued:
                    continue
                state.terminal_enqueued = True
                self.ack_map[uid].append(
                    UserReply(
                        uid=uid,
                        incremental_output="",
                        finished=True,
                        terminal_reason="failed",
                        error=error,
                    )
                )
                self.event_map[uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            self.listener_task = asyncio.create_task(self.listen())
            self.health_task = asyncio.create_task(self.monitor_health())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        try:
            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map.get(uid, [])
                self.ack_map[uid] = []
                ack = None
                for ack in pending:
                    yield ack
                if ack and ack.finished:
                    state = self.request_states.get(uid)
                    if state is not None:
                        state.completed = True
                    break
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        state = self.request_states.get(uid)
        if state is None or state.claim_sse_stop():
            yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        terminal_reason = "stop"
        terminal_error = None
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                terminal_reason = (
                    "stop" if ack.terminal_reason == "completed" else ack.terminal_reason
                )
                terminal_error = ack.error
                break

        state = self.request_states.get(uid)
        if state is None or state.claim_sse_stop():
            end_chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [
                    {"delta": {}, "index": 0, "finish_reason": terminal_reason}
                ],
            }
            if terminal_error is not None:
                end_chunk["error"] = terminal_error
            yield f"data: {json.dumps(end_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        completed = False
        try:
            async for chunk in generator:
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
            completed = True
        finally:
            if not completed and self.config.cancel_on_disconnect:
                await asyncio.shield(
                    self.abort_user(uid, reason="client_disconnected")
                )
                self.ack_map.pop(uid, None)
                self.event_map.pop(uid, None)
            self.request_states.pop(uid, None)

    async def abort_user(
        self, uid: int, *, reason: str = "client_cancelled"
    ) -> bool:
        state = self.request_states.get(uid)
        if state is None or state.completed or state.cancel_sent:
            return False
        state.cancel_sent = True
        state.terminal_enqueued = True
        await self.send_one(AbortMsg(uid=uid, reason=reason))
        if uid in self.ack_map:
            self.ack_map[uid].append(
                UserReply(
                    uid=uid,
                    incremental_output="",
                    finished=True,
                    terminal_reason="cancelled",
                )
            )
            self.event_map[uid].set()
        logger.warning("Aborting request for user %s", uid)
        return True

    def shutdown(self):
        for task in (self.listener_task, self.health_task):
            if task is not None:
                task.cancel()
        if self.supervisor is not None:
            self.supervisor.stop()
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()

    async def request_backend_shutdown(self) -> None:
        await self.send_one(ShutdownMsg())


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        try:
            await asyncio.wait_for(
                _GLOBAL_STATE.request_backend_shutdown(), timeout=2.0
            )
        except Exception:
            logger.exception("Failed to request an orderly backend shutdown")
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
        headers={"X-MiniSGL-Request-ID": str(uid)},
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                deadline_ms=req.deadline_ms,
                ttft_deadline_ms=req.ttft_deadline_ms,
                e2e_deadline_ms=req.e2e_deadline_ms,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(
            state.stream_chat_completions(uid), request, uid
        ),
        media_type="text/event-stream",
        headers={"X-MiniSGL-Request-ID": str(uid)},
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    state.require_ready()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


@app.get("/health")
@app.get("/ready")
async def health():
    state = get_global_state()
    snapshot = state.readiness()
    return JSONResponse(status_code=200 if snapshot["ready"] else 503, content=snapshot)


@app.post("/v1/requests/{uid}/cancel")
async def cancel_request(uid: int):
    state = get_global_state()
    accepted = await state.abort_user(uid, reason="explicit_cancel")
    return {"uid": uid, "cancel_accepted": accepted}


@app.post("/v1/scheduler/metrics/reset")
async def reset_scheduler_metrics():
    state = get_global_state()
    state.require_ready()
    await state.send_one(ResetSchedulerMetricsMsg())
    return {"accepted": True}


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


async def read_stdin():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        line = line.decode().rstrip("\n")


async def async_input(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            need_stop = False
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                if need_stop:
                    break
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(
    config: ServerArgs,
    start_backend: Callable[[], BackendSupervisor],
    run_shell: bool,
) -> None:
    """
    Run the API server using uvicorn.

    Args:
        input_queue: Queue to send requests to the backend
        output_queue: Queue to receive responses from the backend
        host: Host address to bind to
        port: Port number to bind to
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    _GLOBAL_STATE.supervisor = start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
