"""RAISA voice orchestrator: one Pipecat pipeline per robot WebSocket.

robot audio 16k -> Silero VAD + Smart Turn v3.2 -> Whisper (asr) -> Gemma 4 (llama.cpp)
-> VoxCPM2 (per sentence) -> robot audio 24k. See protocol.py for the wire format.
"""

import asyncio
import os
import random
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from loguru import logger

from pipecat.adapters.llm.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsStartedFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternPairAggregator
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
from pipecat.workers.runner import WorkerRunner

from asr_stt import WhisperServerSTTService
from protocol import IN_SAMPLE_RATE, OUT_SAMPLE_RATE, RobotSerializer, gemini_to_openai_tools
from voxcpm_tts import SentenceWords, VoxCPMTTSService

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8000")
ASR_URL = os.environ.get("ASR_URL", "http://localhost:8001")
# Unset only for local development; compose requires it because port 8765 is reachable on the LAN.
VOICE_TOKEN = os.environ.get("VOICE_TOKEN", "")
SIMULATOR = Path(__file__).with_name("simulator.html")
LLM_MODEL = "gemma-4-12b"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
TOOL_TIMEOUT_SECS = 30.0
# While a tool runs, this much silence from the robot gets a short spoken filler.
FILLER_AFTER_SECS = 0.7
FILLERS = [
    "Sebentar ya, saya cek dulu.",
    "Tunggu sebentar ya.",
    "Baik, saya cek dulu sebentar.",
    "Mohon tunggu sebentar.",
]
VOICE_RULES = (
    "Jawabanmu diucapkan lewat speaker, jadi jawab singkat dan natural, maksimal dua kalimat. "
    "Jangan pakai markdown, emoji, atau daftar berpoin. Akhiri tiap kalimat dengan tanda baca. "
    "Jawab dalam bahasa yang dipakai pengunjung: bahasa Indonesia atau bahasa Inggris."
)

filler_audio: list[tuple[bytes, int]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Synthesized once and replayed from memory: no TTS round trip while the robot waits on a tool.
    try:
        async with aiohttp.ClientSession() as session:
            for text in FILLERS:
                async with session.post(f"{TTS_URL}/v1/tts", json={"text": text}) as response:
                    response.raise_for_status()
                    filler_audio.append((await response.read(), int(response.headers["X-Sample-Rate"])))
    except aiohttp.ClientError as e:
        logger.warning(f"fillers unavailable, slow tools will be silent: {e}")
    yield


app = FastAPI(lifespan=lifespan)


class ToolRelay:
    """Runs every LLM tool call on the robot and waits for its tool_response."""

    def __init__(self, events: "ClientEvents"):
        self._pending: dict[str, asyncio.Future] = {}
        self._events = events

    async def call(self, params: FunctionCallParams):
        future = asyncio.get_running_loop().create_future()
        self._pending[params.tool_call_id] = future
        call = {"id": params.tool_call_id, "name": params.function_name, "args": params.arguments}
        await params.llm.push_frame(
            OutputTransportMessageUrgentFrame(message={"type": "tool_call", "functionCalls": [call]})
        )
        filler = self._filler(params, time.monotonic()) if filler_audio else asyncio.sleep(0)
        filler_task = asyncio.create_task(filler)
        try:
            result = await asyncio.wait_for(future, TOOL_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            result = {"error": "Robot tidak membalas pemanggilan fungsi tepat waktu."}
        finally:
            filler_task.cancel()
            self._pending.pop(params.tool_call_id, None)
        await params.result_callback(result)

    async def _filler(self, params: FunctionCallParams, started: float):
        # Waits out any sentence the LLM said before the call, then FILLER_AFTER_SECS of silence.
        events = self._events
        while events.bot_speaking or time.monotonic() - max(started, events.silent_since) < FILLER_AFTER_SECS:
            await asyncio.sleep(0.05)
        audio, rate = random.choice(filler_audio)
        # Plain audio, not a TTS turn: it does not hold up the answer, and an interruption clears it.
        await params.llm.push_frame(OutputAudioRawFrame(audio=audio, sample_rate=rate, num_channels=1))

    def resolve(self, responses: list):
        for response in responses:
            if not isinstance(response, dict):
                continue
            future = self._pending.get(response.get("id"))
            if future is not None and not future.done():
                future.set_result(response.get("response", {}))


class ClientMessages(FrameProcessor):
    """Handles JSON control messages coming from the robot."""

    def __init__(self, relay: ToolRelay, **kwargs):
        super().__init__(**kwargs)
        self._relay = relay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputTransportMessageFrame):
            await self.push_frame(frame, direction)
            return

        message = frame.message
        kind = message["type"]
        if kind == "text_input" and isinstance(message.get("text"), str):
            await self.broadcast_interruption()
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": message["text"]}], run_llm=True
                )
            )
        elif kind == "tool_response" and isinstance(message.get("functionResponses"), list):
            self._relay.resolve(message["functionResponses"])
        elif kind == "interrupt":
            await self.broadcast_interruption()
        elif kind == "speak" and isinstance(message.get("text"), str):
            # Simulator: say this text as is (through the speech normalizer), no LLM.
            await self.push_frame(TTSSpeakFrame(text=message["text"], append_to_context=False))
        elif kind == "clear_context":
            # New visitor: drop the conversation, keep the system instruction (held by the LLM service).
            await self.broadcast_interruption()
            await self.push_frame(LLMMessagesUpdateFrame(messages=[], run_llm=False))
        else:
            logger.warning(f"ignoring client message type {kind!r}")


# Measured against Whisper word times on the robot's own audio: words otherwise showed ~0.12 s
# after they were heard (median), the voiced-span estimate starts late.
WORD_LEAD_SECS = 0.1


class SpokenText(FrameProcessor):
    """Sends the answer to the robot word by word, each word as its audio is played.

    Sits after the output transport, which sends audio at playback pace and only then passes
    each chunk on, so the audio counted here is what the visitor has heard. Each word goes
    out once that much of its sentence has played (times from SentenceWords).
    """

    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self._out = output
        self._sentence: SentenceWords | None = None
        self._heard = 0.0
        self._word_at = 0

    def cut(self):
        """Stop the current answer's words; ClientEvents calls it before reporting `interrupted`."""
        self._sentence = None

    async def finish(self):
        """Send what the last sentence has left; ClientEvents calls it before `turn_complete`."""
        await self._send_words(float("inf"))
        self._sentence = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, SentenceWords):
            await self._send_words(float("inf"))  # whatever the last sentence had left
            self._sentence, self._heard, self._word_at = frame, 0.0, 0
        elif isinstance(frame, OutputAudioRawFrame) and self._sentence:
            self._heard += len(frame.audio) / 2 / frame.sample_rate
            await self._send_words(self._heard + WORD_LEAD_SECS)
        elif isinstance(frame, TTSStoppedFrame):
            await self.finish()  # the sentence's audio has all played
        elif isinstance(frame, InterruptionFrame):
            self._sentence = None
        await self.push_frame(frame, direction)

    async def _send_words(self, played: float):
        sentence = self._sentence
        if not sentence:
            return
        # Times arrive once the TTS has the whole sentence; until then, or if it failed, wait
        # for the end of the audio.
        times = sentence.times or [float("inf")] * len(sentence.words)
        while self._sentence is sentence and self._word_at < len(sentence.words) and times[self._word_at] <= played:
            word = sentence.words[self._word_at]
            self._word_at += 1
            # Clients join consecutive assistant_text pieces of one answer with a space.
            await self._out.send_message(
                OutputTransportMessageUrgentFrame(message={"type": "assistant_text", "text": word})
            )


class ClientEvents(FrameProcessor):
    """Reports the answer's interruption and completion back to the robot."""

    def __init__(self, spoken: SpokenText, **kwargs):
        super().__init__(**kwargs)
        self._spoken = spoken
        self._responding = False
        self.bot_speaking = False
        self.silent_since = 0.0
        self._llm_done = False
        # A tool call means another LLM response follows once the result is in.
        self._tool_followup = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._responding, self._llm_done, self._tool_followup = True, False, False
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._tool_followup = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_done = True
        elif isinstance(frame, BotStartedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            self.bot_speaking = True
        elif isinstance(frame, InterruptionFrame) and self._responding:
            # Only an answer that was actually cut off is reported, like Gemini Live.
            self._responding = False
            self._spoken.cut()  # no word of the cut answer may follow `interrupted`
            await self._send(OutputTransportMessageUrgentFrame, {"type": "interrupted"})
        elif isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            self.bot_speaking = False
            self.silent_since = time.monotonic()
            if self._responding and self._llm_done and not self._tool_followup:
                self._responding = False
                await self._spoken.finish()  # the last words before the turn is closed
                await self._send(OutputTransportMessageUrgentFrame, {"type": "turn_complete"})

    async def _send(self, frame_cls, message: dict):
        await self.push_frame(frame_cls(message=message))


@app.get("/")
def simulator():
    return HTMLResponse(SIMULATOR.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def robot_session(websocket: WebSocket):
    await websocket.accept()
    try:
        setup = await websocket.receive_json()
        if setup.get("type") != "setup":
            raise ValueError("first message must be setup")
        token = setup.get("token")
        if VOICE_TOKEN and not (isinstance(token, str) and secrets.compare_digest(token, VOICE_TOKEN)):
            await websocket.send_json({"type": "error", "message": "unauthorized"})
            await websocket.close(code=1008)
            return
        instruction = setup.get("systemInstruction", "")
        if not isinstance(instruction, str):
            raise ValueError("systemInstruction must be a string")
        tools = gemini_to_openai_tools(setup.get("tools", []))
        vocabulary = setup.get("vocabulary", [])
        if not (isinstance(vocabulary, list) and all(isinstance(w, str) for w in vocabulary)):
            raise ValueError("vocabulary must be a list of strings")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"invalid setup: {e}"})
        await websocket.close()
        return

    system = "\n\n".join(filter(None, [instruction, VOICE_RULES]))
    context = LLMContext(tools=ToolsSchema(standard_tools=[], custom_tools={AdapterType.OPENAI: tools}))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            empty_user_turn=None,
        ),
        # Long conversations: old turns are summarized (tool call pairs are kept whole) so a
        # session stays inside its ~13k-token llama.cpp slot, like Gemini Live's slidingWindow.
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=True,
            auto_context_summarization_config=LLMAutoContextSummarizationConfig(
                max_context_tokens=6000,
                max_unsummarized_messages=None,
                summary_config=LLMContextSummaryConfig(target_context_tokens=800),
            ),
        ),
    )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        if message.content:
            await aggregator.push_frame(
                OutputTransportMessageUrgentFrame(message={"type": "user_transcript", "text": message.content})
            )

    llm = OpenAILLMService(
        api_key="local",
        base_url=LLM_URL,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system,
            extra={"extra_body": NO_THINKING},
        ),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=IN_SAMPLE_RATE,
            audio_out_sample_rate=OUT_SAMPLE_RATE,
            serializer=RobotSerializer(),
        ),
    )

    spoken = SpokenText(transport.output())
    events = ClientEvents(spoken)
    relay = ToolRelay(events)
    # The robot may already be acting on the call; a barge-in must not drop its result.
    llm.register_function(None, relay.call, cancel_on_interruption=False)

    pipeline = Pipeline(
        [
            transport.input(),
            ClientMessages(relay),
            WhisperServerSTTService(base_url=ASR_URL, vocabulary=vocabulary),
            user_aggregator,
            llm,
            LLMTextProcessor(
                text_aggregator=PatternPairAggregator().add_pattern("code", "```", "```", MatchAction.AGGREGATE)
            ),
            VoxCPMTTSService(base_url=TTS_URL),
            events,
            transport.output(),
            spoken,
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=IN_SAMPLE_RATE,
            audio_out_sample_rate=OUT_SAMPLE_RATE,
            enable_metrics=True,
        ),
        enable_rtvi=False,
        idle_timeout_secs=None,
        observers=[MetricsLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Audio sent before this point is dropped, so the robot waits for it (Gemini: setupComplete).
        await worker.queue_frame(OutputTransportMessageUrgentFrame(message={"type": "setup_complete"}))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
