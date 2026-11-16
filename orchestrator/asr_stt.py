"""Pipecat STT service backed by the shared Whisper server (asr/server.py).

SegmentedSTTService sends one WAV per VAD speech segment, transcribed while Smart
Turn is still deciding whether the visitor finished. The user aggregator joins the
segments of one turn, so a pause in mid-sentence does not lose the first half.
"""

from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class WhisperServerSTTService(SegmentedSTTService):
    def __init__(self, *, base_url: str, vocabulary: list[str] | None = None, **kwargs):
        # Model and language are fixed server side.
        super().__init__(settings=STTSettings(model=None, language=None), **kwargs)
        self._url = f"{base_url}/v1/transcribe"
        # The agent's domain words (agent.yaml transcription.customVocabulary); empty means the server default.
        self._params = {"hotwords": ", ".join(vocabulary)} if vocabulary else {}
        self._session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        await self.start_processing_metrics()
        try:
            async with self._session.post(self._url, data=audio, params=self._params) as response:
                if response.status != 200:
                    yield ErrorFrame(error=f"ASR server returned {response.status}")
                    return
                text = (await response.json())["text"]
        except aiohttp.ClientError as e:
            logger.error(f"{self} request failed: {e}")
            yield ErrorFrame(error=f"ASR request failed: {e}")
            return
        finally:
            await self.stop_processing_metrics()
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601())

    async def cleanup(self):
        await super().cleanup()
        if self._session is not None:
            await self._session.close()
