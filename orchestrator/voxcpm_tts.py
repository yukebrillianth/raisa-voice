"""Pipecat TTS service backed by the VoxCPM2 streaming server (tts/server.py).

Pipecat already splits the LLM stream into sentences before calling run_tts,
so each request is one sentence. Cancelling the generator on interruption
closes the HTTP response, which makes the server stop generating.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp
from loguru import logger

from pipecat.frames.frames import DataFrame, ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from speech_text import SPEECH_TRANSFORMS, normalize, screen_text, word_times


@dataclass
class SentenceWords(DataFrame):
    """A sentence's on-screen words, sent ahead of its audio.

    `times` (start of each word, seconds into the sentence's audio) is filled in on this same
    object once the TTS has returned the whole sentence, which at ~10x real time is well
    before most of it has played. SpokenText (bot.py) releases the words as the audio plays.
    """

    words: list[str]
    times: list[float] | None = None


class VoxCPMTTSService(TTSService):
    def __init__(self, *, base_url: str, **kwargs):
        # The voice is fixed server side, so there is no model, voice or language to set here.
        # Stop frames end the bot turn on time; without them Pipecat falls back to a 3 s
        # silence timeout, which delayed the answer after every tool call (same as Piper/OpenAI TTS).
        super().__init__(
            settings=TTSSettings(model=None, voice=None, language=None),
            push_start_frame=True,
            push_stop_frames=True,
            # Markdown only; numbers, money and times are rewritten in run_tts (see speech_text.py).
            text_transforms=SPEECH_TRANSFORMS,
            # Fenced code is tagged "code" by the LLMTextProcessor in bot.py and never spoken.
            skip_text_types=["code"],
            **kwargs,
        )
        self._url = f"{base_url}/v1/tts"
        self._session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        # Frames from run_tts keep their order, so this reaches the output transport right
        # before the sentence's audio. Pipecat's own TTSTextFrame only follows the finished audio.
        sentence = SentenceWords(words=(await screen_text(text)).split())
        if sentence.words:
            yield sentence
        spoken = normalize(text).strip()
        if not spoken:  # emoji only, a table row
            sentence.times = [0.0] * len(sentence.words)
            return
        if self._session is None:
            self._session = aiohttp.ClientSession()
        audio, audio_rate = bytearray(), 1
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(spoken)
            async with self._session.post(self._url, json={"text": spoken}) as response:
                if response.status != 200:
                    yield ErrorFrame(error=f"TTS server returned {response.status}")
                    return
                sample_rate = int(response.headers["X-Sample-Rate"])
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_any(), in_sample_rate=sample_rate, context_id=context_id
                ):
                    await self.stop_ttfb_metrics()
                    # Frames come out resampled to the pipeline rate, not the server's X-Sample-Rate.
                    audio += frame.audio
                    audio_rate = frame.sample_rate
                    yield frame
        except aiohttp.ClientError as e:
            logger.error(f"{self} request failed: {e}")
            yield ErrorFrame(error=f"TTS request failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
            # Also on errors and cut-short audio, so the words are never held back for good.
            sentence.times = word_times(sentence.words, bytes(audio), audio_rate)

    async def cleanup(self):
        await super().cleanup()
        if self._session is not None:
            await self._session.close()
