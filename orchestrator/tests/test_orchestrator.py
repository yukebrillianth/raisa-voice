import asyncio
import json

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)
from protocol import IN_SAMPLE_RATE, RobotSerializer, gemini_to_openai_tools


def run(coro):
    return asyncio.run(coro)


def test_serializer_audio_and_control():
    s = RobotSerializer()
    audio = run(s.deserialize(b"\x01\x00" * 320))
    assert isinstance(audio, InputAudioRawFrame) and audio.sample_rate == IN_SAMPLE_RATE

    msg = run(s.deserialize('{"type": "interrupt"}'))
    assert isinstance(msg, InputTransportMessageFrame) and msg.message == {"type": "interrupt"}
    assert run(s.deserialize("not json")) is None
    assert run(s.deserialize('{"no_type": 1}')) is None

    pcm = b"\x02\x00" * 10
    assert run(s.serialize(OutputAudioRawFrame(audio=pcm, sample_rate=24000, num_channels=1))) == pcm
    assert run(s.serialize(InterruptionFrame())) is None
    out = run(s.serialize(OutputTransportMessageUrgentFrame(message={"type": "turn_complete"})))
    assert json.loads(out) == {"type": "turn_complete"}


def test_gemini_declarations_to_openai_tools():
    tools = gemini_to_openai_tools(
        [
            {
                "name": "antar_ke_titik",
                "description": "Antar pengunjung",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "lokasi": {"type": "STRING", "enum": ["lobi", "lab"]},
                        "titik": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["lokasi"],
                },
            },
            {"name": "cari", "parametersJsonSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            {"name": "halo"},
        ]
    )
    params = tools[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["lokasi"] == {"type": "string", "enum": ["lobi", "lab"]}
    assert params["properties"]["titik"]["items"] == {"type": "string"}
    assert params["required"] == ["lokasi"]
    assert tools[1]["function"]["parameters"]["properties"]["q"]["type"] == "string"
    assert tools[2]["function"]["parameters"] == {"type": "object", "properties": {}}




def test_sentence_words_go_out_before_its_audio_and_tts_gets_spoken_words():
    from aiohttp import web
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.tests.utils import run_test

    from voxcpm_tts import SentenceWords, VoxCPMTTSService

    asked = []

    async def tts(request):
        asked.append((await request.json())["text"])
        return web.Response(body=b"\x01\x00" * 4800, headers={"X-Sample-Rate": "24000"})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/tts", tts)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await run_test(
                VoxCPMTTSService(base_url=f"http://127.0.0.1:{port}"),
                frames_to_send=[TTSSpeakFrame(text="Harga **Rp150.000**.")],
                expected_down_frames=None,
            )
        finally:
            await runner.cleanup()

    down, _ = run(go())
    kinds = [type(f).__name__ for f in down]
    at = kinds.index("SentenceWords")
    assert at < kinds.index("TTSAudioRawFrame")
    assert down[at].words == ["Harga", "Rp150.000."]
    assert down[at].times is not None and len(down[at].times) == 2  # filled in once the audio is in
    assert asked == ["Harga seratus lima puluh ribu rupiah."]


def test_titles_do_not_split_the_sentence():
    from pipecat.utils.text.pattern_pair_aggregator import MatchAction

    from bot import SentenceAggregator

    async def sentences(text):
        aggregator = SentenceAggregator().add_pattern("code", "```", "```", MatchAction.AGGREGATE)
        out = []
        for i in range(0, len(text), 3):  # LLM tokens arrive a few characters at a time
            out += [a.text async for a in aggregator.aggregate(text[i : i + 3])]
        rest = await aggregator.flush()
        return out + ([rest.text] if rest else [])

    assert run(sentences("Pembicaranya Prof. Dr. Ir. Budi, S.T., M.Sc. akan hadir besok. Silakan datang.")) == [
        "Pembicaranya Prof. Dr. Ir. Budi, S.T., M.Sc. akan hadir besok.", "Silakan datang."]
    assert run(sentences("Ada Dr. (HC) Ir. Joko, S.Tr.T., M.Sc.Eng., ASEAN Eng. dan dr. Sinta, Ph.D. Mereka di lobi.")) == [
        "Ada Dr. (HC) Ir. Joko, S.Tr.T., M.Sc.Eng., ASEAN Eng. dan dr. Sinta, Ph.D. Mereka di lobi."]
    # ordinary sentences still split, also right before a title; one-word Indonesian ones are not held
    assert run(sentences("Baik. Saya antar ke lobi! Prof. Budi menunggu. Harga Rp150.000. Oke?")) == [
        "Baik.", "Saya antar ke lobi!", "Prof. Budi menunggu.", "Harga Rp150.000.", "Oke?"]
    # a one-word English sentence goes with the next one, or alone when nothing follows
    assert run(sentences("Sure. Please follow me. The lift is on your right.")) == [
        "Sure. Please follow me.", "The lift is on your right."]
    assert run(sentences("Okay.")) == ["Okay."]
    # a line break ends a sentence: headings and list items have no full stop of their own
    assert run(sentences("**1. Universitas Airlangga (UNAIR)**\n*   Rektor: Prof. Dr. Madyan, S.E., M.Fin. (periode 2025-2030)\n"
                         "*   Sebelumnya: Prof. Nasih.\n\n### Zona Waktu\nWIB itu UTC+7.")) == [
        "**1. Universitas Airlangga (UNAIR)**", "*   Rektor: Prof. Dr. Madyan, S.E., M.Fin. (periode 2025-2030)",
        "*   Sebelumnya: Prof. Nasih.", "### Zona Waktu", "WIB itu UTC+7."]
    # a list number, a street and "No." are not sentence ends; a lone "**5.**" waits for its line
    assert run(sentences("**5. Politeknik Perkapalan (PPNS)** ada di Jl. Teknik Kimia No. 1, Surabaya. Silakan.")) == [
        "**5. Politeknik Perkapalan (PPNS)** ada di Jl. Teknik Kimia No. 1, Surabaya.", "Silakan."]
    assert run(sentences("**5.**\nPoliteknik Perkapalan.")) == ["**5.**\nPoliteknik Perkapalan."]
    # no split inside a code fence
    assert run(sentences("Contoh:\n```\nprint(1)\nprint(2)\n```\nSelesai.")) [-1] == "Selesai."


def test_word_times_follow_spoken_length_and_pauses():
    from speech_text import word_times

    rate = 1000
    audio = (b"\x00\x00" * 100) + (b"\x00\x40" * 1000) + (b"\x00\x00" * 100)  # voiced 0.1-1.1 s
    times = word_times(["Harga", "Rp150.000,", "ya."], audio, rate)
    assert abs(times[0] - 0.1) < 0.01  # leading silence skipped
    assert times[0] < times[1] < times[2] < 1.1
    # "Rp150.000," is spoken as ~10 syllables plus a comma pause: by far the longest word
    assert times[2] - times[1] > 3 * (times[1] - times[0])
    assert word_times(["a"], b"\x00\x00" * 50, rate) == [0.0]
    # "Prof." is said "Profesor": three syllables of the eight, and its dot is no pause
    times = word_times(["Prof.", "Budi", "Pak", "Budi"], audio, rate)
    assert abs(times[1] - times[0] - 3 / 8) < 0.01


def test_spoken_text_sends_each_word_as_its_audio_plays():
    from pipecat.frames.frames import InterruptionFrame, OutputAudioRawFrame, TTSStoppedFrame
    from pipecat.tests.utils import SleepFrame, run_test

    from bot import SpokenText
    from voxcpm_tts import SentenceWords

    sent = []

    class Output:
        async def send_message(self, frame):
            sent.append((frame.message["text"], round(spoken._heard, 2)))  # audio played when sent

    spoken = SpokenText(Output())

    def chunk(secs):
        return OutputAudioRawFrame(audio=b"\x00\x00" * int(1000 * secs), sample_rate=1000, num_channels=1)

    frames = [SentenceWords(words=["Satu", "dua", "tiga."], times=[0.0, 0.35, 0.55])]
    frames += [chunk(0.1) for _ in range(3)] + [TTSStoppedFrame()]
    frames += [SentenceWords(words=["Lagi", "dong."], times=[0.0, 0.6]), chunk(0.1), SleepFrame(0.05), InterruptionFrame(), chunk(0.6)]
    run(run_test(spoken, frames_to_send=frames, expected_down_frames=None))
    # "dua" (0.35 s) goes once 0.3 s has played (0.1 s lead); "tiga." (0.55 s) at the stop; "dong." is cut.
    assert sent == [("Satu", 0.1), ("dua", 0.3), ("tiga.", 0.3), ("Lagi", 0.1)]


def test_agent_vocabulary_reaches_the_asr_server_as_hotwords():
    from aiohttp import web

    from asr_stt import WhisperServerSTTService

    seen = []

    async def transcribe(request):
        seen.append(request.query.get("hotwords"))
        return web.json_response({"text": "Halo"})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/transcribe", transcribe)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            for vocabulary in (["RAISA", "Teaching Factory"], []):
                stt = WhisperServerSTTService(base_url=f"http://127.0.0.1:{port}", vocabulary=vocabulary)
                [f async for f in stt.run_stt(b"RIFF")]
                await stt._session.close()
        finally:
            await runner.cleanup()

    run(go())
    assert seen == ["RAISA, Teaching Factory", None]  # none sent: the server uses its default list
