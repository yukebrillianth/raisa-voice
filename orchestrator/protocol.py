"""Wire protocol between the robot gateway and the orchestrator.

Binary frames carry raw PCM16LE mono audio: 16 kHz uplink, 24 kHz downlink.
Text frames carry JSON control messages with a "type" field. The shapes mirror
the Gemini Live messages the gateway already handles, so its adapter stays thin.

Client to server:
    {"type": "setup", "token": str, "systemInstruction": str, "tools": [functionDeclaration, ...]}
        (a wrong or missing token gets {"type": "error", "message": "unauthorized"} and a close)
    {"type": "text_input", "text": str}
    {"type": "tool_response", "functionResponses": [{"id", "name", "response"}]}
    {"type": "interrupt"}
    {"type": "clear_context"}  (new visitor: forget the conversation, keep the system instruction)
    {"type": "speak", "text": str}  (simulator/debug: say the text as is, no LLM)

Server to client:
    {"type": "setup_complete"}  (pipeline ready, start sending audio)
    {"type": "user_transcript", "text": str}
    {"type": "assistant_text", "text": str}
    {"type": "tool_call", "functionCalls": [{"id", "name", "args"}]}
    {"type": "interrupted"}
    {"type": "turn_complete"}
    {"type": "error", "message": str}
"""

import json
from typing import Any

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

IN_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 24000


class RobotSerializer(FrameSerializer):
    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=IN_SAMPLE_RATE, num_channels=1)
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            return None
        return InputTransportMessageFrame(message=message)


def _convert_schema(schema: Any) -> Any:
    """Gemini Schema (OpenAPI subset, uppercase types) to JSON Schema."""
    if isinstance(schema, list):
        return [_convert_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.lower()
        elif key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: _convert_schema(sub) for name, sub in value.items()}
        else:
            out[key] = _convert_schema(value)
    return out


def gemini_to_openai_tools(declarations: list[dict]) -> list[dict]:
    """Convert @google/genai functionDeclarations into OpenAI chat-completions tools."""
    tools = []
    for decl in declarations:
        if decl.get("parametersJsonSchema") is not None:
            parameters = decl["parametersJsonSchema"]
        elif decl.get("parameters") is not None:
            parameters = _convert_schema(decl["parameters"])
        else:
            parameters = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": decl["name"],
                    "description": decl.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return tools
