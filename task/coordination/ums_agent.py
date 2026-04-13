import json
from typing import Any, Optional

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice
from aidial_sdk.chat_completion.request import CustomContent

_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint.rstrip("/")

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: Optional[str],
    ) -> Message:
        conversation_id = self.__get_ums_conversation_id(request)
        if conversation_id is None:
            conversation_id = await self.__create_ums_conversation()
            choice.set_state({_UMS_CONVERSATION_ID: conversation_id})

        last = request.messages[-1]
        user_text = last.text()
        if additional_instructions:
            user_text = f"{additional_instructions}\n\n{user_text}"

        assistant_text = await self.__call_ums_agent(conversation_id, user_text, stage)
        return Message(
            role=Role.ASSISTANT,
            content=assistant_text,
            custom_content=CustomContent(
                state={_UMS_CONVERSATION_ID: conversation_id},
            ),
        )

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        for msg in reversed(request.messages):
            if msg.role != Role.ASSISTANT:
                continue
            if not msg.custom_content or not msg.custom_content.state:
                continue
            state = msg.custom_content.state
            if not isinstance(state, dict):
                continue
            cid = state.get(_UMS_CONVERSATION_ID)
            if cid:
                return str(cid)
        return None

    async def __create_ums_conversation(self) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.ums_agent_endpoint}/conversations", json={})
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return str(data["id"])

    async def __call_ums_agent(
        self,
        conversation_id: str,
        user_message: str,
        stage: Stage,
    ) -> str:
        url = f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat"
        payload = {
            "message": {"role": "user", "content": user_message},
            "stream": True,
        }
        buffer: list[str] = []
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "conversation_id" in obj and "choices" not in obj:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        buffer.append(piece)
                        stage.append_content(piece)
        return "".join(buffer)
