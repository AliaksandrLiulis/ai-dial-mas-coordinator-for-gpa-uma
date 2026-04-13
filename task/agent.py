import json
import os
from typing import Any, cast

from aidial_client import AsyncDial
from aidial_client.types.chat import Message as ClientChatMessage
from aidial_sdk.chat_completion import Choice, Request, Message, Stage
from aidial_sdk.chat_completion import Role

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


def _message_to_dict(message: Message) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return message.dict(exclude_none=True)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    def _dial_api_key(self) -> str:
        return os.getenv("DIAL_API_KEY", "dial_api_key")

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=self._dial_api_key(),
            api_version="2025-01-01-preview",
        )
        coordination_stage = StageProcessor.open_stage(choice, "Coordination")
        try:
            coordination_request = await self.__prepare_coordination_request(
                client, request
            )
            coordination_stage.append_content(
                json.dumps(
                    coordination_request.model_dump(),
                    ensure_ascii=False,
                )
            )
        finally:
            StageProcessor.close_stage_safely(coordination_stage)

        agent_stage = StageProcessor.open_stage(
            choice, coordination_request.agent_name.value
        )
        try:
            agent_message = await self.__handle_coordination_request(
                coordination_request, choice, agent_stage, request
            )
        finally:
            StageProcessor.close_stage_safely(agent_stage)

        return await self.__final_response(client, choice, request, agent_message)

    async def __prepare_coordination_request(
        self, client: AsyncDial, request: Request
    ) -> CoordinationRequest:
        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)
        extra_body = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": CoordinationRequest.model_json_schema(),
                },
            }
        }
        completion = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=cast(list[ClientChatMessage], messages),
            stream=False,
            extra_body=extra_body,
        )
        raw = completion.choices[0].message.content
        if not raw:
            raise RuntimeError("Coordination model returned empty content")
        data = json.loads(raw)
        return CoordinationRequest.model_validate(data)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            if msg.role == Role.USER and msg.custom_content is not None:
                messages.append({"role": "user", "content": msg.text()})
            else:
                messages.append(_message_to_dict(msg))
        return messages

    async def __handle_coordination_request(
        self,
        coordination_request: CoordinationRequest,
        choice: Choice,
        stage: Stage,
        request: Request,
    ) -> Message:
        if coordination_request.agent_name == AgentName.GPA:
            gpa = GPAGateway(self.endpoint)
            return await gpa.response(
                choice, stage, request, coordination_request.additional_instructions
            )
        ums = UMSAgentGateway(self.ums_agent_endpoint)
        return await ums.response(
            choice, stage, request, coordination_request.additional_instructions
        )

    async def __final_response(
        self,
        client: AsyncDial,
        choice: Choice,
        request: Request,
        agent_message: Message,
    ) -> Message:
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)
        agent_text = agent_message.text()
        if messages and messages[-1].get("role") == "user":
            prev = str(messages[-1].get("content") or "")
            messages[-1]["content"] = (
                f"## Specialist agent output\n{agent_text}\n\n"
                f"## User request\n{prev}\n"
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"## Specialist agent output\n{agent_text}\n\n"
                        f"## User request\n{request.messages[-1].text()}\n"
                    ),
                }
            )

        stream = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=cast(list[ClientChatMessage], messages),
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                choice.append_content(delta.content)
                parts.append(delta.content)
        return Message(role=Role.ASSISTANT, content="".join(parts))
