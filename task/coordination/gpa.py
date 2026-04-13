import os
from copy import deepcopy
from typing import Any, Optional, cast

from aidial_client import AsyncDial
from aidial_client.types.chat import Message as ClientChatMessage
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage, Attachment
from aidial_sdk.chat_completion.request import CustomContent

from task.logging_config import get_logger
from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"

logger = get_logger(__name__)


def _sdk_message_dict(message: Message) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return message.dict(exclude_none=True)


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def _api_key(self) -> str:
        return os.getenv("DIAL_API_KEY", "dial_api_key")

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: Optional[str],
    ) -> Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=self._api_key(),
            api_version="2025-01-01-preview",
        )
        messages = self.__prepare_gpa_messages(request, additional_instructions)
        conv_header = request.headers.get("x-conversation-id")
        extra_headers: dict[str, str] = {}
        if conv_header:
            extra_headers["x-conversation-id"] = conv_header

        stream = await client.chat.completions.create(
            deployment_name="general-purpose-agent",
            messages=cast(list[ClientChatMessage], messages),
            stream=True,
            extra_headers=extra_headers,
        )

        content_parts: list[str] = []
        collected_attachments: list[Attachment] = []
        merged_state: dict[str, Any] = {}
        stages_map: dict[int, Stage] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content is not None:
                logger.info("GPA delta: %s", delta.content)
                stage.append_content(delta.content)
                content_parts.append(delta.content)

            cc = delta.custom_content
            if not cc:
                continue

            if hasattr(cc, "model_dump"):
                cc_dict = cc.model_dump(exclude_none=True)
            elif hasattr(cc, "dict"):
                cc_dict = cc.dict(exclude_none=True)
            else:
                cc_dict = dict(cc)

            for att in cc_dict.get("attachments") or []:
                if isinstance(att, Attachment):
                    collected_attachments.append(att)
                    continue
                if hasattr(att, "model_dump"):
                    ad = att.model_dump(exclude_none=True)
                elif hasattr(att, "dict"):
                    ad = att.dict(exclude_none=True)
                else:
                    ad = dict(att)
                collected_attachments.append(Attachment(**ad))

            st = cc_dict.get("state")
            if isinstance(st, dict):
                merged_state.update(st)

            for stg in cc_dict.get("stages") or []:
                if isinstance(stg, dict):
                    idx = stg.get("index")
                    name = stg.get("name")
                    st_content = stg.get("content")
                    st_attachments = stg.get("attachments") or []
                    st_status = stg.get("status")
                else:
                    idx = getattr(stg, "index", None)
                    name = getattr(stg, "name", None)
                    st_content = getattr(stg, "content", None)
                    st_attachments = getattr(stg, "attachments", None) or []
                    st_status = getattr(stg, "status", None)
                if idx is None:
                    continue
                idx = int(idx)
                if idx in stages_map:
                    st_obj = stages_map[idx]
                    if st_content:
                        st_obj.append_content(st_content)
                    for att in st_attachments:
                        if hasattr(att, "model_dump"):
                            ad = att.model_dump(exclude_none=True)
                        elif hasattr(att, "dict"):
                            ad = att.dict(exclude_none=True)
                        else:
                            ad = dict(att)
                        st_obj.add_attachment(Attachment(**ad))
                    if st_status == "completed":
                        StageProcessor.close_stage_safely(st_obj)
                else:
                    st_obj = StageProcessor.open_stage(choice, name)
                    stages_map[idx] = st_obj
                    if st_content:
                        st_obj.append_content(st_content)
                    for att in st_attachments:
                        if hasattr(att, "model_dump"):
                            ad = att.model_dump(exclude_none=True)
                        elif hasattr(att, "dict"):
                            ad = att.dict(exclude_none=True)
                        else:
                            ad = dict(att)
                        st_obj.add_attachment(Attachment(**ad))
                    if st_status == "completed":
                        StageProcessor.close_stage_safely(st_obj)

        for att in collected_attachments:
            choice.add_attachment(att)

        choice.set_state(
            {
                _IS_GPA: True,
                _GPA_MESSAGES: merged_state,
            }
        )

        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def __prepare_gpa_messages(
        self, request: Request, additional_instructions: Optional[str]
    ) -> list[dict[str, Any]]:
        res_messages: list[dict[str, Any]] = []
        msgs = request.messages
        for i in range(len(msgs)):
            msg = msgs[i]
            if msg.role != Role.ASSISTANT:
                continue
            if not msg.custom_content or not msg.custom_content.state:
                continue
            state = msg.custom_content.state
            if not isinstance(state, dict) or not state.get(_IS_GPA):
                continue
            if i == 0:
                continue
            res_messages.append(_sdk_message_dict(msgs[i - 1]))
            restored = deepcopy(msg)
            inner = state.get(_GPA_MESSAGES)
            if restored.custom_content is None:
                restored.custom_content = CustomContent()
            restored.custom_content.state = inner
            res_messages.append(_sdk_message_dict(restored))

        last_user = _sdk_message_dict(msgs[-1])
        if msgs[-1].role == Role.USER and msgs[-1].custom_content is not None:
            last_user = {"role": "user", "content": msgs[-1].text()}
        res_messages.append(last_user)

        if additional_instructions and res_messages:
            prev = str(res_messages[-1].get("content") or "")
            res_messages[-1]["content"] = f"{additional_instructions}\n\n{prev}"

        return res_messages
