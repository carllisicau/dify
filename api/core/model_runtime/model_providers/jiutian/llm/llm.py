import hashlib
import json
import time
from collections.abc import Generator, Iterator
from decimal import Decimal
from typing import Any, Optional, Union, cast

import jwt
from requests import post

from core.model_runtime.entities import (
    AssistantPromptMessage,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.errors.invoke import InvokeError
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.baichuan.llm.baichuan_turbo_errors import BadRequestError, InternalServerError
from extensions.ext_redis import redis_client

# from extensions.ext_redis import redis_client

TOKEN_EXPIRATION_SECONDS = 36000


class JiutianLanLLmModel(LargeLanguageModel):

    def get_num_tokens(self, model: str, credentials: dict, prompt_messages: list[PromptMessage],
                       tools: Optional[list[PromptMessageTool]] = None) -> int:
        return 0

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        pass

    def trans_prompt_2messages(self, message: PromptMessage) -> dict:
        if isinstance(message, UserPromptMessage):
            message = cast(UserPromptMessage, message)
            if isinstance(message.content, str):
                message_dict = {"role": "user", "content": message.content}
            else:
                raise ValueError("user content is not a string")
        elif isinstance(message, AssistantPromptMessage):
            message = cast(AssistantPromptMessage, message)
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemPromptMessage):
            message = cast(SystemPromptMessage, message)
            message_dict = {"role": "system", "content": message.content}
        else:
            raise ValueError(f"Unknown message type {type(message)}")
        return message_dict

    def _invoke(self, model: str, credentials: dict,
                prompt_messages: list[PromptMessage], model_parameters: dict,
                tools: Optional[list[PromptMessageTool]] = None, stop: Optional[list[str]] = None,
                stream: bool = True, user: Optional[str] = None) \
            -> Union[LLMResult, Generator]:
        token = self._get_token(credentials["dashscope_api_key"])
        return self._generate(model, token, prompt_messages, model_parameters, tools, stop, stream, user)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        token = self._get_token(credentials["api_key"])
        message1 = UserPromptMessage(content="ping")
        prompt_messages = [message1]
        self._generate(model, token, prompt_messages, stream=False)

    @staticmethod
    def generate_token(apikey: str, exp_seconds: int):
        try:
            id, secret = apikey.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)

        payload = {
            "api_key": id,
            "exp": int(round(time.time())) + exp_seconds,
            "timestamp": int(round(time.time())),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "typ": "JWT", "sign_type": "SIGN"},
        )

    def _get_token(self, api_key: str) -> str:
        redis_key = "jiutian_token:" + hashlib.sha256(api_key.encode()).hexdigest()
        # client = Redis(host='localhost', port=6379, db=0, password="difyai123456")
        # ext_redis.redis_client.initialize(client)
        token = redis_client.get(redis_key)
        if token is not None:
            return token.decode("utf-8")
        lock_key = "lock:" + redis_key
        if redis_client.setnx(lock_key, 1):
            redis_client.expire(lock_key, 10)
            try:
                # 生成新的 token 并存入缓存
                token = self.generate_token(api_key, TOKEN_EXPIRATION_SECONDS)
                redis_client.setex(redis_key, TOKEN_EXPIRATION_SECONDS, token)
                return token
            finally:
                # 释放锁
                redis_client.delete(lock_key)
        else:
            # 等待锁被释放，然后重新尝试获取 token
            time.sleep(0.1)
            return self._get_token(api_key)

    def _generate(self, model, credentials, prompt_messages, model_parameters, tools, stop, stream, user):
        messages = [self.trans_prompt_2messages(m) for m in prompt_messages]

        # nothing different between chat model and completion model in tongyi
        response = self._get_response(model=model,
                                      stream=stream,
                                      messages=messages,
                                      credential=credentials
                                      )
        if stream:
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages)

        return self._handle_generate_response(model, credentials, response, prompt_messages)

    def _to_credential_kwargs(self, credentials):
        credentials_kwargs = {
            "api_key": credentials["dashscope_api_key"],
        }
        return credentials_kwargs

    def _convert_prompt_messages_to_tongyi_messages(self, prompt_messages):
        jiutian_messages = []
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, SystemPromptMessage):
                jiutian_messages.append(
                    {
                        "role": "system",
                        "content": prompt_message.content
                    }
                )
            elif isinstance(prompt_message, UserPromptMessage):
                if isinstance(prompt_message.content, str):
                    jiutian_messages.append(
                        {
                            "role": "user",
                            "content": prompt_message.content
                        }
                    )
            else:
                raise ValueError(f"Got unknown type {prompt_message}")

        return jiutian_messages

    def _handle_generate_stream_response(self,
                                         model: str,
                                         credentials: str,
                                         responses: Iterator,
                                         prompt_messages: list[PromptMessage],
                                         ) -> Generator:
        for line in responses:
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                data = json.loads(line)
            except Exception as e:
                if line.strip() == "[DONE]":
                    return
                print(f"faild to parse JSON : {e} with line :{line}")
                continue
            choices = data.get("choices", [])
            if not choices:
                continue
            for choice in choices:
                delta = choice.get("delta", {})
                finish_reason = delta.get("finish_reason", None)
                if finish_reason:
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=delta.get("index", 0),
                            message=AssistantPromptMessage(content=""),
                            finish_reason=finish_reason
                        )
                    )
                content = delta.get("content", "")
                if content:
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=delta.get("index", 0),
                            message=AssistantPromptMessage(content=content),
                            finish_reason=None
                        )
                    )

            if (data.get("usage")):
                # 先不管
                pass

    def _handle_generate_response(self, model, credentials: str, response, prompt_messages) -> LLMResult:
        data = response
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices found")
        usage = data.get("usage")
        prompt_token = usage.get("prompt_tokens")
        completion_token = usage.get("completion_tokens")
        total_token = usage.get("total_tokens")
        for choice in choices:
            finish_reason = choice.get("finish_reason", None)
            index = choice.get("index")
            message = choice.get("message", {})
            if finish_reason:
                content = message.get("content", {})
                role = message.get("role", "assistant")
                prompt_message = AssistantPromptMessage(content=content, role=role)
                return LLMResult(
                    model=model,
                    prompt_messages=prompt_messages,
                    message=prompt_message,
                    usage=self.get_usage(prompt_token, completion_token, total_token),
                )
            if data.get("usage"):
                pass

    @staticmethod
    def get_usage(prompt_tokens: int, completion_tokens: int, total_token: int) -> LLMUsage:
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_token,
            prompt_unit_price=Decimal("0.0"),
            prompt_price_unit=Decimal("0.0"),
            prompt_price=Decimal("0.0"),
            completion_unit_price=Decimal("0.0"),
            completion_price_unit=Decimal("0.0"),
            completion_price=Decimal("0.0"),
            total_price=Decimal("0.0"),
            currency="",
            latency=Decimal("0.0")
        )
        return usage

    def request_headers(self, api_key: str) -> dict[str, Any]:
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        }

    def _get_response(self, model: str, stream: bool, messages: list[dict], credential: str) -> Union[Iterator, dict]:
        if model in self._model_mapping:
            api_base = "http://jiutian.hq.cmcc/largemodel/api/v2/chat/completions"
        else:
            raise BadRequestError(f"Unknown model: {model}")
        post_message = self._build_parameters(model, stream, messages)
        try:
            response = post(url=api_base,
                            headers=self.request_headers(credential),
                            data=json.dumps(post_message),
                            stream=stream)
        except Exception as e:
            raise InternalServerError(f"faild to invoke model : {e}")
        if response.status_code != 200:
            try:
                resp = response.json()
                err = resp["error"]["type"]
                mes = resp["error"]["message"]
            except Exception as e:
                raise InternalServerError(f"failed to convert response to json  : {e} with text {response.text}")
            raise BadRequestError(f"API returned error: {err}, message: {mes}")
        if stream:
            return response.iter_lines()
        else:
            return response.json()

    @property
    def _model_mapping(self) -> dict:
        return {
            "jiutian-lan": "jiutian-lan"
        }

    def _build_parameters(
            self,
            model: str,
            stream: bool,
            messages: list[dict]

    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "stream": stream
        }
