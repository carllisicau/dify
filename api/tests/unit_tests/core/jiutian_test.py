import unittest
from unittest.mock import MagicMock, patch

from redis import Redis

from core.model_runtime.entities import PromptMessage, UserPromptMessage
from core.model_runtime.model_providers.jiutian.llm.llm import JiutianLanLLmModel
from extensions import ext_redis
from extensions.ext_redis import init_app, redis_client


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 手动初始化 Redis 客户端
        cls.redis_client = Redis(host='localhost', port=6379, db=0, password="difyai123456")
        try:
            cls.redis_client.ping()  # 发送 PING 请求来验证 Redis 连接是否可用
            print("Successfully connected to Redis!")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            raise e
        # 如果你的应用代码是使用 ext_redis 的方式引用 redis_client，需要手动赋值
        # 假设 your_project.extensions.ext_redis 是您的 Redis 客户端的引用
        from api.extensions import ext_redis
        ext_redis.redis_client.initialize(cls.redis_client)

        # 延迟检查 redis_client 是否被正确初始化
        if ext_redis.redis_client._client is None:
            raise RuntimeError("Redis client initialization failed in setUpClass")
        else:
            print("Redis client successfully initialized and ready to use.")

    def test_something(self):
        instance = JiutianLanLLmModel()
        model = "jiutian-lan"
        credential = {"api_key": "672ac83494a960152b648290.30ETQfwyvvFoYF1r/pbgVDEydEIaC8EN"}
        prompt_messages = [UserPromptMessage(content="what is the capital of france ?")]
        model_parameter = {"temperature": 0.7}
        stream = False
        user = "bc36bfc9-4fe4-47d0-ae29-f78d70404729"
        result = instance._invoke(model, credential, prompt_messages, model_parameter, None, None, stream, user)


if __name__ == '__main__':
    unittest.main()
