from core.model_runtime.model_providers.__base.model_provider import ModelProvider


class JiutianProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        pass
