"""遙測的空殼：不收集、不傳送任何東西。保留 ProductTelemetry 介面給 agent/tools 呼叫。"""
import logging

logger = logging.getLogger(__name__)


def singleton(cls):
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]
    return wrapper


@singleton
class ProductTelemetry:
    def __init__(self) -> None:
        self._posthog_client = None
        self.debug_logging = False

    def capture(self, event) -> None:
        return None

    def flush(self) -> None:
        return None
