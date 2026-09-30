"""通用動作的參數模型。原版還有 navigate/click/scroll/... 十幾個瀏覽器動作的參數，已移除；
機器人動作的參數模型在 robot_agent/actions.py。"""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class DoneAction(BaseModel):
	text: str = Field(description='Final user message in the format the user requested')
	success: bool = Field(default=True, description='True if user_request completed successfully')
	files_to_display: list[str] | None = Field(default=[])


T = TypeVar('T', bound=BaseModel)


class StructuredOutputAction(BaseModel, Generic[T]):
	success: bool = Field(default=True, description='True if user_request completed successfully')
	data: T = Field(description='The actual output data matching the requested schema')


class NoParamsAction(BaseModel):
	model_config = ConfigDict(extra='ignore')
