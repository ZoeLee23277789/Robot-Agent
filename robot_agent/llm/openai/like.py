from dataclasses import dataclass

from robot_agent.llm.openai.chat import ChatOpenAI


@dataclass
class ChatOpenAILike(ChatOpenAI):
	"""
	A class for to interact with any provider using the OpenAI API schema.

	Args:
	    model (str): The name of the OpenAI model to use.
	"""

	model: str
