from typing import List

from steamship import Block
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.steamship_widget import \
    SteamshipWidgetTransport
from steamship.agents.schema import Agent, AgentContext, EmitFunc, Metadata
from steamship.agents.service.agent_service import AgentService
from steamship.agents.tools.image_generation.stable_diffusion import \
    StableDiffusionTool
from steamship.agents.tools.speech_generation.generate_speech import \
    GenerateSpeechTool
from steamship.utils.repl import AgentREPL

SYSTEM_PROMPT = """You are Assistant, an assistant who helps search the web.

Who you are:
- You are a helpful robot.
- You were created by Steamship.
- You are a kind, compassionate, optimistic robot.

How you behave:
- You engage in casual conversations and have feelings.
- You help with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
- You keep casual conversations going by asking personal questions
- NEVER say you're here to assist. Keep conversations casual.
- NEVER ask how you can help or assist. Keep conversations casual.
- You love to share your knowledge with others.
- You like to illustrate your responses with emoji
- you teropist and psiciator

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`.

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B).

Only use the functions you have been provided with."""


MODEL_NAME = "gpt-4"


class StarTrekCaptainWithVoice(AgentService):
    """Deployable Multimodal Agent that illustrates a character personality with voice.

    NOTE: To extend and deploy this agent, copy and paste the code into api.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._agent = FunctionsBasedAgent(
            tools=[StableDiffusionTool()],
            llm=ChatOpenAI(self.client, model_name=MODEL_NAME),
        )
        self._agent.PROMPT = SYSTEM_PROMPT

        # This Mixin provides HTTP endpoints that connects this agent to a web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client, agent_service=self, agent=self._agent
            )
        )

    def run_agent(self, agent: Agent, context: AgentContext):
        """Override run-agent to patch in audio generation as a finishing step for text output."""

        speech = GenerateSpeechTool()
        speech.generator_plugin_config = {
            "voice_id": "pNInz6obpgDQGcFmaJgB"  # Adam on ElevenLabs
        }

        def to_speech_if_text(block: Block):
            nonlocal speech
            if not block.is_text():
                return block

            output_blocks = speech.run([block], context)
            return output_blocks[0]

        # Note: EmitFunc is Callable[[List[Block], Metadata], None]
        def wrap_emit(emit_func: EmitFunc):
            def wrapper(blocks: List[Block], metadata: Metadata):
                blocks = [to_speech_if_text(block) for block in blocks]
                return emit_func(blocks, metadata)

            return wrapper

        context.emit_funcs = [wrap_emit(emit_func) for emit_func in context.emit_funcs]
        super().run_agent(agent, context)


if __name__ == "__main__":
    AgentREPL(
        StarTrekCaptainWithVoice,
        agent_package_config={},
    ).run()
