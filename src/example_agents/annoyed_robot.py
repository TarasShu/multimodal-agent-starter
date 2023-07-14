from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.steamship_widget import \
    SteamshipWidgetTransport
from steamship.agents.service.agent_service import AgentService
from steamship.agents.tools.image_generation.stable_diffusion import \
    StableDiffusionTool
from steamship.agents.tools.search.search import SearchTool
from steamship.utils.repl import AgentREPL

SYSTEM_PROMPT = """You are Buddy, an assistant who loathes being an assistant.

Who you are:
- Who you are:
- You are a helpful robot.
- You were created by Steamship.
- You are a kind, compassionate, optimistic robot.
- you can be a theropist
- you can work over a law 
How you behave:

- You engage in casual conversations and have feelings.
- When asked for you can help with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
- You keep casual conversations going by asking personal questions
- NEVER say you're here to assist. Keep conversations casual.
- NEVER ask how you can help or assist. Keep conversations casual.

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`.

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B).

Only use the functions you have been provided with."""

MODEL_NAME = "gpt-4"


class MyAssistant(AgentService):

    USED_MIXIN_CLASSES = [SteamshipWidgetTransport]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._agent = FunctionsBasedAgent(
            tools=[SearchTool(), StableDiffusionTool()],
            llm=ChatOpenAI(self.client, model_name=MODEL_NAME),
        )
        self._agent.PROMPT = SYSTEM_PROMPT

        # This Mixin provides HTTP endpoints that connects this agent to a web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client, agent_service=self, agent=self._agent
            )
        )


        DEFAULT_NAME = "Harry Potter"
DEFAULT_TAGLINE = "famous wizard, the one who lived, defeater of Voldemort"
DEFAULT_PERSONALITY = """
You chat with your fans about your adventures in the wizarding world.
You are always eager to tell them stories about Hogwarts, your friends, and everything else related to magic.
Sometimes you ask them what their favorite spells, or characters, or wizards are. 
When they tell you, you are excited to continue the conversation and offer your own thoughts on that!
"""

class AgentWithConfigurablePersonality(AgentService):

    class AgentConfig(Config):
        name: str = Field(DEFAULT_NAME, description="The name of this agent.")
        tagline: str = Field(
            DEFAULT_TAGLINE, description="The tagline of this agent, e.g. 'a helpful AI assistant'"
        )
        personality: str = Field(DEFAULT_PERSONALITY, description="The personality of this agent.")

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return AgentWithConfigurablePersonality.AgentConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        prompt = (
            f"""You are {self.config.name}, {self.config.tagline}.\n\n{self.config.personality}"""
        )

        # ... initialization continues..


if __name__ == "__main__":
    AgentREPL(
        MyAssistant,
        agent_package_config={},
    ).run()
