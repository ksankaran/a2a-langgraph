import click
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent import WeatherAgent
from agent_executor import WeatherAgentExecutor

def app(host: str = None, port: int = None) -> A2AStarletteApplication:
    """Creates and returns the A2AStarletteApplication instance."""
    url = f"http://{host}:{port}"
    capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
    skill = AgentSkill(
        id='get_weather_forecast',
        name='Get Weather Forecast',
        description='Provides weather forecast for a given latitude and longitude.',
        tags=['weather', 'forecast'],
        examples=[
            'get weather forecast for 37.7749,-122.4194',
            'what is the weather in New York City?',
            'forecast for 34.0522,-118.2437',
            'weather in London',
            'tell me the weather in Tokyo',
            'what is the weather like in Paris?',
            'give me the weather forecast for 51.5074,-0.1278',
        ],
    )
    agent_card = AgentCard(
        name='Weather Agent',
        description='An agent that provides weather forecasts based on latitude and longitude.',
        url=url,
        version='1.0.0',
        defaultInputModes=WeatherAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=WeatherAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    httpx_client = httpx.AsyncClient()
    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_notifier=InMemoryPushNotifier(httpx_client),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    return server.build()

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8080)
def main(host, port):
    """Starts the Subscriptions Agent server."""
    try:
        agent_app = app(host, port)

        import uvicorn

        uvicorn.run(agent_app, host=host, port=port)

    except Exception as e:
        print(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()