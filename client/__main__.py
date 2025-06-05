import asyncio
import httpx
import json
import asyncclick as click
from uuid import uuid4

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    TextPart,
    Task,
    Message,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    MessageSendConfiguration,
    SendStreamingMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
    JSONRPCErrorResponse,
)
from rich.console import Console
console = Console()

async def main():
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, "http://localhost:8080")
        card = await card_resolver.get_agent_card()

        print('======= Agent Card ========')
        print(card.model_dump_json(exclude_none=True))
        
        input("Press Enter to stream response from server...")
        prompt = click.prompt(
            '\nWhat do you want to send to the agent?'
        )
        
        client = A2AClient(httpx_client, agent_card=card)
        streaming = card.capabilities.streaming
        
        if not streaming:
            print("Streaming is not supported by the agent.")
            return
        
        message = Message(
            role='user',
            parts=[TextPart(text=prompt)],
            messageId=str(uuid4()),
        )

        payload = MessageSendParams(
            id=str(uuid4()),
            message=message,
            configuration=MessageSendConfiguration(
                acceptedOutputModes=['text'],
            ),
        )
        
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        
        async for result in response_stream:
            if isinstance(result.root, JSONRPCErrorResponse):
                print("Error: ", result.root.error)
                return False, contextId, taskId
            event = result.root.result
            contextId = event.contextId
            if (
                isinstance(event, Task)
            ):
                taskId = event.id
            elif (isinstance(event, TaskStatusUpdateEvent)
                  or isinstance(event, TaskArtifactUpdateEvent)
            ):
                taskId = event.taskId
            elif isinstance(event, Message):
                message = event
            console.print(
                'stream event =>', json.loads(event.model_dump_json(exclude_none=True))
            )
        # Upon completion of the stream. Retrieve the full task if one was made.
        if taskId:
            taskResult = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=taskId),
                )
            )
            taskResult = taskResult.root.result
            
            console.print(
                'Task Result => ', json.loads(taskResult.model_dump_json(exclude_none=True)), style="bold red"
            )


if __name__ == "__main__":
    asyncio.run(main())
