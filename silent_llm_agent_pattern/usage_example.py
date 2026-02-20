# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types

from silent_llm_agent_pattern.agent import SilentLlmAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define a Mock Agent that acts like an LlmAgent but yields pre-defined events
class MockLlmAgent(LlmAgent):
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Simulate an intermediate event (e.g., thinking or tool use)
        yield Event(
            author=self.name,
            content=None,
            actions=EventActions(state_delta={"summary": "User input processed"})
        )

        # Simulate the final text response
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text="I have updated the summary.")])
        )

# 2. Combine SilentLlmAgent (the pattern) with MockLlmAgent (the behavior)
# MRO: SilentMockAgent -> SilentLlmAgent -> MockLlmAgent -> LlmAgent
# SilentLlmAgent._run_async_impl will call super(), which hits MockLlmAgent._run_async_impl
class SilentMockAgent(SilentLlmAgent, MockLlmAgent):
    pass

async def main():
    print("--- Silent Agent Usage Example ---")
    
    # Initialize the silent mock agent
    agent = SilentMockAgent(
        model="mock-model",
        name="silent_worker",
        instruction="Process input and update state silently."
    )

    runner = InMemoryRunner(agent=agent)
    session_id = "session_example"
    
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user1",
        session_id=session_id
    )

    print("Running agent...")
    events_received = []
    async for event in runner.run_async(
        user_id="user1", 
        session_id=session_id, 
        new_message=types.Content(parts=[types.Part(text="Hello")])
    ):
        events_received.append(event)
        if event.content:
            print(f"Received Content: {event.content.parts[0].text}")
        else:
            print(f"Received Event (No Content). Actions: {event.actions}")

    # Verification
    final_event = events_received[-1]
    
    if final_event.content is None:
        print("\nSUCCESS: The final event content was suppressed.")
    else:
        print("\nFAILURE: The final event content was NOT suppressed.")

    if final_event.actions:
        print("Action preserved? Yes (This might be from the merged previous event or the final one if it had actions)")
    
    # Check if the intermediate action was preserved
    action_event = events_received[0]
    if action_event.actions.state_delta and action_event.actions.state_delta.get("summary") == "User input processed":
         print("SUCCESS: Intermediate actions were preserved.")

if __name__ == "__main__":
    asyncio.run(main())