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
from google.adk.runners import InMemoryRunner
from google.genai import types
from function_agent_pattern.agent import FunctionAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the function
def calculate_sum(a: int, b: int) -> int:
    return a + b

# 2. Create the FunctionAgent
sum_agent = FunctionAgent(
    name="sum_agent",
    function=calculate_sum,
    output_key="result",
    # When using the constructor, we explicitly define the list of state keys
    # that correspond to the function arguments in order.
    # calculate_sum(a, b) -> a gets "val1", b gets "val2"
    input_keys=["val1", "val2"],
    return_mode="both" # Update state and respond
)

async def main():
    # 3. Set up runner and session
    runner = InMemoryRunner(agent=sum_agent)
    
    # 4. Initialize session with state
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user1",
        session_id="session1",
        state={"val1": 10, "val2": 20}
    )

    # 5. Run the agent
    print("Running FunctionAgent...")
    async for event in runner.run_async(
        user_id="user1", 
        session_id="session1", 
        new_message=types.Content(parts=[types.Part(text="start")])
    ):
        if event.content:
             print(f"Agent Response: {event.content.parts[0].text}")
        if event.actions and event.actions.state_delta:
             print(f"State Update: {event.actions.state_delta}")

if __name__ == "__main__":
    asyncio.run(main())
