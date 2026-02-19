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
from typing import Callable
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import InMemoryRunner
from google.genai import types

# Import the custom agent
from conditional_sequential_pattern.agent import ConditionalSequentialAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Define Helper Agents ---

# Agent 1: Preprocessor (e.g., checks if an item is in stock)
# Simulating an agent that might set a state variable "valid_sku"
class PreprocessorAgent(LlmAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        # Simulate checking SKU
        logger.info("PreprocessorAgent: Checking SKU...")
        sku_valid = True # Simulate a valid SKU
        # In a real scenario, this decision would come from LLM or tool
        
        # Let's toggle validity based on user input for demonstration
        if "invalid" in ctx.message.content.parts[0].text:
             sku_valid = False
             
        ctx.session.state["valid_sku"] = sku_valid
        logger.info(f"PreprocessorAgent: SKU valid? {sku_valid}")
        
        yield types.Content(parts=[types.Part(text=f"Checked SKU. Valid: {sku_valid}")])

# Agent 2: Parallel Agent (Simulated next step)
class ParallelAgent(LlmAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("ParallelAgent: Processing order...")
        yield types.Content(parts=[types.Part(text="Processing order in parallel...")])

# --- 2. Define Condition Logic ---

def valid_sku_check(ctx: InvocationContext) -> bool:
    """
    Checks if preprocessor_agent signaled to stop (valid_sku is False).
    """
    valid_sku_state = ctx.session.state.get("valid_sku")
    # If explicitly False, stop. If None or True, continue.
    if valid_sku_state is False:
      logger.info("Stopping sequence because SKU is invalid.")
      return False
    return True

# --- 3. Create Agents ---

preprocessor_agent = PreprocessorAgent(
    name="preprocessor_agent",
    model="gemini-2.5-flash", 
    instruction="Check SKU validity."
)

parallel_agent = ParallelAgent(
    name="parallel_agent",
    model="gemini-2.5-flash",
    instruction="Process valid orders."
)

# --- 4. Create Conditional Sequence ---

root_agent = ConditionalSequentialAgent(
    name="root_agent",
    sub_agents=[preprocessor_agent, parallel_agent],
    description="Conditional Sequence: Preprocessor -> (if valid) -> Parallel",
    should_continue=valid_sku_check
)

# --- 5. Run Example ---

async def main():
    runner = InMemoryRunner(agent=root_agent)
    
    print("--- Scenario 1: Valid SKU ---")
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user1",
        session_id="session_valid"
    )
    async for event in runner.run_async(
        user_id="user1", 
        session_id="session_valid", 
        new_message=types.Content(parts=[types.Part(text="Check this valid SKU")])
    ):
        if event.content:
            print(f"Response: {event.content.parts[0].text}")

    print("--- Scenario 2: Invalid SKU ---")
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user1",
        session_id="session_invalid"
    )
    async for event in runner.run_async(
        user_id="user1", 
        session_id="session_invalid", 
        new_message=types.Content(parts=[types.Part(text="Check this invalid SKU")])
    ):
        if event.content:
            print(f"Response: {event.content.parts[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
