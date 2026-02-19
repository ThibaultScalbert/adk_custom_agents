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
import unittest
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from router_agent_pattern.agent import RouterAgent

# Mock LlmAgent to avoid actual API calls and verify execution
class MockLlmAgent(LlmAgent):
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Yield the name as content so we can assert it ran
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"RAN:{self.name}")]),
        )

# A simple sequential agent that wraps a single mock agent
def create_sequential_mock(name: str) -> SequentialAgent:
    mock = MockLlmAgent(
        model="gemini-2.5-flash",
        name=f"mock_{name}",
        instruction="Run"
    )
    return SequentialAgent(
        name=name,
        sub_agents=[mock],
        description=f"Sequential wrapper for {name}"
    )

class TestRouterAgent(unittest.TestCase):
    def test_router_selects_correct_agent(self):
        asyncio.run(self._test_router_selects_correct_agent())

    async def _test_router_selects_correct_agent(self):
        """
        Test that RouterAgent selects exactly ONE sub-agent based on state.
        Scenario:
          - category="A" -> Runs agent_A (sequential)
          - category="B" -> Runs agent_B (sequential)
        """
        
        # 1. Route function
        def route_by_category(ctx: InvocationContext) -> str:
            category = ctx.session.state.get("category")
            if category == "A":
                return "agent_A"
            elif category == "B":
                return "agent_B"
            else:
                return "unknown"

        # 2. Setup Sub-Agents
        agent_A = create_sequential_mock("agent_A")
        agent_B = create_sequential_mock("agent_B")
        
        # 3. Setup Router
        router = RouterAgent(
            name="router",
            sub_agents=[agent_A, agent_B],
            route_function=route_by_category,
            description="Routes to A or B"
        )
        
        runner = InMemoryRunner(agent=router)

        # --- TEST CASE 1: Run with category="A" ---
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user1",
            session_id="session_A",
            state={"category": "A"} # Pre-populate state
        )
        
        events_A = []
        async for event in runner.run_async(user_id="user1", session_id="session_A", new_message=types.Content(parts=[types.Part(text="start")])):
            events_A.append(event)
            
        # Verify agent_A ran
        # Look for events from mock_agent_A (sub-agent of agent_A)
        ran_A = any("RAN:mock_agent_A" in str(e.content) for e in events_A if e.content)
        ran_B = any("RAN:mock_agent_B" in str(e.content) for e in events_A if e.content)
        
        self.assertTrue(ran_A, "Agent A should have run when category='A'")
        self.assertFalse(ran_B, "Agent B should NOT have run when category='A'")

        # --- TEST CASE 2: Run with category="B" ---
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user2",
            session_id="session_B",
            state={"category": "B"} # Pre-populate state
        )
        
        events_B = []
        async for event in runner.run_async(user_id="user2", session_id="session_B", new_message=types.Content(parts=[types.Part(text="start")])):
            events_B.append(event)
            
        # Verify agent_B ran
        ran_A_2 = any("RAN:mock_agent_A" in str(e.content) for e in events_B if e.content)
        ran_B_2 = any("RAN:mock_agent_B" in str(e.content) for e in events_B if e.content)
        
        self.assertFalse(ran_A_2, "Agent A should NOT have run when category='B'")
        self.assertTrue(ran_B_2, "Agent B should have run when category='B'")

        # --- TEST CASE 3: Unknown Category ---
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user3",
            session_id="session_C",
            state={"category": "C"} # Unknown
        )
        
        events_C = []
        async for event in runner.run_async(user_id="user3", session_id="session_C", new_message=types.Content(parts=[types.Part(text="start")])):
            events_C.append(event)
            
        # Verify neither ran
        ran_A_3 = any("RAN:mock_agent_A" in str(e.content) for e in events_C if e.content)
        ran_B_3 = any("RAN:mock_agent_B" in str(e.content) for e in events_C if e.content)
        
        self.assertFalse(ran_A_3, "Agent A should NOT run for unknown category")
        self.assertFalse(ran_B_3, "Agent B should NOT run for unknown category")

if __name__ == "__main__":
    unittest.main()
