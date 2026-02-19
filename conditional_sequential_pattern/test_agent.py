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

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from conditional_sequential_pattern.agent import ConditionalSequentialAgent

from google.adk.events import Event, EventActions

# Mock Agent that just yields its name
class MockAgent(BaseAgent):
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Log run in state for verification
        run_log = ctx.session.state.get("run_log", [])
        new_log = list(run_log) + [self.name]
        
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"RAN:{self.name}")]),
            actions=EventActions(state_delta={"run_log": new_log})
        )

class TestConditionalSequentialAgent(unittest.TestCase):
    def test_sequence_continues_on_true(self):
        asyncio.run(self._test_sequence_continues_on_true())

    async def _test_sequence_continues_on_true(self):
        # Setup
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        
        def always_true(ctx): return True
        
        seq_agent = ConditionalSequentialAgent(
            name="seq_agent",
            sub_agents=[agent1, agent2],
            should_continue=always_true
        )
        
        runner = InMemoryRunner(agent=seq_agent)
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user",
            session_id="session_true",
            state={"run_log": []}
        )
        
        # Run
        async for _ in runner.run_async(user_id="user", session_id="session_true", new_message=types.Content(parts=[types.Part(text="go")])):
            pass
            
        # Verify both ran
        state = await runner.session_service.get_session(app_name=runner.app_name, user_id="user", session_id="session_true")
        self.assertEqual(state.state["run_log"], ["agent1", "agent2"])

    def test_sequence_stops_on_false(self):
        asyncio.run(self._test_sequence_stops_on_false())

    async def _test_sequence_stops_on_false(self):
        # Setup
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        
        # Condition: Stop after agent1
        def stop_after_agent1(ctx):
            log = ctx.session.state.get("run_log", [])
            if "agent1" in log:
                return False
            return True
        
        seq_agent = ConditionalSequentialAgent(
            name="seq_agent",
            sub_agents=[agent1, agent2],
            should_continue=stop_after_agent1
        )
        
        runner = InMemoryRunner(agent=seq_agent)
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user",
            session_id="session_false",
            state={"run_log": []}
        )
        
        # Run
        async for _ in runner.run_async(user_id="user", session_id="session_false", new_message=types.Content(parts=[types.Part(text="go")])):
            pass
            
        # Verify only agent1 ran
        state = await runner.session_service.get_session(app_name=runner.app_name, user_id="user", session_id="session_false")
        self.assertEqual(state.state["run_log"], ["agent1"])

if __name__ == "__main__":
    unittest.main()
