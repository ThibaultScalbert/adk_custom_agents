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

import unittest
import asyncio
from unittest.mock import MagicMock, patch
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types

from silent_llm_agent_pattern.agent import SilentLlmAgent

from datetime import datetime

# Helper to create an async generator
async def async_iter(items):
    for item in items:
        yield item

class TestSilentLlmAgent(unittest.TestCase):
    def test_suppresses_final_content(self):
        asyncio.run(self._test_suppresses_final_content())

    async def _test_suppresses_final_content(self):
        # 1. Setup Mock Events
        # Event 1: Thinking (intermediate)
        event1 = MagicMock(spec=Event)
        event1.is_final_response.return_value = False
        event1.content = types.Content(parts=[types.Part(text="Thinking...")])
        event1.actions = EventActions()
        event1.partial = False
        event1.timestamp = datetime.now()
        
        # Event 2: Final Response (should be suppressed)
        event2 = MagicMock(spec=Event)
        event2.is_final_response.return_value = True
        event2.content = types.Content(parts=[types.Part(text="Done.")])
        event2.actions = EventActions(state_delta={"key": "value"})
        event2.id = "ev2"
        event2.invocation_id = "inv2"
        event2.author = "agent"
        event2.grounding_metadata = None
        event2.partial = False
        event2.timestamp = datetime.now()

        # 2. Patch LlmAgent._run_async_impl to return our mock events
        with patch("google.adk.agents.LlmAgent._run_async_impl", return_value=async_iter([event1, event2])):
            
            agent = SilentLlmAgent(name="silent_agent", model="gemini-flash")
            runner = InMemoryRunner(agent=agent)
            
            await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id="user1",
                session_id="session1"
            )

            events = []
            async for event in runner.run_async(user_id="user1", session_id="session1", new_message=types.Content(parts=[types.Part(text="start")])):
                events.append(event)

            # 3. Assertions
            # We expect event1 to pass through unmodified
            self.assertEqual(events[0].content.parts[0].text, "Thinking...")
            
            # We expect event2 to have content=None but actions preserved
            self.assertIsNone(events[1].content)
            self.assertEqual(events[1].actions.state_delta, {"key": "value"})

if __name__ == "__main__":
    unittest.main()
