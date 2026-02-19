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
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import InMemoryRunner
from google.genai import types

from function_agent_pattern.agent import FunctionAgent

# --- Test Functions ---
def sync_tool(a: int, b: int) -> int:
    return a + b

async def async_tool(a: int, b: int) -> int:
    return a * b

def tool_with_optional(a: int, b: Optional[int] = None) -> int:
    return a + (b if b is not None else 10)

def tool_that_raises(a: int) -> int:
    raise ValueError('Tool Error')

class TestFunctionAgent(unittest.TestCase):

    def setUp(self):
        self.runner = None # Initialized in helper if needed or per test

    async def _run_agent(self, agent: FunctionAgent, state: dict) -> list:
        runner = InMemoryRunner(agent=agent)
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user1",
            session_id="session1",
            state=state
        )
        
        events = []
        async for event in runner.run_async(
            user_id="user1", 
            session_id="session1", 
            new_message=types.Content(parts=[types.Part(text="start")])
        ):
            events.append(event)
        return events

    def test_deterministic_agent_sync_tool(self):
        asyncio.run(self._test_deterministic_agent_sync_tool())

    async def _test_deterministic_agent_sync_tool(self):
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='sum_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.author, agent.name)
        self.assertEqual(event.actions.state_delta, {'sum_result': 8})
        self.assertIsNone(event.content)

    def test_deterministic_agent_async_tool(self):
        asyncio.run(self._test_deterministic_agent_async_tool())

    async def _test_deterministic_agent_async_tool(self):
        agent = FunctionAgent.from_function(
            func=async_tool,
            output_key='product_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.author, agent.name)
        self.assertEqual(event.actions.state_delta, {'product_result': 15})
        self.assertIsNone(event.content)

    def test_deterministic_agent_explicit_input_keys(self):
        asyncio.run(self._test_deterministic_agent_explicit_input_keys())

    async def _test_deterministic_agent_explicit_input_keys(self):
        agent = FunctionAgent(
            name='ExplicitSyncTool',
            function=sync_tool,
            output_key='sum_result',
            input_keys=['val1', 'val2'],
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'sum_result': 8})

    def test_deterministic_agent_input_mapping(self):
        asyncio.run(self._test_deterministic_agent_input_mapping())

    async def _test_deterministic_agent_input_mapping(self):
        def my_func(alpha, beta):
            return alpha - beta

        agent = FunctionAgent.from_function(
            func=my_func,
            output_key='diff_result',
            input_mapping={'alpha': 'val1', 'beta': 'val2'},
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'diff_result': 2})
        self.assertIsNone(events[0].content)

    def test_deterministic_agent_with_optional_arg_none(self):
        asyncio.run(self._test_deterministic_agent_with_optional_arg_none())

    async def _test_deterministic_agent_with_optional_arg_none(self):
        agent = FunctionAgent.from_function(
            func=tool_with_optional,
            output_key='optional_result',
            input_mapping={'a': 'val1', 'b': 'missing_key'},
        )
        # missing_key is not in state
        events = await self._run_agent(agent, {'val1': 5})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'optional_result': 15})

    def test_deterministic_agent_with_optional_arg_provided(self):
        asyncio.run(self._test_deterministic_agent_with_optional_arg_provided())

    async def _test_deterministic_agent_with_optional_arg_provided(self):
        agent = FunctionAgent.from_function(
            func=tool_with_optional,
            output_key='optional_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'optional_result': 8})

    def test_deterministic_agent_none_in_input_keys(self):
        asyncio.run(self._test_deterministic_agent_none_in_input_keys())

    async def _test_deterministic_agent_none_in_input_keys(self):
        agent = FunctionAgent(
            name='ToolWithNone',
            function=tool_with_optional,
            output_key='optional_result',
            input_keys=['val1', None],
        )
        events = await self._run_agent(agent, {'val1': 5})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'optional_result': 15})

    def test_deterministic_agent_error_handling(self):
        asyncio.run(self._test_deterministic_agent_error_handling())

    async def _test_deterministic_agent_error_handling(self):
        agent = FunctionAgent.from_function(
            func=tool_that_raises,
            output_key='error_result',
            input_mapping={'a': 'val1'},
        )
        events = await self._run_agent(agent, {'val1': 5})

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.author, agent.name)
        self.assertEqual(event.actions.state_delta, {})
        self.assertIn("System Error: Error executing tool 'tool_that_raises': Tool Error", event.content.parts[0].text)

    def test_from_function_name(self):
        agent = FunctionAgent.from_function(sync_tool, 'out')
        self.assertEqual(agent.name, 'sync_tool')

    def test_from_function_custom_name(self):
        agent = FunctionAgent.from_function(sync_tool, 'out', name='CustomName')
        self.assertEqual(agent.name, 'CustomName')

    def test_from_function_input_keys(self):
        agent = FunctionAgent.from_function(sync_tool, 'out')
        self.assertEqual(agent.input_keys, ['a', 'b'])

    def test_return_mode_state_update_default(self):
        asyncio.run(self._test_return_mode_state_update_default())

    async def _test_return_mode_state_update_default(self):
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='sum_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'sum_result': 8})
        self.assertIsNone(events[0].content)

    def test_return_mode_respond(self):
        asyncio.run(self._test_return_mode_respond())

    async def _test_return_mode_respond(self):
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='sum_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
            return_mode='respond',
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {})
        self.assertEqual(events[0].content.parts[0].text, '8')

    def test_return_mode_both(self):
        asyncio.run(self._test_return_mode_both())

    async def _test_return_mode_both(self):
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='sum_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
            return_mode='both',
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'sum_result': 8})
        self.assertEqual(events[0].content.parts[0].text, '8')

    def test_invalid_return_mode(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            FunctionAgent(
                name='test',
                function=sync_tool,
                output_key='out',
                return_mode='invalid_mode',
            )

    def test_context_injection_type_hint(self):
        asyncio.run(self._test_context_injection_type_hint())

    async def _test_context_injection_type_hint(self):
        def tool_with_context(a: int, my_ctx: InvocationContext):
            self.assertIsInstance(my_ctx, InvocationContext)
            return a + my_ctx.session.state['val2']

        agent = FunctionAgent.from_function(
            tool_with_context, output_key='ctx_result', input_mapping={'a': 'val1'}
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'ctx_result': 8})
        self.assertEqual(agent.context_arg_name, 'my_ctx')
        self.assertEqual(agent.input_keys, ['val1'])

    def test_context_injection_arg_name(self):
        asyncio.run(self._test_context_injection_arg_name())

    async def _test_context_injection_arg_name(self):
        def tool_with_context_name(a: int, ctx):
            self.assertIsInstance(ctx, InvocationContext)
            return a + ctx.session.state['val2']

        agent = FunctionAgent.from_function(
            tool_with_context_name,
            output_key='ctx_result',
            input_mapping={'a': 'val1'},
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'ctx_result': 8})
        self.assertEqual(agent.context_arg_name, 'ctx')
        self.assertEqual(agent.input_keys, ['val1'])

    def test_overwrite_false_key_exists(self):
        asyncio.run(self._test_overwrite_false_key_exists())

    async def _test_overwrite_false_key_exists(self):
        """Test that execution is skipped if overwrite is False and key exists."""
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='exec_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
            overwrite=False,
        )
        # Pre-exist exec_result
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3, 'exec_result': 'initial_value'})
        
        self.assertEqual(len(events), 0) # No event yielded as it skipped

    def test_overwrite_false_key_not_exists(self):
        asyncio.run(self._test_overwrite_false_key_not_exists())

    async def _test_overwrite_false_key_not_exists(self):
        """Test that execution happens if overwrite is False but key doesn't exist."""
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='new_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
            overwrite=False,
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'new_result': 8})

    def test_overwrite_true_key_exists(self):
        asyncio.run(self._test_overwrite_true_key_exists())

    async def _test_overwrite_true_key_exists(self):
        """Test that execution happens if overwrite is True, even if key exists."""
        agent = FunctionAgent.from_function(
            func=sync_tool,
            output_key='exec_result',
            input_mapping={'a': 'val1', 'b': 'val2'},
            overwrite=True,
        )
        events = await self._run_agent(agent, {'val1': 5, 'val2': 3, 'exec_result': 'initial_value'})
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].actions.state_delta, {'exec_result': 8})

if __name__ == "__main__":
    unittest.main()