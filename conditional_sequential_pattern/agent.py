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

import logging
from typing import AsyncGenerator, Callable
from typing_extensions import override
from pydantic import PrivateAttr

from google.adk.agents import SequentialAgent
from google.adk.agents.sequential_agent import SequentialAgentState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
try:
    from google.adk.utils.context_utils import Aclosing
except ImportError:
    class Aclosing:
        def __init__(self, thing):
            self.thing = thing
        async def __aenter__(self):
            return self.thing
        async def __aexit__(self, *exc_info):
            if hasattr(self.thing, 'aclose'):
                await self.thing.aclose()

logger = logging.getLogger(__name__)

class ConditionalSequentialAgent(SequentialAgent):
    """
    A SequentialAgent that checks a condition after each sub-agent run.
    If should_continue(ctx) returns False, the sequence stops.
    """
    _should_continue: Callable[[InvocationContext], bool] = PrivateAttr()

    def __init__(self, *, should_continue: Callable[[InvocationContext], bool], **kwargs):
        super().__init__(**kwargs)
        self._should_continue = should_continue

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if not self.sub_agents:
            return

        # Initialize or resume the execution state from the agent state.
        agent_state = self._load_agent_state(ctx, SequentialAgentState)
        start_index = self._get_start_index(agent_state)

        pause_invocation = False
        resuming_sub_agent = agent_state is not None
        
        # We iterate through the sub_agents starting from where we left off (or 0)
        for i in range(start_index, len(self.sub_agents)):
            sub_agent = self.sub_agents[i]
            
            # If NOT resuming, we need to set the state for the *current* sub-agent
            # so that if we pause inside it, we know where to resume.
            if not resuming_sub_agent:
                if ctx.is_resumable:
                    agent_state = SequentialAgentState(current_sub_agent=sub_agent.name)
                    ctx.set_agent_state(self.name, agent_state=agent_state)
                    yield self._create_agent_state_event(ctx)

            # Execute the sub-agent
            async with Aclosing(sub_agent.run_async(ctx)) as agen:
                async for event in agen:
                    yield event
                    if ctx.should_pause_invocation(event):
                        pause_invocation = True

            if pause_invocation:
                return

            # Reset resuming flag after the first iteration
            resuming_sub_agent = False
            
            # Check if we should continue to the next agent
            if not self._should_continue(ctx):
                logger.info(f"Condition met to stop sequence after agent {sub_agent.name}")
                break

        # Mark the sequence as complete in the state
        if ctx.is_resumable:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)
