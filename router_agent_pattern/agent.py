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

from google.adk.agents import BaseAgent, SequentialAgent
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

class RouterAgent(BaseAgent):
    """
    A RouterAgent that selects exactly one sub-agent to run based on a routing function.
    """
    _route_function: Callable[[InvocationContext], str] = PrivateAttr()

    def __init__(self, *, route_function: Callable[[InvocationContext], str], **kwargs):
        super().__init__(**kwargs)
        self._route_function = route_function

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Determine the target agent name using the routing function
        target_agent_name = self._route_function(ctx)
        
        # Find the agent in sub_agents
        target_agent = next((agent for agent in self.sub_agents if agent.name == target_agent_name), None)

        if not target_agent:
            logger.warning(f"RouterAgent '{self.name}': No sub-agent found with name '{target_agent_name}'. Skipping execution.")
            return

        logger.info(f"RouterAgent '{self.name}': Routing to '{target_agent_name}'")

        # Run the selected agent
        async with Aclosing(target_agent.run_async(ctx)) as agen:
            async for event in agen:
                yield event
