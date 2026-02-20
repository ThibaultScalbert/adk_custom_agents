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

from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

class SilentLlmAgent(LlmAgent):
    """An LlmAgent that populates state but does not send a visible message."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        async for event in super()._run_async_impl(ctx):
            if event.is_final_response():
                # Yield the final event with content=None to suppress the message
                # while keeping actions for state/artifact updates.
                yield Event(
                    id=event.id,
                    invocation_id=event.invocation_id,
                    author=event.author,
                    content=None,
                    actions=event.actions,
                    grounding_metadata=event.grounding_metadata,
                )
            else:
                yield event
