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

import inspect
import logging
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, List, Literal, Optional, Type

from google.adk.agents.base_agent import BaseAgent, BaseAgentState
from google.adk.agents.base_agent_config import BaseAgentConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event, EventActions
from google.genai import types
from pydantic import Field
from typing_extensions import override

logger = logging.getLogger(__name__)


class FunctionAgentConfig(BaseAgentConfig):
  input_keys: List[Optional[str]] = Field(default_factory=list)
  function: Callable
  output_key: str
  input_mapping: Optional[Dict[str, str]] = None
  return_mode: Literal["state_update", "respond", "both"] = Field(
      default="state_update"
  )
  overwrite: bool = True


class FunctionAgent(BaseAgent):
  """A generic agent that performs deterministically a function call.

  It maps state keys to function arguments, executes a function, and stores the
  result in the state.
  """

  config_type: ClassVar[Type[BaseAgentConfig]] = FunctionAgentConfig

  input_keys: List[Optional[str]] = Field(default_factory=list)
  function: Callable
  output_key: str
  input_mapping: Optional[Dict[str, str]] = None
  return_mode: Literal["state_update", "respond", "both"] = Field(
      default="state_update",
      description="Determines what the agent does with the result",
  )
  context_arg_name: Optional[str] = Field(
      default=None,
      description=(
          "The name of the argument to inject the InvocationContext into"
      ),
  )
  overwrite: bool = Field(
      default=True,
      description=(
          "If False, the function will not run if the output_key is already in"
          " the state"
      ),
  )

  def __init__(self, **data: Any):
    super().__init__(**data)
    if self.return_mode not in ["state_update", "respond", "both"]:
      raise ValueError(
          f"Invalid return_mode: {self.return_mode}. Must be one of"
          " 'state_update', 'respond', 'both'."
      )

  @classmethod
  def from_function(
      cls,
      func: Callable,
      output_key: str,
      name: Optional[str] = None,
      input_mapping: Optional[Dict[str, str]] = None,
      return_mode: Literal["state_update", "respond", "both"] = "state_update",
      overwrite: bool = True,
  ) -> "FunctionAgent":
    sig = inspect.signature(func)
    input_keys = []
    mapping = input_mapping or {}
    context_arg_name = None

    for param_name, param in sig.parameters.items():
      is_context_arg = False
      if param.annotation is InvocationContext:
        is_context_arg = True
      elif (
          param_name in ["ctx", "context"]
          and param.annotation is inspect.Parameter.empty
      ):
        is_context_arg = True

      if is_context_arg:
        if context_arg_name is not None:
          raise ValueError(
              f"Function {func.__name__} can have at most one argument for"
              f" InvocationContext, found {context_arg_name} and {param_name}"
          )
        context_arg_name = param_name
      else:
        state_key = mapping.get(param_name, param_name)
        input_keys.append(state_key)

    agent_name = name or func.__name__
    return cls(
        name=agent_name,
        input_keys=input_keys,
        function=func,
        output_key=output_key,
        input_mapping=mapping,
        return_mode=return_mode,
        context_arg_name=context_arg_name,
        overwrite=overwrite,
    )

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    logger.info(
        f"[{self.name}] Running FunctionAgent tool with return_mode:"
        f" {self.return_mode}, overwrite: {self.overwrite}."
    )

    if not self.overwrite and self.output_key in ctx.session.state:
      logger.info(
          f"[{self.name}] Skipping execution: output_key '{self.output_key}'"
          " already exists in state and overwrite is False."
      )
      return

    tool_kwargs = {}
    if self.context_arg_name:
      tool_kwargs[self.context_arg_name] = ctx

    sig = inspect.signature(self.function)
    state_param_names = [
        name for name in sig.parameters if name != self.context_arg_name
    ]

    if len(state_param_names) != len(self.input_keys):
      raise ValueError(
          f"Mismatch between state parameters [{', '.join(state_param_names)}]"
          f" and input_keys [{', '.join(self.input_keys)}]"
      )

    for i, param_name in enumerate(state_param_names):
      key = self.input_keys[i]
      if key is None:
        tool_kwargs[param_name] = None
      else:
        tool_kwargs[param_name] = ctx.session.state.get(key)

    try:
      if inspect.iscoroutinefunction(self.function):
        result = await self.function(**tool_kwargs)
      else:
        result = self.function(**tool_kwargs)

      event_content = None
      event_actions = EventActions()

      if self.return_mode in ["respond", "both"]:
        event_content = types.Content(parts=[types.Part(text=str(result))])
        logger.info(f"[{self.name}] Responding with result.")

      if self.return_mode in ["state_update", "both"]:
        event_actions.state_delta = {self.output_key: result}
        logger.info(
            f"[{self.name}] Successfully updated state key '{self.output_key}'"
        )

      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          content=event_content,
          actions=event_actions,
      )

    except Exception as e:
      error_msg = f"Error executing tool '{self.function.__name__}': {str(e)}"
      logger.error(f"[{self.name}] {error_msg}")

      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          content=types.Content(
              parts=[types.Part(text=f"System Error: {error_msg}")]
          ),
      )

  @override
  @classmethod
  def _parse_config(
      cls: Type["FunctionAgent"],
      config: FunctionAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    kwargs.update({
        "input_keys": config.input_keys,
        "function": config.function,
        "output_key": config.output_key,
        "input_mapping": config.input_mapping,
        "overwrite": config.overwrite,
    })
    return kwargs