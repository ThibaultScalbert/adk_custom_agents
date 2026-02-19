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
import random
from typing import Callable, Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from router_agent_pattern.agent import RouterAgent

logger = logging.getLogger(__name__)

# --- 1. Define the Agents (Travel Industry Example) ---

# Agent 1: Flight Booking (Simple Agent)
# This demonstrates a simple, single-step interaction handled by one LLM agent.
flight_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="flight_agent",
    description="Handles flight search and booking in a single conversation.",
    instruction="Ask the user for their departure city, destination, and travel dates."
)

# Agent 2: Hotel Booking Flow (Sequential Agent)
# This demonstrates a multi-step process where strict ordering is required.
# Step 1: Search for hotels
hotel_search_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="hotel_search_agent",
    description="Handles hotel search.",
    instruction="Ask the user for their destination city, check-in, and check-out dates to find available hotels."
)

# Step 2: Payment and Confirmation
hotel_payment_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="hotel_payment_agent",
    description="Handles payment and confirmation.",
    instruction="Collect payment details and confirm the hotel booking."
)

# The SequentialAgent ensures these run in order: Search -> Payment
hotel_booking_flow = SequentialAgent(
    name="hotel_booking_flow",
    sub_agents=[hotel_search_agent, hotel_payment_agent],
    description="Sequence for booking hotels: Search -> Payment"
)


# --- 2. Define the Routing Logic ---

def random_travel_routing(ctx: InvocationContext) -> str:
    """
    Randomly decides which travel flow to run.
    Returns the NAME of the target agent.
    """
    # Simulate a decision ("flight booking" or "hotel booking")
    # In a real app, this might check ctx.session.state["intent"]
    decision = random.choice(["flight booking", "hotel booking"])
    
    if decision == "flight booking":
        logger.info("Routing decision: flight booking -> flight_agent")
        return "flight_agent"
    else:
        logger.info("Routing decision: hotel booking -> hotel_booking_flow")
        return "hotel_booking_flow"

# --- 3. Create the Router ---

travel_router = RouterAgent(
    name="travel_router",
    # Pass the sub-agents it can choose from.
    # Note: 'hotel_booking_flow' is a SequentialAgent that wraps multiple sub-agents.
    # The RouterAgent treats it as a single unit of work.
    sub_agents=[flight_agent, hotel_booking_flow],
    # Pass the function that picks one by name
    route_function=random_travel_routing,
    description="Routes users to either flight agent or hotel booking flow"
)
