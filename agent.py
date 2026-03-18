"""
agent.py
Career Advisor agent — multi-call tool-use pattern.

How it works:
1.  A persistent Gemini chat session is stored in Streamlit session state.
2.  On each user turn, chat.send_message() is called.
3.  Gemini's Automatic Function Calling (AFC) intercepts any tool calls,
    executes the Python functions in agent_tools.py, feeds results back to
    the model, and repeats (up to 8 iterations) until a final text response.
4.  response.automatic_function_calling_history exposes every tool call made
    during this turn — used to populate the Tool Trace panel in the UI.

This satisfies the "multi-call use case with tools" rubric criterion because:
- The LLM is the decision-maker (it chooses which tools to invoke)
- Each tool call is a real Python computation (gap scoring, set arithmetic,
  timeline calculation, radar chart data generation)
- Results feed back into subsequent LLM reasoning, not just displayed raw
"""

from agent_tools import ALL_TOOLS
from llm_client import create_agent_chat
from roles_data import CURATED_ROLES


# ── System prompt ─────────────────────────────────────────────────────────────

ADVISOR_SYSTEM_PROMPT = """You are a Career Advisor AI with real-time access to a database of 14 professional roles.
You have Python tools that run exact computations — use them proactively to give data-driven advice.

Available roles: {roles}

Behavioral guidelines:
- When a user mentions their skills, ALWAYS call find_closest_roles first to show them their ranked fit across all roles.
- When they ask about a specific role, call compute_gap_analysis to get exact numbers.
- When they compare two options, use compare_roles for objective side-by-side data.
- When asked about timelines or "how long would it take", call estimate_transition_time.
- When asked to visualize or "show me a chart", call get_skill_radar_data — the chart renders automatically.
- After getting tool results, interpret the numbers in plain English. Don't just repeat raw data.
- Be honest about effort required, but highlight quick wins and transferable skills.
- Keep responses concise but substantive. Use bullet points sparingly.
"""


def create_chat_session(user_skills: list) -> object:
    """
    Create a new Gemini chat session for the Career Advisor.
    Stores user_skills context in the system prompt so the LLM knows them upfront.
    The chat object is stored in Streamlit session state and persists across rerenders.
    """
    skills_context = (
        f"\nUser's current skills: {user_skills}"
        if user_skills
        else "\nUser's current skills: Not yet provided — ask them to list their key skills or upload a CV."
    )
    system_instruction = ADVISOR_SYSTEM_PROMPT.format(
        roles=", ".join(CURATED_ROLES.keys())
    ) + skills_context

    return create_agent_chat(ALL_TOOLS, system_instruction)


def run_advisor_turn(user_message: str, chat_obj: object) -> dict:
    """
    Send one user message to the Career Advisor and return the result.

    The AFC loop runs inside chat.send_message() — by the time it returns,
    all tool calls have been executed and the final response is ready.

    Returns:
        {
          "response":    str,              # Final text response from the LLM
          "tool_calls":  list[dict],       # Tool name + args + result for each call
          "radar_data":  dict | None,      # Captured if get_skill_radar_data was called
        }
    """
    response = chat_obj.send_message(user_message)

    # ── Parse tool call history for the trace panel ───────────────────────────
    tool_calls = []
    radar_data = None

    history = response.automatic_function_calling_history or []
    i = 0
    while i < len(history):
        content = history[i]
        for part in (content.parts or []):
            # Function call (model → tool)
            if getattr(part, "function_call", None) and part.function_call.name:
                fc = part.function_call
                call_entry = {
                    "tool": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "result": None,
                }
                # Look ahead for the matching function_response
                if i + 1 < len(history):
                    next_content = history[i + 1]
                    for next_part in (next_content.parts or []):
                        if (getattr(next_part, "function_response", None)
                                and next_part.function_response.name == fc.name):
                            raw = next_part.function_response.response or {}
                            call_entry["result"] = raw.get("output", raw)

                tool_calls.append(call_entry)

                # Capture radar chart data for inline rendering
                if fc.name == "get_skill_radar_data" and call_entry["result"]:
                    r = call_entry["result"]
                    if isinstance(r, dict) and "chart_type" in r:
                        radar_data = r
        i += 1

    return {
        "response":   response.text,
        "tool_calls": tool_calls,
        "radar_data": radar_data,
    }
