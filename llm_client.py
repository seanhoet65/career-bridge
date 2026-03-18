"""
llm_client.py
Centralized Gemini API client.

API key is read exclusively from .streamlit/secrets.toml — NEVER hardcoded in source.
This file is the single integration point for all LLM calls in the app.
"""

import json
import re
import streamlit as st
from google import genai
from google.genai import types


# ── Client singleton ──────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    """Return a configured Gemini client using the key from Streamlit secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Add it to .streamlit/secrets.toml:\n\nGEMINI_API_KEY = \"your-key-here\""
        )
    return genai.Client(api_key=api_key)


def _clean_json(raw: str) -> str:
    """Strip markdown code fences that some LLMs add around JSON."""
    raw = raw.strip()
    raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return raw.strip()


# ── Core call wrappers ────────────────────────────────────────────────────────

# gemini-2.0-flash: stable model with full function-calling support
MODEL = "gemini-2.0-flash"


def simple_call(prompt: str) -> str:
    """Single text prompt → text response. For CV extraction Call 1."""
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text


def json_call(prompt: str) -> dict | list:
    """
    Single prompt → parsed JSON dict or list.
    Uses response_mime_type='application/json' to guarantee clean JSON output.
    Used for CV normalization, JD analysis, and the 3-call roadmap pipeline.
    """
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    raw = _clean_json(response.text)
    return json.loads(raw)


def create_agent_chat(tools: list, system_instruction: str) -> object:
    """
    Create a persistent Gemini chat session with function calling enabled.
    Automatic Function Calling (AFC) handles the tool loop — the SDK calls
    Python functions directly, sends results back, and repeats until
    Gemini returns a final text response.

    The chat object is stored in Streamlit session state so the full
    conversation history is preserved across rerenders within a session.
    """
    client = _get_client()
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=8,
                ignore_call_history=False,  # Keep history so we can inspect tool calls
            ),
        ),
    )
    # Keep a strong reference to the client on the chat object so Python's GC
    # doesn't close the underlying HTTP connection while the chat is in use.
    chat._parent_client_ref = client
    return chat
