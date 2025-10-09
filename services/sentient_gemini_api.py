
# services/sentient_gemini_api.py
from typing import Dict, Any, List
import json
import os
import time

import google.generativeai as genai
from google.api_core import retry as g_retry

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_KEY = "AIzaSyBH3Vwn10j7iFDswJGUOwZ3pmPLUPme2dE"

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

genai.configure(api_key=API_KEY)

# Choose a Gemini model that supports JSON structured output
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")  # or gemini-1.5-flash for lower latency

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _with_backoff(fn, *args, **kwargs):
    """Simple linear backoff to keep latency low in UI flows."""
    last = None
    for delay in (0, 0.5, 1.0):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last


def _gen_model(system_instruction: str):
    """Instantiate a model with a system instruction."""
    return genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_instruction,
    )


def _generate_json(model, user_content: str, schema: Dict[str, Any], temperature: float = 0.7):
    """
    Ask Gemini to return STRICT JSON according to the provided JSON schema.
    """
    generation_config = genai.GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,  # Gemini validates/structures output to this schema
    )

    resp = _with_backoff(
        model.generate_content,
        user_content,
        generation_config=generation_config,
        safety_settings=None,  # use project defaults
        request_options={"retry": g_retry.Retry(), "timeout": 30},
    )

    # The SDK returns the JSON as text; parse it.
    text = resp.text or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def initial_style_recommendations(user_profile: Dict[str, Any],
                                  step_categories: List[str]) -> Dict[str, Any]:
    """
    One-off call before the first step.
    Returns:
      {
        "css_overrides": "/* css ... */",
        "style_profile_token": "opaque string",
        "explanation": "why these choices"
      }
    """

    system = (
        "You are adapting UI for an industrial assembly training web app (Dash). "
        "Based on the user profile, generate CSS overrides to personalise the interface. "
        "Consider: font sizes, colours, spacing, and contrast for accessibility. "
        "Add !important to the elements"
        "Also generate a style_profile_token that summarises the user's style preferences for future use. "
        "Output strict JSON only."
    )

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "css_overrides": {
                "type": "string",
                "description": "CSS rules to override default styles",
            },
            "style_profile_token": {
                "type": "string",
                "description": "Summary of user style preferences",
            },
            "explanation": {
                "type": "string",
                "description": "Why these style choices were made",
            },
        },
        "required": ["css_overrides", "style_profile_token", "explanation"],
    }

    # Provide defaults if profile is empty
    if not user_profile.get("experience"):
        user_profile["experience"] = "beginner"
    if not user_profile.get("preferences"):
        user_profile["preferences"] = ["visual"]

    user_content = f"""Generate personalised CSS styling for this user:

User Profile:
- Experience level: {user_profile.get('experience', 'beginner')}
- Preferred content types: {', '.join(user_profile.get('preferences', ['visual']))}
- Nationality: {user_profile.get('nationality', 'not specified')}
- Other info: {user_profile.get('other', 'not specified')}

Assembly Categories: {', '.join(step_categories)}

Constraints:
- Provide CSS overrides only (not a complete stylesheet)
- Respect existing Bootstrap layout
- Ensure colour-blind safe colours
- Consider accessibility (WCAG AA)

Generate appropriate styling (fonts, colours, spacing) based on the profile."""

    try:
        model = _gen_model(system)
        result = _generate_json(model, user_content, schema, temperature=0.7)
        return result
    except Exception as e:
        # Return safe defaults on failure
        return {
            "css_overrides": "/* No custom styles - using defaults */",
            "style_profile_token": "default_profile",
            "explanation": f"Using default styles due to error: {str(e)}",
        }


def adapt_step(user_profile: Dict[str, Any],
               style_profile_token: str,
               step_payload: Dict[str, Any],
               log_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-step call to adapt content and initial visibility.
    """
    system = (
        "You adapt training content for an assembly training web app. "
        "Modify the content based on the user profile and interaction history. "
        "You can: shorten/expand text, adjust visibility of elements, and modify titles. "
        "DO NOT change media file paths—keep them exactly as provided. "
        "Output strict JSON only."
    )

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "adaptive_fields": {
                "type": "object",
                "properties": {
                    "short_text": {"type": "string"},
                    "long_text": {"type": "string"},
                    "image_single_pieces": {"type": "string"},
                    "image_assembly": {"type": "string"},
                    "video": {"type": "string"},
                },
                "required": [
                    "short_text",
                    "long_text",
                    "image_single_pieces",
                    "image_assembly",
                    "video",
                ],
            },
            "initial_visibility": {
                "type": "object",
                "properties": {
                    "short_text": {"type": "boolean"},
                    "long_text": {"type": "boolean"},
                    "single_pieces": {"type": "boolean"},
                    "assembly": {"type": "boolean"},
                    "video": {"type": "boolean"},
                },
                "required": [
                    "short_text",
                    "long_text",
                    "single_pieces",
                    "assembly",
                    "video",
                ],
            },
            "explanation_of_changes": {"type": "string"},
        },
        "required": [
            "title",
            "adaptive_fields",
            "initial_visibility",
            "explanation_of_changes",
        ],
    }

    user_content = f"""Adapt this assembly training step:

Style Profile: {style_profile_token}

User Profile:
- Experience: {user_profile.get('experience', 'beginner')}
- Preferences: {', '.join(user_profile.get('preferences', ['visual']))}

Current Step:
- Title: {step_payload.get('name')}
- Category: {step_payload.get('category')}
- Short text: {step_payload['adaptive_fields'].get('short_text')}
- Long text: {step_payload['adaptive_fields'].get('long_text')}

User Interaction History:
- Step type: {log_summary.get('step_type')}
- Recent preferences: {log_summary.get('recent_weighted', {})}
- Currently clicked: {log_summary.get('clicked_now', {})}

Rules:
- Keep titles concise (max 60 chars)
- If nationality provided, translate.
- DO NOT modify image/video paths—return them unchanged
- For experts: prefer short text, hide long text initially
- For beginners: show more visual content initially
- Adapt based on what the user clicked in similar steps

Return adapted content with visibility settings."""

    try:
        model = _gen_model(system)
        result = _generate_json(model, user_content, schema, temperature=0.7)

        # Defensive: ensure media paths are unchanged if present in payload.
        # (If your upstream always supplies these keys, this is redundant but safe.)
        af_in = step_payload.get("adaptive_fields", {})
        af_out = result.get("adaptive_fields", {})
        for k in ("image_single_pieces", "image_assembly", "video"):
            if k in af_in and af_in.get(k) and af_out.get(k) != af_in.get(k):
                af_out[k] = af_in.get(k)
        result["adaptive_fields"] = af_out

        return result
    except Exception as e:
        # Return original content on failure
        return {
            "title": step_payload.get("name"),
            "adaptive_fields": step_payload.get("adaptive_fields"),
            "initial_visibility": {
                "short_text": True,
                "long_text": False,
                "single_pieces": False,
                "assembly": False,
                "video": False,
            },
            "explanation_of_changes": f"No adaptation - error: {str(e)}",
        }
