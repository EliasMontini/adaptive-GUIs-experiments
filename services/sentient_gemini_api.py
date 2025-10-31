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
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

genai.configure(api_key=API_KEY)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

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
        response_schema=schema,
    )

    resp = _with_backoff(
        model.generate_content,
        user_content,
        generation_config=generation_config,
        safety_settings=None,
        request_options={"retry": g_retry.Retry(), "timeout": 30},
    )

    text = resp.text or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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
        "Add !important to the elements. "
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

    # if not user_profile.get("experience"):
    #     user_profile["experience"] = "beginner"
    # if not user_profile.get("preferences"):
    #     user_profile["preferences"] = ["visual"]

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
        return {
            "css_overrides": "/* No custom styles - using defaults */",
            "style_profile_token": "default_profile",
            "explanation": f"Using default styles due to error: {str(e)}",
        }


def adapt_step(user_profile: Dict[str, Any],
               style_profile_token: str,
               step_payload: Dict[str, Any],
               user_history_formatted: str,
               aggregated_preferences: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Per-step call to adapt content and initial visibility.

    Args:
        user_profile: User profile dictionary
        style_profile_token: Style profile token from initial_style_recommendations
        step_payload: Current step data
        user_history_formatted: Formatted history from HistoricalDataManager.format_for_prompt()
        aggregated_preferences: Optional dict of aggregated preferences for current step
    """
    system = (
        "You are an AI assistant helping to personalize an assembly training interface for LEGO fork assembly.\n\n"
        "CONTEXT:\n"
        "Participants withdraw components from a warehouse (4 columns × 4 rows), assemble them, and check quality.\n"
        "The warehouse layout:\n"
        "- A1: GNP21 (small black piece with hole)\n"
        "- B2: SNP1 (L-shaped black piece)\n"
        "- B4: PG1 (grey straight piece)\n"
        "- C3: GNE22 (black piece with double hole)\n"
        "- B3: GPP11 (small black piece)\n"
        "- D1: PN3 (longest straight black piece)\n"
        "- C1: PON (black piece with small sphere)\n"
        "- A3: ELA (elastic band)\n"
        "- D2: F1 (second longest grey straight piece)\n\n"
        "ASSEMBLY STEPS:\n"
        "1. Position two SNP1 pieces mirrored (L shapes pointing same direction)\n"
        "2. Insert two PG1 pieces into cross holes at SNP1 ends (centered)\n"
        "3. Insert two GNE22 pieces at left/right ends of upper axis (protruding parts facing you)\n"
        "4. Place two GPP11 pieces at left/right ends of lower axis (align holes)\n"
        "5. Insert two PN3 pieces into GNP21 holes (parallel to each other, perpendicular to GNP21)\n"
        "6. Insert long sides of PIECE 5 into remaining cross holes of PIECE 3\n"
        "7. Insert PON into center hole of PIECE 4 (round part protrudes, perpendicular)\n"
        "8. Attach ELA from PON round part, pull down, pass around center pieces, rest on L bottom\n"
        "9. Insert two F1 pieces into front holes of PIECE 7 (small overhang side)\n"
        "10. (and QUALITY CONTROL STEP) Push down where grey axles inserted, check ELA tension\n\n"
        "AVAILABLE CONTENT TYPES:\n"
        "- short_text: Brief instruction\n"
        "- long_text: Detailed instruction\n"
        "- single_pieces: Image of components (withdraw: shows warehouse position)\n"
        "- assembly: Image of assembled result\n"
        "- video: Video demonstration\n\n"
        "CONSTRAINTS BY STEP TYPE:\n"
        "- WITHDRAW: Only short_text + single_pieces available\n"
        "- QUALITY CONTROL: No assembly image available\n"
        "- ASSEMBLY: All content types available\n\n"
        "YOUR TASK:\n"
        "Based on user profile, interaction history, and aggregated preferences from other users:\n"
        "1. Determine which content should be INITIALLY VISIBLE (set to true)\n"
        "2. Adapt text complexity to user expertise\n"
        "3. Translate if nationality specified\n"
        "4. Keep titles concise (max 60 chars)\n"
        "5. DO NOT modify media file paths\n\n"
        "ADAPTATION STRATEGY:\n"
        "- Consider what user clicked in previous similar steps\n"
        "- Consider what majority of users preferred for this step\n"
        "- Balance user preferences with pedagogical effectiveness\n\n"
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

    # Format aggregated preferences if provided
    aggregated_prefs_text = ""
    if aggregated_preferences:
        aggregated_prefs_text = "\nAGGREGATED PREFERENCES FOR THIS STEP (from other users):\n"
        for content_type, percentage in aggregated_preferences.items():
            # Handle both Content enum and dict values
            content_name = content_type.value if hasattr(content_type, 'value') else str(content_type)
            # Ensure percentage is a number
            pct_value = percentage if isinstance(percentage, (int, float)) else 0
            if pct_value > 0:
                aggregated_prefs_text += f"  - {content_name}: {pct_value*100:.1f}% of users viewed this\n"

    user_content = f"""Adapt this assembly training step:

STYLE PROFILE: {style_profile_token}

USER PROFILE:
- Experience: {user_profile.get('experience', 'not specified')}
- Preferences: {', '.join(user_profile.get('preferences', ['not specified']))}
- Nationality: {user_profile.get('nationality', 'not specified')}

CURRENT STEP:
- Title: {step_payload.get('name')}
- Category: {step_payload.get('category')}
- Short text: {step_payload['adaptive_fields'].get('short_text', '')}
- Long text: {step_payload['adaptive_fields'].get('long_text', '')}

{user_history_formatted}

{aggregated_prefs_text}

Based on the user's history and aggregated preferences, determine:
1. Which content types should be INITIALLY VISIBLE
2. How to adapt the text complexity
3. Whether to translate (if nationality specified)

Return adapted content with visibility settings and explanation."""

    try:
        model = _gen_model(system)
        result = _generate_json(model, user_content, schema, temperature=0.7)

        # Ensure media paths are unchanged
        af_in = step_payload.get("adaptive_fields", {})
        af_out = result.get("adaptive_fields", {})
        for k in ("image_single_pieces", "image_assembly", "video"):
            if k in af_in and af_in.get(k):
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