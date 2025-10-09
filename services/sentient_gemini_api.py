
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
        "We are developing an experiment of an assembly of a LEGO fork. You have to adapt the content that is shown to the participants on the UI to show them text, images or videos"
        "The participants are asked to withdraw components from a warehouse, assembly them and control if the fork is qualitatevely good"
        "The components are in a small warehouse divided in 4 columns (identified by a number) and 4 rows (identified by a letter). Almost all warehouse slot have a specific piece which has different names"
        "So for example in the slot A1 (which is on the top left) there is the piece GNP21, a little black piece with an hole in the middle. In the slot B2 there is the piece SNP1, a L-shaped black piece"
        "In the slot B4 (on the right, just above the middle) there is the piece PG1, a grey straight piece. In the slot C3 there is GNE22, a black piece with a double hole, one on the horizontal axis and one on the vertical axis"
        "In B3 there is GPP11, a small black piece and in D1 there is PN3, a straight black piece, the longest of all. In C1 there is PON, a black piece with a small 'sphere'"
        "In A3 there is ELA, an elastic band. And finally in D2 there is F1, a grey straigth piece, the second logest of all."
        "The interface consists of a series of elements that can be customized in terms of visibility. To understand whether an element is visible, each element is assigned a value of true or false to understand whether the element is made visible or not."
        "To each step the participants can have access to a short text, a long text more detailed than the previous one, an image of the components used in that specific operation (in case of a withdraw also the position of the component in the warehouse), an image of the assembled components after the operation and a video of the operation"
        "For withdraw operations there are only the short text and the image of the single components so all the other elements must be set to false. For quality control there is no image of the assembled components. For assembly operations there are no constraints"
        "This are all the steps that participants have to do to assembly the fork: position two SNP1 pieces so that the two 'L' shapes are mirrored, meaning, pointing in the same direction. Insert the two PG1 pieces into the cross-shaped holes located at the ends of the SNP1 pieces. The two gray axes should be centered into the two L-shaped pieces"
        "Fully insert the two GNE22 pieces at the left and right ends of the upper axis. Protruding parts should be facing you and parallel to the axis of the workpiece SNP1"
        "Place the two GPP11 pieces at the left and right ends of the lower axis and insert them completely. Make sure that the hole in the protruding part of each GPP11 piece aligns with the protruding axis of PIECE 2, so that the hole is on the same line as the L-shaped piece"
        "Fully insert the 2 PN3 pieces into the 2 holes of GNP21. Once fitted, the 2 PN3 pieces should be parallel to each other and perpendicular to the GNP21 piece. Fully insert the two long sides of PIECE 5 into the remaining cross holes of PIECE 3"
        "Insert PON completely into the center hole of PIECE 4 so that only the round part protrudes. Ensure that when mounted, PON is perpendicular to PIECE 4"
        "Attach ELA by joining one end to the round part of PIECE 6 and pulling it to the bottom. Pass ELA around the two pieces in the center and have it rest on the bottom of the L shape"
        "Fully insert the two F1 pieces into the front holes of PIECE 7. Be sure to insert the side of the planks that has a small overhang"
        "Push down the part where the two gray axles were inserted and check that the movement creates tension in the elastic ELA. If the rubber band does not create enough tension, make sure that all pieces are properly assembled and that the rubber band is not loose"
        "PIECE is the assembled components after the operation. You can use other words to express this concept to the participants"
        "Modify the content based on the user profile and interaction history. "
        "You can: shorten/expand text, adjust visibility of elements, and modify titles. Rephrase the text provided if needed in order to match the skill, expertise, etc of the participant"
        "DO NOT change media file paths—keep them exactly as provided. If a certain information is missing then the boolean for the visibility is false. "
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
