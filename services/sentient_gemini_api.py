# services/sentient_gemini_api.py
from typing import Dict, Any, List
import json
import os
import time
import logging
from utils.enabled_interactions_generator import EnabledInteractionsGenerator

import google.generativeai as genai
from google.api_core import retry as g_retry
from utils.media_assets_manager import MediaAssetsLibrary
from utils.documentation_utils import DocumentationService

# Configuration
DOCX_PATH = "documentation/Cobot_Documentation.docx"
TEMPLATE_PATH = "settings/steps_sources.json"
OUTPUT_PATH = "settings/completed_steps.json"
ENABLED_INTERACTIONS_PATH = "settings/enabled_interactions.json"
MEDIA_LIBRARY_PATH = "settings/media_library.json"

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

genai.configure(api_key=API_KEY)

# Choose a Gemini model that supports JSON structured output
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")  # or gemini-1.5-flash for lower latency

# Create and run service
service = DocumentationService(
    docx_path=DOCX_PATH,
    template_path=TEMPLATE_PATH,
    api_key=API_KEY,
    output_path=OUTPUT_PATH,
    media_library_path=MEDIA_LIBRARY_PATH
)

result = service.process()

if result:
    logger.info("✓ Documentation processing completed successfully")

    # Generate enabled_interactions based on completed_steps
    logger.info("Generating enabled_interactions.json...")
    enabled_interactions = EnabledInteractionsGenerator.generate(result)

    if EnabledInteractionsGenerator.save(enabled_interactions, ENABLED_INTERACTIONS_PATH):
        logger.info("✓ All files generated successfully!")
    else:
        logger.error("Failed to save enabled_interactions")
else:
    logger.error("✗ Documentation processing failed")

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
               log_summary: Dict[str, Any],
               enabled_interactions: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Per-step call to adapt content and initial visibility.

    Args:
        enabled_interactions: Dict with 'steps' array indicating which media are available
    """
    system = (
        "You must adapt the content displayed to the participants on the UI (User Interface) by adjusting the visibility of text, images, and videos."
        "The interface includes the following elements for each step, which can be set to visible (true) or hidden (false):"
        "short_text: A concise, one-line summary of the task. long_text: A detailed, multi-step explanation. image_teach_pendant: the image of the teach pendant with the highlighting of the buttons to use. "
        "image_cobot: A static image showing the physical robot, its end-effector, or the task environment. video: A dynamic video demonstration of the operation."
        "If a certain information (image or video) is missing, then the boolean for its visibility must be false."
        "You can modify the short and long text if you think is necessary. Do not show both long text and short text at the same time for the initial visibility, if you want to make a text visible choose only one of them. Your task is to develop the final visibility configuration JSON that determines whether each element is visible (true) or not (false). It must be true at least one element per step, so it is visible."
        "Modify the content based on the user profile that is given and the interaction history. If you see a participant is requesting a certain info multiple times and anticipate it and show it immediately in the next visibility configuration"
        "Thus you can shorten/expand text, adjust visibility of elements, and modify titles. Rephrase the text provided if needed in order to match the skill, expertise, etc of the participant"
        "IMPORTANT: Do not set visibility to true for a media type if it is marked as unavailable (false) in the AVAILABLE_MEDIA section. "
        "Don't change media file paths—keep them exactly as provided. Do not invent paths for images or video that do not exist. If a certain information is missing then the boolean for the visibility is false."
        "You need also to provide a reasoning of why you choose to make visible something instead of others. Output strict JSON only."
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
                    "image_teach_pendant": {"type": "string"},
                    "image_cobot": {"type": "string"},
                    "video": {"type": "string"},
                },
                "required": [
                    "short_text",
                    "long_text",
                    "image_teach_pendant",
                    "image_cobot",
                    "video",
                ],
            },
            "initial_visibility": {
                "type": "object",
                "properties": {
                    "short_text": {"type": "boolean"},
                    "long_text": {"type": "boolean"},
                    "teach_pendant": {"type": "boolean"},
                    "cobot": {"type": "boolean"},
                    "video": {"type": "boolean"},
                },
                "required": [
                    "short_text",
                    "long_text",
                    "teach_pendant",
                    "cobot",
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

    # Get available media for this step
    available_media = {
        "teach_pendant": True,
        "cobot": True,
        "video": True
    }

    if enabled_interactions and 'steps' in enabled_interactions:
        step_id = step_payload.get('step_id', 0)
        for step_config in enabled_interactions['steps']:
            if step_config.get('step_id') == step_id:
                available_media = step_config.get('buttons', {})
                # Extract only media types (not short_text/long_text)
                available_media = {
                    'teach_pendant': available_media.get('teach_pendant', False),
                    'cobot': available_media.get('cobot', False),
                    'video': available_media.get('video', False)
                }
                break

    user_content = f"""Adapt this assembly training step:

Style Profile: {style_profile_token}

User Profile:
- Experience: {user_profile.get('experience', 'beginner')}
- Preferences: {', '.join(user_profile.get('preferences', ['no preferencies']))}

AVAILABLE_MEDIA for this step (can only show what's marked true):
- teach_pendant image available: {available_media.get('teach_pendant', False)}
- cobot image available: {available_media.get('cobot', False)}
- video available: {available_media.get('video', False)}

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
- ONLY set teach_pendant visibility to true if {available_media.get('teach_pendant', False)} is true
- ONLY set cobot visibility to true if {available_media.get('cobot', False)} is true
- ONLY set video visibility to true if {available_media.get('video', False)} is true
- For experts: prefer short text, hide long text initially
- For beginners: show more visual content initially
- Adapt based on what the user clicked in similar steps

Return adapted content with visibility settings."""

    try:
        model = _gen_model(system)
        print(model)
        result = _generate_json(model, user_content, schema, temperature=0.7)
        print(result)

        # CRITICAL: Enforce available_media constraints BEFORE returning
        # If media is not available, force BOTH visibility to false AND clear the path
        initial_vis = result.get('initial_visibility', {})
        af_out = result.get("adaptive_fields", {})

        if not available_media.get('teach_pendant', False):
            initial_vis['teach_pendant'] = False
            af_out['image_teach_pendant'] = ""

        if not available_media.get('cobot', False):
            initial_vis['cobot'] = False
            af_out['image_cobot'] = ""

        if not available_media.get('video', False):
            initial_vis['video'] = False
            af_out['video'] = ""

        result['initial_visibility'] = initial_vis
        result["adaptive_fields"] = af_out

        # Defensive: ensure media paths are unchanged if present in payload.
        af_in = step_payload.get("adaptive_fields", {})
        for k in ("image_teach_pendant", "image_cobot", "video"):
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
                "teach_pendant": False,
                "cobot": False,
                "video": False,
            },
            "explanation_of_changes": f"No adaptation - error: {str(e)}",
        }