# services/sentient_openAI_api.py
import os
from typing import Dict, Any, List, Optional
import json
import time
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)  # expects OPENAI_API_KEY in env

MODEL = "gpt-4.1"  # fast + supports structured outputs


def _with_backoff(fn, *args, **kwargs):
    # simple linear backoff; keep it short to avoid UI lag
    for delay in (0, 0.5, 1.0):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last



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
    system = """You are adapting UI for an industrial assembly training webapp (Dash). 
    Based on the user profile, generate CSS overrides to personalize the interface.
    Consider: font sizes, colors, spacing, contrast for accessibility.
    Also generate a style_profile_token that summarizes the user's style preferences for future use.
    Output strict JSON only."""

    schema = {
        "type": "object",
        "properties": {
            "css_overrides": {"type": "string", "description": "CSS rules to override default styles"},
            "style_profile_token": {"type": "string", "description": "Summary of user style preferences"},
            "explanation": {"type": "string", "description": "Why these style choices were made"}
        },
        "required": ["css_overrides", "style_profile_token", "explanation"],
        "additionalProperties": False
    }

    # Provide defaults if profile is empty
    if not user_profile.get('experience'):
        user_profile['experience'] = 'beginner'
    if not user_profile.get('preferences'):
        user_profile['preferences'] = ['visual']

    user_content = f"""Generate personalized CSS styling for this user:

User Profile:
- Experience level: {user_profile.get('experience', 'beginner')}
- Preferred content types: {', '.join(user_profile.get('preferences', ['visual']))}
- Nationality: {user_profile.get('nationality', 'not specified')}
- Other info: {user_profile.get('other', 'not specified')}

Assembly Categories: {', '.join(step_categories)}

Constraints:
- Provide CSS overrides only (not a complete stylesheet)
- Respect existing Bootstrap layout
- Ensure color-blind safe colors
- Consider accessibility (WCAG AA)

Generate appropriate styling (fonts, colors, spacing) based on the profile."""

    try:
        print(f"Sending request to OpenAI...")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "StyleOut",
                    "schema": schema,
                    "strict": True
                }
            },
            temperature=0.7
        )

        print(f"Received response from OpenAI")
        result = json.loads(resp.choices[0].message.content)
        print(f"Parsed result: {result.keys()}")
        return result

    except Exception as e:
        print(f"Error in OpenAI call: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return defaults
        return {
            "css_overrides": "/* No custom styles - using defaults */",
            "style_profile_token": "default_profile",
            "explanation": f"Using default styles due to error: {str(e)}"
        }


def adapt_step(user_profile: Dict[str, Any],
               style_profile_token: str,
               step_payload: Dict[str, Any],
               log_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-step call.
    """
    system = """You adapt training content for an assembly training webapp. 
    Modify the content based on user profile and interaction history.
    You can: shorten/expand text, adjust visibility of elements, modify titles.
    DO NOT change media file paths - keep them exactly as provided.
    Output strict JSON only."""

    schema = {
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
                    "video": {"type": "string"}
                },
                "required": ["short_text", "long_text", "image_single_pieces", "image_assembly", "video"],
                "additionalProperties": False
            },
            "initial_visibility": {
                "type": "object",
                "properties": {
                    "short_text": {"type": "boolean"},
                    "long_text": {"type": "boolean"},
                    "single_pieces": {"type": "boolean"},
                    "assembly": {"type": "boolean"},
                    "video": {"type": "boolean"}
                },
                "required": ["short_text", "long_text", "single_pieces", "assembly", "video"],
                "additionalProperties": False
            },
            "explanation_of_changes": {"type": "string"}
        },
        "required": ["title", "adaptive_fields", "initial_visibility", "explanation_of_changes"],
        "additionalProperties": False
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
- DO NOT modify image/video paths - return them unchanged
- For experts: prefer short text, hide long text initially
- For beginners: show more visual content initially
- Adapt based on what user clicked in similar steps

Return adapted content with visibility settings."""

    try:
        print(f"Adapting step: {step_payload.get('name')}")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "AdaptOut",
                    "schema": schema,
                    "strict": True
                }
            },
            temperature=0.7
        )

        result = json.loads(resp.choices[0].message.content)
        print(f"Step adapted successfully")
        return result

    except Exception as e:
        print(f"Error adapting step: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return original content
        return {
            "title": step_payload.get('name'),
            "adaptive_fields": step_payload.get('adaptive_fields'),
            "initial_visibility": {
                "short_text": True,
                "long_text": False,
                "single_pieces": False,
                "assembly": False,
                "video": False
            },
            "explanation_of_changes": f"No adaptation - error: {str(e)}"
        }