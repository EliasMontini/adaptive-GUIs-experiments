# services/sentient_gemini_api.py
from typing import Dict, Any, List, Optional
import json
import os
import time
import re

import google.generativeai as genai
from google.api_core import retry as g_retry

# Configuration
API_KEY = ""

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=API_KEY)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# Utilities
def _iter_prefs(pref_str: Optional[str]):
    """Splits a single string like 'visual, video' into normalized tokens."""
    if not pref_str:
        return []
    return [p.strip().lower() for p in re.split(r'[\s,]+', pref_str) if p.strip()]


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


def initial_style_recommendations(user_profile: Dict[str, Any],
                                  step_categories: List[str],
                                  page_structure: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 80)
    print("🎨 INITIAL STYLE RECOMMENDATIONS - START")
    print("=" * 80)

    system = (
        "You are a UI component stylist for an industrial training app built with Next.js and Tailwind CSS. "
        "Based on UI best design approach and standards, the user requests and profile, generate ONLY Tailwind CSS utility classes for each UI component."
        "Also generate a style_profile_token summarizing the user's preferences. "
        "Output valid JSON only with no markdown formatting."
        "Organise the page and its structure to fit with the device characteristics, if provided (default: laptop)"
    )

    component_properties = {}
    for component_name in page_structure.get("components", {}):
        component_properties[component_name] = {
            "type": "string",
            "description": f"Space-separated Tailwind utility classes for {component_name}"
        }

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "style_profile_token": {
                "type": "string",
                "description": "A concise summary of user style preferences (e.g., 'high-contrast, large-font, visual-first').",
            },
            "explanation": {
                "type": "string",
                "description": "A brief explanation of why these style choices were made for the user.",
            },
            "component_classes": {
                "type": "object",
                "properties": component_properties,
                "description": "Tailwind CSS utility classes for each component. Use only standard Tailwind classes."
            },
        },
        "required": ["style_profile_token", "explanation", "component_classes"],
    }

    user_preferences_list = _iter_prefs(user_profile.get('preferences', ''))
    component_descriptions = "\n".join([
        f"- **{name}**: {desc}"
        for name, desc in page_structure.get("components", {}).items()
    ])

    user_content = f"""Generate Tailwind CSS classes for each UI component based on the user profile.

**User Profile:**
- Experience level: {user_profile.get('experience', 'beginner')}
- Preferred content types: {', '.join(user_preferences_list) or 'not specified'}
- Nationality: {user_profile.get('nationality', 'not specified')}
- Other preferences: {user_profile.get('other', 'not specified')}

**Page Structure:**
{page_structure.get('description', 'Industrial assembly training interface')}

**Components to Style:**
{component_descriptions}

**Requirements:**
1. Use ONLY standard Tailwind utility classes (e.g., bg-blue-600, text-white, px-4, py-2, rounded-lg)
2. Allowed tokens. 
   state variants: hover:, focus:, active:, disabled:, motion-safe:
    colours (all variants apply): bg-, text-, border-, ring-, placeholder-, divide-, stroke-, fill-, from-, via-, to- plus palettes slate, gray, zinc, neutral, stone, red, orange, amber, yellow, lime, green, emerald, teal, cyan, sky, blue, indigo, violet, purple, fuchsia, pink, rose combined with levels 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950
    gradients: bg-gradient-to followed by one of t, tr, r, br, b, bl, l, tl
    opacity: opacity-0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100
    shadows: shadow-sm, md, lg, xl, 2xl, inner, none
    radius: rounded-none, sm, md, lg, full, xl, 2xl
    spacing: p, px, py, m, mx, my, gap combined with 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32; negative margins -m, -mx, -my with same scale; mx-auto; space-x, space-y with same scale
    width: w- followed by 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 1/2, 1/3, 2/3, 1/4, 3/4, full, screen, min, max
    height: h- followed by 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, full, screen, min, max
    min/max size: min-w, min-h combined with 0, full, min, max; max-w-xs to max-w-7xl, full; max-h-0, full, screen
    typography: text-xs to text-8xl; font-thin to font-black; leading-3 to leading-10; tracking-tighter to tracking-widest; whitespace-normal, nowrap, pre, pre-line, pre-wrap
    overflow and object fit: overflow-auto, hidden, visible, scroll; object-contain, cover, fill, none, scale-down
    display and visibility: block, inline, inline-block, flex, inline-flex, grid, contents, table, hidden, visible
    positioning: static, relative, absolute, fixed, sticky; top, right, bottom, left, inset-x, inset-y, -inset, and fractional/full values
    overflow extras: overscroll-auto, contain, none; scroll-smooth
    flex and grid: flex-1, auto, initial, none; basis values (0, 1/2, 1/3, 2/3, 1/4, 3/4, full, auto); grow, shrink; flex-row, row-reverse, col, col-reverse, wrap, nowrap, wrap-reverse; grid-cols-1 to 12; col-span-1 to 12; grid-rows-1 to 12; row-span-1 to 12; grid-flow-row, col, dense; auto-rows, auto-cols with min, max, fr
    alignment and distribution: justify-start, center, end, between, around, evenly; items-start, center, end, baseline, stretch; content-start, center, end, between, around, evenly; place-content and place-items variants
    layering, transform, motion: z-0,10,20,30,40,50,auto; inset-0,1,2,3,4,5,6,8,10,12,16,20,24,32; translate-0,1,2,3,4,5,6,8,10,12,16; scale-0,50,75,90,95,100,105,110,125,150; rotate-0,45,90,180,270; skew-0,1,2,3,6
    aspect ratio: aspect-auto, square, video
    transitions: transition-none, all, colors, opacity, shadow, transform; duration-75,100,150,200,300,500,700,1000; ease-linear, in, out, in-out

3. Match user preferences (e.g., high contrast, large text, dark theme if requested)
4. Apply best practices in UI design (header positioned at the top, interactive buttons).
5. Match user preferences (e.g., high contrast, large text, dark theme if requested)

**Examples:**
- Dark button: "bg-gray-800 text-white hover:bg-gray-700 px-6 py-3 rounded-lg font-semibold shadow-lg"
- High contrast card: "bg-white border-4 border-black p-8 rounded-xl shadow-2xl"
- Large text area: "text-xl leading-relaxed text-gray-900"

Return valid JSON with component_classes, style_profile_token, and explanation."""

    try:
        model = _gen_model(system)
        result = _generate_json(model, user_content, schema, temperature=0.5)
        print(f"✅ Style profile token: {result.get('style_profile_token')}")
        print(f"🎯 Component classes generated: {list(result.get('component_classes', {}).keys())}")
        print("=" * 80)
        return result
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        fallback_classes = {
            name: "bg-white rounded-lg shadow-sm p-4 border border-gray-200"
            for name in page_structure.get("components", {})
        }
        return {
            "style_profile_token": "default_profile",
            "explanation": f"Using default styles due to error: {str(e)}",
            "component_classes": fallback_classes
        }


def adapt_step(user_profile: Dict[str, Any],
               style_profile_token: str,
               step_payload: Dict[str, Any],
               log_summary: Dict[str, Any],
               user_preference: Optional[str] = None) -> Dict[str, Any]:
    print("=" * 80)
    print("📄 ADAPT STEP - START")
    print("=" * 80)

    system = (
        "You adapt training content for an industrial assembly web app. "
        "Modify content based on user profile and interaction history. "
        "You can: adjust text (shorten/expand), modify visibility flags, and change titles. "
        "DO NOT modify media file paths - keep them exactly as provided. "
        "Output valid JSON only with no markdown formatting."
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
                },
                "required": ["short_text", "long_text"],
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
                "required": ["short_text", "long_text", "single_pieces", "assembly", "video"],
            },
            "explanation_of_changes": {"type": "string"},
        },
        "required": ["title", "adaptive_fields", "initial_visibility", "explanation_of_changes"],
    }

    # main.py changes:
    # In InitSessionResponse class, remove:
    #   css_overrides: Optional[str] = None
    #
    # In initialize_session function, remove:
    #   css_overrides = None
    #   css_overrides = ai_response.get("css_overrides")
    #   "ai_css": css_overrides
    #
    # In InitSessionResponse return, remove:
    #   css_overrides=css_overrides
    #
    # In get_initial_data return, remove:
    #   "cssOverrides": session_data.get("ai_css")
    #   "cssOverrides": session_data.get("ai_css") or ""
    #
    # In AdaptStepResponse class, remove:
    #   dynamic_styles: Optional[str] = None
    #
    # In ApplyPreferenceResponse class, remove:
    #   dynamic_styles: Optional[str]

    user_preferences_list = _iter_prefs(user_profile.get('preferences', ''))

    user_content = f"""Adapt this assembly training step for the user.

**User Profile:**
- Experience: {user_profile.get('experience', 'beginner')}
- Preferences: {', '.join(user_preferences_list) or 'not specified'}
- Nationality: {user_profile.get('nationality', 'not specified')}

**User Request:** "{user_preference or 'None'}"

**Current Step:**
- Title: {step_payload.get('name')}
- Category: {step_payload.get('category')}
- Short text: {step_payload['adaptive_fields'].get('short_text')}
- Long text: {step_payload['adaptive_fields'].get('long_text')}

**Interaction History:**
- Recent clicks: {log_summary.get('recent_weighted', {})}
- User prefers: {log_summary.get('content_preference_order', [])}

**Rules:**
- Keep titles under 60 characters
- If user requests translation and nationality is provided, translate text
- DO NOT modify image/video paths
- For experts: prefer short text, use technical language
- For beginners: show visual content first, use simple language
- Respect user's explicit request (e.g., "video only" hides everything else)

Return adapted content with new visibility settings and explanation."""

    try:
        model = _gen_model(system)
        result = _generate_json(model, user_content, schema, temperature=0.7)

        # Preserve media paths
        af_in = step_payload.get("adaptive_fields", {})
        af_out = result.get("adaptive_fields", {})
        af_out["image_single_pieces"] = af_in.get("image_single_pieces", "")
        af_out["image_assembly"] = af_in.get("image_assembly", "")
        af_out["video"] = af_in.get("video", "")
        result["adaptive_fields"] = af_out

        print("✅ Step adapted successfully")
        print("=" * 80)
        return result
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            "title": step_payload.get("name"),
            "adaptive_fields": step_payload.get("adaptive_fields"),
            "initial_visibility": {
                "short_text": True, "long_text": False, "single_pieces": False,
                "assembly": False, "video": False,
            },
            "explanation_of_changes": f"No adaptation due to error: {str(e)}",
        }


def apply_user_preference(preference: str,
                          current_context: Dict[str, Any],
                          session_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 80)
    print("🎯 APPLY USER PREFERENCE - START")
    print("=" * 80)

    system = (
        "You are a UI adaptation engine. The user provides a natural language request to modify the interface. "
        "Translate this into Tailwind CSS classes and visibility flags. "
        "Use ONLY standard Tailwind utility classes - NO custom CSS. "
        "Output valid JSON only with no markdown formatting."
    )

    schema = {
        "type": "object",
        "properties": {
            "component_classes": {
                "type": "object",
                "properties": {
                    "container": {"type": "string"},
                    "header": {"type": "string"},
                    "content": {"type": "string"},
                    "button": {"type": "string"},
                    "text": {"type": "string"},
                    "card": {"type": "string"},
                },
                "description": "Tailwind CSS utility classes for UI components"
            },
            "visibility": {
                "type": "object",
                "properties": {
                    "short_text": {"type": "boolean"},
                    "long_text": {"type": "boolean"},
                    "single_pieces": {"type": "boolean"},
                    "assembly": {"type": "boolean"},
                    "video": {"type": "boolean"},
                },
                "description": "Content visibility flags"
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of changes"
            }
        },
        "required": ["component_classes", "visibility", "explanation"]
    }

    user_content = f"""Apply the user's UI preference request.

**User Request:** "{preference}"

**Current Context:**
- Step ID: {current_context.get('step_id')}
- Currently Visible: {current_context.get('visibility', {})}
- User Style: {session_data.get('ai_style_token', 'default')}

**Task:**
Generate Tailwind classes and visibility flags to fulfill the request.

**Examples:**
- "dark mode" → component_classes with bg-gray-800, text-white, border-gray-700
- "show only pictures" → visibility: all false except single_pieces and assembly true
- "bigger text" → component_classes.text: "text-lg leading-relaxed"
- "red buttons" → component_classes.button: "bg-red-600 hover:bg-red-700 text-white"

Use ONLY standard Tailwind utility classes. Return JSON now."""

    try:
        model = _gen_model(system)
        result = _generate_json(model, user_content, schema, temperature=0.6)
        print("✅ Preference applied successfully")
        print("=" * 80)
        return result
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            "component_classes": {},
            "visibility": current_context.get('visibility', {}),
            "explanation": f"Could not apply preference due to error: {str(e)}"
        }


def _generate_json(model, user_content: str, schema: Dict[str, Any], temperature: float = 0.7):
    """Ask Gemini to return strict JSON according to schema."""
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
        request_options={"retry": g_retry.Retry(), "timeout": 60},
    )

    text = resp.text or "{}"

    try:
        # Clean the response text
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n', '', text)
            text = re.sub(r'\n```\s*$', '', text)

        parsed = json.loads(text)
        print(f"  ✅ JSON parsed successfully")
        return parsed
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON decode error: {str(e)}")
        print(f"  📄 Response text (first 500 chars): {text[:500]}")
        # Extract JSON if wrapped in other content
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            extracted = text[start: end + 1]
            try:
                return json.loads(extracted)
            except:
                pass
        raise
