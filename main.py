"""
FastAPI Backend with Dynamic Gemini-based Adaptability
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
import re
from datetime import datetime
import uuid
from pathlib import Path

# Import Gemini service
from services.sentient_gemini_api import (
    initial_style_recommendations,
    adapt_step,
    apply_user_preference
)

app = FastAPI(
    title="Assembly Training API",
    description="Backend for adaptive assembly training interface",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images_single_pieces"
IMAGES_ASSEMBLY_DIR = BASE_DIR / "images_assembly"
VIDEOS_DIR = BASE_DIR / "videos"

IMAGES_DIR.mkdir(exist_ok=True)
IMAGES_ASSEMBLY_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images_single_pieces")
app.mount("/static/images_assembly", StaticFiles(directory=str(IMAGES_ASSEMBLY_DIR)), name="images_assembly")
app.mount("/static/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

print(f"📁 Serving static files from:")
print(f"   Images: {IMAGES_DIR}")
print(f"   Assembly Images: {IMAGES_ASSEMBLY_DIR}")
print(f"   Videos: {VIDEOS_DIR}")


# ============================================================================
# Models
# ============================================================================

class UserProfile(BaseModel):
    experience: str
    preferences: str = ""
    nationality: Optional[str] = None
    other: Optional[str] = None


class PageStructure(BaseModel):
    description: str = Field(..., description="Overall description of the page/interface")
    components: Dict[str, str] = Field(..., description="Component names and their descriptions")


class InitSessionRequest(BaseModel):
    experiment_id: str = Field(..., description="Unique experiment identifier")
    mode: str = Field(..., description="Training mode")
    profile: Optional[UserProfile] = Field(None, description="User profile for sentient mode")
    page_structure: Optional[PageStructure] = Field(None, description="UI page structure for sentient mode")


class InitSessionResponse(BaseModel):
    session_token: str
    style_token: str
    style_explanation: str
    initial_visibility: Optional[Dict[str, bool]] = None
    mode_config: Optional[Dict[str, Any]] = None
    component_classes: Optional[Dict[str, str]] = None


class AdaptStepRequest(BaseModel):
    session_token: str
    style_token: str
    step_data: Dict[str, Any]
    button_configs: Dict[str, Any]
    current_step: int
    interaction_history: List[Dict[str, Any]]
    user_preference: Optional[str] = None


class AdaptStepResponse(BaseModel):
    visibility: Optional[Dict[str, bool]] = None
    button_configs: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    component_classes: Optional[Dict[str, str]] = None
    title: Optional[str] = None
    adaptive_fields: Optional[Dict[str, str]] = None


class ApplyPreferenceRequest(BaseModel):
    session_token: str
    preference: str
    current_context: Dict[str, Any]


class ApplyPreferenceResponse(BaseModel):
    component_classes: Optional[Dict[str, str]]
    visibility: Optional[Dict[str, bool]]
    explanation: Optional[str]


class LogInteractionRequest(BaseModel):
    experiment_id: str
    timestamp: str
    action: str
    step_id: Optional[int] = None
    content_type: Optional[str] = None


# ============================================================================
# In-memory storage
# ============================================================================

sessions: Dict[str, Dict[str, Any]] = {}
step_categories: List[str] = []
mode_configs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Startup: Load configurations
# ============================================================================

@app.on_event("startup")
async def load_configurations():
    """Load step categories and mode configurations"""
    global step_categories, mode_configs

    # Load step categories
    try:
        steps_file = "settings/steps_sources.json"
        if os.path.exists(steps_file):
            with open(steps_file, 'r') as f:
                data = json.load(f)
                steps_data = data.get('steps', data.get('assembly_process', []))
                categories = list(set(
                    step.get('category', 'Unknown')
                    for step in steps_data
                ))
                step_categories = sorted(categories)
                print(f"✅ Loaded {len(step_categories)} step categories")
        else:
            step_categories = ["foundation", "mechanical", "electrical"]
            print(f"⚠️ Using fallback categories")
    except Exception as e:
        print(f"❌ Error loading categories: {e}")
        step_categories = ["foundation", "assembly", "testing"]

    # Load visibility configurations for different modes
    try:
        visibility_dir = Path("settings/visibility")
        if visibility_dir.exists():
            for config_file in visibility_dir.glob("*.json"):
                mode_name = config_file.stem
                with open(config_file, 'r') as f:
                    mode_configs[mode_name] = json.load(f)
                print(f"✅ Loaded visibility config for mode: {mode_name}")
    except Exception as e:
        print(f"⚠️ Error loading visibility configs: {e}")


# ============================================================================
# Helper function to convert local paths to URLs
# ============================================================================

def convert_path_to_url(file_path: str) -> str:
    """Convert local file paths to served URLs"""
    if not file_path:
        return ""

    file_path = file_path.replace("\\", "/")

    if "images_assembly" in file_path:
        filename = file_path.split("images_assembly/")[-1]
        return f"http://localhost:8000/static/images_assembly/{filename}"
    elif "images_single_pieces" in file_path:
        filename = file_path.split("images_single_pieces/")[-1]
        return f"http://localhost:8000/static/images/{filename}"
    elif "videos" in file_path:
        filename = file_path.split("videos/")[-1]
        return f"http://localhost:8000/static/videos/{filename}"

    return file_path


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Assembly Training API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "static_files": {
            "images_single_pieces": str(IMAGES_DIR),
            "images_assembly": str(IMAGES_ASSEMBLY_DIR),
            "videos": str(VIDEOS_DIR)
        }
    }


@app.get("/api/steps")
async def get_steps():
    """Load and return assembly steps with converted URLs"""
    try:
        steps_file = "settings/steps_sources.json"
        if os.path.exists(steps_file):
            with open(steps_file, 'r') as f:
                data = json.load(f)
                steps_data = data.get('steps', data.get('assembly_process', []))

                for step in steps_data:
                    if 'adaptive_fields' in step:
                        fields = step['adaptive_fields']
                        if 'image_single_pieces' in fields:
                            fields['image_single_pieces'] = convert_path_to_url(fields['image_single_pieces'])
                        if 'image_assembly' in fields:
                            fields['image_assembly'] = convert_path_to_url(fields['image_assembly'])
                        if 'video' in fields:
                            fields['video'] = convert_path_to_url(fields['video'])

                return {"steps": steps_data}
        else:
            raise HTTPException(status_code=404, detail=f"Steps file not found")
    except Exception as e:
        print(f"Error loading steps: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load steps: {str(e)}")


@app.post("/api/v1/session/init", response_model=InitSessionResponse)
async def initialize_session(request: InitSessionRequest):
    """Initialize a new training session with mode-specific visibility"""
    session_token = str(uuid.uuid4())
    style_token = "default"
    style_explanation = "Using default interface styling"
    component_classes = None
    initial_visibility = {
        "short_text": False, "long_text": False, "single_pieces": False, "assembly": False, "video": False
    }

    session_data = {
        "session_token": session_token,
        "experiment_id": request.experiment_id,
        "mode": request.mode,
        "created_at": datetime.now().isoformat(),
        "profile": request.profile.dict() if request.profile else None
    }

    mode_config = mode_configs.get(request.mode)
    if mode_config:
        style_explanation = f"Using {request.mode} configuration"
        print(f"📋 Loaded config for mode: {request.mode}")

    if request.mode == "sentient":
        if not request.profile:
            raise HTTPException(status_code=400, detail="Profile required for sentient mode")
        if not request.page_structure:
            raise HTTPException(status_code=400, detail="Page structure required for sentient mode")

        try:
            ai_response = initial_style_recommendations(
                user_profile=request.profile.dict(),
                step_categories=step_categories,
                page_structure=request.page_structure.dict()
            )
            print("---------- GEMINI INITIAL STYLE RESPONSE ----------")
            print(ai_response)
            print("-------------------------------------------------")

            style_token = ai_response.get("style_profile_token", "default")
            style_explanation = ai_response.get("explanation", "AI-generated styling applied")
            component_classes = ai_response.get("component_classes")
            initial_visibility = _get_initial_visibility_for_profile(request.profile)

            session_data.update({
                "ai_style_token": style_token,
                "component_classes": component_classes
            })
        except Exception as e:
            print(f"❌ Error generating AI recommendations: {e}")
            import traceback
            traceback.print_exc()
            style_explanation = f"Using default styling (AI error: {str(e)})"
    elif request.mode == "static":
        initial_visibility = {k: True for k in initial_visibility}
    elif request.mode in ["dynamically_adaptive", "rule_based"]:
        initial_visibility["short_text"] = True

    sessions[session_token] = session_data
    print(f"✅ Session initialized: {session_token} (mode: {request.mode})")

    return InitSessionResponse(
        session_token=session_token,
        style_token=style_token,
        style_explanation=style_explanation,
        initial_visibility=initial_visibility,
        mode_config=mode_config,
        component_classes=component_classes
    )


@app.post("/api/get-initial-data")
async def get_initial_data(request: dict):
    """Get initial data for training page"""
    session_token = request.get("session_token")

    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_token]

    # Load steps
    try:
        steps_file = "settings/steps_sources.json"
        if os.path.exists(steps_file):
            with open(steps_file, 'r') as f:
                data = json.load(f)
                steps_data = data.get('steps', data.get('assembly_process', []))

                # Convert paths to URLs
                for step in steps_data:
                    if 'adaptive_fields' in step:
                        fields = step['adaptive_fields']
                        if 'image_single_pieces' in fields:
                            fields['image_single_pieces'] = convert_path_to_url(fields['image_single_pieces'])
                        if 'image_assembly' in fields:
                            fields['image_assembly'] = convert_path_to_url(fields['image_assembly'])
                        if 'video' in fields:
                            fields['video'] = convert_path_to_url(fields['video'])
        else:
            steps_data = []
    except Exception as e:
        print(f"Error loading steps: {e}")
        steps_data = []

    return {
        "session": {
            "experimentId": session_data.get("experiment_id"),
            "mode": session_data.get("mode"),
            "sessionToken": session_token,
            "styleToken": session_data.get("ai_style_token", "default"),
            "styleExplanation": f"Session initialized for {session_data.get('mode')} mode",
            "componentClasses": session_data.get("component_classes"),
            "modeConfig": mode_configs.get(session_data.get("mode"))
        },
        "steps": steps_data,
        "componentClasses": session_data.get("component_classes") or {}
    }

@app.post("/api/adapt-step", response_model=AdaptStepResponse)
async def adapt_step_endpoint(request: AdaptStepRequest):
    """Adapt step content based on mode and user interactions"""
    if request.session_token not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[request.session_token]
    mode = session_data.get("mode")

    if mode == "sentient":
        try:
            log_summary = _analyze_interaction_history(request.interaction_history, request.step_data)
            user_profile = session_data.get("profile", {"experience": "beginner", "preferences": ["visual"]})

            ai_response = adapt_step(
                user_profile=user_profile,
                style_profile_token=request.style_token,
                step_payload=request.step_data,
                log_summary=log_summary,
                user_preference=request.user_preference
            )

            return AdaptStepResponse(
                visibility=ai_response.get("initial_visibility"),
                explanation=ai_response.get("explanation_of_changes"),
                component_classes=ai_response.get("component_classes"),
                title=ai_response.get("title"),
                adaptive_fields=ai_response.get("adaptive_fields")
            )
        except Exception as e:
            print(f"❌ Error adapting step with AI: {e}")
            # Fallback to default on error

    # Default/Fallback behavior for non-sentient modes or on AI error
    visibility = {"short_text": True, "long_text": False, "single_pieces": False, "assembly": False, "video": False}
    explanation = f"{mode} mode - default visibility"

    # Check for pre-defined mode configurations
    if mode in mode_configs:
        config = mode_configs[mode]
        step_config = next((s for s in config.get("steps", []) if s["step_id"] == request.step_data.get("id")), None)
        if step_config:
            visibility = step_config.get("content", visibility)
            explanation = f"{mode} configuration for step {request.step_data.get('id')}"

    return AdaptStepResponse(visibility=visibility, explanation=explanation)


@app.post("/api/apply-preference", response_model=ApplyPreferenceResponse)
async def apply_preference_endpoint(request: ApplyPreferenceRequest):
    """Apply user preference dynamically using Gemini"""
    if request.session_token not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session_data = sessions[request.session_token]

    try:
        result = apply_user_preference(
            preference=request.preference,
            current_context=request.current_context,
            session_data=session_data
        )
        return ApplyPreferenceResponse(**result)
    except Exception as e:
        print(f"❌ Error applying preference: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply preference: {str(e)}")


@app.post("/api/log-interaction")
async def log_interaction_endpoint(request: LogInteractionRequest):
    """Log user interaction"""
    try:
        os.makedirs("logs", exist_ok=True)
        interaction_log_file = f"logs/interactions_{request.experiment_id}.jsonl"
        with open(interaction_log_file, 'a') as f:
            f.write(json.dumps(request.dict()) + '\n')
        return {"status": "logged", "timestamp": request.timestamp}
    except Exception as e:
        print(f"❌ Error logging interaction: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Helper Functions
# ============================================================================
def _iter_prefs(pref_str: Optional[str]):
    """Splits a single string like 'visual, video' into normalized tokens."""
    if not pref_str:
        return []
    return [p.strip().lower() for p in re.split(r'[\s,]+', pref_str) if p.strip()]


def _get_initial_visibility_for_profile(profile: UserProfile) -> Dict[str, bool]:
    """Determine initial content visibility based on user profile"""
    visibility = {"short_text": False, "long_text": False, "single_pieces": False, "assembly": False, "video": False}

    if profile.experience in ["novice", "beginner"]:
        visibility["assembly"] = True
    else:
        visibility["short_text"] = True

    for pref in _iter_prefs(profile.preferences):
        if pref in ["video", "videos"]:
            visibility["video"] = True
        elif pref in ["images", "image", "visual", "pictures"]:
            visibility["single_pieces"] = True
            visibility["assembly"] = True
        elif pref in ["text", "reading", "short_text"]:
            visibility["short_text"] = True
        elif pref in ["detailed", "long_text"]:
            visibility["long_text"] = True

    return visibility


def _analyze_interaction_history(interactions: List[Dict[str, Any]],
                                 current_step: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze interaction history to find user preferences."""
    content_clicks = {"short_text": 0, "long_text": 0, "single_pieces": 0, "assembly": 0, "video": 0}
    recent_interactions = interactions[-10:]

    recent_weighted = {}
    for i, interaction in enumerate(recent_interactions):
        if interaction.get("action") == "content_shown":
            content_type = interaction.get("content_type")
            if content_type in content_clicks:
                content_clicks[content_type] += 1
                weight = (i + 1) / len(recent_interactions)
                recent_weighted[content_type] = recent_weighted.get(content_type, 0) + weight

    return {
        "step_type": current_step.get("category", "unknown"),
        "recent_weighted": {k: round(v, 2) for k, v in recent_weighted.items()},
        "content_preference_order": sorted(content_clicks.items(), key=lambda x: x[1], reverse=True)
    }




# ============================================================================
# Run the application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)