"""
FastAPI Backend for Assembly Training Application
WITH STATIC FILE SERVING FOR IMAGES AND VIDEOS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime
import uuid
from pathlib import Path

# Import Gemini service
from services.sentient_gemini_api import initial_style_recommendations

app = FastAPI(
    title="Assembly Training API",
    description="Backend for adaptive assembly training interface",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STATIC FILE SERVING - Mount directories for media assets
# ============================================================================

# Define base paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images_single_pieces"
IMAGES_ASSEMBLY_DIR = BASE_DIR / "images_assembly"
VIDEOS_DIR = BASE_DIR / "videos"

# Create directories if they don't exist
IMAGES_DIR.mkdir(exist_ok=True)
IMAGES_ASSEMBLY_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

# Mount static file directories
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
    experience: str = Field(..., description="novice, intermediate, or expert")
    preferences: List[str] = Field(default_factory=list, description="Preferred content types")
    nationality: Optional[str] = Field(None, description="User nationality")
    other: Optional[str] = Field(None, description="Additional information")


class InitSessionRequest(BaseModel):
    experiment_id: str = Field(..., description="Unique experiment identifier")
    mode: str = Field(..., description="Training mode")
    profile: Optional[UserProfile] = Field(None, description="User profile for sentient mode")


class InitSessionResponse(BaseModel):
    session_token: str
    style_token: str
    style_explanation: str
    css_overrides: Optional[str] = None
    initial_visibility: Optional[Dict[str, bool]] = None
    mode_config: Optional[Dict[str, Any]] = None



class AdaptStepRequest(BaseModel):
    session_token: str
    style_token: str
    step_data: Dict[str, Any]
    button_configs: Dict[str, Any]
    current_step: int
    interaction_history: List[Dict[str, Any]]


class AdaptStepResponse(BaseModel):
    visibility: Optional[Dict[str, bool]] = None
    button_configs: Optional[Dict[str, Any]] = None
    dynamic_styles: Optional[str] = None
    explanation: Optional[str] = None


class OptimizeButtonRequest(BaseModel):
    session_token: str
    button_id: str
    click_count: int
    interaction_history: List[Dict[str, Any]]


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
            print(f"⚠️  Using fallback categories")
    except Exception as e:
        print(f"❌ Error loading categories: {e}")
        step_categories = ["foundation", "assembly", "testing"]

    # Load visibility configurations for different modes
    try:
        visibility_dir = Path("settings/visibility")
        if visibility_dir.exists():
            for config_file in visibility_dir.glob("*.json"):
                mode_name = config_file.stem  # filename without .json
                with open(config_file, 'r') as f:
                    mode_configs[mode_name] = json.load(f)
                print(f"✅ Loaded visibility config for mode: {mode_name}")
    except Exception as e:
        print(f"⚠️  Error loading visibility configs: {e}")


# ============================================================================
# Helper function to convert local paths to URLs
# ============================================================================

def convert_path_to_url(file_path: str) -> str:
    """
    Convert local file paths to served URLs

    Example:
      images/part1.jpg -> http://localhost:8000/static/images/part1.jpg
      images_assembly/step1.jpg -> http://localhost:8000/static/images_assembly/step1.jpg
    """
    if not file_path:
        return ""

    # Normalize path separators
    file_path = file_path.replace("\\", "/")

    # Check which directory it belongs to
    if "images_assembly" in file_path:
        filename = file_path.split("images_assembly/")[-1]
        return f"http://localhost:8000/static/images_assembly/{filename}"
    elif "images" in file_path:
        filename = file_path.split("images_single_pieces/")[-1]
        return f"http://localhost:8000/static/images/{filename}"
    elif "videos" in file_path:
        filename = file_path.split("videos/")[-1]
        return f"http://localhost:8000/static/videos/{filename}"

    # If no match, return as-is (will likely fail, but good for debugging)
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
        "version": "1.0.0",
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

                # Convert all file paths to URLs
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
    css_overrides = None
    initial_visibility = {
        "short_text": False,
        "long_text": False,
        "single_pieces": False,
        "assembly": False,
        "video": False
    }

    session_data = {
        "session_token": session_token,
        "experiment_id": request.experiment_id,
        "mode": request.mode,
        "created_at": datetime.now().isoformat(),
        "profile": request.profile.dict() if request.profile else None
    }

    # Load mode configuration if available
    mode_config = None
    if request.mode in mode_configs:
        mode_config = mode_configs[request.mode]
        style_explanation = f"Using {request.mode} configuration"
        print(f"📋 Loaded config for mode: {request.mode}")

    # Sentient mode: AI-powered adaptations
    if request.mode == "sentient":
        if not request.profile:
            raise HTTPException(status_code=400, detail="Profile required for sentient mode")

        try:
            ai_response = initial_style_recommendations(
                user_profile=request.profile.dict(),
                step_categories=step_categories
            )

            style_token = ai_response.get("style_profile_token", "default")
            style_explanation = ai_response.get("explanation", "AI-generated styling applied")
            css_overrides = ai_response.get("css_overrides", None)
            initial_visibility = _get_initial_visibility_for_profile(request.profile)

            session_data["ai_style_token"] = style_token
            session_data["ai_css"] = css_overrides

        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            style_explanation = f"Using default styling (AI error: {str(e)})"

    # Other modes use loaded configurations
    elif request.mode == "data_collection":
        style_explanation = "Data collection mode - all content hidden initially"
    elif request.mode == "static":
        style_explanation = "Static mode - all content visible"
        initial_visibility = {k: True for k in initial_visibility}
    elif request.mode in ["dynamically_adaptive", "rule_based"]:
        style_explanation = f"{request.mode.replace('_', ' ').title()} mode active"
        initial_visibility["short_text"] = True

    sessions[session_token] = session_data

    print(f"✅ Session initialized: {session_token} (mode: {request.mode})")

    return InitSessionResponse(
        session_token=session_token,
        style_token=style_token,
        style_explanation=style_explanation,
        css_overrides=css_overrides,
        initial_visibility=initial_visibility,
        mode_config=mode_config
    )


@app.post("/api/adapt-step", response_model=AdaptStepResponse)
async def adapt_step_endpoint(request: AdaptStepRequest):
    """Adapt step content based on mode and user interactions"""

    if request.session_token not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[request.session_token]
    mode = session_data.get("mode")
    print("mode is", mode)
    print("content is", mode_configs)

    # Map mode names to config keys
    mode_mapping = {
        "data_collection": "initial_visibility_data_collection",
        "dynamically_adaptive": "initial_visibility_dynamically_adaptive",
        "rule_based": "initial_visibility_rule_based_adaptive",
        "static": "initial_visibility_static_mode",
        "sentient": "sentient"
    }

    # Get the correct config key for this mode
    config_key = mode_mapping.get(mode)

    # For modes with pre-defined configurations, use those
    if config_key and config_key in mode_configs:
        config = mode_configs[config_key]
        step_config = next(
            (s for s in config.get("steps", []) if s["step_id"] == request.step_data.get("id")),
            None
        )

        if step_config:
            print("Step configured!")
            return AdaptStepResponse(
                visibility=step_config["content"],
                explanation=f"{mode} configuration for step {request.step_data.get('id')}"
            )

    # Sentient mode: Use AI
    if mode == "sentient":
        try:
            log_summary = _analyze_interaction_history(
                request.interaction_history,
                request.step_data
            )
            user_profile = session_data.get("profile", {
                "experience": "beginner",
                "preferences": ["visual"]
            })

            from services.sentient_gemini_api import adapt_step

            ai_response = adapt_step(
                user_profile=user_profile,
                style_profile_token=request.style_token,
                step_payload=request.step_data,
                log_summary=log_summary
            )

            visibility = ai_response.get("initial_visibility", {
                "short_text": True,
                "long_text": False,
                "single_pieces": False,
                "assembly": False,
                "video": False
            })

            button_configs = _adapt_button_configs(request.button_configs, log_summary)
            explanation = ai_response.get("explanation_of_changes", "AI-adapted content")

            return AdaptStepResponse(
                visibility=visibility,
                button_configs=button_configs,
                explanation=explanation
            )

        except Exception as e:
            print(f"Error adapting step: {e}")

    # Default fallback
    return AdaptStepResponse(
        visibility={"short_text": True, "long_text": False,
                    "single_pieces": False, "assembly": False, "video": False},
        explanation=f"{mode} mode - default visibility"
    )


@app.post("/api/optimize-button")
async def optimize_button_endpoint(request: OptimizeButtonRequest):
    """Optimize button placement/size based on usage"""

    if request.session_token not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    click_analysis = _analyze_button_clicks(request.button_id, request.interaction_history)

    button_config = {
        "size": "lg" if request.click_count >= 5 else "md",
        "position": click_analysis.get("suggested_position", {"x": 0, "y": 0}),
        "priority": click_analysis.get("priority", "normal")
    }

    return {
        "button_config": button_config,
        "explanation": f"Optimized based on {request.click_count} clicks"
    }


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
        print(f"Error logging interaction: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Helper Functions
# ============================================================================

def _get_initial_visibility_for_profile(profile: UserProfile) -> Dict[str, bool]:
    """Determine initial content visibility based on user profile"""
    visibility = {
        "short_text": False,
        "long_text": False,
        "single_pieces": False,
        "assembly": False,
        "video": False
    }

    if profile.experience in ["novice", "beginner"]:
        visibility["assembly"] = True
    elif profile.experience == "expert":
        visibility["short_text"] = True
    else:
        visibility["short_text"] = True

    if profile.preferences:
        for pref in profile.preferences:
            pref_lower = pref.lower().strip()
            if pref_lower in ["video", "videos"]:
                visibility["video"] = True
            elif pref_lower in ["images", "image", "visual", "pictures"]:
                visibility["single_pieces"] = True
                visibility["assembly"] = True
            elif pref_lower in ["text", "reading", "short_text"]:
                visibility["short_text"] = True
            elif pref_lower in ["detailed", "long_text"]:
                visibility["long_text"] = True

    return visibility


def _analyze_interaction_history(interactions: List[Dict[str, Any]],
                                 current_step: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze interaction history"""
    step_category = current_step.get("category", "unknown")

    content_clicks = {
        "short_text": 0,
        "long_text": 0,
        "single_pieces": 0,
        "assembly": 0,
        "video": 0
    }

    recent_interactions = interactions[-10:] if len(interactions) > 10 else interactions

    for interaction in recent_interactions:
        if interaction.get("action") == "content_shown":
            content_type = interaction.get("content_type")
            if content_type in content_clicks:
                content_clicks[content_type] += 1

    recent_weighted = {}
    for i, interaction in enumerate(recent_interactions):
        if interaction.get("action") == "content_shown":
            content_type = interaction.get("content_type")
            if content_type:
                weight = (i + 1) / len(recent_interactions)
                recent_weighted[content_type] = recent_weighted.get(content_type, 0) + weight

    clicked_now = {k: v > 0 for k, v in content_clicks.items()}

    return {
        "step_type": step_category,
        "total_interactions": len(interactions),
        "recent_weighted": recent_weighted,
        "clicked_now": clicked_now,
        "content_preference_order": sorted(
            content_clicks.items(),
            key=lambda x: x[1],
            reverse=True
        )
    }


def _adapt_button_configs(current_configs: Dict[str, Any],
                          log_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt button configurations"""
    adapted = {}
    preference_order = log_summary.get("content_preference_order", [])

    for content_type, config in current_configs.items():
        new_config = config.copy()
        rank = next((i for i, (ct, _) in enumerate(preference_order) if ct == content_type), 999)

        if rank == 0:
            new_config["size"] = "lg"
        elif rank <= 2:
            new_config["size"] = "md"
        else:
            new_config["size"] = "sm"

        adapted[content_type] = new_config

    return adapted


def _analyze_button_clicks(button_id: str,
                           interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze button click patterns"""
    button_interactions = [
        i for i in interactions
        if i.get("content_type") == button_id
    ]

    return {
        "total_clicks": len(button_interactions),
        "suggested_position": {"x": 0, "y": 0},
        "priority": "high" if len(button_interactions) >= 5 else "normal"
    }


# ============================================================================
# Run the application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )