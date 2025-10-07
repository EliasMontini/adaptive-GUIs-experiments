"""
Adaptive UI system using Ollama for content and CSS adaptation
"""
from __future__ import annotations

import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class AdaptationContext:
    """Context for UI adaptation"""
    user_id: str
    program_id: str
    item_id: str
    device_type: str  # 'desktop', 'tablet', 'mobile'
    environment: str  # 'shop_floor', 'classroom', 'office', 'home'
    lighting: str  # 'bright', 'normal', 'dim'
    noise_level: str  # 'quiet', 'moderate', 'loud'


import json
import requests
from typing import Optional


class OllamaAdapter:
    """Handles communication with local Ollama instance"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model
        self._verify_connection()

    def _verify_connection(self):
        """Verify Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Ollama not accessible at {self.base_url}")

    def generate(self, prompt: str, system: Optional[str] = None, timeout: int = 180) -> str:
        """Generate text using Ollama API with streaming for better performance"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,  # Streaming gives better feedback
            "system": system,
            "options": {
                "temperature": 0.7,
                "num_predict": 2000,  # Limit response length
            }
        }

        try:
            print(f"Requesting generation from {self.model}...")
            response = requests.post(url, json=payload, timeout=timeout, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                            # Print progress indicator
                            if len(full_response) % 100 == 0:
                                print(".", end="", flush=True)
                        if chunk.get("done", False):
                            print(" ✓")
                            break
                    except json.JSONDecodeError:
                        continue

            return full_response
        except requests.exceptions.Timeout:
            print(f"\n✗ Request timed out after {timeout} seconds")
            print("Try using a smaller model: ollama pull llama3.2:1b")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"\n✗ Ollama API error: {e}")
            return ""

    def chat(self, messages: list[dict], timeout: int = 180) -> str:
        """Chat using Ollama API with streaming"""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 2000,
            }
        }

        try:
            print(f"Requesting chat from {self.model}...")
            response = requests.post(url, json=payload, timeout=timeout, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            full_response += chunk["message"]["content"]
                            if len(full_response) % 100 == 0:
                                print(".", end="", flush=True)
                        if chunk.get("done", False):
                            print(" ✓")
                            break
                    except json.JSONDecodeError:
                        continue

            return full_response
        except requests.exceptions.Timeout:
            print(f"\n✗ Request timed out after {timeout} seconds")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"\n✗ Ollama API error: {e}")
            return ""


class ContentAdapter:
    """Adapts learning content based on user context"""

    def __init__(self, ollama: OllamaAdapter):
        self.ollama = ollama

    def _load_json(self, filename: str) -> list:
        return json.loads((DATA / filename).read_text(encoding='utf-8'))

    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        users = self._load_json("users.json")
        return next((u for u in users if u["id"] == user_id), {})

    def _get_program(self, program_id: str) -> Dict[str, Any]:
        programs = self._load_json("programs.json")
        return next((p for p in programs if p["id"] == program_id), {})

    def _get_learning_item(self, item_id: str) -> Dict[str, Any]:
        items = self._load_json("learning_paths.json")
        return next((i for i in items if i["id"] == item_id), {})

    def adapt_content(self, context: AdaptationContext, original_content: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Adapt content based on user context

        Returns: (adapted_content, explanation)
        """

        user = self._get_user_profile(context.user_id)
        program = self._get_program(context.program_id)
        item = self._get_learning_item(context.item_id)

        # Build context prompt
        system_prompt = """You are an expert instructional designer specializing in adaptive learning for manufacturing training.
    Your task is to adapt learning content to match the learner's profile, environment, and device.
    You must maintain technical accuracy while adjusting complexity, language, and presentation style."""

        user_prompt = f"""
    LEARNER PROFILE:
    {json.dumps(user, indent=2)}

    PROGRAM CONTEXT:
    Title: {program.get('title', 'N/A')}
    Expertise Level: {program.get('expertise', 'N/A')}
    About: {program.get('about', 'N/A')}

    CURRENT LEARNING ITEM:
    {json.dumps(item, indent=2)}

    ENVIRONMENT:
    - Device: {context.device_type}
    - Location: {context.environment}
    - Lighting: {context.lighting}
    - Noise Level: {context.noise_level}

    ADAPTATION TASK:
    Based on the learner's profile, adapt the following content:

    ORIGINAL CONTENT:
    {json.dumps(original_content, indent=2)}

    ADAPTATION RULES:
    1. For beginners with limited experience: simplify terminology, add more explanations, use analogies
    2. For experts: be concise, focus on advanced details, assume prior knowledge
    3. For visual learners: emphasize where images/diagrams would help
    4. For hands-on learners: structure as actionable steps
    5. For shop floor environment: make content scannable, use larger text mentally
    6. For mobile devices: break content into smaller chunks
    7. For dim lighting: avoid suggesting small text
    8. For noisy environments: emphasize visual cues over audio

    Return your response in this exact format:
    JSON:
    [your adapted JSON here]

    EXPLANATION:
    [Short explanation of key adaptations relating to: Learner Profile, Program Context, Environment, Device. Keep it concise - 3-5 sentences max.]
    """

        response = self.ollama.generate(user_prompt, system=system_prompt)

        # Parse JSON and explanation
        json_marker = response.find("JSON:")
        exp_marker = response.find("EXPLANATION:")

        if json_marker != -1 and exp_marker != -1:
            json_section = response[json_marker + 5:exp_marker].strip()
            explanation = response[exp_marker + 12:].strip()
        else:
            json_section = response
            explanation = "No explanation provided."

        try:
            # Try to extract JSON
            start = json_section.find('{')
            end = json_section.rfind('}') + 1
            if start != -1 and end > start:
                json_str = json_section[start:end]
                adapted = json.loads(json_str)
                return adapted, explanation
            else:
                print("Could not extract JSON from Ollama response")
                return original_content, "Failed to parse response."
        except json.JSONDecodeError:
            print("Failed to parse Ollama response as JSON")
            return original_content, "Failed to parse response."


class StyleAdapter:
    """Adapts CSS styling based on user context"""

    def __init__(self, ollama: OllamaAdapter):
        self.ollama = ollama

    def generate_adaptive_css(self, context: AdaptationContext, user: Dict[str, Any],
                              program: Dict[str, Any]) -> tuple[str, str]:
        """Generate adaptive CSS based on context

        Returns: (css, explanation)
        """

        system_prompt = """You are a UX designer specializing in accessible, context-aware interfaces for industrial training.
    Generate CSS that adapts to the user's environment, device, and learning needs. 
    Focus on readability, accessibility, and reducing cognitive load in challenging environments."""

        user_prompt = f"""
    LEARNER PROFILE:
    - Name: {user.get('name')}
    - Expertise: {user.get('expertise')}
    - Description: {user.get('description', 'N/A')}

    PROGRAM TYPE:
    - Title: {program.get('title')}
    - Type: {program.get('type')} (PC/VR)
    - Duration: {program.get('duration_min')} minutes

    ENVIRONMENT CONTEXT:
    - Device: {context.device_type}
    - Location: {context.environment}
    - Lighting: {context.lighting}
    - Noise Level: {context.noise_level}

    Generate CSS rules for class ".adaptive-learning-content" that:

    LEARNER PROFILE:
    - Use the description to define best colors and styles

    DEVICE ADAPTATIONS:
    - Mobile: larger touch targets (48px min), single column, larger fonts
    - Tablet: comfortable reading width, medium fonts
    - Desktop: multi-column possible, standard fonts

    ENVIRONMENT ADAPTATIONS:
    - Shop floor: high contrast, large fonts (18px+ body), prominent buttons, reduce animations
    - Classroom: standard sizing, collaborative-friendly
    - Dim lighting: increase contrast, avoid pure white backgrounds, use 14-16px minimum font
    - Loud environments: use visual indicators (icons, colors) prominently

    EXPERTISE ADAPTATIONS:
    - Beginner: more white space, clear section breaks, prominent help indicators
    - Expert: denser information, faster animations, compact layout

    LEARNING STYLE (from description):
    - Visual learners: emphasize image containers, larger media
    - Detail-oriented: clear typography hierarchy, organized lists

    Return your response in this exact format:
    CSS:
    [your CSS code here]

    EXPLANATION:
    [Short explanation of key choices relating to: Learner Profile adaptations, Device adaptations, Environment adaptations, Expertise adaptations. Keep it concise - 3-5 sentences max.]
    """

        response = self.ollama.generate(user_prompt, system=system_prompt)

        # Parse CSS and explanation
        css_marker = response.find("CSS:")
        exp_marker = response.find("EXPLANATION:")

        if css_marker != -1 and exp_marker != -1:
            css_section = response[css_marker + 4:exp_marker].strip()
            explanation = response[exp_marker + 12:].strip()
        else:
            css_section = response
            explanation = "No explanation provided."

        # Clean up CSS
        css = css_section.strip()
        if css.startswith('```css'):
            css = css[6:]
        if css.startswith('```'):
            css = css[3:]
        if css.endswith('```'):
            css = css[:-3]

        return css.strip(), explanation


class AdaptiveUIManager:
    """Main manager for adaptive UI system"""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.ollama = OllamaAdapter(ollama_url, model)
        self.content_adapter = ContentAdapter(self.ollama)
        self.style_adapter = StyleAdapter(self.ollama)

        # Cache to avoid repeated LLM calls
        self._css_cache: Dict[str, str] = {}
        self._content_cache: Dict[str, Dict] = {}

    def get_adaptive_css(self, context: AdaptationContext) -> str:
        """Get adaptive CSS for the learning path context"""

        # Create cache key
        cache_key = f"{context.user_id}_{context.program_id}_{context.device_type}_{context.environment}_{context.lighting}"

        if cache_key in self._css_cache:
            return self._css_cache[cache_key]

        user = self.content_adapter._get_user_profile(context.user_id)
        program = self.content_adapter._get_program(context.program_id)

        css = self.style_adapter.generate_adaptive_css(context, user, program)

        self._css_cache[cache_key] = css
        return css

    def get_adaptive_content(self, context: AdaptationContext, original_content: Dict[str, Any]) -> Dict[str, Any]:
        """Get adapted content for a specific learning item"""

        # Create cache key
        content_hash = json.dumps(original_content, sort_keys=True)
        cache_key = f"{context.user_id}_{context.item_id}_{hash(content_hash)}"

        if cache_key in self._content_cache:
            return self._content_cache[cache_key]

        adapted = self.content_adapter.adapt_content(context, original_content)

        self._content_cache[cache_key] = adapted
        return adapted

    def detect_environment(self, user_agent: str = None, time_of_day: int = 12) -> tuple[str, str, str]:
        """Detect environment context from user agent and time

        Returns: (device_type, environment, lighting)
        """
        device_type = "desktop"
        if user_agent:
            ua_lower = user_agent.lower()
            if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
                device_type = "mobile"
            elif "tablet" in ua_lower or "ipad" in ua_lower:
                device_type = "tablet"

        # Estimate lighting based on time
        if 6 <= time_of_day <= 18:
            lighting = "bright"
        elif 18 < time_of_day <= 20 or 5 <= time_of_day < 6:
            lighting = "normal"
        else:
            lighting = "dim"

        # Default environment (could be enhanced with location data)
        environment = "office"

        return device_type, environment, lighting


# Usage example functions for integration with Dash

def get_learning_path_css(user_id: str, program_id: str, device_type: str = "desktop",
                          environment: str = "office", lighting: str = "normal") -> str:
    """Get adaptive CSS when loading a learning path"""

    manager = AdaptiveUIManager()

    context = AdaptationContext(
        user_id=user_id,
        program_id=program_id,
        item_id="",  # Not needed for CSS
        device_type=device_type,
        environment=environment,
        lighting=lighting,
        noise_level="moderate"
    )

    return manager.get_adaptive_css(context)


def adapt_learning_item(user_id: str, program_id: str, item_id: str,
                        original_item: Dict[str, Any],
                        device_type: str = "desktop",
                        environment: str = "office") -> Dict[str, Any]:
    """Adapt a learning item's content based on context"""

    manager = AdaptiveUIManager()

    # Detect time-based lighting
    from datetime import datetime
    hour = datetime.now().hour
    _, _, lighting = manager.detect_environment(time_of_day=hour)

    context = AdaptationContext(
        user_id=user_id,
        program_id=program_id,
        item_id=item_id,
        device_type=device_type,
        environment=environment,
        lighting=lighting,
        noise_level="moderate"
    )

    return manager.get_adaptive_content(context, original_item)