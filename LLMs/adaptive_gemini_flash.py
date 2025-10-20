from __future__ import annotations

import json
import re
import google.generativeai as genai
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

DATA = Path(__file__).resolve().parent.parent / "data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class AdaptationContext:
    """Context for UI adaptation"""
    user_id: str
    program_id: str
    item_id: str
    device_type: str  # Keep for compatibility but won't use heavily
    environment: str
    lighting: str
    noise_level: str


class GeminiAdapter:
    """Handles communication with Google Gemini API with improved error handling"""

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-exp"):
        if api_key is None:
            api_key = GEMINI_API_KEY

        if not api_key:
            raise ValueError("API key must be provided")

        genai.configure(api_key=api_key)
        self.model_name = model

        self.safety_settings = {
            'HARASSMENT': 'BLOCK_NONE',
            'HATE_SPEECH': 'BLOCK_NONE',
            'SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        }

        self.model = genai.GenerativeModel(
            model,
            safety_settings=self.safety_settings
        )
        self._verify_connection()

    def _verify_connection(self):
        """Verify Gemini API is accessible"""
        try:
            response = self.model.generate_content("Hello")
            logger.info(f"✓ Connected to Gemini ({self.model_name})")
        except Exception as e:
            logger.error(f"✗ Cannot connect to Gemini: {e}")
            raise ConnectionError(f"Gemini not accessible: {e}")

    def generate(self, prompt: str, system: Optional[str] = None) -> Tuple[str, bool]:
        """Generate text using Gemini API

        Returns: (response_text, success)
        """
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            logger.info(f"Requesting generation from {self.model_name}...")

            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # Increased for more creative adaptations
                    max_output_tokens=3000,
                ),
                safety_settings=self.safety_settings
            )

            if response.prompt_feedback.block_reason:
                logger.error(f"Response blocked: {response.prompt_feedback.block_reason}")
                return "", False

            if not response.parts:
                logger.error(f"No response parts. Finish reason: {response.candidates[0].finish_reason}")
                return "", False

            logger.info("✓ Generation successful")
            return response.text, True

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "", False


class ContentAdapter:
    """Adapts learning content based on user context"""

    def __init__(self, gemini: GeminiAdapter):
        self.gemini = gemini

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

    def adapt_content(self, context: AdaptationContext, original_content: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Adapt content based on user context"""

        user = self._get_user_profile(context.user_id)
        program = self._get_program(context.program_id)

        expertise = user.get('expertise', 'intermediate')

        system_prompt = """You are an expert instructional designer for manufacturing training.
                            Adapt learning content to match the learner's profile and environment.
                            Make changes to vocabulary, detail level, and structure based on expertise.
                            Always return valid JSON followed by a brief explanation."""

        user_prompt = f"""Adapt this learning content for maximum effectiveness:

LEARNER PROFILE:
- Name: {user.get('name')}
- Expertise: {expertise}
- Background: {user.get('description', 'N/A')}
- Learning preference: {user.get('learning_preference', 'visual')}

ORIGINAL CONTENT:
{json.dumps(original_content, indent=2)}

ENVIRONMENT CONTEXT:
- Location: {context.environment}
- Lighting: {context.lighting}
- Noise level: {context.noise_level}

ADAPTATION RULES BY EXPERTISE:

BEGINNER ({expertise == 'beginner'}):
- Use simple, everyday language
- Add detailed explanations for technical terms
- Break complex steps into smaller sub-steps
- Add safety warnings and "why this matters" context
- Include more encouragement and guidance

INTERMEDIATE ({expertise == 'intermediate'}):
- Use standard technical terminology
- Provide balanced detail
- Focus on practical application
- Mention common pitfalls

EXPERT ({expertise == 'expert'}):
- Use precise technical language
- Be concise and direct
- Highlight critical parameters and tolerances
- Focus on efficiency and troubleshooting

Consider also the information provided in  {user.get('description')}


ENVIRONMENT ADAPTATIONS:
- Shop floor: Add safety emphasis, make scannable with bold key terms
- Dim lighting: Ensure clarity with shorter sentences
- High noise: Structure for quick reference

MAINTAIN: All original JSON structure and field names.
ADAPT: Text content, explanations, step descriptions, titles if needed.

Return EXACTLY this format:
```json
{{your fully adapted JSON here}}
```

EXPLANATION: Describe the 3-4 key adaptations you made based on expertise and environment."""

        response_text, success = self.gemini.generate(user_prompt, system=system_prompt)

        if not success:
            logger.warning("Content adaptation failed, using original")
            return original_content, "Adaptation failed - using original content"

        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            try:
                adapted = json.loads(json_match.group(1) if json_match.lastindex else json_match.group(0))

                explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n\n|$)', response_text, re.DOTALL)
                explanation = explanation_match.group(
                    1).strip() if explanation_match else "Content adapted successfully"

                return adapted, explanation
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return original_content, "Failed to parse adapted content"
        else:
            logger.error("Could not extract JSON from response")
            return original_content, "Failed to extract adapted content"

    def adapt_support_material(self, context: AdaptationContext, material: Dict[str, Any]) -> Tuple[
        Dict[str, Any], str]:
        """Adapt support materials (text in offcanvas panels)"""

        if material.get('kind') != 'text' or not material.get('text'):
            return material, "No text to adapt"

        user = self._get_user_profile(context.user_id)
        expertise = user.get('expertise', 'intermediate')

        system_prompt = """You are adapting supplementary training materials for manufacturing workers.
Tailor the text content to the learner's expertise level."""

        user_prompt = f"""Adapt this support material text:

LEARNER: {user.get('name')} - {expertise} level - these are its preferences {user.get('description')} 

ORIGINAL TEXT:
{material.get('text')}

ADAPTATION RULES:
- Beginner: Simplify language, add context, explain "why"
- Intermediate: Keep balanced, add practical tips
- Expert: Be concise, focus on edge cases and optimization

Consider also the information provided in  {user.get('description')}

- Shop floor context: Add bold safety notes, use bullet points
- Dim lighting: Shorter paragraphs for easier reading

Return adapted text directly (no JSON wrapper needed).
Then add "EXPLANATION:" with 1-2 sentences about changes."""

        response_text, success = self.gemini.generate(user_prompt, system=system_prompt)

        if not success:
            return material, "Material adaptation failed"

        # Split response into adapted text and explanation
        parts = response_text.split("EXPLANATION:")
        adapted_text = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else "Material adapted"

        adapted_material = material.copy()
        adapted_material['text'] = adapted_text

        return adapted_material, explanation


class StyleAdapter:
    """Adapts CSS styling based on user context"""

    def __init__(self, gemini: GeminiAdapter):
        self.gemini = gemini

    def generate_adaptive_css(self, context: AdaptationContext, user: Dict[str, Any],
                              program: Dict[str, Any]) -> Tuple[str, str]:
        """Generate adaptive CSS based on context with strong color and layout variations"""

        expertise = user.get('expertise', 'intermediate')
        learning_pref = user.get('learning_preference', 'visual')

        system_prompt = """You are a UX designer specializing in adaptive interfaces for industrial training. Create DISTINCT visual experiences based on user expertise and environment. Use bold color schemes, varied layouts, and clear button styling.
Return only valid CSS."""

        user_prompt = f"""Design an adaptive interface for manufacturing training:

USER PROFILE:
- Expertise: {expertise}
- Learning preference: {learning_pref}
- Program: {program.get('title', 'Training')}

Consider also the information on preferences provided in {user.get('description')}

ENVIRONMENT:
- Location: {context.environment}
- Lighting: {context.lighting}
- Noise level: {context.noise_level}

CREATE DISTINCT STYLING FOR CLASS ".adaptive-learning-content":

EXPERTISE-BASED COLOR SCHEMES:
BEGINNER:
- Use warm, encouraging colors (soft blues #4A90E2, greens #27AE60)
- Light background (#F8F9FA or #FFFFFF)
- High contrast for readability
- Larger fonts (18px base)
- More padding and spacing
- Friendly rounded corners (8px)

INTERMEDIATE:
- Professional, balanced colors (navy #2C3E50, teal #16A085)
- Neutral backgrounds (#F5F5F5)
- Standard spacing
- Medium fonts (16px base)
- Clean, modern design

EXPERT:
- Technical, efficient colors (dark theme: #1E1E1E bg, #00D9FF accents)
- Can use darker backgrounds for focus
- Compact layout with less padding
- Smaller fonts acceptable (14-15px base)
- Sharp, minimal design
- Information-dense layout

ENVIRONMENT ADAPTATIONS:
SHOP FLOOR:
- HIGH visibility: bright accent colors (#FF6B35, #FFC107)
- Extra large buttons (min 56px height)
- Bold text (font-weight: 600)
- Strong borders (3px solid)

DIM LIGHTING:
- Dark mode: background #1A1A1A or #121212
- Light text: #E0E0E0
- Softer contrast to reduce eye strain
- Glowing accent colors (#00E5FF, #76FF03)

BRIGHT LIGHTING:
- High contrast mode
- Crisp white or light backgrounds
- Dark text (#212121)
- Stronger shadows for depth

BUTTON STYLING (vary significantly):
- Position buttons prominently (consider flex positioning)
- Vary button colors based on expertise theme
- Add hover effects and transitions
- Use different shapes/sizes for different expertise levels
- Consider button position: top-right for experts, centered for beginners

LAYOUT POSITIONING:
- Beginners: centered, generous margins (2-3rem)
- Intermediate: standard flow, 1.5rem padding
- Expert: edge-to-edge efficiency, 1rem padding

Use modern CSS: flexbox, grid, transitions, box-shadow, gradients where appropriate.

Return EXACTLY this format:
```css
.adaptive-learning-content {{
  /* Base layout */

  /* Colors and backgrounds */

  /* Typography */

  /* Buttons and interactive elements */

  /* Spacing and positioning */
}}

.adaptive-learning-content button {{
  /* Button-specific overrides */
}}

.adaptive-learning-content h2 {{
  /* Heading styles */
}}
```

EXPLANATION: Describe your color scheme choice and key layout decisions."""

        response_text, success = self.gemini.generate(user_prompt, system=system_prompt)

        if not success:
            logger.warning("CSS generation failed, using defaults")
            return self._get_default_css(context, expertise), "Using default CSS due to generation failure"

        css_match = re.search(r'```css\s*(.*?)\s*```', response_text, re.DOTALL)
        if not css_match:
            css_match = re.search(r'\.adaptive-learning-content\s*\{[^}]+\}', response_text, re.DOTALL)

        if css_match:
            css = css_match.group(1) if css_match.lastindex else css_match.group(0)
            css = css.strip()

            explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n\n|$)', response_text, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "CSS generated successfully"

            return css, explanation
        else:
            logger.error("Could not extract CSS from response")
            return self._get_default_css(context, expertise), "Failed to extract CSS, using defaults"

    def _get_default_css(self, context: AdaptationContext, expertise: str = "intermediate") -> str:
        """Fallback CSS with clear expertise-based variations"""

        if expertise == "beginner":
            return """
.adaptive-learning-content {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    color: #212529;
    padding: 2.5rem;
    border-radius: 12px;
    font-size: 18px;
    line-height: 1.8;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.adaptive-learning-content button {
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
    border: none;
    padding: 14px 28px;
    font-size: 17px;
    border-radius: 8px;
    min-height: 52px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s;
}
.adaptive-learning-content button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
}
.adaptive-learning-content h2 {
    color: #27AE60;
    font-size: 2.2rem;
    margin-bottom: 1.5rem;
}
"""
        elif expertise == "expert":
            return """
.adaptive-learning-content {
    font-family: 'Courier New', monospace;
    background: #1E1E1E;
    color: #E0E0E0;
    padding: 1rem;
    font-size: 14px;
    line-height: 1.5;
    border-left: 3px solid #00D9FF;
}
.adaptive-learning-content button {
    background: #00D9FF;
    color: #1E1E1E;
    border: 1px solid #00D9FF;
    padding: 8px 16px;
    font-size: 13px;
    border-radius: 2px;
    min-height: 36px;
    font-weight: 500;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.adaptive-learning-content button:hover {
    background: transparent;
    color: #00D9FF;
}
.adaptive-learning-content h2 {
    color: #00D9FF;
    font-size: 1.4rem;
    margin-bottom: 0.75rem;
    font-weight: 700;
}
"""
        else:  # intermediate
            return """
.adaptive-learning-content {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #FFFFFF;
    color: #2C3E50;
    padding: 1.75rem;
    border-radius: 6px;
    font-size: 16px;
    line-height: 1.6;
    border: 1px solid #E0E0E0;
}
.adaptive-learning-content button {
    background: #16A085;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 15px;
    border-radius: 4px;
    min-height: 42px;
    font-weight: 500;
    cursor: pointer;
}
.adaptive-learning-content button:hover {
    background: #138D75;
}
.adaptive-learning-content h2 {
    color: #2C3E50;
    font-size: 1.8rem;
    margin-bottom: 1.2rem;
    font-weight: 600;
}
"""


class AdaptiveUIManager:
    """Main manager for adaptive UI system"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.gemini = GeminiAdapter(api_key, model)
        self.content_adapter = ContentAdapter(self.gemini)
        self.style_adapter = StyleAdapter(self.gemini)
        self._css_cache: Dict[str, Tuple[str, str]] = {}
        self._content_cache: Dict[str, Tuple[Dict, str]] = {}

    def get_adaptive_css(self, context: AdaptationContext) -> Tuple[str, str]:
        """Get adaptive CSS with caching"""
        user = self.content_adapter._get_user_profile(context.user_id)
        expertise = user.get('expertise', 'intermediate')

        # Cache key now emphasizes user profile over device
        cache_key = f"{context.user_id}_{expertise}_{context.environment}_{context.lighting}"

        print("parameters:", context.user_id, expertise, context.environment, context.lighting)

        if cache_key in self._css_cache:
            logger.info("Using cached CSS")
            return self._css_cache[cache_key]

        program = self.content_adapter._get_program(context.program_id)

        css, explanation = self.style_adapter.generate_adaptive_css(context, user, program)
        self._css_cache[cache_key] = (css, explanation)
        return css, explanation

    def get_adaptive_content(self, context: AdaptationContext, original_content: Dict[str, Any]) -> Tuple[
        Dict[str, Any], str]:
        """Get adapted content with caching"""
        content_hash = json.dumps(original_content, sort_keys=True)
        cache_key = f"{context.user_id}_{context.item_id}_{hash(content_hash)}"

        if cache_key in self._content_cache:
            logger.info("Using cached content")
            return self._content_cache[cache_key]

        adapted, explanation = self.content_adapter.adapt_content(context, original_content)
        self._content_cache[cache_key] = (adapted, explanation)
        return adapted, explanation

    def get_adaptive_support_material(self, context: AdaptationContext, material: Dict[str, Any]) -> Tuple[
        Dict[str, Any], str]:
        """Get adapted support material"""
        return self.content_adapter.adapt_support_material(context, material)