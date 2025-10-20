# adaptive_openai.py
from __future__ import annotations

import os
from openai import OpenAI

# choose a fast, cheap general model
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # override in env if needed

class AdaptationContext:
    def __init__(self, user_id, program_id, item_id, device_type, environment, lighting, noise_level):
        self.user_id = user_id
        self.program_id = program_id
        self.item_id = item_id
        self.device_type = device_type
        self.environment = environment
        self.lighting = lighting
        self.noise_level = noise_level

class AdaptiveUIManager:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _respond(self, instructions: str, user_input: str) -> str:
        # Responses API; returns plain text for simplicity
        r = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
        )
        return r.output_text  # convenient text accessor in the SDK
    # --- CSS ---
    def get_adaptive_css(self, ctx: AdaptationContext) -> tuple[str, str]:
        instr = (
            "You produce only CSS. Tailor sizes, spacing, and contrasts to context. "
            "Explain choices plainly when asked. Avoid unsafe CSS."
        )
        prompt = (
            f"Context: device={ctx.device_type}, env={ctx.environment}, "
            f"lighting={ctx.lighting}, noise={ctx.noise_level}. "
            "Generate a compact CSS block for class .adaptive-learning-content "
            "that improves readability and focus in this context. "
            "Prefer small side nav, tighter badges, and large target areas for taps."
            "\n---\nAfter the CSS, write a 3–4 sentence explanation prefixed with EXPLAIN: "
            "Keep it concise."
        )
        out = self._respond(instr, prompt)
        # split CSS and explanation
        css, explain = out.split("EXPLAIN:", 1) if "EXPLAIN:" in out else (out, "")
        return css.strip(), explain.strip()

    # --- Learning item adaptation ---
    def get_adaptive_content(self, ctx: AdaptationContext, item: dict) -> tuple[dict, str]:
        instr = "Rewrite content to match user expertise and context. Return Markdown only."
        md = self._respond(instr,
            f"User is {ctx.user_id}. Program {ctx.program_id}. Item {ctx.item_id}.\n"
            f"Environment: {ctx.environment}, lighting: {ctx.lighting}. "
            f"Rewrite the following to be concise and scannable with bullets and short code samples when relevant:\n\n"
            f"{item.get('content','')}\n\n"
            "Then add 'EXPLAIN:' with a brief rationale."
        )
        text, explain = md.split("EXPLAIN:", 1) if "EXPLAIN:" in md else (md, "")
        return {"text": text.strip()}, explain.strip()

    # --- Support material adaptation (text items) ---
    def get_adaptive_support_material(self, ctx: AdaptationContext, material: dict) -> tuple[dict, str]:
        instr = "Adapt support text to be quick to skim, with 3–6 bullets."
        md = self._respond(instr,
            f"Device={ctx.device_type}. Simplify and compress:\n{material.get('text','')}\n\n"
            "End with 'EXPLAIN:' and a one-sentence reason."
        )
        text, explain = md.split("EXPLAIN:", 1) if "EXPLAIN:" in md else (md, "")
        return {"text": text.strip()}, explain.strip()
