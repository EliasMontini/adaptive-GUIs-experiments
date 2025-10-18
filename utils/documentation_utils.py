"""
Gemini API Service for processing Cobot documentation and generating structured JSON steps.

This module handles:
- Loading and extracting text from DOCX files
- Preparing prompts for Gemini API
- Calling Gemini API to generate structured JSON
- Parsing and saving the results
"""

from typing import Dict, Any, Optional
import json
import os
import logging
import requests
from docx import Document
from utils.media_assets_manager import MediaAssetsLibrary, MediaMatcher
from utils.json_validator import StepsValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DocumentationProcessor:
    """Handles extraction and loading of documentation files."""

    def __init__(self, docx_path: str):
        """
        Initialize the processor with a DOCX file path.

        Args:
            docx_path: Path to the DOCX documentation file
        """
        self.docx_path = docx_path

    def extract_text(self) -> str:
        """
        Extract text from DOCX file, filtering out empty paragraphs.

        Returns:
            Extracted text as a single string
        """
        logger.info(f"Loading documentation from {self.docx_path}")
        doc = Document(self.docx_path)

        # Extract non-empty paragraphs
        text_content = "\n".join(
            [para.text for para in doc.paragraphs if para.text.strip()]
        )

        logger.info(f"Successfully extracted {len(text_content)} characters")
        return text_content


class TemplateLoader:
    """Handles loading and managing JSON templates."""

    def __init__(self, template_path: str):
        """
        Initialize the loader with a JSON template path.

        Args:
            template_path: Path to the JSON template file
        """
        self.template_path = template_path

    def load(self) -> Dict[str, Any]:
        """
        Load JSON template from file.

        Returns:
            Parsed JSON template as dictionary
        """
        logger.info(f"Loading JSON template from {self.template_path}")
        with open(self.template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        logger.info(f"Template loaded with {len(template.get('steps', []))} base steps")
        return template


class PromptBuilder:
    """Constructs prompts for Gemini API."""

    @staticmethod
    def build_documentation_prompt(
            documentation_text: str,
            json_template: Dict[str, Any],
            media_library: Optional[MediaAssetsLibrary] = None
    ) -> str:
        """
        Build a comprehensive prompt for Gemini to process documentation.

        Args:
            documentation_text: Extracted documentation content
            json_template: JSON template structure to use as reference for format only
            media_library: Optional media assets library for reference

        Returns:
            Formatted prompt string
        """
        # Build media reference section if library is provided
        media_section = ""
        if media_library:
            teach_pendant_assets = media_library.get_assets_by_category('teach_pendant')
            cobot_action_assets = media_library.get_assets_by_category('cobot_action')
            all_videos = [a for a in media_library.get_all_assets() if a.asset_type == 'video']

            media_section = "\nAVAILABLE MEDIA ASSETS:\n\n"

            if teach_pendant_assets:
                media_section += "TEACH PENDANT IMAGES:\n"
                for asset in teach_pendant_assets:
                    media_section += f"- {asset.filename}: {asset.description}\n"
                media_section += "\n"

            if cobot_action_assets:
                media_section += "COBOT ACTION IMAGES:\n"
                for asset in cobot_action_assets:
                    media_section += f"- {asset.filename}: {asset.description}\n"
                media_section += "\n"

            if all_videos:
                media_section += "VIDEOS:\n"
                for asset in all_videos:
                    media_section += f"- {asset.filename}: {asset.description}\n"
                media_section += "\n"

            if not teach_pendant_assets and not cobot_action_assets and not all_videos:
                media_section += "No media assets loaded.\n"

        prompt = f"""{documentation_text}

This is the documentation for a hands-on exercise using the Universal Robots UR5e Cobot involving screwdriver installation and a subsequent programming task.
{media_section}

TASK:
Identify ALL main steps from the documentation in a logical order that covers the entire content completely.

STRUCTURE FOR EACH STEP:
Each step MUST have exactly this structure:
{{
  "id": <sequential_number>,
  "name": "<action_oriented_title>",
  "category": "<functional_category>",
  "target_cycle_time": <time_in_seconds>,
  "adaptive_fields": {{
    "short_text": "<brief_1_2_sentence_description>",
    "long_text": "<detailed_step_by_step_instructions>",
    "image_cobot": "<filename_or_empty_string>",
    "image_teach_pendant": "<filename_or_empty_string>",
    "video": "<filename_or_empty_string>"
  }}
}}

CRITICAL REQUIREMENTS:
1. Generate as many steps as needed to cover ALL content in the documentation - DO NOT truncate or skip content
2. Each step MUST have all 5 adaptive_fields (short_text, long_text, image_cobot, image_teach_pendant, video)
3. Use only filenames from the available media assets or empty strings
4. DO NOT add any extra fields or arrays
5. Return ONLY valid JSON with "steps" array

MEDIA MATCHING RULES:
- Use ONLY filenames from the available assets listed above
- Leave media fields as empty strings "" if no suitable asset exists
- Match media only if it directly relates to the step

Format reference (use only the structure, not the content):
{json.dumps(json_template, indent=2)}

Return the JSON with key "steps" containing all identified steps:
"""
        return prompt


class GeminiAPIClient:
    """Handles communication with Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini API client.

        Args:
            api_key: Google API key for Gemini
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate_content(self, prompt: str, temperature: float = 0.2, max_tokens: int = 8000) -> str:
        """
        Call Gemini API to generate content.

        Args:
            prompt: The prompt to send to Gemini
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text content

        Raises:
            requests.exceptions.HTTPError: If API request fails
        """
        logger.info(f"Sending request to Gemini ({self.model})")

        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            params={"key": self.api_key}
        )

        response.raise_for_status()

        result = response.json()
        generated_text = result['candidates'][0]['content']['parts'][0]['text']

        logger.info("Response received successfully")
        logger.debug(f"Response length: {len(generated_text)} characters")

        return generated_text


class JSONParser:
    """Handles parsing and validation of JSON responses."""

    @staticmethod
    def parse(json_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON text with fallback extraction if needed.
        Also normalizes key names from 'assembly_process' to 'steps'.
        Handles markdown code blocks (```json ... ```).

        Args:
            json_text: Raw text potentially containing JSON

        Returns:
            Parsed JSON as dictionary, or None if parsing fails
        """
        # Remove markdown code block wrapper if present
        if json_text.strip().startswith('```json'):
            logger.info("Removing markdown code block wrapper")
            json_text = json_text.replace('```json', '').replace('```', '').strip()
        elif json_text.strip().startswith('```'):
            logger.info("Removing markdown code block wrapper")
            json_text = json_text.replace('```', '').strip()

        # First attempt: direct parsing (no cleaning, keep original)
        try:
            logger.info("Attempting to parse JSON directly")
            parsed = json.loads(json_text)
            logger.info("✓ JSON parsed successfully")
            return JSONParser._normalize_keys(parsed)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct parsing failed: {e}")

        # Second attempt: extract JSON from text (no cleaning)
        try:
            logger.info("Attempting to extract JSON from text")
            start = json_text.find('{')
            end = json_text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = json_text[start:end]
                logger.debug(f"Extracted JSON substring (first 200 chars): {json_str[:200]}")
                parsed = json.loads(json_str)
                logger.info("✓ JSON extracted and parsed successfully")
                return JSONParser._normalize_keys(parsed)
        except json.JSONDecodeError as extract_err:
            logger.error(f"JSON decode error: {extract_err}")
            logger.error(f"Error at position {extract_err.pos}: {extract_err.msg}")
            # Show the problematic area
            if hasattr(extract_err, 'pos'):
                start_pos = max(0, extract_err.pos - 100)
                end_pos = min(len(json_text), extract_err.pos + 100)
                logger.error(f"Context around error: ...{json_text[start_pos:end_pos]}...")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")

        logger.error("✗ Unable to parse JSON from response")
        logger.error(f"Raw response preview (first 500 chars):\n{json_text[:500]}")

        return None

    @staticmethod
    def _normalize_keys(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize key names in the JSON structure.
        Maps 'assembly_process' to 'steps' if needed.

        Args:
            data: Dictionary potentially with non-standard keys

        Returns:
            Normalized dictionary with 'steps' key
        """
        # If 'assembly_process' exists, rename it to 'steps'
        if 'assembly_process' in data and 'steps' not in data:
            data['steps'] = data.pop('assembly_process')
            logger.info("Normalized 'assembly_process' key to 'steps'")

        return data


class FileWriter:
    """Handles writing output to files."""

    @staticmethod
    def save_json(data: Dict[str, Any], output_path: str) -> bool:
        """
        Save dictionary as JSON to file.

        Args:
            data: Dictionary to save
            output_path: Full path where to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ JSON saved successfully to {output_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save JSON: {e}")
            return False


class DocumentationService:
    """Main service orchestrating the documentation processing workflow."""

    def __init__(
            self,
            docx_path: str,
            template_path: str,
            api_key: str,
            output_path: str,
            media_library_path: Optional[str] = None,
            model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize the documentation service.

        Args:
            docx_path: Path to documentation DOCX file
            template_path: Path to JSON template file
            api_key: Google API key
            output_path: Path where to save output JSON
            media_library_path: Optional path to media_library.json
            model: Gemini model to use
        """
        self.docx_processor = DocumentationProcessor(docx_path)
        self.template_loader = TemplateLoader(template_path)
        self.gemini_client = GeminiAPIClient(api_key, model)
        self.output_path = output_path

        # Load media library if provided
        self.media_library = None
        if media_library_path and os.path.exists(media_library_path):
            self.media_library = MediaAssetsLibrary.load_from_json(media_library_path)

    def process(self) -> Optional[Dict[str, Any]]:
        """
        Execute the complete documentation processing workflow.

        Returns:
            Processed JSON dictionary, or None if processing fails
        """
        logger.info("Starting documentation processing workflow")

        # Step 1: Load documentation
        documentation_text = self.docx_processor.extract_text()

        # Step 2: Load template
        json_template = self.template_loader.load()

        # Step 3: Build prompt (with media library if available)
        prompt = PromptBuilder.build_documentation_prompt(
            documentation_text,
            json_template,
            self.media_library
        )

        # Step 4: Call Gemini API
        try:
            gemini_response = self.gemini_client.generate_content(prompt)
        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ API request failed: {e.response.status_code}")
            logger.error(f"Response: {e.response.text}")
            return None

        # Save raw response for debugging
        debug_path = os.path.join(os.path.dirname(self.output_path), "gemini_raw_response.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(gemini_response)
            logger.info(f"Raw response saved to {debug_path}")
        except Exception as e:
            logger.warning(f"Could not save debug response: {e}")

        # Step 5: Parse response
        parsed_json = JSONParser.parse(gemini_response)

        if parsed_json is None:
            logger.error("Failed to parse Gemini response")
            logger.error(f"Raw response:\n{gemini_response}")
            return None

        logger.info(f"Parsed JSON structure with {len(parsed_json.get('steps', []))} steps")

        # Step 6: Save output (no validation needed, structure is already good)
        success = FileWriter.save_json(parsed_json, self.output_path)

        if success:
            logger.info("✓ Workflow completed successfully")
            return parsed_json
        else:
            logger.error("Failed to save output")
            return None