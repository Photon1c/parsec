"""
Parallel Agent Crime Reconstruction System
==========================================
A LangGraph-based multi-agent system that transforms police blotter text
into educational crime reconstruction videos using parallel processing.

⚠️  LEGAL DISCLAIMER: All outputs are AI-generated simulations for 
    TRAINING AND HYPOTHESIS PURPOSES ONLY. NOT admissible as evidence.

Usage: python parallel_test.py --blotter "10-4, suspicious vehicle..." --output_dir ./videos

License: Reconstruction-Only License v1.0
- Prohibits use in legal proceedings as evidence
- Prohibits removal of watermarks or voiceover disclaimers
- Requires attribution to this repository
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from tqdm import tqdm
from google import genai

load_dotenv()

# =============================================================================
# LEGAL DISCLAIMER BANNER (DISPLAYED EVERY RUN)
# =============================================================================

LEGAL_BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                      ⚠️  THIS IS NOT REAL FOOTAGE ⚠️                           ║
║                                                                               ║
║   Generated reconstructions are ILLUSTRATIVE SIMULATIONS only.                ║
║   They are NOT admissible as evidence in ANY court of law.                    ║
║                                                                               ║
║   • Based on AI interpretation of radio logs/dispatcher notes                 ║
║   • May contain errors, assumptions, or inaccuracies                          ║
║   • NOT actual video evidence of any incident                                 ║
║                                                                               ║
║   Removing watermarks, disclaimers, or using in legal proceedings             ║
║   violates the license and may constitute FRAUD.                              ║
║                                                                               ║
║   See: U.S. v. Schaffer on AI evidence risks                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

LICENSE_TERMS = """
═══════════════════════════════════════════════════════════════════════════════
                    RECONSTRUCTION-ONLY LICENSE v1.0
═══════════════════════════════════════════════════════════════════════════════

By proceeding, you acknowledge and agree to the following terms:

1. PURPOSE LIMITATION: All generated content is for TRAINING, EDUCATIONAL, 
   and INTERNAL HYPOTHESIS purposes ONLY.

2. LEGAL PROHIBITION: You will NOT use any output as evidence in any legal 
   proceeding, including but not limited to: court cases, depositions, 
   discovery, witness interviews, jury presentations, or administrative hearings.

3. WATERMARK INTEGRITY: You will NOT remove, obscure, or alter any watermarks, 
   disclaimers, or voiceover warnings embedded in the output.

4. NO WITNESS EXPOSURE: You will NOT show generated content to witnesses, 
   victims, suspects, jurors, or any party involved in an active investigation 
   without explicit written consent and clear disclosure of AI generation.

5. ATTRIBUTION: Any sharing or publication requires attribution to this tool 
   and clear labeling as "AI-GENERATED SIMULATION."

6. AUDIT COMPLIANCE: You consent to logging of generation activities for 
   compliance and audit purposes.

7. LIABILITY: You assume all responsibility for misuse. The creators of this 
   tool bear no liability for improper use or legal consequences.

VIOLATION OF THESE TERMS MAY CONSTITUTE FRAUD AND OBSTRUCTION OF JUSTICE.
═══════════════════════════════════════════════════════════════════════════════
"""

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

MAX_SCENES = 5
MAX_VIDEO_LENGTH_SECONDS = 60
MAX_TOKENS_PER_AGENT = 8000
MAX_CONCURRENT_AGENTS = 6
MAX_RETRIES = 1
BUDGET_CAP_MINUTES = 450

# Watermark text that MUST appear in all generated content
WATERMARK_TEXT = "⚠️ SIMULATED RECONSTRUCTION – NOT ACTUAL FOOTAGE – NOT ADMISSIBLE IN COURT ⚠️"
WATERMARK_SHORT = "AI SIMULATION - NOT EVIDENCE"

# Voiceover disclaimer script
VOICEOVER_DISCLAIMER = """This is an AI-generated reconstruction based solely on police radio logs 
and dispatcher notes. It is not actual video evidence and may contain errors or assumptions. 
It is not admissible in any court of law."""

# Session tracking
SESSION_ID = str(uuid.uuid4())[:8]
GENERATION_TIMESTAMP = datetime.now().isoformat()

# Logging setup with audit trail
log_filename = f'reconstruction_audit_{datetime.now():%Y%m%d_%H%M%S}_{SESSION_ID}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# AUDIT & PROVENANCE TRACKING
# =============================================================================

class AuditTrail:
    """Tracks all operations for compliance and provenance reporting."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.events = []
        self.user_acknowledgments = []
        self.input_hash = ""
        self.output_hash = ""
        
    def log_event(self, event_type: str, details: dict):
        """Log an auditable event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "details": details
        }
        self.events.append(event)
        logger.info(f"AUDIT: {event_type} - {json.dumps(details)[:200]}")
        
    def log_acknowledgment(self, ack_type: str, user_response: bool):
        """Log user acknowledgment of terms/warnings."""
        self.user_acknowledgments.append({
            "timestamp": datetime.now().isoformat(),
            "type": ack_type,
            "accepted": user_response
        })
        
    def set_input_hash(self, content: str):
        """Hash input for provenance tracking."""
        self.input_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
    def set_output_hash(self, filepath: str):
        """Hash output file for provenance tracking."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.output_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                
    def generate_provenance_report(self) -> dict:
        """Generate full provenance report for compliance."""
        return {
            "report_type": "AI_RECONSTRUCTION_PROVENANCE",
            "version": "1.0",
            "session_id": self.session_id,
            "generation_timestamp": self.start_time.isoformat(),
            "completion_timestamp": datetime.now().isoformat(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "user_acknowledgments": self.user_acknowledgments,
            "events": self.events,
            "disclaimer": VOICEOVER_DISCLAIMER,
            "license": "Reconstruction-Only License v1.0",
            "legal_notice": "NOT ADMISSIBLE AS EVIDENCE IN ANY LEGAL PROCEEDING",
            "ai_models_used": {
                "llm": "gpt-4o-mini",
                "video": "veo-3.0-generate-001"
            }
        }
        
    def save_report(self, output_dir: str, case_id: str = "unknown"):
        """Save provenance report as JSON sidecar."""
        report = self.generate_provenance_report()
        report_path = os.path.join(
            output_dir, 
            f"PROVENANCE_SIMULATION_{datetime.now():%Y-%m-%d}_{case_id}_{self.session_id}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Provenance report saved: {report_path}")
        return report_path

# Global audit trail
audit = AuditTrail(SESSION_ID)

# =============================================================================
# STATE DEFINITION
# =============================================================================

class ReconstructionState(TypedDict):
    """State shared across all agents in the pipeline."""
    blotter_input: str
    case_id: str
    transcript: str
    scenes: list[dict]
    visual_prompts: list[dict]
    video_script: dict
    critic_feedback: dict
    uncertainty_indicators: list[dict]
    final_approved: bool
    video_path: str
    provenance_path: str
    error: str | None
    token_usage: int
    start_time: float
    safe_mode: bool
    training_mode: bool

# =============================================================================
# USER ACKNOWLEDGMENT & CONSENT FUNCTIONS
# =============================================================================

def display_legal_banner():
    """Display the legal disclaimer banner."""
    print(LEGAL_BANNER)

def require_license_acceptance(safe_mode: bool = False) -> bool:
    """
    Require user to explicitly accept license terms.
    Returns True if accepted, False otherwise.
    """
    print(LICENSE_TERMS)
    
    if safe_mode:
        print("\n🔒 SAFE MODE ENABLED - Enhanced disclaimers and logging active.\n")
    
    print("\nTo proceed, you must type 'I ACCEPT' exactly as shown.")
    print("Any other response will terminate the program.\n")
    
    response = input("Your response: ").strip()
    
    accepted = response == "I ACCEPT"
    audit.log_acknowledgment("license_terms", accepted)
    
    if not accepted:
        print("\n❌ License not accepted. Exiting.")
        logger.warning("User declined license terms")
        return False
        
    # Additional certification in safe mode
    if safe_mode:
        print("\n" + "="*60)
        print("SAFE MODE CERTIFICATION REQUIRED")
        print("="*60)
        print("\nPlease type your name/identifier to certify:")
        print("'I certify this will NOT be used in witness interviews,")
        print(" court prep, or any legal proceeding.'")
        
        cert_name = input("\nYour name/ID: ").strip()
        cert_statement = input("Type 'I CERTIFY': ").strip()
        
        if cert_statement != "I CERTIFY" or not cert_name:
            print("\n❌ Certification failed. Exiting.")
            audit.log_acknowledgment("safe_mode_certification", False)
            return False
            
        audit.log_event("safe_mode_certification", {
            "certifier": cert_name,
            "statement": "Will not use in legal proceedings",
            "timestamp": datetime.now().isoformat()
        })
        audit.log_acknowledgment("safe_mode_certification", True)
        print(f"\n✓ Certification recorded for: {cert_name}")
    
    logger.info("License terms accepted by user")
    return True

def confirm_prompt(message: str, require_reason: bool = False) -> tuple[bool, str]:
    """
    Human-in-the-loop confirmation prompt with optional reason field.
    Returns (accepted, reason).
    """
    print(f"\n{'='*60}")
    print(message)
    print('='*60)
    
    while True:
        response = input("Proceed? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            reason = ""
            if require_reason:
                reason = input("Reason for proceeding (required for audit): ").strip()
                if not reason:
                    print("Reason is required. Please provide a reason.")
                    continue
            audit.log_acknowledgment(f"checkpoint_{message[:30]}", True)
            return True, reason
        elif response in ('n', 'no'):
            audit.log_acknowledgment(f"checkpoint_{message[:30]}", False)
            return False, ""
        print("Please enter 'y' or 'n'")

# =============================================================================
# VALIDATION & SAFETY FUNCTIONS
# =============================================================================

def validate_blotter(text: str) -> tuple[bool, str]:
    """Check if blotter input is valid (not too vague or long)."""
    if len(text) < 20:
        return False, "Input too short – provide more detail."
    if len(text) > 2000:
        return False, "Input too long – limit to 2000 characters."
    
    action_keywords = ['vehicle', 'suspect', 'officer', 'scene', 'call', 'report', 
                       'respond', 'arrive', 'dispatch', '10-', 'code']
    if not any(kw in text.lower() for kw in action_keywords):
        return False, "Input too ambiguous – include specific incident details."
    return True, ""

def extract_case_id(blotter: str) -> str:
    """Extract or generate a case ID from the blotter."""
    import re
    # Try to find case number patterns
    patterns = [
        r'case[#\s-]*(\d+[-\d]*)',
        r'incident[#\s-]*(\d+[-\d]*)',
        r'report[#\s-]*(\d+[-\d]*)',
        r'#(\d{4,})'
    ]
    for pattern in patterns:
        match = re.search(pattern, blotter, re.I)
        if match:
            return match.group(1)
    # Generate random case ID if none found
    return f"GEN-{SESSION_ID}"

def generate_safe_filename(case_id: str, safe_mode: bool = False) -> str:
    """Generate a legally-safe filename that clearly indicates simulation."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    prefix = "SIMULATION_NOT_EVIDENCE"
    if safe_mode:
        prefix = "SAFE_MODE_SIMULATION_NOT_EVIDENCE"
    return f"{prefix}_{timestamp}_Case-{case_id}.mp4"

async def check_content_sensitivity(client: AsyncOpenAI, content: str) -> dict:
    """
    Quick LLM check for sensitive content that requires extra review.
    Returns sensitivity assessment.
    """
    system = """You are a content safety reviewer. Analyze the following content for sensitivity issues.
    
    Flag ANY of these:
    - Violence against persons (especially graphic)
    - Minors involved in any capacity
    - Sexual content or assault
    - Hate crimes or discriminatory content
    - Terrorism or mass casualty events
    - Suicide or self-harm
    - Identifiable victim information
    
    Output JSON:
    {
        "sensitivity_level": "low|medium|high|critical",
        "flags": ["list of specific concerns"],
        "requires_extra_review": boolean,
        "recommendation": "proceed|caution|block",
        "reasoning": "brief explanation"
    }
    
    Be conservative - when in doubt, flag it."""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content[:2000]}
            ],
            max_tokens=500
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except:
        return {"sensitivity_level": "medium", "requires_extra_review": True, 
                "recommendation": "caution", "flags": ["Parse error - manual review"]}

# =============================================================================
# ENHANCED PROMPTS WITH DISCLAIMERS
# =============================================================================

def build_video_prompt_with_disclaimers(script: dict, safe_mode: bool = False) -> str:
    """Build video prompt with mandatory disclaimer requirements."""
    
    scenes = script.get("scenes", [])
    scene_descriptions = []
    for s in scenes[:3]:
        prompt = s.get("primary_visual_prompt", s.get("description", ""))
        scene_descriptions.append(prompt)
    
    # Watermark instructions for AI
    watermark_instruction = f"""
MANDATORY VISUAL REQUIREMENTS:
- Display prominent semi-transparent watermark throughout: "{WATERMARK_TEXT}"
- Show text overlay: "AI RECONSTRUCTION - FOR TRAINING PURPOSES ONLY"
- Include timestamp and "SIMULATED" label in corner
- Add uncertainty indicators for any speculative elements (blur, "ESTIMATED" labels)
"""
    
    if safe_mode:
        watermark_instruction += """
SAFE MODE ADDITIONAL REQUIREMENTS:
- DOUBLE watermark opacity (very prominent)
- Add flashing "NOT REAL FOOTAGE" at scene transitions
- Include "TRAINING USE ONLY" footer throughout
"""
    
    # Voiceover instruction
    voiceover_instruction = f"""
MANDATORY AUDIO REQUIREMENTS:
- BEGIN video with spoken disclaimer: "{VOICEOVER_DISCLAIMER[:100]}..."
- END video with spoken disclaimer repeating non-admissibility
- Use calm, professional narrator voice
"""
    
    combined_prompt = f"""Educational crime reconstruction documentary simulation:

{watermark_instruction}

{voiceover_instruction}

SCENE CONTENT:
{chr(10).join(f'Scene {i+1}: {desc}' for i, desc in enumerate(scene_descriptions))}

STYLE REQUIREMENTS:
- Professional documentary reconstruction aesthetic
- Realistic but clearly SIMULATED law enforcement elements
- Environmental context with "RECONSTRUCTION" labels
- Semi-transparent uncertainty bars on speculative elements (e.g., "80% confidence")
- Final frame: 5-second static disclaimer card

TONE: Educational, informative, factual - explicitly NOT real footage
"""
    
    # Truncate if too long
    if len(combined_prompt) > 2500:
        combined_prompt = combined_prompt[:2500]
    
    return combined_prompt

# =============================================================================
# ASYNC LLM HELPER
# =============================================================================

async def call_llm(client: AsyncOpenAI, system: str, user: str, 
                   max_tokens: int = 2000, model: str = "gpt-4o-mini") -> tuple[str, int]:
    """Make an async LLM call with error handling and token tracking."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        audit.log_event("llm_call", {"model": model, "tokens": tokens})
        return content, tokens
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        audit.log_event("llm_error", {"error": str(e)})
        raise

# =============================================================================
# AGENT IMPLEMENTATIONS
# =============================================================================

async def transcript_agent(state: ReconstructionState, client: AsyncOpenAI) -> dict:
    """Transcript Agent: Converts police codes to narrative with uncertainty markers."""
    logger.info("🎙️ Transcript Agent: Converting blotter to narrative...")
    audit.log_event("agent_start", {"agent": "transcript"})
    
    system = """You are a police communications expert. Convert police blotter text 
    and radio codes into a clear, professional narrative transcript.
    
    Rules:
    - Expand all 10-codes (e.g., 10-4 = "Acknowledged", 10-20 = "Location")
    - Convert abbreviations to full terms
    - Add realistic radio dialogue format (Dispatch/Unit call-response)
    - Keep it factual, professional, educational
    - Output should be 150-300 words
    - Format as a continuous narrative with speaker labels
    
    IMPORTANT: Mark any ASSUMED or INFERRED details with [ESTIMATED] tags.
    If information is unclear, note it as [AMBIGUOUS: reason].
    
    This is for TRAINING SIMULATION purposes only."""
    
    content, tokens = await call_llm(client, system, state["blotter_input"], max_tokens=1500)
    
    logger.info(f"✓ Transcript complete ({tokens} tokens)")
    audit.log_event("agent_complete", {"agent": "transcript", "tokens": tokens})
    return {"transcript": content, "token_usage": state["token_usage"] + tokens}

async def scene_breakdown_agent(state: ReconstructionState, client: AsyncOpenAI) -> dict:
    """Scene Breakdown Agent with uncertainty indicators."""
    logger.info("🎬 Scene Breakdown Agent: Analyzing for key scenes...")
    audit.log_event("agent_start", {"agent": "scene_breakdown"})
    
    system = f"""You are a documentary scene planner for TRAINING SIMULATIONS.
    Break this police incident transcript into {MAX_SCENES} or fewer distinct scenes.
    
    For each scene provide:
    - scene_id: Sequential number
    - title: Short descriptive title
    - duration_seconds: Estimated length (5-15 seconds each, total max {MAX_VIDEO_LENGTH_SECONDS}s)
    - description: What happens (2-3 sentences)
    - key_elements: Visual elements needed
    - audio_cue: Narration/sound
    - confidence_level: "high|medium|low" - how certain is this interpretation?
    - uncertainty_notes: What's speculative or assumed?
    
    Output as JSON array. Mark speculative elements clearly.
    Remember: This is for TRAINING, not evidence."""
    
    content, tokens = await call_llm(client, system, state["transcript"], max_tokens=2000)
    
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        scenes = json.loads(content.strip())
        scenes = scenes[:MAX_SCENES]
        
        total_duration = sum(s.get("duration_seconds", 10) for s in scenes)
        if total_duration > MAX_VIDEO_LENGTH_SECONDS:
            scale = MAX_VIDEO_LENGTH_SECONDS / total_duration
            for s in scenes:
                s["duration_seconds"] = int(s.get("duration_seconds", 10) * scale)
                
    except json.JSONDecodeError:
        scenes = [{"scene_id": 1, "title": "Incident Overview", 
                   "duration_seconds": 30, "description": content[:500],
                   "key_elements": ["overview"], "audio_cue": "narration",
                   "confidence_level": "low", "uncertainty_notes": "Parse error"}]
    
    # Extract uncertainty indicators
    uncertainties = [{"scene": s.get("scene_id"), "level": s.get("confidence_level", "medium"),
                      "notes": s.get("uncertainty_notes", "")} for s in scenes]
    
    logger.info(f"✓ Identified {len(scenes)} scenes ({tokens} tokens)")
    audit.log_event("agent_complete", {"agent": "scene_breakdown", "scenes": len(scenes)})
    return {"scenes": scenes, "uncertainty_indicators": uncertainties, 
            "token_usage": state["token_usage"] + tokens}

async def visual_reconstruction_agent(scene: dict, scene_idx: int, 
                                       client: AsyncOpenAI, semaphore: asyncio.Semaphore) -> dict:
    """Visual Reconstruction Agent with mandatory simulation labeling."""
    async with semaphore:
        logger.info(f"🎨 Visual Agent {scene_idx+1}: Processing '{scene.get('title', 'Scene')}'...")
        
        system = """You are a crime documentary visual director creating TRAINING SIMULATIONS.
        Generate detailed video prompts for reconstructing a police incident scene.
        
        Generate 2-3 camera angles/views:
        1. Primary POV (bodycam style, dashcam)
        2. Overhead/tactical view (bird's eye)
        3. Environmental context (establishing shot)
        
        For each view provide:
        - view_type: Type of shot
        - prompt: Veo3-compatible prompt (50-100 words)
        - annotations: Text overlays (timestamps, labels, "SIMULATED" markers)
        - duration_seconds: Length
        - confidence: "high|medium|low" for this visual interpretation
        
        MANDATORY: Include "SIMULATION" or "RECONSTRUCTION" labels in every prompt.
        Add uncertainty indicators (blur, "ESTIMATED" text) for speculative elements.
        
        Output as JSON with 'views' array."""
        
        user = f"""Scene: {scene.get('title', 'Untitled')}
        Description: {scene.get('description', '')}
        Key Elements: {scene.get('key_elements', [])}
        Confidence Level: {scene.get('confidence_level', 'medium')}
        Uncertainty Notes: {scene.get('uncertainty_notes', 'None')}
        Target Duration: {scene.get('duration_seconds', 10)} seconds
        
        Remember: This is a TRAINING SIMULATION, not real footage."""
        
        content, tokens = await call_llm(client, system, user, max_tokens=1500)
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            visual_data = json.loads(content.strip())
        except json.JSONDecodeError:
            visual_data = {"views": [{"view_type": "primary", "prompt": content[:500], 
                                      "annotations": ["SIMULATED"], "duration_seconds": 10,
                                      "confidence": "low"}]}
        
        logger.info(f"✓ Visual prompts for scene {scene_idx+1} complete")
        
        return {
            "scene_id": scene.get("scene_id", scene_idx + 1),
            "scene_title": scene.get("title", f"Scene {scene_idx + 1}"),
            "visuals": visual_data.get("views", visual_data) if isinstance(visual_data, dict) else visual_data,
            "tokens_used": tokens
        }

async def run_parallel_visual_agents(state: ReconstructionState, client: AsyncOpenAI) -> dict:
    """Orchestrates parallel visual reconstruction agents."""
    logger.info(f"🚀 Launching {len(state['scenes'])} parallel Visual Agents...")
    audit.log_event("parallel_visual_start", {"scene_count": len(state['scenes'])})
    
    semaphore = asyncio.Semaphore(3)
    
    tasks = [
        visual_reconstruction_agent(scene, idx, client, semaphore)
        for idx, scene in enumerate(state["scenes"])
    ]
    
    results = []
    with tqdm(total=len(tasks), desc="Visual Reconstruction", unit="scene") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    results.sort(key=lambda x: x.get("scene_id", 0))
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    
    audit.log_event("parallel_visual_complete", {"total_tokens": total_tokens})
    return {"visual_prompts": results, "token_usage": state["token_usage"] + total_tokens}

async def synthesis_agent(state: ReconstructionState, client: AsyncOpenAI) -> dict:
    """Synthesis Agent: Merges outputs with mandatory disclaimers."""
    logger.info("🔧 Synthesis Agent: Building video script...")
    audit.log_event("agent_start", {"agent": "synthesis"})
    
    system = f"""You are a video production coordinator for TRAINING SIMULATIONS.
    Combine scene breakdowns and visual prompts into a unified video script.
    
    Output JSON:
    {{
        "title": "SIMULATION: [Video title]",
        "total_duration_seconds": (max 60),
        "mandatory_disclaimers": {{
            "opening_voiceover": "{VOICEOVER_DISCLAIMER[:100]}...",
            "closing_voiceover": "This was an AI-generated simulation...",
            "watermark_text": "{WATERMARK_SHORT}",
            "end_card_duration_seconds": 5
        }},
        "scenes": [
            {{
                "sequence": order,
                "title": "scene title",
                "duration": seconds,
                "primary_visual_prompt": "prompt with SIMULATION labels",
                "narration": "voiceover text",
                "annotations": ["on-screen text", "SIMULATED", "timestamps"],
                "confidence_indicator": "high|medium|low",
                "transition": "transition type"
            }}
        ],
        "metadata": {{
            "disclaimer": "AI-generated reconstruction for training only",
            "not_admissible": true,
            "generation_timestamp": "{GENERATION_TIMESTAMP}",
            "session_id": "{SESSION_ID}"
        }}
    }}
    
    CRITICAL: Every scene MUST include "SIMULATION" or "RECONSTRUCTION" in annotations."""
    
    user = f"""Transcript: {state['transcript'][:1000]}
    Scenes: {json.dumps(state['scenes'], indent=2)[:2000]}
    Visual Prompts: {json.dumps(state['visual_prompts'], indent=2)[:3000]}
    Uncertainties: {json.dumps(state['uncertainty_indicators'], indent=2)}"""
    
    content, tokens = await call_llm(client, system, user, max_tokens=3000)
    
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        video_script = json.loads(content.strip())
    except json.JSONDecodeError:
        video_script = {"title": "SIMULATION: Crime Reconstruction", 
                        "scenes": state["scenes"], "raw": content,
                        "mandatory_disclaimers": {"watermark_text": WATERMARK_SHORT}}
    
    logger.info(f"✓ Video script synthesized ({tokens} tokens)")
    audit.log_event("agent_complete", {"agent": "synthesis"})
    return {"video_script": video_script, "token_usage": state["token_usage"] + tokens}

async def critic_agent(state: ReconstructionState, client: AsyncOpenAI) -> dict:
    """Critic Agent: Reviews for accuracy, sensitivity, and legal compliance."""
    logger.info("🔍 Critic Agent: Reviewing for quality, sensitivity, and compliance...")
    audit.log_event("agent_start", {"agent": "critic"})
    
    # First, run sensitivity check
    sensitivity = await check_content_sensitivity(client, 
        f"{state['blotter_input']}\n{state['transcript']}")
    
    system = """You are an editorial and LEGAL COMPLIANCE reviewer for crime reconstruction simulations.
    
    Review the video script for:
    1. ACCURACY: Sticks to facts? No fabrication beyond source material?
    2. SENSITIVITY: Violence, minors, privacy concerns?
    3. LEGAL COMPLIANCE: 
       - Are "SIMULATION" labels present in every scene?
       - Are disclaimers included?
       - Could this be mistaken for real footage?
    4. EDUCATIONAL VALUE: Informative and appropriate?
    5. UNCERTAINTY: Are speculative elements clearly marked?
    
    Output JSON:
    {{
        "approved": boolean,
        "accuracy_score": 1-10,
        "compliance_score": 1-10,
        "sensitivity_flags": [],
        "legal_concerns": [],
        "missing_disclaimers": [],
        "suggestions": [],
        "requires_human_review": boolean,
        "review_summary": "summary",
        "recommendation": "proceed|caution|block"
    }}
    
    Be VERY conservative on legal compliance. Block if disclaimers are missing."""
    
    user = f"""Original Blotter: {state['blotter_input']}
    Video Script: {json.dumps(state['video_script'], indent=2)[:4000]}
    Sensitivity Pre-Check: {json.dumps(sensitivity)}"""
    
    content, tokens = await call_llm(client, system, user, max_tokens=1000)
    
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        feedback = json.loads(content.strip())
    except json.JSONDecodeError:
        feedback = {"approved": False, "review_summary": content, 
                    "sensitivity_flags": ["Parse error"], "recommendation": "caution"}
    
    # Merge sensitivity check
    feedback["sensitivity_precheck"] = sensitivity
    if sensitivity.get("recommendation") == "block":
        feedback["approved"] = False
        feedback["recommendation"] = "block"
        feedback["sensitivity_flags"].extend(sensitivity.get("flags", []))
    
    logger.info(f"✓ Critic review: {feedback.get('recommendation', 'unknown').upper()}")
    audit.log_event("critic_review", {"recommendation": feedback.get("recommendation"),
                                       "approved": feedback.get("approved")})
    return {"critic_feedback": feedback, "token_usage": state["token_usage"] + tokens}

async def generate_video(state: ReconstructionState, output_dir: str) -> dict:
    """Video Generation with mandatory disclaimers and safe filenames."""
    logger.info("🎥 Starting video generation with Veo3...")
    audit.log_event("video_generation_start", {"safe_mode": state.get("safe_mode", False)})
    
    script = state["video_script"]
    
    if not script.get("scenes"):
        return {"error": "No scenes to generate", "video_path": ""}
    
    # Build prompt with all disclaimers
    combined_prompt = build_video_prompt_with_disclaimers(script, state.get("safe_mode", False))
    
    logger.info(f"Prompt length: {len(combined_prompt)} chars")
    audit.log_event("video_prompt", {"length": len(combined_prompt)})
    
    try:
        client = genai.Client()
        operation = client.models.generate_videos(
            model="veo-3.0-generate-001",
            prompt=combined_prompt,
        )
        
        with tqdm(desc="Generating Video", unit="check") as pbar:
            while not operation.done:
                time.sleep(10)
                operation = client.operations.get(operation)
                pbar.update(1)
        
        if getattr(operation, "error", None):
            raise RuntimeError(f"Veo3 error: {operation.error}")
        
        try:
            operation = client.operations.wait(operation)
        except:
            pass
        
        if operation.response and operation.response.generated_videos:
            generated_video = operation.response.generated_videos[0]
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Use safe filename
            safe_filename = generate_safe_filename(state["case_id"], state.get("safe_mode", False))
            video_path = os.path.join(output_dir, safe_filename)
            
            client.files.download(file=generated_video.video)
            generated_video.video.save(video_path)
            
            # Set output hash for provenance
            audit.set_output_hash(video_path)
            
            # Save provenance report as sidecar
            provenance_path = audit.save_report(output_dir, state["case_id"])
            
            logger.info(f"✓ Video saved: {video_path}")
            audit.log_event("video_generation_complete", {"path": video_path})
            
            # Print post-processing reminder
            print("\n" + "="*60)
            print("⚠️  POST-PROCESSING REQUIRED FOR FULL COMPLIANCE:")
            print("="*60)
            print("The AI was instructed to include watermarks, but verify:")
            print(f"1. Watermark visible: '{WATERMARK_SHORT}'")
            print("2. Opening/closing voiceover disclaimers present")
            print("3. Run through ffmpeg to burn in metadata:")
            print(f"   ffmpeg -i {safe_filename} -metadata comment='{VOICEOVER_DISCLAIMER[:100]}' ...")
            print("="*60 + "\n")
            
            return {"video_path": video_path, "provenance_path": provenance_path, "error": None}
        else:
            return {"error": "No video generated - check quotas", "video_path": ""}
            
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        audit.log_event("video_generation_error", {"error": str(e)})
        return {"error": str(e), "video_path": ""}

# =============================================================================
# LANGGRAPH WORKFLOW NODES
# =============================================================================

def create_workflow(output_dir: str, safe_mode: bool = False, training_mode: bool = True):
    """Create the LangGraph workflow with all safety checkpoints."""
    
    openai_client = AsyncOpenAI()
    
    async def validate_node(state: ReconstructionState) -> ReconstructionState:
        valid, error = validate_blotter(state["blotter_input"])
        if not valid:
            state["error"] = error
            logger.error(f"Validation failed: {error}")
            audit.log_event("validation_failed", {"error": error})
        else:
            audit.set_input_hash(state["blotter_input"])
            audit.log_event("validation_passed", {"input_length": len(state["blotter_input"])})
        return state
    
    async def transcript_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        result = await transcript_agent(state, openai_client)
        state.update(result)
        return state
    
    async def confirm_transcript_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        print(f"\n📜 TRANSCRIPT:\n{'-'*40}\n{state['transcript']}\n{'-'*40}")
        print("\n⚠️  Review for accuracy. [ESTIMATED] tags mark assumptions.")
        accepted, reason = confirm_prompt("Transcript generated. Review above.", 
                                          require_reason=safe_mode)
        if not accepted:
            state["error"] = "User rejected transcript"
        return state
    
    async def breakdown_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        result = await scene_breakdown_agent(state, openai_client)
        state.update(result)
        return state
    
    async def confirm_scenes_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        print(f"\n🎬 SCENES ({len(state['scenes'])} total):")
        for s in state["scenes"]:
            conf = s.get('confidence_level', '?')
            print(f"  • {s.get('title', 'Untitled')} ({s.get('duration_seconds', '?')}s) [Confidence: {conf}]")
            if s.get('uncertainty_notes'):
                print(f"    ⚠️  Uncertainty: {s.get('uncertainty_notes')}")
        
        accepted, reason = confirm_prompt("Scene breakdown complete. Review confidence levels above.",
                                          require_reason=safe_mode)
        if not accepted:
            state["error"] = "User rejected scene breakdown"
        return state
    
    async def visual_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        result = await run_parallel_visual_agents(state, openai_client)
        state.update(result)
        return state
    
    async def synthesis_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        result = await synthesis_agent(state, openai_client)
        state.update(result)
        return state
    
    async def critic_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        result = await critic_agent(state, openai_client)
        state.update(result)
        
        feedback = state["critic_feedback"]
        
        # Display all flags
        if feedback.get("sensitivity_flags"):
            print(f"\n⚠️  SENSITIVITY FLAGS: {feedback['sensitivity_flags']}")
        if feedback.get("legal_concerns"):
            print(f"\n⚖️  LEGAL CONCERNS: {feedback['legal_concerns']}")
        if feedback.get("missing_disclaimers"):
            print(f"\n📋 MISSING DISCLAIMERS: {feedback['missing_disclaimers']}")
            
        print(f"\n📋 CRITIC REVIEW:")
        print(f"   Recommendation: {feedback.get('recommendation', 'unknown').upper()}")
        print(f"   Accuracy Score: {feedback.get('accuracy_score', '?')}/10")
        print(f"   Compliance Score: {feedback.get('compliance_score', '?')}/10")
        print(f"   Summary: {feedback.get('review_summary', 'No summary')}")
        
        # Block if critic recommends blocking
        if feedback.get("recommendation") == "block":
            print("\n❌ CRITIC BLOCKED: Content flagged as inappropriate for generation.")
            state["error"] = "Critic blocked content generation"
            return state
        
        if feedback.get("requires_human_review") or not feedback.get("approved", True):
            accepted, reason = confirm_prompt("Critic flagged issues. Continue anyway?",
                                              require_reason=True)
            if not accepted:
                state["error"] = "User rejected after critic review"
        return state
    
    async def confirm_video_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        
        elapsed = time.time() - state["start_time"]
        est_cost = state["token_usage"] * 0.00001
        
        print(f"\n{'='*60}")
        print("💰 FINAL CONFIRMATION BEFORE VIDEO GENERATION")
        print('='*60)
        print(f"   Tokens used: {state['token_usage']}")
        print(f"   Estimated LLM cost: ${est_cost:.4f}")
        print(f"   Time elapsed: {elapsed:.1f}s")
        print(f"   Video generation: ~1-2 min additional")
        print(f"   Case ID: {state['case_id']}")
        print(f"   Output filename: {generate_safe_filename(state['case_id'], state.get('safe_mode', False))}")
        print('='*60)
        print("\n⚠️  FINAL REMINDER:")
        print("   The generated video is a SIMULATION for training only.")
        print("   It is NOT admissible as evidence in any legal proceeding.")
        print('='*60)
        
        accepted, reason = confirm_prompt("Ready to generate video with Veo3.",
                                          require_reason=safe_mode)
        if not accepted:
            state["error"] = "User cancelled video generation"
            audit.log_event("user_cancelled_video", {"reason": reason})
        return state
    
    async def video_node(state: ReconstructionState) -> ReconstructionState:
        if state.get("error"):
            return state
        result = await generate_video(state, output_dir)
        state.update(result)
        state["final_approved"] = not state.get("error")
        return state
    
    # Build graph
    workflow = StateGraph(ReconstructionState)
    
    workflow.add_node("validate", validate_node)
    workflow.add_node("transcript", transcript_node)
    workflow.add_node("confirm_transcript", confirm_transcript_node)
    workflow.add_node("breakdown", breakdown_node)
    workflow.add_node("confirm_scenes", confirm_scenes_node)
    workflow.add_node("visual", visual_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("confirm_video", confirm_video_node)
    workflow.add_node("video", video_node)
    
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "transcript")
    workflow.add_edge("transcript", "confirm_transcript")
    workflow.add_edge("confirm_transcript", "breakdown")
    workflow.add_edge("breakdown", "confirm_scenes")
    workflow.add_edge("confirm_scenes", "visual")
    workflow.add_edge("visual", "synthesis")
    workflow.add_edge("synthesis", "critic")
    workflow.add_edge("critic", "confirm_video")
    workflow.add_edge("confirm_video", "video")
    workflow.add_edge("video", END)
    
    return workflow.compile()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_reconstruction(blotter: str, output_dir: str, 
                             safe_mode: bool = False, training_mode: bool = True) -> dict:
    """Main async entry point."""
    
    # Display legal banner
    display_legal_banner()
    
    # Require license acceptance
    if not require_license_acceptance(safe_mode):
        return {"error": "License not accepted"}
    
    logger.info("="*60)
    logger.info("🚔 CRIME RECONSTRUCTION SIMULATION SYSTEM")
    logger.info(f"   Session: {SESSION_ID}")
    logger.info(f"   Safe Mode: {safe_mode}")
    logger.info(f"   Training Mode: {training_mode}")
    logger.info("="*60)
    
    case_id = extract_case_id(blotter)
    audit.log_event("session_start", {"case_id": case_id, "safe_mode": safe_mode})
    
    initial_state: ReconstructionState = {
        "blotter_input": blotter,
        "case_id": case_id,
        "transcript": "",
        "scenes": [],
        "visual_prompts": [],
        "video_script": {},
        "critic_feedback": {},
        "uncertainty_indicators": [],
        "final_approved": False,
        "video_path": "",
        "provenance_path": "",
        "error": None,
        "token_usage": 0,
        "start_time": time.time(),
        "safe_mode": safe_mode,
        "training_mode": training_mode
    }
    
    workflow = create_workflow(output_dir, safe_mode, training_mode)
    
    try:
        final_state = await workflow.ainvoke(initial_state)
        
        elapsed = time.time() - initial_state["start_time"]
        
        print("\n" + "="*60)
        if final_state.get("error"):
            logger.error(f"❌ Pipeline terminated: {final_state['error']}")
            print(f"❌ TERMINATED: {final_state['error']}")
        elif final_state.get("video_path"):
            logger.info(f"✅ SUCCESS! Video: {final_state['video_path']}")
            print(f"✅ VIDEO SAVED: {final_state['video_path']}")
            print(f"📋 PROVENANCE: {final_state.get('provenance_path', 'N/A')}")
            print("\n⚠️  REMINDER: This is a SIMULATION - NOT real footage.")
            print("   NOT admissible as evidence in any court.")
        else:
            logger.warning("⚠️ Completed but no video generated")
        
        print(f"\n📊 Stats: {final_state.get('token_usage', 0)} tokens, {elapsed:.1f}s")
        print("="*60)
        
        # Save final provenance report
        audit.log_event("session_complete", {
            "success": bool(final_state.get("video_path")),
            "total_tokens": final_state.get("token_usage", 0),
            "duration_seconds": elapsed
        })
        
        return final_state
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        audit.log_event("session_error", {"error": str(e)})
        return {"error": str(e)}

def main():
    """CLI entry point with legal safeguards."""
    parser = argparse.ArgumentParser(
        description="Parallel Agent Crime Reconstruction SIMULATION System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  LEGAL NOTICE: All outputs are AI-generated SIMULATIONS.
    NOT admissible as evidence in any court of law.
    
Examples:
  python parallel_test.py --blotter "10-4, suspicious vehicle at 123 Main St"
  python parallel_test.py --blotter-file incident.txt --output_dir ./videos
  python parallel_test.py --safe-mode --blotter "..."
        """
    )
    parser.add_argument('--blotter', type=str, help='Police blotter text input')
    parser.add_argument('--blotter-file', type=str, help='File containing blotter text')
    parser.add_argument('--output_dir', type=str, default='./videos', help='Output directory')
    parser.add_argument('--safe-mode', action='store_true', 
                        help='Enable enhanced disclaimers, double watermarks, and signed certification')
    parser.add_argument('--training-mode', action='store_true', default=True,
                        help='Lock to training-only mode (always on)')
    
    args = parser.parse_args()
    
    # Get blotter input
    if args.blotter:
        blotter = args.blotter
    elif args.blotter_file:
        with open(args.blotter_file, 'r') as f:
            blotter = f.read().strip()
    else:
        blotter = """10-4, dispatch to all units. Suspicious vehicle reported at 123 Main Street, 
        plates Alpha-Bravo-Charlie-123. Possible 10-31 (crime in progress), suspected B&E. 
        Witness reports two individuals in dark clothing exiting rear of building. 
        Units 7 and 12 responding. 10-20 confirmed, ETA 3 minutes. Advise caution, 
        possible 10-32 (person with weapon). Requesting backup and K-9 unit."""
        logger.info("Using default test blotter input")
    
    result = asyncio.run(run_reconstruction(
        blotter, 
        args.output_dir, 
        safe_mode=args.safe_mode,
        training_mode=True  # Always forced on
    ))
    
    if result.get("error"):
        exit(1)
    exit(0)

if __name__ == "__main__":
    main()
