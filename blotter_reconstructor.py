"""
Blotter Reconstructor - AI Video Generation from Police Radio
==============================================================
Generates educational reconstruction videos from police radio/blotter input.

⚠️  LEGAL DISCLAIMER: All outputs are AI-generated SIMULATIONS for 
    TRAINING AND HYPOTHESIS PURPOSES ONLY. NOT admissible as evidence.

Usage: python blotter_reconstructor.py --url <youtube_url> --duration <seconds>

License: Reconstruction-Only License v1.0
- Prohibits use in legal proceedings as evidence
- Prohibits removal of watermarks or voiceover disclaimers
- Requires attribution to this repository
"""

import argparse
import hashlib
import json
import re
import time
import subprocess
import tempfile
import os
import uuid
from datetime import datetime

import whisper
from dotenv import load_dotenv
load_dotenv()
from google import genai

# =============================================================================
# LEGAL DISCLAIMER BANNER
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
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

LICENSE_TERMS = """
═══════════════════════════════════════════════════════════════════════════════
                    RECONSTRUCTION-ONLY LICENSE v1.0
═══════════════════════════════════════════════════════════════════════════════

By proceeding, you acknowledge and agree:

1. All generated content is for TRAINING/EDUCATIONAL purposes ONLY.
2. You will NOT use any output as evidence in any legal proceeding.
3. You will NOT remove watermarks or disclaimers from outputs.
4. You will NOT show content to witnesses, victims, or suspects.
5. You assume all responsibility for misuse.

VIOLATION MAY CONSTITUTE FRAUD AND OBSTRUCTION OF JUSTICE.
═══════════════════════════════════════════════════════════════════════════════
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

WATERMARK_TEXT = "⚠️ SIMULATED RECONSTRUCTION – NOT ACTUAL FOOTAGE – NOT ADMISSIBLE IN COURT ⚠️"
WATERMARK_SHORT = "AI SIMULATION - NOT EVIDENCE"

VOICEOVER_DISCLAIMER = """This is an AI-generated reconstruction based solely on police radio logs 
and dispatcher notes. It is not actual video evidence and may contain errors or assumptions. 
It is not admissible in any court of law."""

SESSION_ID = str(uuid.uuid4())[:8]
GENERATION_TIMESTAMP = datetime.now().isoformat()

# =============================================================================
# PROVENANCE & AUDIT
# =============================================================================

class ProvenanceTracker:
    """Tracks provenance for compliance."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.events = []
        self.user_acknowledgments = []
        
    def log_event(self, event_type: str, details: dict):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        })
        
    def log_acknowledgment(self, ack_type: str, accepted: bool):
        self.user_acknowledgments.append({
            "timestamp": datetime.now().isoformat(),
            "type": ack_type,
            "accepted": accepted
        })
        
    def generate_report(self, input_text: str, output_path: str) -> dict:
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        output_hash = ""
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                output_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                
        return {
            "report_type": "AI_RECONSTRUCTION_PROVENANCE",
            "version": "1.0",
            "session_id": self.session_id,
            "generation_timestamp": self.start_time.isoformat(),
            "completion_timestamp": datetime.now().isoformat(),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "user_acknowledgments": self.user_acknowledgments,
            "events": self.events,
            "disclaimer": VOICEOVER_DISCLAIMER,
            "license": "Reconstruction-Only License v1.0",
            "legal_notice": "NOT ADMISSIBLE AS EVIDENCE",
            "ai_models_used": {
                "transcription": "whisper-base",
                "video": "veo-3.0-generate-001"
            }
        }
        
    def save_report(self, output_dir: str, report: dict):
        report_path = os.path.join(
            output_dir,
            f"PROVENANCE_SIMULATION_{datetime.now():%Y-%m-%d}_{self.session_id}.json"
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📋 Provenance report saved: {report_path}")
        return report_path

provenance = ProvenanceTracker(SESSION_ID)

# =============================================================================
# USER CONSENT FUNCTIONS
# =============================================================================

def display_legal_banner():
    """Display the legal disclaimer banner."""
    print(LEGAL_BANNER)

def require_license_acceptance(safe_mode: bool = False) -> bool:
    """Require explicit license acceptance."""
    print(LICENSE_TERMS)
    
    if safe_mode:
        print("\n🔒 SAFE MODE ENABLED - Enhanced disclaimers active.\n")
    
    print("\nTo proceed, type 'I ACCEPT' exactly as shown:")
    response = input("Your response: ").strip()
    
    accepted = response == "I ACCEPT"
    provenance.log_acknowledgment("license_terms", accepted)
    
    if not accepted:
        print("\n❌ License not accepted. Exiting.")
        return False
        
    if safe_mode:
        print("\n" + "="*60)
        print("SAFE MODE CERTIFICATION")
        print("="*60)
        cert_name = input("Your name/ID: ").strip()
        cert_statement = input("Type 'I CERTIFY' to confirm non-legal use: ").strip()
        
        if cert_statement != "I CERTIFY" or not cert_name:
            print("\n❌ Certification failed. Exiting.")
            provenance.log_acknowledgment("safe_mode_cert", False)
            return False
            
        provenance.log_event("safe_mode_certification", {
            "certifier": cert_name,
            "timestamp": datetime.now().isoformat()
        })
        provenance.log_acknowledgment("safe_mode_cert", True)
        print(f"✓ Certified: {cert_name}")
    
    return True

def confirm_prompt(message: str) -> bool:
    """Human-in-the-loop confirmation."""
    print(f"\n{'='*60}")
    print(message)
    print('='*60)
    while True:
        response = input("Proceed? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            provenance.log_acknowledgment(f"checkpoint_{message[:20]}", True)
            return True
        elif response in ('n', 'no'):
            provenance.log_acknowledgment(f"checkpoint_{message[:20]}", False)
            return False
        print("Please enter 'y' or 'n'")

def generate_safe_filename(safe_mode: bool = False) -> str:
    """Generate legally-safe filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    prefix = "SIMULATION_NOT_EVIDENCE"
    if safe_mode:
        prefix = "SAFE_MODE_SIMULATION_NOT_EVIDENCE"
    return f"{prefix}_{timestamp}_{SESSION_ID}.mp4"

# =============================================================================
# PROMPT BUILDING WITH DISCLAIMERS
# =============================================================================

def build_video_prompt(transcript: str, scene_desc: str, safe_mode: bool = False) -> str:
    """Build video prompt with mandatory disclaimer instructions."""
    
    watermark_instruction = f"""
MANDATORY VISUAL REQUIREMENTS:
- Display prominent semi-transparent watermark throughout: "{WATERMARK_TEXT}"
- Show text overlay: "AI RECONSTRUCTION - FOR TRAINING PURPOSES ONLY"
- Include "SIMULATED" label in corner throughout video
- Add uncertainty indicators for speculative elements
"""
    
    if safe_mode:
        watermark_instruction += """
SAFE MODE REQUIREMENTS:
- DOUBLE watermark opacity
- Add flashing "NOT REAL FOOTAGE" at transitions
- Include "TRAINING USE ONLY" footer
"""
    
    voiceover_instruction = f"""
MANDATORY AUDIO REQUIREMENTS:
- BEGIN with spoken disclaimer about AI-generated reconstruction
- END with spoken reminder: "This is not admissible as evidence"
- Professional narrator voice throughout
"""
    
    prompt = f"""Educational crime reconstruction SIMULATION:

{watermark_instruction}

{voiceover_instruction}

CONTENT TO RECONSTRUCT:
Transcript: {transcript[:500]}
Scene Elements: {scene_desc}

STYLE:
- Professional documentary reconstruction aesthetic
- Clearly SIMULATED law enforcement elements
- "RECONSTRUCTION" labels on all scenes
- Final frame: 5-second static disclaimer card

CRITICAL: This is a TRAINING SIMULATION, NOT real footage.
Tone: Educational, informative, factual - explicitly simulated.
"""
    
    if len(prompt) > 2000:
        prompt = prompt[:2000]
    
    return prompt

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    # Parse args
    parser = argparse.ArgumentParser(
        description="Blotter Reconstructor - AI Video SIMULATION Generator",
        epilog="⚠️ All outputs are SIMULATIONS - NOT admissible as evidence."
    )
    parser.add_argument('--url', required=True, help='YouTube live stream URL')
    parser.add_argument('--duration', type=int, default=2, help='Seconds of video to capture')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--safe-mode', action='store_true', 
                        help='Enable enhanced disclaimers and certification')
    args = parser.parse_args()
    
    # Display legal banner and require acceptance
    display_legal_banner()
    
    if not require_license_acceptance(args.safe_mode):
        return 1
    
    provenance.log_event("session_start", {
        "url": args.url,
        "duration": args.duration,
        "safe_mode": args.safe_mode
    })
    
    print(f"\n🚔 Starting reconstruction (Session: {SESSION_ID})")
    print("="*60)
    
    # Step 1: Download limited video segment with yt-dlp
    audio_path = os.path.join(tempfile.gettempdir(), f"blotter_audio_{os.getpid()}")
    final_audio = audio_path + ".mp4"

    duration_str = f"0:{args.duration:02d}"
    cmd = [
        'yt-dlp',
        '-f', 'worst[ext=mp4]/worst',
        '--no-playlist',
        '--progress',
        '--download-sections', f'*0:00-{duration_str}',
        '--force-keyframes-at-cuts',
        '-o', final_audio,
        args.url
    ]
    print(f"📥 Downloading {args.duration}s from {args.url}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Failed to download video")

    print(f"✓ Downloaded to {final_audio}")
    provenance.log_event("download_complete", {"path": final_audio})

    # Step 2: Transcribe with Whisper
    print("🎙️ Transcribing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(final_audio)
    transcript = result["text"].strip()
    print(f"\n📜 Transcript:\n{'-'*40}\n{transcript}\n{'-'*40}")
    os.unlink(final_audio)
    
    provenance.log_event("transcription_complete", {"length": len(transcript)})
    
    # Human checkpoint: Confirm transcript
    if not transcript:
        print("⚠️ No speech detected in audio. Using placeholder.")
        transcript = "[No clear speech detected - ambient audio only]"
    
    if not confirm_prompt("Review transcript above. Proceed with video generation?"):
        print("❌ User cancelled at transcript review.")
        return 1

    # Step 3: Parse elements
    codes = re.findall(r'10-\d+', transcript)
    locations = re.findall(r'(alleyway|street|building|scene)', transcript, re.I)
    weather = re.findall(r'(rain|snow|dark|night)', transcript, re.I)
    actions = re.findall(r'(approaching|arrived|pursuit|responding)', transcript, re.I)
    scene_desc = f"Police {' '.join(actions)} {' '.join(locations)} in {' '.join(weather)} weather, {' '.join(codes)} context."
    
    print(f"\n🎬 Scene Analysis:\n{scene_desc}")
    
    # Human checkpoint: Confirm scene analysis
    if not confirm_prompt("Scene analysis complete. Proceed to video generation?"):
        print("❌ User cancelled at scene analysis.")
        return 1

    # Step 4: Build prompt with all disclaimers
    prompt = build_video_prompt(transcript, scene_desc, args.safe_mode)
    
    provenance.log_event("prompt_built", {"length": len(prompt)})

    # Step 5: Final confirmation before video generation
    print("\n" + "="*60)
    print("💰 FINAL CONFIRMATION")
    print("="*60)
    print(f"   Session ID: {SESSION_ID}")
    print(f"   Output: {generate_safe_filename(args.safe_mode)}")
    print(f"   Estimated time: 1-2 minutes")
    print("="*60)
    print("\n⚠️  FINAL REMINDER:")
    print("   The generated video is a SIMULATION for training only.")
    print("   It is NOT admissible as evidence in any legal proceeding.")
    print("="*60)
    
    if not confirm_prompt("Generate video with Veo3?"):
        print("❌ User cancelled video generation.")
        return 1

    # Step 6: Generate video with Veo 3
    print("\n🎥 Generating video with Veo3...")
    provenance.log_event("video_generation_start", {})
    
    client = genai.Client()
    operation = client.models.generate_videos(
        model="veo-3.0-generate-001",
        prompt=prompt,
    )

    # Poll until done
    while not operation.done:
        print("⏳ Generating video...")
        time.sleep(10)
        operation = client.operations.get(operation)

    if getattr(operation, "error", None):
        raise RuntimeError(f"Veo failed: {operation.error}")

    # Final wait
    try:
        operation = client.operations.wait(operation)
    except:
        pass

    if operation.response and operation.response.generated_videos:
        generated_video = operation.response.generated_videos[0]
        
        # Use safe filename
        os.makedirs(args.output_dir, exist_ok=True)
        safe_filename = generate_safe_filename(args.safe_mode)
        video_path = os.path.join(args.output_dir, safe_filename)
        
        client.files.download(file=generated_video.video)
        generated_video.video.save(video_path)
        
        print(f"\n✅ Video saved: {video_path}")
        
        # Save provenance report
        report = provenance.generate_report(transcript, video_path)
        provenance.save_report(args.output_dir, report)
        
        # Post-processing reminder
        print("\n" + "="*60)
        print("⚠️  POST-PROCESSING REMINDER:")
        print("="*60)
        print("Verify the AI included watermarks. If not, add manually:")
        print(f"1. Watermark: '{WATERMARK_SHORT}'")
        print("2. Opening/closing voiceover disclaimers")
        print("3. Burn metadata with ffmpeg:")
        print(f"   ffmpeg -i {safe_filename} -metadata comment='AI SIMULATION' ...")
        print("="*60)
        
        print("\n" + "="*60)
        print("⚠️  REMINDER: This is a SIMULATION - NOT real footage.")
        print("   NOT admissible as evidence in any court.")
        print("="*60)
        
        provenance.log_event("video_generation_complete", {"path": video_path})
        return 0
    else:
        print("❌ No video generated—check prompt/quotas.")
        provenance.log_event("video_generation_failed", {})
        return 1

if __name__ == "__main__":
    exit(main())
