import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

prompt = """

An elegant visualization for an app that visualizes option contracts as rockets in space, titled "Optionaut 4D", fades out to a professional italicized title with a rocket emoji.

"""

operation = client.models.generate_videos(
    model="veo-3.0-generate-001",
    prompt=prompt,
)

# Poll the operation status until the video is ready.
while not operation.done:
    print("Waiting for video generation to complete...")
    time.sleep(10)
    # Refresh operation status using the operation object
    operation = client.operations.get(operation)

# Ensure the operation completed successfully
if getattr(operation, "error", None):
    raise RuntimeError(f"Video generation failed: {operation.error}")

# Final wait to hydrate response if needed (SDK-dependent)
try:
    operation = client.operations.wait(operation)  # may block briefly
except Exception:
    pass

# Validate response and generated videos
if not getattr(operation, "response", None) or not getattr(operation.response, "generated_videos", None):
    # Print diagnostics to help understand why no videos were returned
    print("Operation object:", operation)
    print("Operation done:", getattr(operation, "done", None))
    print("Operation error:", getattr(operation, "error", None))
    print("Operation metadata:", getattr(operation, "metadata", None))
    print("Operation response:", getattr(operation, "response", None))
    if getattr(operation, "response", None):
        try:
            print("Response attributes:", [a for a in dir(operation.response) if not a.startswith("_")])
        except Exception:
            pass
    raise RuntimeError("Operation completed but no generated videos were returned.")

# Download the generated video
generated_video = operation.response.generated_videos[0]
client.files.download(file=generated_video.video)
generated_video.video.save("dialogue_example.mp4")
print("Generated video saved to dialogue_example.mp4")