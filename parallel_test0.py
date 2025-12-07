# batch_parallel.py — runs 8 tasks truly in parallel
import asyncio
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

async def run_task(task_prompt):
    resp = await acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": task_prompt}],
        temperature=0.7,
        max_tokens=800
    )
    return resp.choices[0].message.content

async def main():
    tasks = [
        "Write a report on adequate court admissible evidence in a criminal case.",
        "Write a report on effectice crime reconstruction techniques.",
        # ← your 60–100 tasks here
    ]
    
    # This line is where the magic happens:
    results = await asyncio.gather(*[run_task(t) for t in tasks[:2]], return_exceptions=True)
    
    # Then just loop with chunks of 8 until done
    # (add a for loop + semaphore if you want exactly N concurrent)

asyncio.run(main())