"""Part VII - Concurrency and Performance."""

PART = "Concurrency and Performance"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Threads, Processes and concurrent.futures for Data Loading",
        "blocks": [
            {"t": "p", "text":
                "Data loading is often the real bottleneck in training: the GPU "
                "sits idle while the CPU decodes images or tokenises text. "
                "Python gives you two kinds of parallelism to hide that "
                "latency, and choosing between them comes down to one acronym: "
                "the **GIL**."},
            {"t": "h", "text": "The GIL: threads vs processes"},
            {"t": "p", "text":
                "The Global Interpreter Lock lets only one thread run Python "
                "bytecode at a time. So **threads** win for I/O-bound work "
                "(downloading, reading files) because the GIL is released while "
                "waiting, but **processes** are needed for CPU-bound work "
                "(heavy pure-Python computation) since each process has its own "
                "interpreter and GIL."},
            {"t": "bullets", "items": [
                "**Threads** - I/O-bound: downloading shards, reading files, "
                "calling APIs.",
                "**Processes** - CPU-bound: heavy pixel/audio decoding, pure-"
                "Python math.",
                "Most tensor math already runs in C and releases the GIL, so "
                "threads go further than beginners expect.",
            ]},
            {"t": "h", "text": "ThreadPoolExecutor.map for parallel I/O"},
            {"t": "p", "text":
                "`concurrent.futures` gives a single clean API for both pools. "
                "`executor.map` applies a function across an iterable "
                "concurrently and returns results **in input order** - ideal "
                "for preprocessing a batch of samples. We simulate I/O latency "
                "with `time.sleep`."},
            {"t": "code", "run": True, "caption": "Parallel preprocessing with a thread pool",
             "code": '''\
import time
from concurrent.futures import ThreadPoolExecutor

def load_sample(i):
    time.sleep(0.1)          # pretend this is a network / disk read
    return i * i

items = range(8)

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(load_sample, items))
elapsed = time.perf_counter() - start

print("results:", results)
print(f"8 x 0.1s work done in {elapsed:.2f}s (concurrent, not 0.8s)")
'''},
            {"t": "h", "text": "submit + as_completed"},
            {"t": "p", "text":
                "When you want each result the moment it is ready (say, to "
                "update a progress bar) use `submit` to schedule work and "
                "`as_completed` to iterate over futures as they finish, in "
                "**completion order** rather than submission order."},
            {"t": "code", "run": True, "caption": "Process results as they finish",
             "code": '''\
import time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch(url):
    time.sleep(random.uniform(0.02, 0.12))
    return f"{url} -> 200 OK"

urls = [f"shard-{i}" for i in range(5)]

with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {pool.submit(fetch, u): u for u in urls}
    for fut in as_completed(futures):
        print("done:", fut.result())
'''},
            {"t": "h", "text": "Switching to processes for CPU work"},
            {"t": "p", "text":
                "The API is identical - just swap the executor class. The one "
                "catch is that process pools re-import your module, so the "
                "launch code must sit under an `if __name__ == '__main__':` "
                "guard. The snippet below is *illustrative* (process pools are "
                "awkward inside tiny scripts, but this is exactly the shape you "
                "use in a real training script)."},
            {"t": "code", "run": False, "caption": "Illustrative: ProcessPoolExecutor for CPU-bound work",
             "code": '''\
from concurrent.futures import ProcessPoolExecutor

def heavy_decode(chunk):
    # Pure-Python CPU work that the GIL would otherwise serialise.
    return sum(i * i for i in range(chunk))

if __name__ == "__main__":         # required for process pools
    with ProcessPoolExecutor(max_workers=4) as pool:
        totals = list(pool.map(heavy_decode, [10_000] * 4))
    print(totals)
'''},
            {"t": "note", "text":
                "Why it matters for AI: this is exactly what a PyTorch "
                "`DataLoader(num_workers=N)` does under the hood - fan out "
                "sample loading across workers so the GPU never starves. Reach "
                "for threads on I/O-bound pipelines and processes only when the "
                "work is genuinely CPU-bound Python."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "async / await for Concurrent LLM and API Calls",
        "blocks": [
            {"t": "p", "text":
                "When you call an LLM API, almost all the time is spent waiting "
                "for the network. **async/await** lets a single thread juggle "
                "hundreds of in-flight requests, making it the standard tool "
                "for batching prompts, building agents and serving inference."},
            {"t": "h", "text": "Defining and running a coroutine"},
            {"t": "p", "text":
                "`async def` defines a *coroutine*; calling it returns an object "
                "that does nothing until awaited. `asyncio.run` starts the event "
                "loop and drives one coroutine to completion. `await` yields "
                "control while waiting, so other tasks can proceed."},
            {"t": "code", "run": True, "caption": "A first coroutine",
             "code": '''\
import asyncio

async def call_model(prompt):
    await asyncio.sleep(0.1)          # stand-in for a network round-trip
    return f"response to: {prompt!r}"

async def main():
    reply = await call_model("hello")
    print(reply)

asyncio.run(main())
'''},
            {"t": "h", "text": "gather: fire many requests at once"},
            {"t": "p", "text":
                "`asyncio.gather` schedules many coroutines concurrently and "
                "waits for all of them, returning results in order. Ten "
                "requests that each take 0.1s finish in about 0.1s total, not "
                "1.0s - the whole point of async."},
            {"t": "code", "run": True, "caption": "Concurrent requests with gather",
             "code": '''\
import asyncio, time

async def call_model(prompt):
    await asyncio.sleep(0.1)
    return prompt.upper()

async def main():
    prompts = [f"q{i}" for i in range(10)]
    start = time.perf_counter()
    results = await asyncio.gather(*(call_model(p) for p in prompts))
    elapsed = time.perf_counter() - start
    print("results:", results)
    print(f"10 requests in {elapsed:.2f}s (concurrent, not 1.0s)")

asyncio.run(main())
'''},
            {"t": "h", "text": "Semaphore: respecting rate limits"},
            {"t": "p", "text":
                "Real APIs cap how many requests you may have in flight. An "
                "`asyncio.Semaphore` bounds concurrency: acquire it before each "
                "call and only N run at once, while the rest wait their turn. "
                "This is the idiomatic way to stay under a rate limit."},
            {"t": "code", "run": True, "caption": "Bounding concurrency to N at a time",
             "code": '''\
import asyncio, time

sem = asyncio.Semaphore(3)          # at most 3 concurrent calls

async def call_model(i):
    async with sem:
        await asyncio.sleep(0.1)
        return i

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(call_model(i) for i in range(9)))
    elapsed = time.perf_counter() - start
    # 9 tasks, 3 at a time, 0.1s each => ~3 waves => ~0.3s
    print("results:", results)
    print(f"9 calls, limit 3, took {elapsed:.2f}s (~3 waves)")

asyncio.run(main())
'''},
            {"t": "h", "text": "What a real async LLM client looks like"},
            {"t": "p", "text":
                "The pattern is identical with a real SDK: build an async "
                "client, wrap each call in a semaphore, and `gather` them. The "
                "snippet below is *illustrative* only - no network call is made "
                "- but it mirrors the shape of production batch-inference code."},
            {"t": "code", "run": False, "caption": "Illustrative: concurrent real API calls",
             "code": '''\
import asyncio
from some_llm_sdk import AsyncClient      # e.g. an async Anthropic/OpenAI client

client = AsyncClient()
sem = asyncio.Semaphore(5)               # honour the provider's rate limit

async def ask(prompt):
    async with sem:
        resp = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

async def main(prompts):
    return await asyncio.gather(*(ask(p) for p in prompts))

answers = asyncio.run(main(["Summarise X", "Translate Y", "Classify Z"]))
'''},
            {"t": "note", "text":
                "Why it matters for AI: batch inference, agent loops and "
                "retrieval pipelines all issue many network calls whose cost is "
                "latency, not CPU. `asyncio.gather` for concurrency plus a "
                "`Semaphore` for rate limiting is the backbone of fast, "
                "well-behaved LLM clients."},
        ],
    },
]
