"""Part III - Iteration, Generators and Data Pipelines."""

PART = "Iteration, Generators and Data Pipelines"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Iterables, Iterators and the Iterator Protocol",
        "blocks": [
            {"t": "p", "text":
                "Every `for` loop in Python, including the one that walks your "
                "training data, is built on the *iterator protocol*. "
                "Understanding it lets you write custom data loaders that plug "
                "straight into `for batch in loader:` syntax."},
            {"t": "h", "text": "iter() and next()"},
            {"t": "p", "text":
                "`iter(obj)` asks an *iterable* for an *iterator*; `next(it)` "
                "pulls the next value and raises `StopIteration` when it is "
                "exhausted. A `for` loop is just this in disguise."},
            {"t": "code", "run": True, "caption": "Driving an iterator by hand",
             "code": '''\
data = [10, 20, 30]
it = iter(data)
print(next(it))
print(next(it))
print(next(it))
try:
    next(it)
except StopIteration:
    print("StopIteration: stream exhausted")
'''},
            {"t": "h", "text": "How a for-loop really works"},
            {"t": "p", "text":
                "A `for` loop calls `iter()` once, then `next()` repeatedly, "
                "catching `StopIteration` to stop. This desugaring is worth "
                "seeing once so iterators feel less magical."},
            {"t": "code", "run": True, "caption": "for-loop desugared",
             "code": '''\
def manual_for(iterable, body):
    it = iter(iterable)
    while True:
        try:
            item = next(it)
        except StopIteration:
            break
        body(item)

manual_for(["a", "b", "c"], lambda x: print("got", x))
'''},
            {"t": "h", "text": "__iter__ and __next__: a custom iterator"},
            {"t": "p", "text":
                "An object is iterable if it defines `__iter__`; it is an "
                "iterator if it also defines `__next__`. Here is a mini data "
                "loader that yields shuffled batch *indices*, the same shape as "
                "a real `DataLoader`."},
            {"t": "code", "run": True, "caption": "A minimal index DataLoader",
             "code": '''\
import random

class BatchSampler:
    def __init__(self, n, batch_size, seed=0):
        self.idx = list(range(n))
        random.Random(seed).shuffle(self.idx)
        self.batch_size = batch_size

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        if self.pos >= len(self.idx):
            raise StopIteration
        batch = self.idx[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return batch

for batch in BatchSampler(7, batch_size=3):
    print("batch indices:", batch)
'''},
            {"t": "h", "text": "Iterable vs iterator: not the same thing"},
            {"t": "p", "text":
                "The distinction trips people up. An *iterable* can produce a "
                "fresh iterator every time you call `iter()` on it, so a `list` "
                "supports many independent passes. An *iterator* is a one-shot "
                "cursor: calling `iter()` on it returns *itself*, so two names "
                "share the same position and one pass exhausts it."},
            {"t": "code", "run": True, "caption": "One is reusable, one is not",
             "code": '''\
data = [1, 2, 3]
a = iter(data)
b = iter(data)                       # independent cursor
print("independent iterators:", a is b)
print("first of a:", next(a), "first of b:", next(b))

it = iter(data)
print("iter(iterator) is itself:", iter(it) is it)
'''},
            {"t": "h", "text": "iter(callable, sentinel): read until a marker"},
            {"t": "p", "text":
                "The two-argument form of `iter()` calls a zero-argument "
                "*callable* repeatedly and stops when the result equals the "
                "*sentinel*. It is the clean way to drain a source, such as "
                "fixed-size chunks from a file, until an end marker appears, "
                "with no `while True` / `break` boilerplate."},
            {"t": "code", "run": True, "caption": "Chunk a stream until sentinel",
             "code": '''\
import io

buffer = io.StringIO("abcdefghij")
# read 4 chars at a time; "" (EOF) is the sentinel that stops iteration
read_chunk = lambda: buffer.read(4)
for chunk in iter(read_chunk, ""):
    print("chunk:", chunk)
'''},
            {"t": "note", "text":
                "Why it matters for AI: PyTorch `Dataset`/`DataLoader`, Hugging "
                "Face streaming datasets and tf.data all speak the iterator "
                "protocol. Implement `__iter__`/`__next__` and your data source "
                "works with `for batch in loader:` and `enumerate` for free."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Generators and yield: Streaming Datasets and Mini-Batches",
        "blocks": [
            {"t": "p", "text":
                "A *generator* is the easiest way to build an iterator: write a "
                "normal function but use `yield` instead of `return`. Execution "
                "pauses at each `yield` and resumes on the next request, so "
                "values are produced lazily, one at a time, using almost no "
                "memory."},
            {"t": "h", "text": "yield basics"},
            {"t": "p", "text":
                "Calling a generator function does not run it; it returns a "
                "generator object. Each `next()` runs until the following "
                "`yield`."},
            {"t": "code", "run": True, "caption": "A generator counts lazily",
             "code": '''\
def warmup_lrs(base, steps):
    for s in range(1, steps + 1):
        yield base * s / steps       # produced on demand

gen = warmup_lrs(1e-3, 4)
print("type:", type(gen).__name__)
for lr in gen:
    print(f"lr = {lr:.4e}")
'''},
            {"t": "h", "text": "Laziness = memory efficiency"},
            {"t": "p", "text":
                "A generator never holds the whole sequence in memory. This lets "
                "you stream a dataset of millions of rows and process it one "
                "record at a time, which a list of the same data could not fit."},
            {"t": "code", "run": True, "caption": "Stream a huge dataset lazily",
             "code": '''\
import sys

def read_huge_dataset(n):
    for i in range(n):               # pretend each line is read from disk
        yield f"sample_{i}"

stream = read_huge_dataset(10_000_000)
# only the generator object exists, not 10M strings
print("generator size (bytes):", sys.getsizeof(stream))
# consume just the first three without materialising the rest
for _ in range(3):
    print(next(stream))
'''},
            {"t": "h", "text": "A mini-batch generator"},
            {"t": "p", "text":
                "Generators shine as batchers: pull items from an upstream "
                "iterable and `yield` fixed-size lists. Because it is lazy, it "
                "works on infinite or on-disk streams too."},
            {"t": "code", "run": True, "caption": "Group a stream into batches",
             "code": '''\
def batches(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:                        # yield the short final batch
        yield batch

for b in batches(range(7), 3):
    print("batch:", b)
'''},
            {"t": "h", "text": "Generators are single-use"},
            {"t": "p", "text":
                "A generator carries its own execution state and is exhausted "
                "after one full pass. To iterate again (a new epoch) you must "
                "create a fresh generator, a common source of silent empty "
                "second epochs."},
            {"t": "code", "run": True, "caption": "Exhaustion after one pass",
             "code": '''\
def gen():
    yield from range(3)

g = gen()
print("epoch 1:", list(g))
print("epoch 2:", list(g))          # empty: already exhausted
'''},
            {"t": "h", "text": "Two-way generators with .send()"},
            {"t": "p", "text":
                "`yield` is also an expression: `x = yield value` sends `value` "
                "out and, on the next `.send(v)`, resumes with `x == v`. This "
                "turns a generator into a tiny stateful coroutine, ideal for a "
                "running metric that updates as batches arrive without a class."},
            {"t": "code", "run": True, "caption": "A running-average generator",
             "code": '''\
def running_average():
    total, count = 0.0, 0
    avg = None
    while True:
        x = yield avg                # emit current avg, wait for next value
        total += x
        count += 1
        avg = total / count

loss = running_average()
next(loss)                           # prime: advance to the first yield
for batch_loss in [1.0, 0.6, 0.2, 0.4]:
    print(f"loss={batch_loss:.1f}  avg={loss.send(batch_loss):.3f}")
'''},
            {"t": "h", "text": ".throw() and .close(): the generator lifecycle"},
            {"t": "p", "text":
                "Beyond `.send()`, `.throw(exc)` raises an exception *inside* the "
                "generator at the paused `yield` (letting it clean up or skip a "
                "bad record), and `.close()` raises `GeneratorExit` so `finally` "
                "blocks run, releasing files or sockets. This is the whole "
                "generator lifecycle."},
            {"t": "code", "run": True, "caption": "throw, close and finally",
             "code": '''\
def worker():
    try:
        while True:
            item = yield
            print("processed", item)
    except ValueError as e:
        print("recovered from:", e)
    finally:
        print("cleanup: releasing resources")

w = worker()
next(w)                              # prime
w.send("a")
try:
    w.throw(ValueError("bad record"))   # caught inside the generator
except StopIteration:
    print("generator finished after throw")
w.close()                           # no-op: already closed
'''},
            {"t": "h", "text": "Infinite streams tamed with islice"},
            {"t": "p", "text":
                "A generator can loop forever; `itertools.islice` slices a lazy "
                "stream without materialising it, giving you exactly *k* items. "
                "This is the standard way to take a bounded sample from an "
                "endless source such as an augmentation pipeline."},
            {"t": "code", "run": True, "caption": "islice a never-ending stream",
             "code": '''\
import itertools

def augment_forever(seed):
    i = 0
    while True:                      # never stops on its own
        yield f"{seed}_aug{i}"
        i += 1

first_five = list(itertools.islice(augment_forever("img"), 5))
print(first_five)
'''},
            {"t": "h", "text": "An epoch loader that reshuffles each pass"},
            {"t": "p", "text":
                "Because a generator is single-use, wrap the logic in a function "
                "you call once per epoch. Here each call reshuffles and yields "
                "fresh mini-batches, exactly how a training loop requests a new "
                "iterator every epoch."},
            {"t": "code", "run": True, "caption": "One fresh generator per epoch",
             "code": '''\
import random

def epoch_loader(data, batch_size, seed):
    order = list(data)
    random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield order[i:i + batch_size]

data = list(range(6))
for epoch in range(2):
    batches = epoch_loader(data, batch_size=2, seed=epoch)
    print(f"epoch {epoch}:", list(batches))
'''},
            {"t": "note", "text":
                "Why it matters for AI: streaming data loaders, tokenizing "
                "corpora that do not fit in RAM, and yielding mini-batches are "
                "all generator patterns. Remember that a generator is "
                "single-use, so wrap it in a function you call once per epoch."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Generator Expressions, yield from and Lazy Pipelines",
        "blocks": [
            {"t": "p", "text":
                "Generator expressions give you a generator with comprehension "
                "syntax, and `yield from` delegates to a sub-generator. "
                "Together they let you assemble multi-stage data pipelines that "
                "stay lazy from end to end."},
            {"t": "h", "text": "Generator expression vs list comprehension"},
            {"t": "p", "text":
                "Swap the `[]` of a list comprehension for `()` and you get a "
                "generator expression: same syntax, but it computes items on "
                "demand instead of building the whole list. For large data the "
                "memory difference is enormous."},
            {"t": "code", "run": True, "caption": "() is lazy, [] is eager",
             "code": '''\
import sys

n = 1_000_000
listcomp = [x * x for x in range(n)]     # builds 1M ints now
genexpr = (x * x for x in range(n))      # builds nothing yet

print("list bytes:", sys.getsizeof(listcomp))
print("gen  bytes:", sys.getsizeof(genexpr))
# feed a genexpr straight into a reducer, no intermediate list
print("sum of squares:", sum(x * x for x in range(n)))
'''},
            {"t": "h", "text": "Chaining generators into a pipeline"},
            {"t": "p", "text":
                "Because each stage consumes and produces a lazy stream, you "
                "can wire *read -> clean -> tokenize -> batch* together. Data "
                "flows one record at a time, so peak memory is just one record "
                "plus one batch."},
            {"t": "code", "run": True, "caption": "read -> clean -> tokenize -> batch",
             "code": '''\
def read(lines):
    for line in lines:
        yield line

def clean(stream):
    for line in stream:
        yield line.strip().lower()

def tokenize(stream):
    for line in stream:
        yield line.split()

def batch(stream, size):
    buf = []
    for item in stream:
        buf.append(item)
        if len(buf) == size:
            yield buf; buf = []
    if buf:
        yield buf

raw = ["  Hello WORLD ", "Deep  Learning", " GPUs go BRRR "]
pipeline = batch(tokenize(clean(read(raw))), size=2)
for b in pipeline:
    print(b)
'''},
            {"t": "h", "text": "yield from: delegation and flattening"},
            {"t": "p", "text":
                "`yield from sub` yields every item of `sub` in turn, "
                "forwarding cleanly without a manual loop. It is the natural "
                "way to flatten nested shards into one stream."},
            {"t": "code", "run": True, "caption": "Flatten shards with yield from",
             "code": '''\
def flatten(shards):
    for shard in shards:
        yield from shard             # splice each shard's items in

shards = [["a", "b"], ["c"], ["d", "e", "f"]]
print("flattened:", list(flatten(shards)))
'''},
            {"t": "h", "text": "Recursive yield from for nested structure"},
            {"t": "p", "text":
                "Because `yield from` accepts any iterable, including another "
                "call to the same generator, it flattens arbitrarily nested "
                "structures in a few lines, handy for ragged config or nested "
                "shard directories."},
            {"t": "code", "run": True, "caption": "Deep-flatten a nested list",
             "code": '''\
def deep_flatten(obj):
    for item in obj:
        if isinstance(item, list):
            yield from deep_flatten(item)   # recurse
        else:
            yield item

nested = [1, [2, [3, 4], 5], [[6]], 7]
print("deep flat:", list(deep_flatten(nested)))
'''},
            {"t": "h", "text": "Memory: a full pipeline stays flat"},
            {"t": "p", "text":
                "The payoff of composing generators is that peak memory does "
                "not grow with the dataset. Below, the lazy pipeline processes a "
                "large stream while holding only a running total; the eager list "
                "version allocates every intermediate at once."},
            {"t": "code", "run": True, "caption": "Lazy vs eager peak memory",
             "code": '''\
import sys

n = 1_000_000
# eager: materialise all squares, then all evens, then sum
eager = [x for x in [v * v for v in range(n)] if x % 2 == 0]
print("eager list bytes:", sys.getsizeof(eager))

# lazy: one value in flight at a time, nothing materialised
lazy = (x for x in (v * v for v in range(n)) if x % 2 == 0)
print("lazy gen bytes:  ", sys.getsizeof(lazy))
print("both sums equal:", sum(eager) == sum(lazy))
'''},
            {"t": "h", "text": "The same idea in the frameworks"},
            {"t": "p", "text":
                "These generator patterns are exactly what production data APIs "
                "expose. A `torch` `IterableDataset` implements `__iter__` as a "
                "generator, and `tf.data.Dataset.from_generator` wraps one "
                "directly; both then add batching, shuffling and prefetching."},
            {"t": "code", "run": False, "caption": "Illustrative framework wrappers",
             "code": '''\
# Illustrative only (frameworks not required to run this book).
from torch.utils.data import IterableDataset

class Stream(IterableDataset):
    def __iter__(self):
        yield from clean(read(open("corpus.txt")))

# import tensorflow as tf
# ds = tf.data.Dataset.from_generator(
#     lambda: clean(read(open("corpus.txt"))),
#     output_types=tf.string,
# ).batch(32).prefetch(1)
'''},
            {"t": "note", "text":
                "Why it matters for AI: lazy pipelines are how you preprocess "
                "corpora larger than memory. Use generator expressions to avoid "
                "materialising intermediates, compose stages as generators, and "
                "use `yield from` to flatten shards, keeping peak memory to a "
                "single record plus a batch."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Building a Streaming Data Pipeline End to End",
        "blocks": [
            {"t": "p", "text":
                "This chapter combines the pieces into one small but realistic "
                "pipeline: a resource-safe reader, lazy cleaning and tokenizing "
                "stages, batching, and a live metric fed with `.send()`. Every "
                "stage is a generator, so the whole thing streams and stays "
                "flat in memory."},
            {"t": "h", "text": "contextlib for resource-safe iteration"},
            {"t": "p", "text":
                "A data source usually owns a resource (a file, socket or DB "
                "cursor) that must close even if iteration stops early or "
                "raises. `contextlib.contextmanager` lets you write that "
                "open/yield/close guarantee as a generator, and the `finally` "
                "block always runs."},
            {"t": "code", "run": True, "caption": "A generator-based context manager",
             "code": '''\
from contextlib import contextmanager

@contextmanager
def open_shard(name, rows):
    print("open", name)
    try:
        yield iter(rows)             # hand the caller a lazy cursor
    finally:
        print("close", name)         # runs even on early break / error

with open_shard("shard-0", ["  A ", "b"]) as cur:
    for row in cur:
        print("row:", row.strip())
'''},
            {"t": "h", "text": "The full worked pipeline"},
            {"t": "p", "text":
                "Now we wire the stages together. `read` streams rows from the "
                "managed shard, `clean` and `tokenize` transform lazily, "
                "`batch` groups, and a `running_average` coroutine tracks mean "
                "tokens-per-batch as batches flow past, no list of the corpus "
                "ever exists."},
            {"t": "code", "run": True, "caption": "Read to metric, fully lazy",
             "code": '''\
from contextlib import contextmanager

@contextmanager
def shard(rows):
    try:
        yield iter(rows)
    finally:
        print("shard closed")

def clean(stream):
    for line in stream:
        s = line.strip().lower()
        if s:                        # drop blank lines lazily
            yield s

def tokenize(stream):
    for line in stream:
        yield line.split()

def batch(stream, size):
    buf = []
    for item in stream:
        buf.append(item)
        if len(buf) == size:
            yield buf; buf = []
    if buf:
        yield buf

def running_average():
    total, count, avg = 0, 0, None
    while True:
        x = yield avg
        total += x; count += 1
        avg = total / count

rows = ["  Hello WORLD ", "", "Deep  Learning here", " GPUs go BRRR "]
mean_tokens = running_average()
next(mean_tokens)                    # prime the metric

with shard(rows) as src:
    pipeline = batch(tokenize(clean(src)), size=2)
    for i, b in enumerate(pipeline):
        n_tokens = sum(len(sent) for sent in b)
        avg = mean_tokens.send(n_tokens)
        print(f"batch {i}: {b}  avg_tokens={avg:.1f}")
'''},
            {"t": "note", "text":
                "Why it matters for AI: real ingestion code looks exactly like "
                "this: resource-safe readers, composed lazy transforms, "
                "batching, and streaming metrics. Master the generator toolkit "
                "and you can preprocess corpora far larger than RAM while "
                "guaranteeing every file handle is released."},
        ],
    },
]
