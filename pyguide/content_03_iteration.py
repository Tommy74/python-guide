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
            {"t": "note", "text":
                "Why it matters for AI: lazy pipelines are how you preprocess "
                "corpora larger than memory. Use generator expressions to avoid "
                "materialising intermediates, compose stages as generators, and "
                "use `yield from` to flatten shards, keeping peak memory to a "
                "single record plus a batch."},
        ],
    },
]
