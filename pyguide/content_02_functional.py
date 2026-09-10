"""Part II - Functions and Functional Programming."""

PART = "Functions and Functional Programming"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Functions: Arguments, Defaults and the Mutable-Default Trap",
        "blocks": [
            {"t": "p", "text":
                "Functions are the unit of reuse in every AI codebase, from a "
                "one-line loss to a full training step. Python's calling "
                "conventions are flexible: positional and keyword arguments, "
                "defaults, `*args`, `**kwargs`, and keyword-only parameters. "
                "Knowing them well is what lets you pass a whole config into a "
                "model with `model(**config)`."},
            {"t": "h", "text": "Positional, keyword and default arguments"},
            {"t": "p", "text":
                "Arguments can be given by position or by name. Defaults make "
                "parameters optional, which is how hyperparameters get sensible "
                "fallbacks."},
            {"t": "code", "run": True, "caption": "Defaults and keyword calls",
             "code": '''\
def train_step(lr=1e-3, momentum=0.9, clip=None):
    msg = f"lr={lr}, momentum={momentum}, clip={clip}"
    return msg

print(train_step())                 # all defaults
print(train_step(0.01))             # positional lr
print(train_step(clip=1.0, lr=0.1)) # keyword, any order
'''},
            {"t": "h", "text": "*args and **kwargs"},
            {"t": "p", "text":
                "`*args` collects extra positional arguments into a tuple; "
                "`**kwargs` collects extra keyword arguments into a dict. "
                "Frameworks use them everywhere to forward options down a stack "
                "of layers without listing every parameter."},
            {"t": "code", "run": True, "caption": "Collecting variadic arguments",
             "code": '''\
def build_layer(name, *shape, **options):
    print("name  :", name)
    print("shape :", shape)      # a tuple
    print("opts  :", options)    # a dict

build_layer("dense", 784, 128, activation="relu", bias=False)
'''},
            {"t": "h", "text": "Keyword-only arguments"},
            {"t": "p", "text":
                "Anything after a bare `*` in the signature must be passed by "
                "name. This prevents accidentally swapping two positional flags "
                "and makes call sites self-documenting."},
            {"t": "code", "run": True, "caption": "Forcing callers to be explicit",
             "code": '''\
def sample(logits, *, temperature=1.0, greedy=False):
    return f"temp={temperature}, greedy={greedy}"

print(sample([0.1, 0.9], temperature=0.7))
try:
    sample([0.1, 0.9], 0.7)          # positional -> error
except TypeError as e:
    print("TypeError:", e)
'''},
            {"t": "h", "text": "The mutable-default trap"},
            {"t": "p", "text":
                "A default value is evaluated **once**, when the function is "
                "defined, not on every call. A mutable default like `[]` is "
                "therefore *shared* across calls, a bug that silently corrupts "
                "accumulated data. The fix is the `None`-sentinel pattern."},
            {"t": "code", "run": True, "caption": "The bug and the fix",
             "code": '''\
def buggy(x, acc=[]):            # DON'T: shared list
    acc.append(x)
    return acc

print("buggy:", buggy(1), buggy(2), buggy(3))

def fixed(x, acc=None):          # DO: fresh list each call
    if acc is None:
        acc = []
    acc.append(x)
    return acc

print("fixed:", fixed(1), fixed(2), fixed(3))
'''},
            {"t": "h", "text": "Unpacking arguments at the call site"},
            {"t": "p", "text":
                "The mirror image of `*args`/`**kwargs`: use `*` to spread a "
                "sequence into positional arguments and `**` to spread a dict "
                "into keyword arguments. This is the idiom behind `model(**config)`."},
            {"t": "code", "run": True, "caption": "Spreading a config dict",
             "code": '''\
def make_optimizer(lr, weight_decay, betas):
    return f"Adam(lr={lr}, wd={weight_decay}, betas={betas})"

config = {"lr": 3e-4, "weight_decay": 0.01, "betas": (0.9, 0.999)}
print(make_optimizer(**config))     # dict -> keyword args

pair = (0.9, 0.999)
print("betas unpacked:", *pair)     # tuple -> positional args
'''},
            {"t": "note", "text":
                "Why it matters for AI: config-driven training relies on "
                "`**config` unpacking, and the mutable-default trap is a real "
                "source of data-leak bugs (e.g. a metric list that never "
                "resets between runs). Prefer `None` sentinels for any "
                "list/dict/set default."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Lambdas, map, filter and sorted(key=)",
        "blocks": [
            {"t": "p", "text":
                "A `lambda` is a small anonymous function, most useful as the "
                "`key=` argument to `sorted`, `min` and `max`. Combined with "
                "`map` and `filter`, lambdas express data transforms compactly, "
                "though comprehensions are often clearer."},
            {"t": "h", "text": "lambda basics"},
            {"t": "code", "run": True, "caption": "Anonymous one-liners",
             "code": '''\
square = lambda x: x * x
scale = lambda x, s=2.0: x * s
print(square(5), scale(3), scale(3, 10))
'''},
            {"t": "h", "text": "map and filter (and why comprehensions win)"},
            {"t": "p", "text":
                "`map` applies a function to every item; `filter` keeps items "
                "where a predicate is true. Both return lazy iterators. For "
                "readability, a list/generator comprehension usually expresses "
                "the same thing more clearly."},
            {"t": "code", "run": True, "caption": "map/filter vs comprehension",
             "code": '''\
nums = [-2, -1, 0, 1, 2, 3]

# functional style
sq = list(map(lambda x: x * x, filter(lambda x: x > 0, nums)))
print("map/filter :", sq)

# comprehension style (usually preferred)
sq2 = [x * x for x in nums if x > 0]
print("comprehension:", sq2)
'''},
            {"t": "h", "text": "sorted(key=): the workhorse"},
            {"t": "p", "text":
                "`key=` maps each element to a sort key. Sorting samples by "
                "sequence length (to build efficient batches) or ranking "
                "candidates by score are everyday uses."},
            {"t": "code", "run": True, "caption": "Sort samples by length",
             "code": '''\
batch = ["hi", "hello there", "hey", "good morning all"]
by_len = sorted(batch, key=len)
print("shortest first:", by_len)
print("longest first :", sorted(batch, key=len, reverse=True))
'''},
            {"t": "h", "text": "Top-k with sorted, min and max"},
            {"t": "p", "text":
                "`max`/`min` also take `key=`. Selecting the best prediction "
                "or the top-k scoring candidates is a one-liner."},
            {"t": "code", "run": True, "caption": "Best and top-k candidates",
             "code": '''\
candidates = [("cat", 0.12), ("dog", 0.71), ("fox", 0.55), ("owl", 0.33)]

best = max(candidates, key=lambda c: c[1])
print("argmax:", best)

top2 = sorted(candidates, key=lambda c: c[1], reverse=True)[:2]
print("top-2 :", top2)
'''},
            {"t": "note", "text":
                "Why it matters for AI: `sorted(..., key=...)` and "
                "`max(..., key=...)` implement argmax, ranking, top-k selection "
                "and length-bucketing without any library. Reach for a "
                "comprehension over `map`/`filter` unless you already have a "
                "named function to pass."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "functools: partial, lru_cache, reduce, wraps",
        "blocks": [
            {"t": "p", "text":
                "The `functools` module contains higher-order tools that show "
                "up constantly: freezing arguments with `partial`, memoizing "
                "with `lru_cache`, folding sequences with `reduce`, and "
                "preserving metadata with `wraps`."},
            {"t": "h", "text": "partial: freezing hyperparameters"},
            {"t": "p", "text":
                "`functools.partial` returns a new callable with some arguments "
                "already filled in. It is the clean way to preconfigure an "
                "optimizer, activation or metric before passing it somewhere "
                "that will call it with the remaining arguments."},
            {"t": "code", "run": True, "caption": "Pre-binding arguments",
             "code": '''\
from functools import partial

def scaled_dropout(x, rate, training):
    return f"x={x}, rate={rate}, training={training}"

# freeze rate=0.1; callers only pass x and training
dropout = partial(scaled_dropout, rate=0.1)
print(dropout(1.0, training=True))
print(dropout(2.0, training=False))
'''},
            {"t": "h", "text": "lru_cache / cache: memoization"},
            {"t": "p", "text":
                "`@lru_cache` stores results keyed by arguments, so repeated "
                "calls with the same inputs return instantly. Great for "
                "expensive pure functions like tokenizer lookups or recursive "
                "computations. `@cache` is an unbounded shortcut (Python 3.9+)."},
            {"t": "code", "run": True, "caption": "Cache expensive pure calls",
             "code": '''\
from functools import lru_cache

calls = {"n": 0}

@lru_cache(maxsize=None)
def fib(n):
    calls["n"] += 1
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print("fib(30) =", fib(30))
print("actual calls:", calls["n"])   # linear, not exponential
print("cache info :", fib.cache_info())
'''},
            {"t": "h", "text": "reduce: folding a sequence"},
            {"t": "p", "text":
                "`functools.reduce` collapses a sequence to a single value by "
                "repeatedly applying a two-argument function. Useful for "
                "chaining transforms or multiplying a shape into a flat size."},
            {"t": "code", "run": True, "caption": "Product of a tensor shape",
             "code": '''\
from functools import reduce

shape = (32, 3, 224, 224)               # (batch, channels, H, W)
num_elements = reduce(lambda a, b: a * b, shape)
print("elements per batch tensor:", num_elements)
'''},
            {"t": "h", "text": "wraps: honest decorators"},
            {"t": "p", "text":
                "When you write a decorator (see the Decorators chapter), the "
                "wrapper replaces the original function and loses its name and "
                "docstring. `functools.wraps` copies that metadata back, which "
                "matters for debugging and introspection."},
            {"t": "code", "run": True, "caption": "Preserving __name__ and __doc__",
             "code": '''\
from functools import wraps

def logged(fn):
    @wraps(fn)                       # copy metadata from fn
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

@logged
def predict(x):
    "Run the model forward pass."
    return x

print("name:", predict.__name__)     # 'predict', not 'wrapper'
print("doc :", predict.__doc__)
'''},
            {"t": "note", "text":
                "Why it matters for AI: `partial` preconfigures optimizers and "
                "transforms in data pipelines, `lru_cache` avoids recomputing "
                "tokenization or feature lookups, and `wraps` keeps your "
                "training-loop decorators debuggable."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "itertools: Batching and Combining Iterables",
        "blocks": [
            {"t": "p", "text":
                "`itertools` provides memory-efficient building blocks for "
                "iteration. They operate lazily on streams, so you can process "
                "datasets far larger than RAM. The stars for AI work are "
                "`chain`, `islice`, `product`, `combinations` and `batched`."},
            {"t": "h", "text": "chain: concatenate iterables"},
            {"t": "p", "text":
                "`chain` treats several iterables as one continuous stream, for "
                "example merging train shards without building a giant list."},
            {"t": "code", "run": True, "caption": "Merge dataset shards lazily",
             "code": '''\
from itertools import chain

shard_a = ["a1", "a2"]
shard_b = ["b1", "b2", "b3"]
for i, item in enumerate(chain(shard_a, shard_b)):
    print(i, item)
'''},
            {"t": "h", "text": "islice and count: windows into infinite streams"},
            {"t": "p", "text":
                "`count` is an infinite counter and `islice` takes a finite "
                "slice from any iterator, so you can peek at the first N items "
                "of an endless generator without materialising it."},
            {"t": "code", "run": True, "caption": "Take the first N of a stream",
             "code": '''\
from itertools import count, islice

steps = count(start=0, step=10)          # 0, 10, 20, ... forever
first5 = list(islice(steps, 5))
print("first 5 lr-warmup steps:", first5)
'''},
            {"t": "h", "text": "product: hyperparameter grids"},
            {"t": "p", "text":
                "`product` is the Cartesian product, i.e. nested loops without "
                "the nesting. It is the natural way to enumerate a grid search "
                "over hyperparameters."},
            {"t": "code", "run": True, "caption": "Grid search over hyperparameters",
             "code": '''\
from itertools import product

lrs = [1e-2, 1e-3]
batch_sizes = [16, 32]
for lr, bs in product(lrs, batch_sizes):
    print(f"run: lr={lr}, batch_size={bs}")
'''},
            {"t": "h", "text": "combinations: pairs without repetition"},
            {"t": "p", "text":
                "`combinations` yields unordered subsets, handy for building "
                "positive/negative pairs in contrastive learning or comparing "
                "every model against every other."},
            {"t": "code", "run": True, "caption": "All unique model pairs",
             "code": '''\
from itertools import combinations

models = ["A", "B", "C"]
for m1, m2 in combinations(models, 2):
    print(f"compare {m1} vs {m2}")
'''},
            {"t": "h", "text": "batched: grouping a stream into mini-batches"},
            {"t": "p", "text":
                "Since Python 3.12, `itertools.batched` chops an iterable into "
                "tuples of a fixed size, which is exactly what a data loader "
                "does to turn a stream of samples into mini-batches. The final "
                "batch may be shorter."},
            {"t": "code", "run": True, "caption": "Stream -> mini-batches",
             "code": '''\
from itertools import batched

samples = range(10)                      # pretend: 10 samples
for step, mb in enumerate(batched(samples, 4)):
    print(f"step {step}: batch={mb} (size {len(mb)})")
'''},
            {"t": "note", "text":
                "Why it matters for AI: itertools is the pure-Python toolkit "
                "for data pipelines. `chain` merges shards, `islice` bounds "
                "infinite generators, `product` enumerates hyperparameter "
                "grids, and `batched` builds mini-batches, all lazily and "
                "without loading everything into memory."},
        ],
    },
]
