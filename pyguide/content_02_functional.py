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
            {"t": "h", "text": "Returning multiple values"},
            {"t": "p", "text":
                "A function returns one object, but that object can be a tuple, "
                "which Python packs and unpacks automatically. This is how a "
                "training step hands back both loss and accuracy, or how a data "
                "split returns train and test sets at once."},
            {"t": "code", "run": True, "caption": "Tuple returns and unpacking",
             "code": '''\
def evaluate(preds, labels):
    correct = sum(p == y for p, y in zip(preds, labels))
    acc = correct / len(labels)
    loss = 1.0 - acc                # toy loss
    return loss, acc                # packed into a tuple

loss, acc = evaluate([1, 0, 1, 1], [1, 0, 0, 1])
print(f"loss={loss:.2f}, acc={acc:.2f}")
'''},
            {"t": "h", "text": "First-class functions"},
            {"t": "p", "text":
                "In Python functions are ordinary objects: you can store them in "
                "variables, put them in a dict, and pass them as arguments. This "
                "is what makes a registry of activations or a table of metrics "
                "possible, and it underpins `map`, `sorted(key=)` and "
                "decorators."},
            {"t": "code", "run": True, "caption": "Functions as values in a registry",
             "code": '''\
def relu(x):
    return max(0.0, x)

def leaky(x):
    return x if x > 0 else 0.01 * x

ACTIVATIONS = {"relu": relu, "leaky": leaky}   # a dispatch table

def apply(name, x):
    fn = ACTIVATIONS[name]          # look up a function by name
    return fn(x)

print("relu(-3) =", apply("relu", -3.0))
print("leaky(-3):", apply("leaky", -3.0))
'''},
            {"t": "h", "text": "Closures: functions that remember state"},
            {"t": "p", "text":
                "A function defined inside another function *captures* the "
                "enclosing variables, forming a **closure**. The inner function "
                "keeps a live reference to that state after the outer function "
                "returns. A learning-rate scheduler factory is the classic "
                "example: it captures the base rate and decay, and each call "
                "advances a private step counter via `nonlocal`."},
            {"t": "code", "run": True, "caption": "A learning-rate scheduler factory",
             "code": '''\
def make_scheduler(base_lr, decay=0.5, every=2):
    step = 0                         # captured, private state
    def next_lr():
        nonlocal step
        lr = base_lr * (decay ** (step // every))
        step += 1
        return lr
    return next_lr                   # a closure over base_lr/step

sched = make_scheduler(1.0, decay=0.5, every=2)
print([round(sched(), 3) for _ in range(6)])
'''},
            {"t": "h", "text": "Positional-only parameters with /"},
            {"t": "p", "text":
                "Parameters before a `/` in the signature can only be passed by "
                "position, never by name. This lets you rename them freely "
                "later, and mirrors many C-implemented builtins. It pairs with "
                "the keyword-only `*` to give precise control over how a "
                "function is called."},
            {"t": "code", "run": True, "caption": "Locking a parameter to position",
             "code": '''\
def scale(x, /, factor=2.0):
    return x * factor               # x is positional-only

print("by position:", scale(10))
print("keyword factor:", scale(10, factor=3.0))
try:
    scale(x=10)                     # naming x -> error
except TypeError as e:
    print("TypeError:", e)
'''},
            {"t": "note", "text":
                "Why it matters for AI: config-driven training relies on "
                "`**config` unpacking, and the mutable-default trap is a real "
                "source of data-leak bugs (e.g. a metric list that never "
                "resets between runs). Prefer `None` sentinels for any "
                "list/dict/set default. Closures capture scheduler and "
                "counter state cleanly, and first-class functions let you "
                "build registries of layers, losses and metrics keyed by name."},
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
            {"t": "h", "text": "operator: faster, clearer sort keys"},
            {"t": "p", "text":
                "The `operator` module provides ready-made key functions: "
                "`itemgetter(i)` pulls an element by index or key, "
                "`attrgetter('x')` pulls an attribute, and `methodcaller('m')` "
                "calls a method. They read better than a `lambda` and, being "
                "written in C, run faster on large datasets, exactly the sort "
                "keys you want when ordering millions of samples."},
            {"t": "code", "run": True, "caption": "itemgetter vs lambda",
             "code": '''\
from operator import itemgetter

candidates = [("cat", 0.12), ("dog", 0.71), ("fox", 0.55)]
by_score = sorted(candidates, key=itemgetter(1), reverse=True)
print("by score:", by_score)

# clearer and faster than: key=lambda c: c[1]
best = max(candidates, key=itemgetter(1))
print("argmax  :", best)
'''},
            {"t": "h", "text": "Multi-key sort with itemgetter"},
            {"t": "p", "text":
                "Pass several indices to `itemgetter` to sort by a tuple of "
                "keys: `itemgetter(1, 0)` sorts by field 1, breaking ties with "
                "field 0. Python's sort is **stable**, meaning equal keys keep "
                "their original order, so you can also get a secondary sort by "
                "sorting twice, least-significant key first."},
            {"t": "code", "run": True, "caption": "Sort by score, then name",
             "code": '''\
from operator import itemgetter

rows = [("dog", 2), ("cat", 1), ("ant", 2), ("bee", 1)]

# single pass: primary count, secondary name (both ascending)
combined = sorted(rows, key=itemgetter(1, 0))
print("count then name:", combined)

# two-pass via stable sort: sort by name, then by count
tmp = sorted(rows, key=itemgetter(0))          # secondary key first
stable = sorted(tmp, key=itemgetter(1))        # primary key last
print("stable two-pass:", stable)
'''},
            {"t": "h", "text": "attrgetter and methodcaller"},
            {"t": "p", "text":
                "`attrgetter` sorts objects by an attribute, common when your "
                "samples are dataclasses. `methodcaller('m', *args)` builds a "
                "key that calls a method on each element, for example sorting "
                "strings case-insensitively via `str.lower`."},
            {"t": "code", "run": True, "caption": "Sort objects and by method",
             "code": '''\
from operator import attrgetter, methodcaller
from dataclasses import dataclass

@dataclass
class Sample:
    text: str
    length: int

data = [Sample("BB", 2), Sample("a", 1), Sample("CCC", 3)]
print("by length:", [s.text for s in sorted(data, key=attrgetter("length"))])

words = ["banana", "Apple", "cherry"]
print("caseless :", sorted(words, key=methodcaller("lower")))
'''},
            {"t": "note", "text":
                "Why it matters for AI: `sorted(..., key=...)` and "
                "`max(..., key=...)` implement argmax, ranking, top-k selection "
                "and length-bucketing without any library. Reach for a "
                "comprehension over `map`/`filter` unless you already have a "
                "named function to pass, and prefer `operator.itemgetter`/"
                "`attrgetter` over `lambda` for sort keys, they are clearer "
                "and faster on large datasets."},
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
            {"t": "h", "text": "reduce with an initializer"},
            {"t": "p", "text":
                "A third argument to `reduce` is the starting accumulator. It "
                "fixes the result type and, crucially, makes `reduce` safe on "
                "an empty sequence (without it, an empty input raises). Here we "
                "compose a list of preprocessing steps into one pipeline "
                "function, starting from the identity."},
            {"t": "code", "run": True, "caption": "Fold transforms into a pipeline",
             "code": '''\
from functools import reduce

steps = [lambda s: s.strip(),
         lambda s: s.lower(),
         lambda s: s.replace(" ", "_")]

def compose(f, g):
    return lambda s: g(f(s))         # apply f, then g

pipeline = reduce(compose, steps, lambda s: s)   # start = identity
print(repr(pipeline("  Hello World  ")))

# initializer also guards the empty case
print("empty sum:", reduce(lambda a, b: a + b, [], 0))
'''},
            {"t": "h", "text": "cache vs lru_cache and eviction"},
            {"t": "p", "text":
                "`@cache` is an unbounded memo (never evicts, keeps everything). "
                "`@lru_cache(maxsize=N)` keeps only the N most-recently-used "
                "entries and evicts the least-recently-used when full, trading "
                "some recomputation for bounded memory. Watch the `misses` and "
                "`currsize` fields of `cache_info()` to see eviction happen."},
            {"t": "code", "run": True, "caption": "Bounded LRU eviction in action",
             "code": '''\
from functools import lru_cache, cache

@lru_cache(maxsize=2)               # room for 2 entries only
def embed(token):
    return len(token)               # pretend: expensive lookup

for tok in ["a", "bb", "a", "ccc", "bb"]:
    embed(tok)
print("lru  :", embed.cache_info())  # evictions once 3rd key seen

@cache                              # unbounded, never evicts
def square(n):
    return n * n

square(2); square(3); square(2)
print("cache:", square.cache_info())
'''},
            {"t": "h", "text": "partial on methods"},
            {"t": "p", "text":
                "`partial` also freezes the first argument of a bound method, "
                "letting you turn a general method into a specialised callable, "
                "for example a metric object whose threshold is fixed once and "
                "then called on many prediction batches."},
            {"t": "code", "run": True, "caption": "Specialising a bound method",
             "code": '''\
from functools import partial

class Metric:
    def score(self, preds, threshold):
        hits = sum(p >= threshold for p in preds)
        return hits / len(preds)

m = Metric()
at_half = partial(m.score, threshold=0.5)   # freeze the threshold
print("frac >= 0.5:", at_half([0.2, 0.6, 0.9, 0.4]))
print("frac >= 0.5:", at_half([0.7, 0.8]))
'''},
            {"t": "h", "text": "singledispatch: type-based dispatch"},
            {"t": "p", "text":
                "`@singledispatch` turns a function into a generic that picks an "
                "implementation based on the type of its **first** argument. It "
                "is a clean alternative to a chain of `isinstance` checks, ideal "
                "for a preprocessing function that must accept strings, numbers "
                "or lists and normalise each differently."},
            {"t": "code", "run": True, "caption": "One function, many input types",
             "code": '''\
from functools import singledispatch

@singledispatch
def to_tokens(x):
    raise TypeError(f"unsupported: {type(x).__name__}")

@to_tokens.register
def _(x: str):
    return x.split()

@to_tokens.register
def _(x: int):
    return [str(x)]

@to_tokens.register(list)
def _(x):
    return [t for item in x for t in to_tokens(item)]

print(to_tokens("hello world"))
print(to_tokens(42))
print(to_tokens(["a b", 7]))
'''},
            {"t": "h", "text": "cached_property: compute once per instance"},
            {"t": "p", "text":
                "`@cached_property` turns a method into an attribute that is "
                "computed on first access and then stored on the instance, so "
                "later reads are free. Handy for a derived value like a "
                "vocabulary size that is expensive to build but never changes."},
            {"t": "code", "run": True, "caption": "Lazy, memoized attribute",
             "code": '''\
from functools import cached_property

class Corpus:
    def __init__(self, docs):
        self.docs = docs
    @cached_property
    def vocab(self):
        print("(building vocab...)")     # runs only once
        return sorted({w for d in self.docs for w in d.split()})

c = Corpus(["a b", "b c"])
print(c.vocab)                            # builds
print(c.vocab)                            # cached, no rebuild
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
                "training-loop decorators debuggable. `singledispatch` gives "
                "clean type-based preprocessing, and `cached_property` memoizes "
                "expensive derived values like vocabularies per instance."},
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
            {"t": "h", "text": "accumulate: running totals"},
            {"t": "p", "text":
                "`accumulate` yields a running reduction, a cumulative sum by "
                "default (like `cumsum`), or any binary function you pass. Use "
                "it to turn per-step increments into a schedule, or per-batch "
                "counts into a cumulative sample position."},
            {"t": "code", "run": True, "caption": "Cumulative sum of a schedule",
             "code": '''\
from itertools import accumulate
import operator

warmup = [0.1, 0.2, 0.3, 0.4]
print("cumsum   :", list(accumulate(warmup)))
print("running max:", list(accumulate([3, 1, 4, 1, 5], max)))
print("cumprod  :", list(accumulate([1, 2, 3, 4], operator.mul)))
'''},
            {"t": "h", "text": "groupby: group consecutive items"},
            {"t": "p", "text":
                "`groupby` groups **adjacent** items sharing a key, so the input "
                "must be sorted by that key first. It is the standard way to "
                "bucket samples by label or split a stream into runs."},
            {"t": "code", "run": True, "caption": "Group samples by label",
             "code": '''\
from itertools import groupby
from operator import itemgetter

samples = [("cat", 0), ("dog", 1), ("cat", 2), ("dog", 3)]
samples.sort(key=itemgetter(0))          # sort before grouping!
for label, group in groupby(samples, key=itemgetter(0)):
    ids = [s[1] for s in group]
    print(f"{label}: {ids}")
'''},
            {"t": "h", "text": "zip_longest: pad ragged sequences"},
            {"t": "p", "text":
                "Plain `zip` stops at the shortest input; `zip_longest` runs to "
                "the longest and fills gaps with a `fillvalue`. This is exactly "
                "how you pad variable-length sequences into a rectangular batch."},
            {"t": "code", "run": True, "caption": "Pad sequences to equal length",
             "code": '''\
from itertools import zip_longest

seqs = [[1, 2, 3], [4], [5, 6]]
padded = list(zip_longest(*seqs, fillvalue=0))
# transpose back to per-sequence rows
rows = [list(col) for col in zip(*padded)]
print("padded batch:", rows)
'''},
            {"t": "h", "text": "starmap: apply over pre-zipped args"},
            {"t": "p", "text":
                "`starmap` is like `map` but unpacks each item as the function's "
                "arguments, i.e. `f(*args)`. Perfect when your data already "
                "comes as tuples of arguments, such as (prediction, label) pairs."},
            {"t": "code", "run": True, "caption": "Map a 2-arg function over pairs",
             "code": '''\
from itertools import starmap

pairs = [(0.9, 1), (0.2, 0), (0.6, 1)]
def loss(p, y):
    return round((p - y) ** 2, 3)
print("per-sample loss:", list(starmap(loss, pairs)))
'''},
            {"t": "h", "text": "pairwise: sliding windows of two"},
            {"t": "p", "text":
                "`pairwise` (Python 3.10+) yields overlapping consecutive pairs, "
                "the simplest sliding window. Use it to compute step-to-step "
                "deltas of a loss curve or gaps between checkpoints."},
            {"t": "code", "run": True, "caption": "Deltas between consecutive losses",
             "code": '''\
from itertools import pairwise

losses = [2.0, 1.6, 1.5, 1.1]
deltas = [round(b - a, 2) for a, b in pairwise(losses)]
print("step-to-step change:", deltas)
'''},
            {"t": "h", "text": "cycle + islice: repeat a finite stream"},
            {"t": "p", "text":
                "`cycle` repeats an iterable forever; bound it with `islice` to "
                "draw a fixed number of items. Cycling class labels is a quick "
                "way to build a round-robin assignment or a repeating schedule."},
            {"t": "code", "run": True, "caption": "Round-robin over a small set",
             "code": '''\
from itertools import cycle, islice

folds = cycle(["train", "val"])          # infinite alternation
assignment = list(islice(folds, 5))
print("fold assignment:", assignment)
'''},
            {"t": "h", "text": "takewhile and dropwhile"},
            {"t": "p", "text":
                "`takewhile` yields items until the predicate first fails, then "
                "stops; `dropwhile` skips items until it first fails, then "
                "yields the rest. Handy for trimming a warmup prefix or reading "
                "an early-stopping run."},
            {"t": "code", "run": True, "caption": "Split a stream at a threshold",
             "code": '''\
from itertools import takewhile, dropwhile

acc = [0.1, 0.3, 0.5, 0.49, 0.6]
rising = list(takewhile(lambda a: a < 0.5, acc))
rest = list(dropwhile(lambda a: a < 0.5, acc))
print("before 0.5:", rising)
print("from 0.5  :", rest)
'''},
            {"t": "h", "text": "tee: fork an iterator"},
            {"t": "p", "text":
                "`tee` splits one iterator into several independent ones, so you "
                "can traverse a stream twice, for example to compute a running "
                "mean and a max over the same data. Do not keep using the "
                "original iterator after teeing it."},
            {"t": "code", "run": True, "caption": "Two passes over one stream",
             "code": '''\
from itertools import tee

data = iter([3, 1, 4, 1, 5])
a, b = tee(data, 2)                       # two independent copies
total = sum(a)
peak = max(b)
print(f"mean={total / 5:.1f}, max={peak}")
'''},
            {"t": "note", "text":
                "Why it matters for AI: itertools is the pure-Python toolkit "
                "for data pipelines. `chain` merges shards, `islice` bounds "
                "infinite generators, `product` enumerates hyperparameter "
                "grids, and `batched` builds mini-batches, all lazily and "
                "without loading everything into memory. `accumulate` builds "
                "schedules, `groupby` buckets samples by label, `zip_longest` "
                "pads ragged batches, and `pairwise` gives sliding windows over "
                "a loss curve."},
        ],
    },
]
