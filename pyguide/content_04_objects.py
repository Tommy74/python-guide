"""Part IV - Objects and Abstractions."""

PART = "Objects and Abstractions"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Classes, Dunder Methods and __call__: the nn.Module Pattern",
        "blocks": [
            {"t": "p", "text":
                "A **class** bundles data (attributes) with behaviour (methods). "
                "Every neural-network layer you will ever use, `nn.Linear`, "
                "`nn.Conv2d`, a whole `nn.Module`, is just a class instance that "
                "holds parameters and knows how to transform an input."},
            {"t": "h", "text": "__init__ and instance vs class attributes"},
            {"t": "p", "text":
                "`__init__` runs when you build an instance. Attributes set on "
                "`self` are *per-instance*; attributes set on the class body are "
                "*shared* by every instance, handy for constants like a default "
                "learning rate."},
            {"t": "code", "run": True, "caption": "Instance vs class attributes",
             "code": '''class Neuron:
    activation = "relu"          # class attribute, shared

    def __init__(self, n_inputs):
        self.weights = [0.0] * n_inputs   # instance attribute

a, b = Neuron(3), Neuron(5)
print("a.weights:", a.weights)
print("b.weights:", b.weights)
print("shared activation:", a.activation, b.activation)
Neuron.activation = "gelu"       # change once, seen everywhere
print("after change:", a.activation, b.activation)'''},
            {"t": "h", "text": "__repr__ makes objects printable"},
            {"t": "p", "text":
                "Dunder (double-underscore) methods hook into Python syntax. "
                "`__repr__` controls how an object shows up in the REPL and in "
                "logs, priceless when you print a model or a config."},
            {"t": "code", "run": True, "caption": "A readable __repr__",
             "code": '''class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"

layer = Linear(768, 256)
print(layer)
print([Linear(4, 4), Linear(4, 1)])'''},
            {"t": "h", "text": "__call__: why layer(x) works"},
            {"t": "p", "text":
                "When an object defines `__call__`, you can invoke the instance "
                "like a function. This is exactly why `output = layer(x)` works "
                "in PyTorch instead of `layer.forward(x)`, the framework routes "
                "the call through `__call__`. Here is a pure-Python `Linear` "
                "layer that computes `y = xW + b` and is *callable*."},
            {"t": "code", "run": True, "caption": "A callable Linear layer (pure Python)",
             "code": '''class Linear:
    def __init__(self, W, b):
        self.W = W            # out x in
        self.b = b            # length out

    def __call__(self, x):    # y = W @ x + b
        return [sum(w_i * x_j for w_i, x_j in zip(row, x)) + b_i
                for row, b_i in zip(self.W, self.b)]

layer = Linear(W=[[1.0, 0.0], [0.5, 0.5]], b=[0.1, -0.2])
x = [2.0, 4.0]
print("layer(x) =", layer(x))   # calls __call__, just like PyTorch'''},
            {"t": "note", "text":
                "Why it matters for AI: the whole `model(x)` ergonomics of "
                "PyTorch and Keras rests on `__call__`. Understanding it "
                "demystifies why layers are 'called' and lets you write your own "
                "composable, callable components."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Inheritance, super(), Abstract Base Classes and Properties",
        "blocks": [
            {"t": "p", "text":
                "Inheritance lets a class reuse and specialise another. In "
                "practice you subclass a framework base (`nn.Module`, a "
                "scikit-learn `BaseEstimator`) and override one or two methods."},
            {"t": "h", "text": "Subclassing and super()"},
            {"t": "p", "text":
                "`super().__init__(...)` runs the parent's setup so the child "
                "does not have to re-implement it, the same call you write at "
                "the top of every custom `nn.Module.__init__`."},
            {"t": "code", "run": True, "caption": "super() calls the parent",
             "code": '''class Layer:
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        raise NotImplementedError

class ReLU(Layer):
    def __init__(self):
        super().__init__("relu")          # run parent setup
        self.calls = 0

    def forward(self, x):                 # override
        self.calls += 1
        return [max(0.0, v) for v in x]

r = ReLU()
print(r.name, "->", r.forward([-1.0, 2.0, -3.0]))
print("forward calls:", r.calls)'''},
            {"t": "h", "text": "Abstract base classes define a contract"},
            {"t": "p", "text":
                "`abc.ABC` plus `@abstractmethod` forces every subclass to "
                "implement required methods. Instantiating an incomplete "
                "subclass fails loudly, which is how frameworks guarantee your "
                "model actually has a `forward`."},
            {"t": "code", "run": True, "caption": "@abstractmethod enforces the interface",
             "code": '''from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def forward(self, x): ...

class Identity(Model):
    def forward(self, x):
        return x

print("Identity works:", Identity().forward(42))
try:
    Model()                       # abstract, cannot instantiate
except TypeError as e:
    print("TypeError:", e)'''},
            {"t": "h", "text": "@property for computed attributes"},
            {"t": "p", "text":
                "A `@property` looks like an attribute but runs code on access, "
                "perfect for derived values such as a model's parameter count, "
                "which should never fall out of sync with the weights."},
            {"t": "code", "run": True, "caption": "num_parameters as a property",
             "code": '''class Linear:
    def __init__(self, in_f, out_f):
        self.in_f, self.out_f = in_f, out_f

    @property
    def num_parameters(self):
        return self.in_f * self.out_f + self.out_f   # weights + bias

layer = Linear(768, 256)
print("params:", layer.num_parameters)   # accessed like an attribute'''},
            {"t": "note", "text":
                "Why it matters for AI: every custom model you write subclasses "
                "a base and calls `super().__init__()`. ABCs catch missing "
                "methods before training starts, and properties keep derived "
                "stats (param counts, output shapes) always correct."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Dataclasses for Configs and Records",
        "blocks": [
            {"t": "p", "text":
                "Training runs are drowning in configuration: learning rate, "
                "batch size, number of layers. A `@dataclass` turns a plain "
                "class into a typed record with a free `__init__`, `__repr__` "
                "and equality, ideal for hyper-parameter configs."},
            {"t": "h", "text": "A typed config with defaults"},
            {"t": "code", "run": True, "caption": "@dataclass with defaults",
             "code": '''from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 10

cfg = TrainConfig(lr=1e-3)
print(cfg)                       # auto __repr__
print("batch_size:", cfg.batch_size)
print("equal?", cfg == TrainConfig(lr=1e-3))'''},
            {"t": "h", "text": "Mutable defaults need default_factory"},
            {"t": "p", "text":
                "A list or dict default must be created per instance with "
                "`field(default_factory=...)`, otherwise every config would "
                "*share* the same list, a classic bug."},
            {"t": "code", "run": True, "caption": "field(default_factory=list)",
             "code": '''from dataclasses import dataclass, field, asdict

@dataclass
class ModelConfig:
    hidden_sizes: list = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1

cfg = ModelConfig()
cfg.hidden_sizes.append(64)
print(cfg)
print("as dict:", asdict(cfg))   # easy to serialise to JSON'''},
            {"t": "h", "text": "frozen=True and __post_init__ validation"},
            {"t": "p", "text":
                "`frozen=True` makes a config immutable (hashable, safe to "
                "share), and `__post_init__` runs validation right after "
                "construction so bad hyper-parameters fail immediately."},
            {"t": "code", "run": True, "caption": "Immutable, self-validating config",
             "code": '''from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    lr: float
    def __post_init__(self):
        if not 0 < self.lr < 1:
            raise ValueError(f"lr out of range: {self.lr}")

print(Config(lr=0.01))
try:
    Config(lr=5.0)
except ValueError as e:
    print("rejected:", e)
try:
    Config(lr=0.01).lr = 0.02     # frozen, cannot mutate
except Exception as e:
    print(type(e).__name__, "on assignment")'''},
            {"t": "note", "text":
                "Why it matters for AI: Hugging Face `TrainingArguments` and "
                "most research configs are dataclasses. They give you typed, "
                "self-documenting, serialisable hyper-parameters with almost no "
                "boilerplate, and validation catches typos before a GPU-hour is "
                "wasted."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Enums and NamedTuples",
        "blocks": [
            {"t": "p", "text":
                "Magic strings (`'cuda'`, `'adam'`) and bare tuples are "
                "easy to mistype and hard to read. Enums and named tuples give "
                "those values names and structure."},
            {"t": "h", "text": "Enum for fixed choices"},
            {"t": "p", "text":
                "An `enum.Enum` defines a closed set of options, a device or an "
                "optimizer, so a typo becomes an error instead of a silent "
                "wrong run."},
            {"t": "code", "run": True, "caption": "enum.Enum for device selection",
             "code": '''from enum import Enum

class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

def to(device: Device):
    print("moving tensors to", device.value)

to(Device.CUDA)
print("members:", [d.name for d in Device])
print("lookup:", Device("cpu"))'''},
            {"t": "h", "text": "NamedTuple for lightweight records"},
            {"t": "p", "text":
                "A `typing.NamedTuple` is an immutable record with named fields. "
                "It is perfect for a training `Batch`: you get "
                "`batch.inputs` instead of `batch[0]`, but it still unpacks and "
                "behaves like a tuple."},
            {"t": "code", "run": True, "caption": "typing.NamedTuple as a Batch",
             "code": '''from typing import NamedTuple

class Batch(NamedTuple):
    inputs: list
    labels: list
    weight: float = 1.0

b = Batch(inputs=[1, 2, 3], labels=[0, 1, 1])
print("named access:", b.inputs, b.labels, b.weight)
x, y, w = b                       # still unpacks like a tuple
print("unpacked:", x, y, w)
print("as dict:", b._asdict())'''},
            {"t": "note", "text":
                "Why it matters for AI: dataloaders, RL transitions and model "
                "outputs are often named tuples (PyTorch returns them from many "
                "ops). Enums prevent `'cuda'`/`'Cuda'` typos and make valid "
                "options discoverable in your editor."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Decorators: Timing, Caching and no_grad-style Wrappers",
        "blocks": [
            {"t": "p", "text":
                "A **decorator** wraps a function to add behaviour without "
                "touching its body. You have already seen `@dataclass` and "
                "`@property`; the AI ecosystem is full of them: "
                "`@torch.no_grad()`, `@tf.function`, `@functools.lru_cache`."},
            {"t": "h", "text": "A timing decorator with functools.wraps"},
            {"t": "p", "text":
                "`functools.wraps` copies the original function's name and "
                "docstring onto the wrapper so tracebacks and `help()` stay "
                "meaningful."},
            {"t": "code", "run": True, "caption": "@timeit measures a function",
             "code": '''import time, functools

def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"{fn.__name__} took {dt*1e3:.2f} ms")
        return out
    return wrapper

@timeit
def matmul_size(n):
    return sum(i * i for i in range(n))

print("result:", matmul_size(100_000))
print("name preserved:", matmul_size.__name__)'''},
            {"t": "h", "text": "Caching repeated work"},
            {"t": "p", "text":
                "`functools.lru_cache` memoises results, turning an expensive "
                "recomputation into a dictionary lookup, useful for tokeniser "
                "lookups or pure feature functions."},
            {"t": "code", "run": True, "caption": "@lru_cache memoises",
             "code": '''from functools import lru_cache

calls = {"n": 0}

@lru_cache(maxsize=None)
def fib(n):
    calls["n"] += 1
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print("fib(30) =", fib(30))
print("unique calls:", calls["n"])   # linear, not exponential
print(fib.cache_info())'''},
            {"t": "h", "text": "A decorator that takes arguments"},
            {"t": "p", "text":
                "Add a layer of nesting and a decorator can accept parameters, "
                "for example a `@retry(n)` that re-runs a flaky call, exactly "
                "what you want around an unreliable model-serving API."},
            {"t": "code", "run": True, "caption": "@retry(n) with an argument",
             "code": '''import functools

def retry(times):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            for attempt in range(1, times + 1):
                try:
                    return fn(*a, **k)
                except Exception as e:
                    print(f"attempt {attempt} failed: {e}")
            raise RuntimeError("all retries exhausted")
        return wrapper
    return deco

state = {"n": 0}

@retry(3)
def flaky():
    state["n"] += 1
    if state["n"] < 3:
        raise ValueError("temporary")
    return "ok"

print("result:", flaky())'''},
            {"t": "p", "text":
                "In real code you rarely write these, you *apply* them. The "
                "snippet below is **illustrative** (PyTorch is not installed): "
                "`@torch.no_grad()` wraps a function so it runs without building "
                "the autograd graph, saving memory during evaluation."},
            {"t": "code", "run": False, "caption": "Framework decorators (illustrative)",
             "code": '''import torch

@torch.no_grad()               # disables gradient tracking for the call
def evaluate(model, loader):
    model.eval()
    return sum(accuracy(model(x), y) for x, y in loader)

@torch.compile               # JIT-optimise the function/module
def forward(x):
    return model(x)'''},
            {"t": "note", "text":
                "Why it matters for AI: decorators are the vocabulary of modern "
                "frameworks. Recognising `@no_grad`, `@lru_cache`, `@compile` "
                "and `@jit` lets you read library code, and writing your own "
                "`@timeit`/`@retry` cleans up training and serving scripts."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Context Managers and the with Statement",
        "blocks": [
            {"t": "p", "text":
                "A **context manager** guarantees setup and teardown around a "
                "block, even if it raises. Files are the classic case, but in "
                "AI the pattern is everywhere: `with torch.no_grad():`, "
                "autocast, distributed scopes, timers."},
            {"t": "h", "text": "with for files closes them for you"},
            {"t": "code", "run": True, "caption": "with always closes the file",
             "code": '''import tempfile, os

path = os.path.join(tempfile.gettempdir(), "metrics.txt")
with open(path, "w") as f:
    f.write("epoch,loss\\n0,0.91\\n1,0.42\\n")
# file is guaranteed closed here, even on error

with open(path) as f:
    print(f.read().strip())
os.remove(path)'''},
            {"t": "h", "text": "Writing your own with __enter__/__exit__"},
            {"t": "p", "text":
                "Any class with `__enter__` and `__exit__` is a context "
                "manager. A block timer is a tidy example, `__exit__` runs "
                "whether or not the body succeeded."},
            {"t": "code", "run": True, "caption": "A Timer context manager",
             "code": '''import time

class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        print(f"{self.label}: {dt*1e3:.2f} ms")
        return False              # do not suppress exceptions

with Timer("training step"):
    total = sum(i * i for i in range(200_000))
print("total:", total)'''},
            {"t": "h", "text": "contextlib.contextmanager: the easy way"},
            {"t": "p", "text":
                "The `@contextmanager` decorator lets you write a context "
                "manager as a generator: code before `yield` is setup, code "
                "after is teardown. Here we temporarily set and restore a "
                "random seed, a common trick for reproducible sampling."},
            {"t": "code", "run": True, "caption": "@contextmanager for a temporary seed",
             "code": '''import random
from contextlib import contextmanager

@contextmanager
def temporary_seed(seed):
    state = random.getstate()     # setup: save
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)    # teardown: restore

with temporary_seed(0):
    print("seeded:", [random.randint(0, 9) for _ in range(4)])
with temporary_seed(0):
    print("repeat:", [random.randint(0, 9) for _ in range(4)])'''},
            {"t": "p", "text":
                "The same protocol powers the most common inference idiom. This "
                "snippet is **illustrative** (PyTorch not installed): the "
                "`with` block disables gradient tracking just for its body, "
                "then restores it."},
            {"t": "code", "run": False, "caption": "with torch.no_grad() (illustrative)",
             "code": '''import torch

with torch.no_grad():              # __enter__ turns grad off
    logits = model(inputs)         # no autograd graph built -> less memory
    preds = logits.argmax(dim=-1)
# __exit__ restores gradient tracking here'''},
            {"t": "note", "text":
                "Why it matters for AI: `with torch.no_grad()`, "
                "`with autocast()` and `with device:` are all this one "
                "protocol. Knowing it means you can write reproducible-seed, "
                "timing and resource-cleanup blocks that never leak state."},
        ],
    },
]
