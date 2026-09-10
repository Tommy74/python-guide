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
            {"t": "h", "text": "__str__ vs __repr__: two audiences"},
            {"t": "p", "text":
                "`__repr__` is the *unambiguous* developer view (used in the "
                "REPL, logs and inside containers); `__str__` is the *friendly* "
                "user view (used by `print` and `str`). If you only define one, "
                "define `__repr__`, `str` falls back to it."},
            {"t": "code", "run": True, "caption": "One object, two renderings",
             "code": '''class Tensor:
    def __init__(self, shape):
        self.shape = shape
    def __repr__(self):
        return f"Tensor(shape={self.shape})"   # for devs/logs
    def __str__(self):
        dims = "x".join(map(str, self.shape))
        return f"<{dims} tensor>"              # for humans

t = Tensor([32, 768])
print(str(t))            # uses __str__
print(repr(t))           # uses __repr__
print([t])               # containers use __repr__'''},
            {"t": "h", "text": "__eq__ and __hash__ travel together"},
            {"t": "p", "text":
                "`__eq__` defines value equality; `__hash__` lets an object live "
                "in a `set` or `dict` key. If two objects are equal they *must* "
                "hash equal, so define both from the same fields. Beware: "
                "defining `__eq__` alone makes Python set `__hash__` to `None`, "
                "silently dropping hashability."},
            {"t": "code", "run": True, "caption": "__eq__ plus __hash__ from the same fields",
             "code": '''class Token:
    def __init__(self, text, idx):
        self.text, self.idx = text, idx
    def __repr__(self):
        return f"Token({self.text!r}, {self.idx})"
    def __eq__(self, other):
        return (isinstance(other, Token)
                and (self.text, self.idx)
                == (other.text, other.idx))
    def __hash__(self):
        return hash((self.text, self.idx))

a, b = Token("cat", 5), Token("cat", 5)
print("equal?", a == b)
print("same hash?", hash(a) == hash(b))
print("set dedups:", len({a, b}))   # collapses to one'''},
            {"t": "code", "run": True, "caption": "__eq__ without __hash__ drops hashability",
             "code": '''class NoHash:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        return self.v == other.v
    # no __hash__ defined -> Python sets __hash__ = None

try:
    {NoHash(1)}                      # now unhashable
except TypeError as e:
    print("TypeError:", e)'''},
            {"t": "h", "text": "__len__ and __getitem__: make a Dataset"},
            {"t": "p", "text":
                "Defining `__len__` and `__getitem__` makes an object behave "
                "like a sequence: `len(obj)` works, `obj[i]` works, and, because "
                "`__getitem__` accepts integer indices from `0` upward, the "
                "object becomes *iterable* too. This is exactly the PyTorch "
                "`Dataset` protocol."},
            {"t": "code", "run": True, "caption": "An indexable, iterable Dataset",
             "code": '''class Dataset:
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]

ds = Dataset([("a", 0), ("b", 1), ("c", 0)])
print("len:", len(ds))
print("index:", ds[1])               # __getitem__
for text, label in ds:               # iterable for free
    print("sample:", text, label)'''},
            {"t": "h", "text": "__slots__ saves memory on many objects"},
            {"t": "p", "text":
                "By default each instance carries a `__dict__` to hold "
                "attributes. Declaring `__slots__` replaces it with a fixed set "
                "of slots, cutting per-instance memory (useful when you create "
                "millions of tiny records) and blocking accidental new "
                "attributes."},
            {"t": "code", "run": True, "caption": "__slots__ removes the per-instance __dict__",
             "code": '''class Small:
    __slots__ = ("x", "y")           # fixed attributes, no __dict__
    def __init__(self, x, y):
        self.x, self.y = x, y

class Big:
    def __init__(self, x, y):
        self.x, self.y = x, y

s, b = Small(1, 2), Big(1, 2)
print("slots __dict__?", hasattr(s, "__dict__"))
print("plain __dict__?", hasattr(b, "__dict__"))
try:
    s.z = 3                          # cannot add new attrs
except AttributeError as e:
    print("AttributeError:", e)'''},
            {"t": "h", "text": "classmethod and staticmethod"},
            {"t": "p", "text":
                "A `@classmethod` receives the class as `cls` and is the "
                "idiomatic way to write *alternate constructors* like "
                "`from_config`. A `@staticmethod` receives neither `self` nor "
                "`cls`, it is a plain utility that lives in the class namespace "
                "for organisation."},
            {"t": "code", "run": True, "caption": "from_config constructor and a static helper",
             "code": '''class TrainConfig:
    def __init__(self, lr, batch_size):
        self.lr, self.batch_size = lr, batch_size
    def __repr__(self):
        return (f"TrainConfig(lr={self.lr}, "
                f"batch_size={self.batch_size})")

    @classmethod
    def from_config(cls, d):         # alternate constructor
        return cls(lr=d["lr"],
                   batch_size=d.get("batch_size", 32))

    @staticmethod
    def is_valid_lr(lr):             # no self/cls needed
        return 0 < lr < 1

cfg = TrainConfig.from_config({"lr": 1e-3})
print(cfg)
print("valid lr?", TrainConfig.is_valid_lr(1e-3))'''},
            {"t": "h", "text": "Pitfall: a shared mutable class attribute"},
            {"t": "p", "text":
                "Because class attributes are shared, a *mutable* one (a list or "
                "dict) is shared across all instances. Mutating it through one "
                "instance changes it for every instance, a bug that hides for a "
                "long time. Keep per-instance state in `__init__`."},
            {"t": "code", "run": True, "caption": "The shared-mutable-class-attribute trap",
             "code": '''class Logger:
    history = []                     # shared mutable class attr!

    def log(self, msg):
        self.history.append(msg)     # mutates the shared list

a, b = Logger(), Logger()
a.log("from a")
b.log("from b")
print("a sees:", a.history)          # both entries leak in
print("same list?", a.history is b.history)'''},
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
            {"t": "h", "text": "An unfinished subclass still fails"},
            {"t": "p", "text":
                "Abstract methods are checked at *instantiation*: even a "
                "concrete-looking subclass cannot be created until it implements "
                "every `@abstractmethod`. This catches a half-written model long "
                "before the training loop runs."},
            {"t": "code", "run": True, "caption": "Missing method blocks instantiation",
             "code": '''from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def forward(self, x): ...
    @abstractmethod
    def parameters(self): ...

class Half(Model):                # forgets parameters()
    def forward(self, x):
        return x

try:
    Half()
except TypeError as e:
    print("TypeError:", e)'''},
            {"t": "h", "text": "@property with a validating setter"},
            {"t": "p", "text":
                "Pair a `@property` getter with a matching `@x.setter` to run "
                "validation on every assignment. Storing the real value in a "
                "'private' `_lr` lets the public `lr` reject out-of-range "
                "hyper-parameters the moment they are set."},
            {"t": "code", "run": True, "caption": "A setter that validates lr in [0, 1]",
             "code": '''class Optimizer:
    def __init__(self, lr):
        self._lr = 0.0
        self.lr = lr                 # goes through the setter

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if not 0 <= value <= 1:
            raise ValueError(f"lr must be in [0, 1], got {value}")
        self._lr = value

opt = Optimizer(0.01)
print("lr:", opt.lr)
opt.lr = 0.5
print("updated lr:", opt.lr)
try:
    opt.lr = 5.0
except ValueError as e:
    print("rejected:", e)'''},
            {"t": "h", "text": "Multiple inheritance, mixins and the MRO"},
            {"t": "p", "text":
                "A class can inherit from several parents. *Mixins*, small "
                "classes that add one capability, are a common use: combine a "
                "`SerializeMixin` and a base `Module` to compose behaviour. "
                "Python resolves which method wins via the **MRO** (method "
                "resolution order), visible as `__mro__`."},
            {"t": "code", "run": True, "caption": "Mixins and __mro__",
             "code": '''class Module:
    def forward(self, x):
        return x

class SerializeMixin:
    def save(self):
        return f"saved {type(self).__name__}"

class CountMixin:
    def n_params(self):
        return 0

class Net(SerializeMixin, CountMixin, Module):
    pass

net = Net()
print("forward:", net.forward(9))
print("save:", net.save())
print("mro:", [c.__name__ for c in Net.__mro__])'''},
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
            {"t": "h", "text": "order=True and per-field field() options"},
            {"t": "p", "text":
                "`order=True` generates comparison methods so instances sort. "
                "Per field you can opt out of behaviours: `compare=False` "
                "excludes a field from equality/ordering, and `repr=False` hides "
                "it from the printout. Here checkpoints sort purely by score."},
            {"t": "code", "run": True, "caption": "Sortable checkpoints with field() tuning",
             "code": '''from dataclasses import dataclass, field

@dataclass(order=True)
class Checkpoint:
    score: float
    path: str = field(compare=False)     # ignored when sorting
    notes: str = field(default="", repr=False)

a = Checkpoint(0.91, "a.pt")
b = Checkpoint(0.95, "b.pt", notes="best so far")
print("b better?", b > a)          # compares score only
print("best:", max([a, b]))        # notes hidden from repr'''},
            {"t": "h", "text": "asdict/astuple and slots=True"},
            {"t": "p", "text":
                "`asdict` and `astuple` recursively convert a dataclass for "
                "serialisation or unpacking. Passing `slots=True` (Python 3.10+) "
                "builds the class with `__slots__`, saving memory when you hold "
                "many records, exactly like the manual `__slots__` earlier."},
            {"t": "code", "run": True, "caption": "asdict, astuple and slots=True",
             "code": '''from dataclasses import dataclass, asdict, astuple

@dataclass(slots=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
print("as dict:", asdict(p))
print("as tuple:", astuple(p))
print("has __dict__?", hasattr(p, "__dict__"))  # False'''},
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
            {"t": "h", "text": "auto() and IntEnum"},
            {"t": "p", "text":
                "`auto()` fills in values so you do not repeat yourself. An "
                "`IntEnum` *is* an `int`, so members compare and arithmetic like "
                "numbers, ideal for ordered levels such as logging verbosity "
                "where `INFO >= DEBUG` should just work."},
            {"t": "code", "run": True, "caption": "auto() values and comparable IntEnum",
             "code": '''from enum import Enum, IntEnum, auto

class Stage(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30

print("auto values:", [s.value for s in Stage])
print("INFO >= DEBUG?", LogLevel.INFO >= LogLevel.DEBUG)
print("as int:", LogLevel.WARNING + 1)   # behaves like int'''},
            {"t": "h", "text": "Flag for combinable options"},
            {"t": "p", "text":
                "A `Flag` (or `IntFlag`) lets members be combined with `|` and "
                "tested with `in`, the natural fit for a set of toggles such as "
                "an augmentation pipeline. One value carries several independent "
                "choices."},
            {"t": "code", "run": True, "caption": "Flag: OR-combined augmentations",
             "code": '''from enum import Flag, auto

class Augment(Flag):
    NONE = 0
    FLIP = auto()
    ROTATE = auto()
    CROP = auto()

pipeline = Augment.FLIP | Augment.CROP
print("pipeline:", pipeline)
print("has FLIP?", Augment.FLIP in pipeline)
print("has ROTATE?", Augment.ROTATE in pipeline)'''},
            {"t": "h", "text": "NamedTuple class syntax with methods"},
            {"t": "p", "text":
                "The class form of `NamedTuple` adds type hints and even methods "
                "while staying an immutable tuple. `_replace` returns a modified "
                "copy (never mutating the original) and `_asdict` gives a plain "
                "dict."},
            {"t": "code", "run": True, "caption": "A NamedTuple with a method and _replace",
             "code": '''from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
    def norm(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3.0, 4.0)
print("norm:", p.norm())
q = p._replace(y=0.0)             # new immutable copy
print("replaced:", q)
print("original intact:", p)
print("as dict:", p._asdict())'''},
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
            {"t": "h", "text": "Stacking decorators, order matters"},
            {"t": "p", "text":
                "You can apply several decorators to one function. They wrap "
                "*bottom-up*: the one nearest the `def` runs first (innermost), "
                "the top one wraps last (outermost). Reading them top-to-bottom "
                "tells you the order the wrappers execute around the call."},
            {"t": "code", "run": True, "caption": "Two decorators wrap bottom-up",
             "code": '''import functools

def bold(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        return "<b>" + fn(*a, **k) + "</b>"
    return w

def emph(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        return "<i>" + fn(*a, **k) + "</i>"
    return w

@bold                # applied second -> outermost
@emph                # applied first  -> innermost
def greet(name):
    return f"hi {name}"

print(greet("ada"))  # <b><i>hi ada</i></b>'''},
            {"t": "h", "text": "A class-based decorator"},
            {"t": "p", "text":
                "A class whose instances are callable (via `__call__`) can also "
                "be a decorator, handy when the wrapper needs to hold state such "
                "as a call counter. `functools.update_wrapper` copies the "
                "wrapped function's metadata, the class-based twin of `wraps`."},
            {"t": "code", "run": True, "caption": "Counting calls with a class decorator",
             "code": '''import functools

class CountCalls:
    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.count = 0
    def __call__(self, *a, **k):
        self.count += 1
        return self.fn(*a, **k)

@CountCalls
def step():
    return "stepped"

step(); step(); step()
print("name:", step.__name__)      # preserved by update_wrapper
print("calls:", step.count)'''},
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
            {"t": "h", "text": "__exit__ can handle an exception"},
            {"t": "p", "text":
                "`__exit__` receives the exception (type, value, traceback) if "
                "the block raised. Returning a truthy value *suppresses* it, so "
                "execution continues after the `with`. Returning `False` (or "
                "`None`) lets it propagate, as the `Timer` above did."},
            {"t": "code", "run": True, "caption": "A context manager that swallows errors",
             "code": '''class ignore_errors:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print("suppressed:", exc_type.__name__)
        return True                  # truthy -> swallow it

with ignore_errors():
    raise ValueError("boom")         # handled, not raised
print("execution continues")'''},
            {"t": "h", "text": "contextlib.suppress for expected errors"},
            {"t": "p", "text":
                "`contextlib.suppress` is a ready-made context manager that "
                "silently ignores the exception types you name, cleaner than an "
                "empty `except`. Great for best-effort cleanup like deleting a "
                "checkpoint that may not exist."},
            {"t": "code", "run": True, "caption": "suppress a specific exception",
             "code": '''import os
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("does_not_exist.ckpt")   # missing? no problem
print("cleanup done")'''},
            {"t": "h", "text": "ExitStack for a dynamic set of resources"},
            {"t": "p", "text":
                "When the number of resources is only known at runtime, "
                "`contextlib.ExitStack` manages them all and unwinds them in "
                "reverse on exit, so you never hand-write nested `with` "
                "statements for a variable list of open shards."},
            {"t": "code", "run": True, "caption": "Open many files with ExitStack",
             "code": '''import os, tempfile
from contextlib import ExitStack

tmp = tempfile.gettempdir()
paths = [os.path.join(tmp, f"shard_{i}.txt") for i in range(3)]
for p in paths:
    open(p, "w").close()

with ExitStack() as stack:
    files = [stack.enter_context(open(p)) for p in paths]
    print("open files:", len(files))
# every file closed here, in reverse order
for p in paths:
    os.remove(p)
print("all closed and removed")'''},
            {"t": "note", "text":
                "Why it matters for AI: `with torch.no_grad()`, "
                "`with autocast()` and `with device:` are all this one "
                "protocol. Knowing it means you can write reproducible-seed, "
                "timing and resource-cleanup blocks that never leak state."},
        ],
    },
]
