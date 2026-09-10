"""Part V - Typing and Robustness."""

PART = "Typing and Robustness"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Type Hints and the typing Module",
        "blocks": [
            {"t": "p", "text":
                "Type hints annotate what a function expects and returns. Python "
                "does **not** enforce them at runtime, they are for humans, "
                "editors and tools like `mypy`, but modern ML libraries lean on "
                "them heavily for autocomplete and clarity."},
            {"t": "h", "text": "Function and variable annotations"},
            {"t": "p", "text":
                "Since Python 3.9 you write built-in generics directly: "
                "`list[int]`, `dict[str, float]`, `tuple[int, int]`. They "
                "document shapes-of-data without any import."},
            {"t": "code", "run": True, "caption": "Annotations are documentation",
             "code": '''def normalise(xs: list[float]) -> list[float]:
    total: float = sum(xs)
    return [x / total for x in xs]

probs = normalise([2.0, 3.0, 5.0])
print("probs:", probs, "sum:", sum(probs))
print("annotations:", normalise.__annotations__)'''},
            {"t": "h", "text": "Optional, Union and the | syntax"},
            {"t": "p", "text":
                "`X | None` (equivalently `Optional[X]`) marks a value that may "
                "be missing, a device that defaults to auto-detect, say. "
                "`A | B` means 'either type'."},
            {"t": "code", "run": True, "caption": "Optional parameters with | None",
             "code": '''def build_model(hidden: int, device: str | None = None) -> dict:
    if device is None:
        device = "cpu"            # auto-detect fallback
    return {"hidden": hidden, "device": device}

print(build_model(256))
print(build_model(256, device="cuda"))'''},
            {"t": "h", "text": "Callable and simple generics"},
            {"t": "p", "text":
                "`Callable` types a function argument (an activation or a "
                "metric passed in), and a `TypeVar` writes a function that "
                "works for *any* type while keeping the relationship between "
                "input and output."},
            {"t": "code", "run": True, "caption": "Callable and TypeVar",
             "code": '''from typing import Callable, TypeVar

def apply(fn: Callable[[float], float], xs: list[float]) -> list[float]:
    return [fn(x) for x in xs]

relu = lambda x: max(0.0, x)
print("relu:", apply(relu, [-1.0, 0.5, 2.0]))

T = TypeVar("T")
def first(seq: list[T]) -> T:
    return seq[0]

print("first int:", first([10, 20]))
print("first str:", first(["a", "b"]))'''},
            {"t": "h", "text": "Hints are not enforced at runtime"},
            {"t": "p", "text":
                "This surprises newcomers: passing the wrong type still runs. "
                "Use hints for tooling, and add explicit checks (next chapter) "
                "when you truly need runtime guarantees."},
            {"t": "code", "run": True, "caption": "The hint is a lie the runtime ignores",
             "code": '''from typing import get_type_hints

def scale(x: int) -> int:
    return x * 3

print("wrong type still runs:", scale("ab"))   # 'ababab'
print("declared hints:", get_type_hints(scale))'''},
            {"t": "h", "text": "Literal for a fixed set of choices"},
            {"t": "p", "text":
                "`Literal` pins a parameter to specific values, a device string, "
                "a dtype, a reduction mode. A checker rejects anything outside "
                "the set, which catches the classic `device=\"gpu\"` typo before "
                "it wastes a run."},
            {"t": "code", "run": True, "caption": "Constrain a param with Literal",
             "code": '''from typing import Literal

Device = Literal["cpu", "cuda", "mps"]

def to_device(name: str, device: Device = "cpu") -> str:
    return f"{name} -> {device}"

print(to_device("model", "cuda"))
# A wrong literal still runs, but a type checker flags it:
print(to_device("model", "tpu"))   # mypy: not a valid Device'''},
            {"t": "h", "text": "Type aliases (classic and the 3.12 type statement)"},
            {"t": "p", "text":
                "A type alias gives a meaningful name to a repeated shape. The "
                "classic form is a plain assignment; Python 3.12+ adds a `type` "
                "statement that creates a first-class, lazily-evaluated alias "
                "object. Keep the classic form for readers on 3.10/3.11."},
            {"t": "code", "run": True, "caption": "Naming shapes with aliases",
             "code": '''from typing import TypeAlias

Vector: TypeAlias = list[float]    # classic alias, works on 3.10+

def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))

# Python 3.12+ 'type' statement makes a real alias object:
type Matrix = list[list[float]]

print("dot:", dot([1.0, 2.0], [3.0, 4.0]))
print("alias object:", Matrix, "->", Matrix.__value__)'''},
            {"t": "h", "text": "Final, Any vs object, and cast"},
            {"t": "p", "text":
                "`Final` marks a constant a checker should stop you reassigning. "
                "`Any` opts *out* of type checking entirely, while `object` "
                "accepts any value yet keeps you honest, you must narrow it "
                "before using type-specific methods. `cast` tells the checker "
                "'trust me, it is this type' and does nothing at runtime."},
            {"t": "code", "run": True, "caption": "Final, Any, object and cast",
             "code": '''from typing import Final, Any, cast

MAX_TOKENS: Final = 512        # reassigning is flagged by checkers
print("MAX_TOKENS:", MAX_TOKENS)

def describe(x: object) -> str:    # accepts anything, stays strict
    return f"{type(x).__name__}={x!r}"

raw: Any = "42"                # Any: checker asks no questions
n = cast(int, int(raw))        # cast: pure annotation, no-op at run
print(describe(raw), "| cast ->", n + 1)'''},
            {"t": "h", "text": "Reading annotations at runtime"},
            {"t": "p", "text":
                "`__annotations__` holds the raw annotations (possibly as "
                "strings), while `get_type_hints` resolves them into real type "
                "objects. Config systems and serialisers use this to build "
                "themselves from a class definition."},
            {"t": "code", "run": True, "caption": "Introspecting a config class",
             "code": '''from typing import get_type_hints

class Config:
    lr: float = 0.001
    steps: int = 1000

print("raw   :", Config.__annotations__)
print("hints :", get_type_hints(Config))'''},
            {"t": "h", "text": "Overloading with @overload"},
            {"t": "p", "text":
                "`@overload` declares several typed signatures for one function "
                "so a checker knows the return type depends on the argument "
                "type. Only the final, unannotated implementation runs; the "
                "stub bodies (`...`) are erased."},
            {"t": "code", "run": True, "caption": "Precise return types via overload",
             "code": '''from typing import overload

@overload
def encode(x: str) -> list[int]: ...
@overload
def encode(x: list[str]) -> list[list[int]]: ...

def encode(x):                 # the single real implementation
    if isinstance(x, str):
        return [ord(c) for c in x]
    return [[ord(c) for c in s] for s in x]

print("str  ->", encode("hi"))
print("list ->", encode(["hi", "yo"]))'''},
            {"t": "note", "text":
                "Why it matters for AI: framework APIs (PyTorch, Hugging Face, "
                "Pydantic-based configs) are richly typed, so hints power your "
                "editor's autocomplete and catch shape/argument mistakes before "
                "you launch an expensive run. Combine them with dataclasses for "
                "self-documenting configs."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Protocols and Structural Typing (Duck Typing, Checked)",
        "blocks": [
            {"t": "p", "text":
                "Python has always been duck-typed: if it has a `.forward()`, "
                "treat it as a model. `typing.Protocol` makes that idea *checked* "
                "without inheritance, you describe the methods an object must "
                "have, and any object that has them structurally satisfies the "
                "protocol. This is how you type 'anything with a `.forward()` or "
                "`.encode()` method'."},
            {"t": "h", "text": "Defining a Protocol"},
            {"t": "p", "text":
                "A `Protocol` lists method signatures with empty (`...`) bodies. "
                "A class does **not** need to subclass it, it just needs matching "
                "methods. Checkers verify the fit; nothing changes at runtime."},
            {"t": "code", "run": True, "caption": "An interface without inheritance",
             "code": '''from typing import Protocol

class SupportsForward(Protocol):
    def forward(self, x: float) -> float: ...

class Linear:                  # note: does NOT inherit anything
    def __init__(self, w: float) -> None:
        self.w = w
    def forward(self, x: float) -> float:
        return self.w * x

def run(model: SupportsForward, x: float) -> float:
    return model.forward(x)    # any object with .forward fits

print("out:", run(Linear(2.0), 3.0))'''},
            {"t": "h", "text": "runtime_checkable and isinstance"},
            {"t": "p", "text":
                "Add `@runtime_checkable` and you can use `isinstance` against "
                "the protocol. The check only verifies that the method *names* "
                "exist, not their signatures, so treat it as a cheap smoke test, "
                "not a guarantee."},
            {"t": "code", "run": True, "caption": "isinstance against a Protocol",
             "code": '''from typing import Protocol, runtime_checkable

@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

tok = CharTokenizer()
print("is Tokenizer :", isinstance(tok, Tokenizer))
print("encoded      :", tok.encode("hi"))
print("int is Tokenizer:", isinstance(42, Tokenizer))'''},
            {"t": "h", "text": "Protocols vs abstract base classes"},
            {"t": "p", "text":
                "An ABC uses *nominal* typing: a class opts in by subclassing and "
                "cannot be instantiated until it implements every abstract "
                "method. A Protocol uses *structural* typing: no subclassing, the "
                "shape is enough. Use ABCs for a base you control and want to "
                "share code from; use Protocols to accept third-party objects you "
                "do not own."},
            {"t": "bullets", "items": [
                "ABC: explicit inheritance, enforced at instantiation.",
                "Protocol: implicit fit, enforced by the type checker.",
                "Both let one function accept many interchangeable types."]},
            {"t": "code", "run": True, "caption": "An ABC enforces implementation",
             "code": '''from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def __call__(self, pred: float, target: float) -> float: ...

class MSE(Loss):
    def __call__(self, pred: float, target: float) -> float:
        return (pred - target) ** 2

print("mse:", MSE()(3.0, 1.0))
try:
    Loss()                     # abstract: cannot instantiate
except TypeError as e:
    print("abstract:", e)'''},
            {"t": "note", "text":
                "Why it matters for AI: model zoos, tokenizers and datasets come "
                "from many libraries that share no base class. Protocols let you "
                "write `def train(model: SupportsForward, ...)` and accept a "
                "PyTorch module, a wrapper, or a mock in tests, anything with the "
                "right methods, without forcing everyone into one hierarchy."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "TypedDict and Typed Configs / Messages",
        "blocks": [
            {"t": "p", "text":
                "LLM tooling passes dictionaries everywhere: chat messages, JSON "
                "configs, API payloads. `typing.TypedDict` types the *keys and "
                "value types* of a plain `dict`, so your editor autocompletes "
                "`msg[\"role\"]` and a checker catches a misspelled key, while at "
                "runtime it stays an ordinary dict."},
            {"t": "h", "text": "Typing a chat message"},
            {"t": "p", "text":
                "A chat turn is `{\"role\": ..., \"content\": ...}`. A `TypedDict` "
                "names those keys once and reuses them across the whole "
                "conversation list."},
            {"t": "code", "run": True, "caption": "Typed chat messages",
             "code": '''from typing import TypedDict

class Message(TypedDict):
    role: str
    content: str

chat: list[Message] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]
for m in chat:                 # still a plain dict at runtime
    print(f"{m['role']:>6}: {m['content']}")'''},
            {"t": "h", "text": "Required, NotRequired and total=False"},
            {"t": "p", "text":
                "By default every key is required. `NotRequired` (3.11+) marks an "
                "optional key while keeping the rest mandatory, and "
                "`total=False` flips the default so *all* keys become optional, "
                "handy for a config 'patch' that overrides a few fields."},
            {"t": "code", "run": True, "caption": "Optional keys in configs",
             "code": '''from typing import TypedDict, NotRequired

class TrainConfig(TypedDict):
    lr: float                  # required
    epochs: int                # required
    seed: NotRequired[int]     # may be omitted

cfg: TrainConfig = {"lr": 1e-3, "epochs": 10}
full: TrainConfig = {"lr": 1e-3, "epochs": 10, "seed": 42}
print("cfg :", cfg)
print("full:", full)

class Patch(TypedDict, total=False):
    lr: float
    epochs: int

patch: Patch = {"lr": 5e-4}    # any subset is allowed
print("patch:", patch)'''},
            {"t": "h", "text": "TypedDict vs dataclass vs Pydantic"},
            {"t": "p", "text":
                "A `TypedDict` is the right tool when the data *must* stay a dict "
                "(it came from JSON, or an API expects one). A `@dataclass` (from "
                "Part IV) is better when you want a real object with methods, "
                "defaults and dot access. Neither validates at runtime, a "
                "`TypedDict` will happily hold a wrong-typed value."},
            {"t": "bullets", "items": [
                "TypedDict: dict in, dict out; zero runtime cost; no validation.",
                "dataclass: object with attributes, defaults and methods.",
                "Pydantic (third-party): parses and *validates* at runtime, the "
                "de-facto choice for LLM/API schemas, shown here for context "
                "only."]},
            {"t": "code", "run": False, "caption": "Pydantic Validates at Runtime",
             "code": '''# Illustrative only: Pydantic is a third-party library.
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

# Unlike TypedDict, this RAISES on bad data at runtime:
Message(role="user", content=123)   # ValidationError'''},
            {"t": "note", "text":
                "Why it matters for AI: the OpenAI/Anthropic chat format is a "
                "list of typed dicts, and training configs are nested JSON. "
                "TypedDict documents those shapes for free and turns silent key "
                "typos into editor warnings; when you need real validation of "
                "untrusted input, reach for Pydantic on top."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Errors, Exceptions and Custom Exception Types",
        "blocks": [
            {"t": "p", "text":
                "AI code fails in predictable ways: mismatched tensor shapes, "
                "missing files, out-of-memory. Handling exceptions well means "
                "your training loop can log, recover, or fail with a message "
                "that actually tells you what went wrong."},
            {"t": "h", "text": "try / except / else / finally"},
            {"t": "p", "text":
                "`else` runs only if no exception fired; `finally` always runs, "
                "ideal for releasing a resource such as a GPU handle or a file."},
            {"t": "code", "run": True, "caption": "The full try structure",
             "code": '''def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("cannot divide by zero")
        return None
    else:
        print("division ok")
        return result
    finally:
        print("cleanup always runs")

print("=>", safe_divide(10, 2))
print("=>", safe_divide(10, 0))'''},
            {"t": "h", "text": "Catch specific exceptions, not everything"},
            {"t": "p", "text":
                "Catch the narrowest exception you can handle. A bare "
                "`except:` hides real bugs; naming the type keeps unexpected "
                "errors loud."},
            {"t": "code", "run": True, "caption": "Handle only what you expect",
             "code": '''config = {"lr": 0.001}
for key in ["lr", "batch_size"]:
    try:
        print(f"{key} = {config[key]}")
    except KeyError:
        print(f"{key} missing, using default")'''},
            {"t": "h", "text": "Custom exceptions for domain errors"},
            {"t": "p", "text":
                "Defining your own exception type makes failures self-"
                "describing and lets callers catch exactly your error. A "
                "shape-mismatch check is the canonical example."},
            {"t": "code", "run": True, "caption": "A ShapeMismatchError",
             "code": '''class ShapeMismatchError(ValueError):
    """Raised when two tensor shapes are incompatible."""

def matmul_shapes(a, b):
    if a[1] != b[0]:
        raise ShapeMismatchError(f"cannot matmul {a} with {b}")
    return (a[0], b[1])

print("ok:", matmul_shapes((4, 8), (8, 2)))
try:
    matmul_shapes((4, 8), (16, 2))
except ShapeMismatchError as e:
    print("caught:", e)'''},
            {"t": "h", "text": "Exception chaining with raise ... from"},
            {"t": "p", "text":
                "`raise NewError(...) from original` preserves the underlying "
                "cause, so a friendly, high-level message keeps the low-level "
                "traceback for debugging."},
            {"t": "code", "run": True, "caption": "Wrap a low-level error clearly",
             "code": '''class DataLoadError(Exception):
    pass

def load_batch(raw):
    try:
        return int(raw)
    except ValueError as e:
        raise DataLoadError(f"bad sample: {raw!r}") from e

try:
    load_batch("not-a-number")
except DataLoadError as e:
    print("high level:", e)
    print("caused by :", type(e.__cause__).__name__)'''},
            {"t": "h", "text": "Re-raising with a bare raise"},
            {"t": "p", "text":
                "Sometimes you want to log an error and still let it propagate. A "
                "bare `raise` (no argument) inside an `except` re-throws the "
                "*same* exception with its original traceback intact, do not "
                "write `raise e`, which resets the traceback."},
            {"t": "code", "run": True, "caption": "Log, then re-raise unchanged",
             "code": '''import logging
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    force=True)

def checkpoint(step):
    try:
        if step < 0:
            raise ValueError("negative step")
        return f"saved@{step}"
    except ValueError:
        logging.info("logging, then re-raising...")
        raise                  # bare: keep the original traceback

try:
    checkpoint(-1)
except ValueError as e:
    print("caller saw:", e)'''},
            {"t": "h", "text": "Suppressing expected errors with suppress"},
            {"t": "p", "text":
                "When an exception is genuinely fine to ignore, "
                "`contextlib.suppress` reads better than an empty `except` block "
                "and makes the intent explicit."},
            {"t": "code", "run": True, "caption": "contextlib.suppress",
             "code": '''from contextlib import suppress

cache = {"warm": True}
# Delete a key that may not exist, no try/except noise:
with suppress(KeyError):
    del cache["cold"]
print("cache still fine:", cache)

with suppress(FileNotFoundError):
    open("no_such_checkpoint.pt")
print("missing checkpoint ignored")'''},
            {"t": "h", "text": "A custom exception hierarchy"},
            {"t": "p", "text":
                "Group related failures under one base class. Callers can catch "
                "the *base* to handle the whole family, or a specific subclass "
                "when they need finer control, exactly how framework error "
                "hierarchies are built."},
            {"t": "code", "run": True, "caption": "Catch a whole family at the base",
             "code": '''class TrainingError(Exception):
    """Base for anything that can go wrong during training."""

class DataError(TrainingError):
    pass

class OptimizerError(TrainingError):
    pass

def step(kind):
    if kind == "data":
        raise DataError("corrupt batch")
    if kind == "optim":
        raise OptimizerError("nan gradient")
    return "ok"

for kind in ["data", "optim", "fine"]:
    try:
        print(kind, "->", step(kind))
    except TrainingError as e:     # one catch for the family
        print(kind, "-> caught", type(e).__name__, ":", e)'''},
            {"t": "h", "text": "Gathering failures with ExceptionGroup / except*"},
            {"t": "p", "text":
                "When several concurrent tasks fail, you want *all* the errors, "
                "not just the first. Python 3.11 added `ExceptionGroup` to bundle "
                "them and `except*` to handle each type in the bundle "
                "separately, the model behind `asyncio.TaskGroup`."},
            {"t": "code", "run": True, "caption": "Bundle and split errors with except*",
             "code": '''def run_task(i):
    if i % 2 == 0:
        raise ValueError(f"task {i} bad value")
    raise KeyError(f"task {i} missing key")

errors = []
for i in range(3):             # imagine these ran concurrently
    try:
        run_task(i)
    except Exception as e:
        errors.append(e)

try:
    raise ExceptionGroup("some tasks failed", errors)
except* ValueError as eg:
    print("value errors:", len(eg.exceptions))
except* KeyError as eg:
    print("key errors  :", len(eg.exceptions))'''},
            {"t": "note", "text":
                "Why it matters for AI: the most common PyTorch error you will "
                "ever read is a shape mismatch. Validating shapes early with a "
                "clear custom exception, and chaining low-level errors into "
                "readable ones, turns cryptic stack traces into a two-second "
                "fix."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Logging Instead of print for Training Runs",
        "blocks": [
            {"t": "p", "text":
                "`print` is fine for a quick script, but real training loops "
                "need timestamps, severity levels and the ability to silence or "
                "redirect output. The `logging` module gives you all of that "
                "with one configuration call."},
            {"t": "h", "text": "Levels and basicConfig"},
            {"t": "p", "text":
                "Messages have levels (`DEBUG` < `INFO` < `WARNING` < `ERROR`). "
                "You set a threshold once; anything below it is dropped, so you "
                "can dial verbosity up or down without deleting log lines."},
            {"t": "code", "run": True, "caption": "Configure logging once",
             "code": '''import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("train")

log.debug("this is hidden (below INFO)")
log.info("starting training")
log.warning("learning rate looks high")'''},
            {"t": "h", "text": "Logging per-step metrics"},
            {"t": "p", "text":
                "Inside a loop, logging structured metrics beats printing: you "
                "get consistent formatting and can later route the same lines "
                "to a file or a service without touching the loop body."},
            {"t": "code", "run": True, "caption": "A tiny training loop with logging",
             "code": '''import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | step=%(message)s")
log = logging.getLogger("loop")

loss = 1.0
for step in range(1, 4):
    loss *= 0.5                    # pretend the model improves
    log.info("%d loss=%.4f", step, loss)
log.info("done, final loss=%.4f", loss)'''},
            {"t": "h", "text": "Timestamps and richer formats"},
            {"t": "p", "text":
                "Adding `asctime` and the logger name to the format is what "
                "makes logs from long runs actually searchable after the fact."},
            {"t": "code", "run": True, "caption": "Timestamped, named logs",
             "code": '''import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,                   # override any earlier config
)
logging.getLogger("data").info("loaded %d samples", 50000)
logging.getLogger("model").warning("checkpoint dir is nearly full")'''},
            {"t": "h", "text": "Logging exceptions with tracebacks"},
            {"t": "p", "text":
                "Inside an `except` block, `log.exception(...)` records the "
                "message at `ERROR` level *and* attaches the full traceback, so a "
                "failed step is captured in your logs without crashing the run. "
                "`log.error(msg, exc_info=True)` does the same at a chosen level."},
            {"t": "code", "run": True, "caption": "Capture a failure and continue",
             "code": '''import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)
log = logging.getLogger("train")

try:
    weights = 1 / 0                # simulate a step blowing up
except ZeroDivisionError:
    log.exception("training step failed")   # ERROR + traceback
log.info("recovered and continued")'''},
            {"t": "note", "text":
                "Why it matters for AI: training runs last hours and produce "
                "thousands of lines. Levels let you keep DEBUG detail available "
                "but quiet by default, and structured, timestamped logs are what "
                "you grep through when a run diverges at 3 a.m. Frameworks and "
                "experiment trackers all build on `logging`."},
        ],
    },
]
