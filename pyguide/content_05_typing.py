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
            {"t": "note", "text":
                "Why it matters for AI: training runs last hours and produce "
                "thousands of lines. Levels let you keep DEBUG detail available "
                "but quiet by default, and structured, timestamped logs are what "
                "you grep through when a run diverges at 3 a.m. Frameworks and "
                "experiment trackers all build on `logging`."},
        ],
    },
]
