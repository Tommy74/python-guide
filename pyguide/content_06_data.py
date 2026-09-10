"""Part VI - Data, Files and Serialization."""

PART = "Data, Files and Serialization"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "File I/O and pathlib",
        "blocks": [
            {"t": "p", "text":
                "Training runs live and die by files: datasets on disk, "
                "checkpoints, logs, configs. Python's `open()` plus the modern "
                "`pathlib` module give you a clean, cross-platform way to read, "
                "write and discover them without fragile string surgery on "
                "paths."},
            {"t": "h", "text": "Reading and writing with a context manager"},
            {"t": "p", "text":
                "Always open files inside a `with` block. The file is closed "
                "automatically even if an exception fires mid-write, so you "
                "never leak a half-flushed checkpoint. The mode string picks "
                "the operation: `'w'` write, `'r'` read, `'a'` append."},
            {"t": "code", "run": True, "caption": "Write then read a text file",
             "code": '''\
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "notes.txt")
with open(path, "w") as f:
    f.write("epoch 1 loss 0.42\\n")
    f.write("epoch 2 loss 0.31\\n")

with open(path) as f:
    content = f.read()
print(content, end="")
os.remove(path)
'''},
            {"t": "h", "text": "Iterating lines lazily"},
            {"t": "p", "text":
                "A file object is itself an iterator: looping over it yields "
                "one line at a time without loading the whole file into memory. "
                "This is how you stream a multi-gigabyte `.jsonl` dataset that "
                "would never fit in RAM."},
            {"t": "code", "run": True, "caption": "Stream a large file line by line",
             "code": '''\
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "data.txt")
with open(path, "w") as f:
    for i in range(5):
        f.write(f"sample-{i}\\n")

# Never loads the whole file at once - one line per iteration.
with open(path) as f:
    for lineno, line in enumerate(f, start=1):
        print(lineno, line.strip())
os.remove(path)
'''},
            {"t": "h", "text": "pathlib: paths as objects"},
            {"t": "p", "text":
                "`pathlib.Path` treats a path as an object with useful "
                "attributes and methods. The `/` operator joins path parts, "
                "`.stem`/`.suffix` split a filename, and `.mkdir(parents=True, "
                "exist_ok=True)` creates directory trees safely."},
            {"t": "code", "run": True, "caption": "Building and inspecting paths",
             "code": '''\
from pathlib import Path
import tempfile

root = Path(tempfile.mkdtemp())
ckpt_dir = root / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

ckpt = ckpt_dir / "model_epoch3.pt"
ckpt.write_bytes(b"fake-weights")
print("name  :", ckpt.name)
print("stem  :", ckpt.stem)
print("suffix:", ckpt.suffix)
print("exists:", ckpt.exists())
print("parent:", ckpt.parent.name)
'''},
            {"t": "h", "text": "Globbing for checkpoints"},
            {"t": "p", "text":
                "`Path.glob` finds files by pattern, perfect for locating every "
                "checkpoint or shard in a directory. Combine it with `sorted` "
                "and `max` to grab the latest one."},
            {"t": "code", "run": True, "caption": "Find all checkpoints, pick the newest",
             "code": '''\
from pathlib import Path
import tempfile

d = Path(tempfile.mkdtemp())
for epoch in (1, 5, 12, 3):
    (d / f"ckpt_{epoch:03d}.pt").write_bytes(b"x")

ckpts = sorted(d.glob("ckpt_*.pt"))
print("found:", [p.name for p in ckpts])
latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
print("latest:", latest.name)
'''},
            {"t": "note", "text":
                "Why it matters for AI: dataset loaders stream files line by "
                "line to stay within memory, and resuming training means "
                "globbing a checkpoint directory and loading the newest file. "
                "`pathlib` makes both robust across Linux, macOS and Windows."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "JSON and CSV: Configs and Datasets",
        "blocks": [
            {"t": "p", "text":
                "Two text formats dominate practical AI work: **JSON** for "
                "configs, API payloads and `.jsonl` datasets, and **CSV** for "
                "tabular data. Both ship in the standard library, so you can "
                "read and write them without any dependency."},
            {"t": "h", "text": "JSON: strings to objects and back"},
            {"t": "p", "text":
                "`json.dumps` serialises a Python object to a string (use "
                "`indent=` for a readable config file); `json.loads` parses it "
                "back. The `dump`/`load` pair does the same directly to and "
                "from file objects."},
            {"t": "code", "run": True, "caption": "Serialise and parse a config",
             "code": '''\
import json

config = {"model": "gpt", "layers": 12, "lr": 3e-4, "dropout": 0.1}

text = json.dumps(config, indent=2)
print(text)

restored = json.loads(text)
print("layers x2 =", restored["layers"] * 2)
'''},
            {"t": "h", "text": "JSON Lines: one example per line"},
            {"t": "p", "text":
                "The `.jsonl` format stores one JSON object per line. It is the "
                "de-facto standard for LLM fine-tuning datasets because you can "
                "stream it line by line and append new records cheaply."},
            {"t": "code", "run": True, "caption": "Write and read a .jsonl dataset",
             "code": '''\
import json, io

records = [
    {"prompt": "2+2?", "completion": "4"},
    {"prompt": "capital of France?", "completion": "Paris"},
]

buf = io.StringIO()
for r in records:
    buf.write(json.dumps(r) + "\\n")

buf.seek(0)
for line in buf:
    obj = json.loads(line)
    print(obj["prompt"], "->", obj["completion"])
'''},
            {"t": "h", "text": "CSV with DictReader / DictWriter"},
            {"t": "p", "text":
                "The `csv` module handles quoting and escaping for you. "
                "`DictWriter` writes rows from dictionaries and `DictReader` "
                "yields each row as a dict keyed by the header, which keeps "
                "column access readable."},
            {"t": "code", "run": True, "caption": "Round-trip a small dataset",
             "code": '''\
import csv, io

rows = [
    {"text": "great movie", "label": 1},
    {"text": "boring plot", "label": 0},
]

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=["text", "label"])
writer.writeheader()
writer.writerows(rows)

buf.seek(0)
reader = csv.DictReader(buf)
pos = sum(1 for row in reader if row["label"] == "1")
print("CSV written:")
print(buf.getvalue(), end="")
print("positive examples:", pos)
'''},
            {"t": "note", "text":
                "Why it matters for AI: fine-tuning datasets are almost always "
                "`.jsonl`, hyperparameter configs are JSON or YAML, and classic "
                "tabular ML lives in CSV. Note that `csv` reads every field as a "
                "**string** - cast to `int`/`float` yourself before training."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "pickle and Model / Artifact Serialization",
        "blocks": [
            {"t": "p", "text":
                "JSON only handles simple types. To save an arbitrary Python "
                "object - a fitted scaler, a vocabulary, a dict of NumPy arrays "
                "- you reach for **pickle**, Python's native binary "
                "serialisation format. It is what most ML checkpoint formats "
                "are built on under the hood."},
            {"t": "h", "text": "Pickling arbitrary objects"},
            {"t": "p", "text":
                "`pickle.dumps` turns almost any object into bytes and "
                "`pickle.loads` reconstructs it. Here we save a small "
                "preprocessing artifact: a vocabulary plus normalisation "
                "statistics."},
            {"t": "code", "run": True, "caption": "Round-trip a preprocessing artifact",
             "code": '''\
import pickle

artifact = {
    "vocab": {"<pad>": 0, "cat": 1, "dog": 2},
    "mean": 0.5,
    "std": 0.25,
}

blob = pickle.dumps(artifact)
print("serialised bytes:", len(blob))

restored = pickle.loads(blob)
print("vocab size:", len(restored["vocab"]))
print("dog id   :", restored["vocab"]["dog"])
'''},
            {"t": "h", "text": "Pickling NumPy arrays to disk"},
            {"t": "p", "text":
                "NumPy arrays pickle cleanly, so a dict of weights survives a "
                "save/load round-trip exactly. In real code you would prefer "
                "`numpy.save`/`load` (`.npy`) for plain arrays, but pickle wins "
                "when you need to bundle arrays together with other Python "
                "objects."},
            {"t": "code", "run": True, "caption": "Save a weight dict, then reload it",
             "code": '''\
import pickle, tempfile, os
import numpy as np

weights = {"w": np.arange(6).reshape(2, 3), "b": np.zeros(2)}

path = os.path.join(tempfile.mkdtemp(), "weights.pkl")
with open(path, "wb") as f:      # note the binary mode "wb"
    pickle.dump(weights, f)

with open(path, "rb") as f:
    loaded = pickle.load(f)

print("shapes:", {k: v.shape for k, v in loaded.items()})
print("equal :", np.array_equal(weights["w"], loaded["w"]))
os.remove(path)
'''},
            {"t": "h", "text": "The framework equivalent"},
            {"t": "p", "text":
                "Deep-learning frameworks wrap pickle in their own save/load "
                "helpers. The snippet below is *illustrative* (PyTorch is not "
                "installed here) but shows the identical mental model: one call "
                "to persist state, one to restore it."},
            {"t": "code", "run": False, "caption": "Illustrative: torch.save / torch.load",
             "code": '''\
import torch

# Save just the learned parameters (the recommended pattern).
torch.save(model.state_dict(), "model.pt")

# Later, rebuild the architecture, then load weights into it.
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()
'''},
            {"t": "note", "text":
                "Why it matters for AI - and a security warning: **never "
                "unpickle a file you did not create.** Unpickling can execute "
                "arbitrary code, so a malicious checkpoint downloaded from the "
                "internet is a remote-code-execution risk. This is exactly why "
                "safer formats like `safetensors` (weights only, no code) and "
                "`joblib` (efficient for large arrays) exist."},
        ],
    },
]
