"""Part VIII - The Numeric AI Stack (NumPy, pandas, frameworks)."""

PART = "The Numeric AI Stack"

CHAPTERS = [
    # ==================================================================
    {
        "title": "NumPy Arrays, dtypes and Vectorization",
        "blocks": [
            {"t": "p", "text":
                "NumPy is the foundation of the entire Python AI stack. A tensor "
                "in PyTorch, TensorFlow or JAX is essentially a NumPy array with "
                "a GPU and autograd bolted on, so the mental model you build here "
                "transfers directly. The core object is the **ndarray**: a "
                "contiguous block of numbers with a single `dtype` and a `shape`."},
            {"t": "h", "text": "Creating arrays"},
            {"t": "p", "text":
                "You rarely type numbers by hand. `zeros`, `ones`, `arange` and "
                "`linspace` build the weight matrices, index ranges and learning-"
                "rate schedules you actually use."},
            {"t": "code", "run": True, "caption": "Ways to make an array",
             "code": '''import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print("from list:\\n", a)
print("zeros:", np.zeros(3))
print("ones (2x2):\\n", np.ones((2, 2)))
print("arange:", np.arange(0, 10, 2))
print("linspace:", np.linspace(0.0, 1.0, 5))'''},
            {"t": "h", "text": "dtype: why models use float32"},
            {"t": "p", "text":
                "Every array has one `dtype`. Deep-learning models default to "
                "`float32` (not Python's 64-bit float) because it halves memory "
                "and doubles throughput on GPUs with negligible accuracy loss. "
                "Knowing when a cast happens saves you from silent slowdowns."},
            {"t": "code", "run": True, "caption": "dtypes and casting",
             "code": '''import numpy as np

x = np.array([1.0, 2.0, 3.0])
print("default dtype:", x.dtype)          # float64
w = np.ones(3, dtype=np.float32)
print("model weights:", w.dtype)
print("bytes each:", x.nbytes, "vs", w.nbytes)
ids = np.array([5, 1, 9], dtype=np.int64)  # token ids are ints
print("ids dtype:", ids.dtype)'''},
            {"t": "h", "text": "shape, ndim, size and reshape"},
            {"t": "p", "text":
                "The `shape` is the grammar of tensor code. A batch of images "
                "might be `(32, 3, 224, 224)`; a batch of token embeddings "
                "`(batch, seq_len, dim)`. `reshape` rearranges the same data "
                "without copying it, and `-1` means \"infer this axis\"."},
            {"t": "code", "run": True, "caption": "Inspecting and reshaping",
             "code": '''import numpy as np

a = np.arange(12)
print("shape:", a.shape, "ndim:", a.ndim, "size:", a.size)
b = a.reshape(3, 4)
print("reshaped:\\n", b)
print("flatten with -1:", b.reshape(-1).shape)
print("add batch axis:", a.reshape(1, -1).shape)'''},
            {"t": "h", "text": "Vectorization beats Python loops"},
            {"t": "p", "text":
                "The golden rule of numeric Python: *never loop over elements if "
                "an array operation exists*. Vectorized operations run in "
                "optimized C, so they are typically 10-100x faster and read like "
                "the math they implement."},
            {"t": "code", "run": True, "caption": "Loop vs vectorized (same result, one is fast)",
             "code": '''import numpy as np, time

x = np.arange(1_000_000, dtype=np.float64)

t0 = time.perf_counter()
loop = [v * 2.0 + 1.0 for v in x]      # slow, pure Python
t1 = time.perf_counter()
vec = x * 2.0 + 1.0                     # fast, vectorized C
t2 = time.perf_counter()

print("same result:", np.allclose(loop, vec))
print(f"loop:  {t1 - t0:.4f}s")
print(f"vector:{t2 - t1:.4f}s")'''},
            {"t": "h", "text": "Aggregations along an axis"},
            {"t": "p", "text":
                "Reductions like `sum`, `mean` and `max` take an `axis` argument. "
                "`axis=0` collapses rows (per-feature stats), `axis=1` collapses "
                "columns (per-sample stats). This is how you compute batch "
                "statistics and pool over sequence dimensions."},
            {"t": "code", "run": True, "caption": "Reductions with axis",
             "code": '''import numpy as np

batch = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])   # (2 samples, 3 features)
print("grand mean:", batch.mean())
print("per-feature mean (axis=0):", batch.mean(axis=0))
print("per-sample sum  (axis=1):", batch.sum(axis=1))
print("argmax per sample:", batch.argmax(axis=1))'''},
            {"t": "note", "text":
                "Why it matters for AI: every forward pass is array creation, "
                "reshaping and vectorized math over a fixed dtype. If you can "
                "read shapes and reach for an axis-aware reduction instead of a "
                "for-loop, you can read almost any model's code."},
        ],
    },
    # ==================================================================
    {
        "title": "Broadcasting, Indexing and Linear Algebra: the Math of a Layer",
        "blocks": [
            {"t": "p", "text":
                "A neural network layer is mostly one line of linear algebra: "
                "`y = X @ W + b`. To write and debug it you need three skills: "
                "**broadcasting** (combining arrays of different shapes), "
                "**indexing** (pulling out the elements you want), and the "
                "**matmul** operator `@`."},
            {"t": "h", "text": "Broadcasting: adding a bias to a batch"},
            {"t": "p", "text":
                "Broadcasting lets NumPy stretch a smaller array across a larger "
                "one without copying. Adding a bias vector of shape `(features,)` "
                "to a batch of shape `(batch, features)` just works, because the "
                "trailing dimensions match."},
            {"t": "code", "run": True, "caption": "Broadcasting a bias across a batch",
             "code": '''import numpy as np

batch = np.zeros((3, 4))          # 3 samples, 4 features
bias = np.array([10, 20, 30, 40]) # one bias per feature
print("batch + bias:\\n", batch + bias)
# column vector broadcasts the other way:
scale = np.array([[1], [2], [3]]) # shape (3, 1) -> per-sample
print("per-sample scale:\\n", (batch + bias) * scale)'''},
            {"t": "h", "text": "Slicing and boolean masking"},
            {"t": "p", "text":
                "Slices select contiguous regions; boolean masks select elements "
                "by condition. Masking is how you apply a ReLU, filter padded "
                "tokens, or clip outliers, all without a loop."},
            {"t": "code", "run": True, "caption": "Slices and masks",
             "code": '''import numpy as np

a = np.arange(1, 13).reshape(3, 4)
print("first row:", a[0])
print("last column:", a[:, -1])
print("top-left 2x2:\\n", a[:2, :2])

mask = a > 6
print("mask:\\n", mask)
relu = np.where(a - 6 > 0, a - 6, 0)   # a mini ReLU
print("relu(a-6):\\n", relu)'''},
            {"t": "h", "text": "Fancy indexing: gathering rows"},
            {"t": "p", "text":
                "Indexing with an array of integers gathers arbitrary rows in "
                "any order, with repeats. This is exactly an **embedding "
                "lookup**: token ids index into an embedding table."},
            {"t": "code", "run": True, "caption": "Embedding lookup by fancy indexing",
             "code": '''import numpy as np

embed = np.round(np.arange(15).reshape(5, 3) / 10, 1)  # vocab=5, dim=3
print("embedding table:\\n", embed)
token_ids = np.array([0, 2, 2, 4])
print("looked up:\\n", embed[token_ids])   # shape (4, 3)'''},
            {"t": "h", "text": "The matmul operator @ and a linear layer"},
            {"t": "p", "text":
                "`@` is matrix multiplication. Combined with broadcasting for the "
                "bias, it is a complete dense layer. Watch the shapes: "
                "`(batch, in) @ (in, out) -> (batch, out)`."},
            {"t": "code", "run": True, "caption": "A batched linear layer y = X @ W + b",
             "code": '''import numpy as np
rng = np.random.default_rng(0)

X = rng.normal(size=(2, 3))      # batch=2, in_features=3
W = rng.normal(size=(3, 4))      # in=3, out=4
b = np.zeros(4)                  # one bias per output

y = X @ W + b
print("X:", X.shape, " W:", W.shape, " ->  y:", y.shape)
print("y:\\n", np.round(y, 3))'''},
            {"t": "h", "text": "A little linear algebra"},
            {"t": "p", "text":
                "`np.linalg` gives you norms (for gradient clipping and "
                "regularization), dot products (for cosine similarity), and "
                "solvers. Cosine similarity between embeddings is one of the most "
                "common operations in retrieval and RAG systems."},
            {"t": "code", "run": True, "caption": "Cosine similarity between two vectors",
             "code": '''import numpy as np

u = np.array([1.0, 2.0, 2.0])
v = np.array([2.0, 0.0, 1.0])
cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
print("dot:", u @ v)
print("|u|:", round(float(np.linalg.norm(u)), 3))
print("cosine similarity:", round(float(cos), 3))'''},
            {"t": "note", "text":
                "Why it matters for AI: forward passes, attention scores and "
                "similarity search are all matmuls plus broadcast adds. Fancy "
                "indexing *is* the embedding layer. Get comfortable reading "
                "shapes through an `@` and you can trace data through any model."},
        ],
    },
    # ==================================================================
    {
        "title": "Random, Statistics and Activation Functions from Scratch",
        "blocks": [
            {"t": "p", "text":
                "Randomness (for weight initialization and sampling) and "
                "statistics (for normalization) sit at the heart of training. "
                "In this chapter we build the classic activation and loss "
                "functions in pure NumPy so you know exactly what a framework "
                "computes under the hood."},
            {"t": "h", "text": "Reproducible randomness"},
            {"t": "p", "text":
                "Always seed your generator. The modern API is "
                "`np.random.default_rng(seed)`, which gives an isolated stream, "
                "so your experiments are reproducible. Weight initialization "
                "typically draws from a scaled normal distribution."},
            {"t": "code", "run": True, "caption": "Seeded init and sampling",
             "code": '''import numpy as np
rng = np.random.default_rng(0)

W = rng.normal(0, 0.02, size=(2, 3))   # GPT-style small init
print("weights:\\n", np.round(W, 3))
print("uniform:", np.round(rng.uniform(size=3), 3))
print("random choice:", rng.choice([10, 20, 30], size=4))'''},
            {"t": "h", "text": "Statistics and normalization"},
            {"t": "p", "text":
                "Normalizing inputs (zero mean, unit variance) makes training "
                "stable. The z-score below is the same computation LayerNorm and "
                "BatchNorm perform, minus the learnable scale and shift."},
            {"t": "code", "run": True, "caption": "Z-score normalization (a mini LayerNorm)",
             "code": '''import numpy as np
rng = np.random.default_rng(1)

x = rng.normal(5, 3, size=(2, 4))      # not centered
mu = x.mean(axis=1, keepdims=True)
sd = x.std(axis=1, keepdims=True)
norm = (x - mu) / (sd + 1e-5)
print("mean before:", np.round(x.mean(axis=1), 2))
print("mean after :", np.round(norm.mean(axis=1), 6))
print("std  after :", np.round(norm.std(axis=1), 3))'''},
            {"t": "h", "text": "Sigmoid and ReLU"},
            {"t": "p", "text":
                "Activations are just elementwise functions over an array. "
                "Vectorization means the whole tensor is transformed at once."},
            {"t": "code", "run": True, "caption": "Two activation functions",
             "code": '''import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0.0, z)

z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("sigmoid:", np.round(sigmoid(z), 3))
print("relu   :", relu(z))'''},
            {"t": "h", "text": "Numerically stable softmax"},
            {"t": "p", "text":
                "Softmax turns logits into probabilities. The naive version "
                "overflows on large logits, so the standard trick is to subtract "
                "the max first, this changes nothing mathematically but keeps "
                "`exp` in range. Every framework does exactly this."},
            {"t": "code", "run": True, "caption": "Softmax with the max-subtraction trick",
             "code": '''import numpy as np

def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)  # stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

logits = np.array([[2.0, 1.0, 0.1],
                   [0.0, 0.0, 5.0]])
probs = softmax(logits)
print("probs:\\n", np.round(probs, 3))
print("rows sum to 1:", np.round(probs.sum(axis=1), 6))'''},
            {"t": "h", "text": "Cross-entropy loss"},
            {"t": "p", "text":
                "Cross-entropy measures how far predicted probabilities are from "
                "the true labels. It combines fancy indexing (to pull the "
                "probability of the correct class) with a log and a mean, the "
                "loss that trains most classifiers and language models."},
            {"t": "code", "run": True, "caption": "Cross-entropy from probabilities",
             "code": '''import numpy as np

probs = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.1, 0.8]])
labels = np.array([0, 2])              # correct class per sample
n = len(labels)
correct_p = probs[np.arange(n), labels]     # fancy indexing
loss = -np.mean(np.log(correct_p + 1e-9))
print("prob of correct class:", correct_p)
print("cross-entropy loss:", round(float(loss), 4))'''},
            {"t": "note", "text":
                "Why it matters for AI: softmax, ReLU and cross-entropy are the "
                "building blocks of nearly every model. Written in NumPy they are "
                "a few lines each, the framework versions add autograd and GPU "
                "support but compute the same thing, stability tricks included."},
        ],
    },
    # ==================================================================
    {
        "title": "pandas Series and DataFrame: Loading Datasets",
        "blocks": [
            {"t": "p", "text":
                "Before data becomes tensors it usually lives in a table: CSVs, "
                "logs, feature stores. **pandas** is the standard tool for "
                "loading, inspecting and slicing that tabular data. Its two "
                "types are the 1-D `Series` and the 2-D `DataFrame` (a dict of "
                "aligned Series sharing an index)."},
            {"t": "h", "text": "Series: a labeled array"},
            {"t": "code", "run": True, "caption": "A Series is values plus an index",
             "code": '''import pandas as pd

s = pd.Series([0.9, 0.4, 0.7], index=["cat", "dog", "fox"])
print(s)
print("by label:", s["dog"])
print("mean:", round(s.mean(), 3))
print("above 0.5:\\n", s[s > 0.5])'''},
            {"t": "h", "text": "Building a DataFrame"},
            {"t": "p", "text":
                "You will usually read a DataFrame with `pd.read_csv(...)`, but "
                "constructing one from a dict is the clearest way to learn it. "
                "Think of each key as a column of a dataset."},
            {"t": "code", "run": True, "caption": "A small dataset as a DataFrame",
             "code": '''import pandas as pd

df = pd.DataFrame({
    "text_len": [12, 45, 7, 33, 20],
    "label":    [0, 1, 0, 1, 0],
    "source":   ["web", "book", "web", "book", "web"],
})
print(df)
print("\\ndtypes:\\n", df.dtypes)
print("\\nshape:", df.shape)'''},
            {"t": "h", "text": "Inspecting: head and describe"},
            {"t": "p", "text":
                "The first thing you do with any dataset is look at it. `head` "
                "shows the top rows; `describe` gives summary statistics for the "
                "numeric columns, an instant sanity check on ranges and scale."},
            {"t": "code", "run": True, "caption": "Quick look at the data",
             "code": '''import pandas as pd

df = pd.DataFrame({
    "text_len": [12, 45, 7, 33, 20],
    "label":    [0, 1, 0, 1, 0],
})
print(df.head(2))
print("\\nsummary:\\n", df.describe().round(2))'''},
            {"t": "h", "text": "Selecting with loc and iloc"},
            {"t": "p", "text":
                "`loc` selects by label, `iloc` selects by integer position. "
                "Pick columns with a list, and rows with a slice or a mask. This "
                "is how you separate features from labels before training."},
            {"t": "code", "run": True, "caption": "Rows, columns, features vs labels",
             "code": '''import pandas as pd

df = pd.DataFrame({
    "text_len": [12, 45, 7, 33, 20],
    "n_words":  [2, 8, 1, 6, 4],
    "label":    [0, 1, 0, 1, 0],
})
X = df[["text_len", "n_words"]]      # feature columns
y = df["label"]                       # target column
print("features:\\n", X.iloc[:2])
print("\\nfirst label by iloc:", y.iloc[0])
print("label via loc:", df.loc[3, "label"])'''},
            {"t": "h", "text": "Filtering rows by condition"},
            {"t": "p", "text":
                "Boolean masks work on DataFrames just like on NumPy arrays, and "
                "you can combine conditions with `&` and `|` (each condition in "
                "parentheses). This is your everyday tool for subsetting data."},
            {"t": "code", "run": True, "caption": "Boolean filtering",
             "code": '''import pandas as pd

df = pd.DataFrame({
    "text_len": [12, 45, 7, 33, 20],
    "label":    [0, 1, 0, 1, 0],
})
long_pos = df[(df["text_len"] > 15) & (df["label"] == 1)]
print(long_pos)
print("\\ncount per label:\\n", df["label"].value_counts())'''},
            {"t": "note", "text":
                "Why it matters for AI: real projects spend most of their time in "
                "data wrangling, not modeling. `head`/`describe`/`loc`/masks are "
                "how you understand and slice a dataset before a single tensor is "
                "created, and `df[features]`, `df[target]` is the split every "
                "training script begins with."},
        ],
    },
    # ==================================================================
    {
        "title": "pandas Cleaning, Grouping and Feature Engineering",
        "blocks": [
            {"t": "p", "text":
                "Raw data is messy: missing values, wrong scales, categorical "
                "strings a model cannot consume. This chapter covers the "
                "transforms that turn a table into model-ready features."},
            {"t": "h", "text": "Missing values"},
            {"t": "p", "text":
                "`isna` finds gaps, `fillna` imputes them (often with the column "
                "mean), and `dropna` removes them. Leaving NaNs in will poison "
                "your gradients, so this is step one of cleaning."},
            {"t": "code", "run": True, "caption": "Detect and fill missing values",
             "code": '''import pandas as pd
import numpy as np

df = pd.DataFrame({"age": [25, np.nan, 40, np.nan, 30]})
print("missing per column:\\n", df.isna().sum())
df["age"] = df["age"].fillna(df["age"].mean())
print("\\nafter fillna:\\n", df.round(1))'''},
            {"t": "h", "text": "New columns with vectorized ops and apply"},
            {"t": "p", "text":
                "Feature engineering is creating new columns from existing ones. "
                "Prefer vectorized expressions; use `map`/`apply` for logic that "
                "does not vectorize cleanly. Both return a new Series you assign "
                "back to the frame."},
            {"t": "code", "run": True, "caption": "Derived features",
             "code": '''import pandas as pd

df = pd.DataFrame({"chars": [40, 10, 200, 75]})
df["words"] = (df["chars"] / 5).round().astype(int)   # vectorized
df["bucket"] = df["chars"].apply(
    lambda c: "long" if c > 100 else "short")          # apply
print(df)'''},
            {"t": "h", "text": "Grouping and aggregation"},
            {"t": "p", "text":
                "`groupby` splits rows into groups and aggregates each, the "
                "table equivalent of a pivot report. It answers questions like "
                "\"what is the mean label per data source?\", essential for "
                "spotting bias and class imbalance."},
            {"t": "code", "run": True, "caption": "groupby + agg",
             "code": '''import pandas as pd

df = pd.DataFrame({
    "source": ["web", "book", "web", "book", "web"],
    "len":    [12, 45, 7, 33, 20],
    "label":  [0, 1, 0, 1, 0],
})
summary = df.groupby("source").agg(
    n=("label", "size"),
    mean_len=("len", "mean"),
    pos_rate=("label", "mean"),
)
print(summary.round(2))'''},
            {"t": "h", "text": "One-hot encoding categoricals"},
            {"t": "p", "text":
                "Models need numbers, not strings. `pd.get_dummies` expands a "
                "categorical column into 0/1 indicator columns, one per category, "
                "the standard encoding for non-ordinal features."},
            {"t": "code", "run": True, "caption": "get_dummies for categorical features",
             "code": '''import pandas as pd

df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
onehot = pd.get_dummies(df, columns=["color"], dtype=int)
print(onehot)'''},
            {"t": "h", "text": "Scaling a numeric column"},
            {"t": "p", "text":
                "Features on wildly different scales make optimization hard. "
                "Standardizing each numeric column to zero mean and unit "
                "variance, exactly as in the NumPy chapter, is a routine final "
                "step before handing data to a model."},
            {"t": "code", "run": True, "caption": "Standardize a feature column",
             "code": '''import pandas as pd

df = pd.DataFrame({"income": [30000, 80000, 55000, 120000]})
col = df["income"]
df["income_z"] = (col - col.mean()) / col.std()
print(df.round(3))'''},
            {"t": "note", "text":
                "Why it matters for AI: fillna, get_dummies and standardization "
                "are the classic tabular preprocessing pipeline, and scikit-learn "
                "transformers do exactly these steps. groupby is your fastest way "
                "to audit a dataset for imbalance and leakage before training."},
        ],
    },
    # ==================================================================
    {
        "title": "Visualization and the Deep-Learning Frameworks",
        "blocks": [
            {"t": "p", "text":
                "This closing chapter connects the Python features from the whole "
                "book to the frameworks you will actually use. The punchline: "
                "PyTorch and friends invent very little new *language*, they lean "
                "on dunder methods, decorators, context managers and operator "
                "overloading you have already seen."},
            {"t": "h", "text": "Plotting a loss curve (matplotlib, illustrative)"},
            {"t": "p", "text":
                "matplotlib is the default way to visualize training. The snippet "
                "below is illustrative (matplotlib is not installed in this "
                "build), but this is the exact idiom for saving a loss curve to "
                "a file during training."},
            {"t": "code", "run": False, "caption": "Illustrative: save a training loss curve",
             "code": '''import matplotlib.pyplot as plt   # illustrative, not executed here

steps = list(range(0, 100, 10))
loss = [2.5, 1.8, 1.3, 1.0, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5]

plt.figure(figsize=(5, 3))
plt.plot(steps, loss, marker="o")
plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss")
plt.tight_layout()
plt.savefig("loss.png", dpi=150)   # write to disk, view later'''},
            {"t": "h", "text": "The nn.Module pattern is just __init__ + __call__"},
            {"t": "p", "text":
                "A PyTorch layer is a class: `__init__` registers parameters, "
                "`forward` defines the math, and calling the instance `layer(x)` "
                "works because `nn.Module` implements `__call__` (which then "
                "invokes `forward`). This is the *exact* dunder pattern from the "
                "objects chapter, illustrative below since torch is not installed."},
            {"t": "code", "run": False, "caption": "Illustrative: a real PyTorch module",
             "code": '''import torch                      # illustrative, not executed here
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))    # same relu as our NumPy one
        return self.fc2(x)

model = MLP(3, 8, 2)
y = model(torch.randn(4, 3))           # __call__ -> forward
print(y.shape)                         # torch.Size([4, 2])'''},
            {"t": "h", "text": "Decorators and context managers in real training"},
            {"t": "p", "text":
                "Turning off gradient tracking during evaluation uses the same "
                "two features from earlier chapters: `@torch.no_grad()` as a "
                "**decorator** on a function, or `with torch.no_grad():` as a "
                "**context manager**. Same tool, two syntaxes."},
            {"t": "code", "run": False, "caption": "Illustrative: no_grad as decorator and context",
             "code": '''import torch                      # illustrative, not executed here

@torch.no_grad()                       # decorator form
def evaluate(model, x):
    return model(x).argmax(dim=-1)

# ... inside a training loop ...
for step in range(100):
    with torch.no_grad():              # context-manager form
        val_preds = model(val_x)       # no autograd graph built
    # train step here would use gradients as normal'''},
            {"t": "h", "text": "A verified gradient-descent step in NumPy"},
            {"t": "p", "text":
                "To keep this chapter grounded, here is a real (executed) "
                "training step for linear regression in pure NumPy: forward "
                "pass, mean-squared-error loss, analytic gradient, and a weight "
                "update. Watch the loss fall, this is what autograd automates."},
            {"t": "code", "run": True, "caption": "One epoch loop of gradient descent",
             "code": '''import numpy as np
rng = np.random.default_rng(0)

X = rng.normal(size=(50, 3))
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + 0.1 * rng.normal(size=50)   # noisy targets

w = np.zeros(3)
lr = 0.1
for step in range(0, 101):
    pred = X @ w                     # forward
    err = pred - y
    loss = np.mean(err ** 2)         # MSE
    grad = 2.0 / len(y) * (X.T @ err)   # gradient
    w -= lr * grad                   # update
    if step % 25 == 0:
        print(f"step {step:3d}  loss={loss:.4f}")
print("learned w:", np.round(w, 2))'''},
            {"t": "h", "text": "The whole book, in one sentence"},
            {"t": "bullets", "items": [
                "**Containers & comprehensions** build datasets and batches.",
                "**Generators** stream data too big for memory.",
                "**Dunder methods** (`__call__`) make models callable like functions.",
                "**Decorators & context managers** toggle behavior like `no_grad`.",
                "**Dataclasses & type hints** describe configs and tensors.",
                "**NumPy** vectorization is the math every framework accelerates.",
            ]},
            {"t": "note", "text":
                "Why it matters for AI: once you see that `model(x)` is "
                "`__call__`, that `@torch.no_grad()` is a decorator, and that a "
                "tensor is an ndarray with autograd, framework code stops looking "
                "like magic and starts looking like the Python you already know."},
        ],
    },
]
