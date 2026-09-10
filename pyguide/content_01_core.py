"""Part I - Python Core for AI."""

PART = "Python Core for AI"

CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Numbers, Booleans and the Object Model",
        "blocks": [
            {"t": "p", "text":
                "Every value in Python is an **object** with a type, and even "
                "integers have unbounded precision. In AI you constantly move "
                "between Python's `int`/`float` and the fixed-width numeric "
                "types used by tensors (`float32`, `int64`), so it helps to "
                "know what the pure-Python side actually guarantees."},
            {"t": "h", "text": "Integers never overflow"},
            {"t": "p", "text":
                "Unlike C or the fixed-width dtypes inside NumPy, a Python "
                "`int` grows as large as memory allows. This is why counting "
                "tokens or computing large factorials in plain Python never "
                "silently wraps around."},
            {"t": "code", "run": True, "caption": "Arbitrary precision integers",
             "code":
                "n = 2 ** 200\n"
                "print(n)\n"
                "print('bit length:', n.bit_length())\n"
                "# floor division and modulo, useful for batching\n"
                "samples, batch = 1003, 32\n"
                "print('full batches:', samples // batch, 'remainder:', samples % batch)"},
            {"t": "h", "text": "Floats, and why comparisons need tolerance"},
            {"t": "p", "text":
                "Floating-point math is inexact. Loss values and probabilities "
                "almost never compare exactly, so you test *closeness* with "
                "`math.isclose` (or `numpy.allclose` for arrays) instead of "
                "`==`."},
            {"t": "code", "run": True, "caption": "Never compare floats with ==",
             "code":
                "import math\n"
                "a = 0.1 + 0.2\n"
                "print('a =', a)\n"
                "print('a == 0.3 ?', a == 0.3)\n"
                "print('isclose  ?', math.isclose(a, 0.3, rel_tol=1e-9))"},
            {"t": "h", "text": "Booleans are integers"},
            {"t": "p", "text":
                "`True` and `False` are `1` and `0`. Summing a list of boolean "
                "conditions is the idiomatic way to *count* how many are true, "
                "which is exactly how you compute accuracy from a list of "
                "correct/incorrect predictions."},
            {"t": "code", "run": True, "caption": "Counting correct predictions",
             "code":
                "preds  = [1, 0, 1, 1, 0, 1]\n"
                "labels = [1, 0, 0, 1, 0, 1]\n"
                "correct = sum(p == y for p, y in zip(preds, labels))\n"
                "accuracy = correct / len(labels)\n"
                "print(f'{correct} correct out of {len(labels)}')\n"
                "print(f'accuracy = {accuracy:.2%}')"},
            {"t": "note", "text":
                "Why it matters for AI: metrics like accuracy, precision and "
                "recall are all just sums of boolean conditions divided by a "
                "count. Master `sum(cond for ...)` and you can write most "
                "metrics from scratch without a library."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Strings and f-strings for Text Data",
        "blocks": [
            {"t": "p", "text":
                "Text is the raw material of NLP and of every LLM prompt. "
                "Python strings are immutable Unicode sequences, and f-strings "
                "are the fastest, most readable way to build prompts and format "
                "numbers in logs."},
            {"t": "h", "text": "f-strings: interpolation and formatting"},
            {"t": "code", "run": True, "caption": "Building a prompt and formatting metrics",
             "code":
                "role, topic = 'assistant', 'linear algebra'\n"
                "prompt = f'You are a helpful {role}. Explain {topic} simply.'\n"
                "print(prompt)\n\n"
                "loss, lr = 0.03718, 3e-4\n"
                "# format spec: fixed decimals, scientific, padding\n"
                "print(f'loss={loss:.4f}  lr={lr:.1e}')\n"
                "for step in [1, 50, 1000]:\n"
                "    print(f'step {step:>5} | {step/1000:6.1%} done')"},
            {"t": "p", "text":
                "The `=` specifier is a debugging superpower: `f'{x=}'` prints "
                "both the expression and its value."},
            {"t": "code", "run": True, "caption": "Self-documenting debug output",
             "code":
                "temperature = 0.7\n"
                "top_p = 0.95\n"
                "print(f'{temperature=}, {top_p=}')"},
            {"t": "h", "text": "Common text-cleaning methods"},
            {"t": "code", "run": True, "caption": "Normalising text before tokenising",
             "code":
                "raw = '  Hello,  WORLD!  \\n'\n"
                "clean = raw.strip().lower().replace(',', '')\n"
                "print(repr(clean))\n"
                "tokens = clean.split()\n"
                "print('tokens:', tokens)\n"
                "print('joined:', '|'.join(tokens))\n"
                "print('starts with hi?', clean.startswith('hi'))"},
            {"t": "note", "text":
                "Why it matters for AI: prompt templates, chat formatting and "
                "cleaning scraped text are all string work. f-strings keep "
                "prompt-building readable, and `str.strip/split/join/replace` "
                "cover most preprocessing before a real tokenizer runs."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Lists, Tuples, Sets and Dicts",
        "blocks": [
            {"t": "p", "text":
                "Four built-in containers carry almost all data in Python "
                "before it becomes a tensor. Knowing which to reach for is half "
                "of writing clean data code: **lists** for ordered, mutable "
                "sequences (a batch of samples); **tuples** for fixed, "
                "immutable records (an array's `shape`); **sets** for fast "
                "membership and de-duplication (a vocabulary); **dicts** for "
                "keyed lookup (token to id, config values)."},
            {"t": "h", "text": "Lists: the workhorse sequence"},
            {"t": "code", "run": True, "caption": "Building and slicing a list of samples",
             "code":
                "data = [5, 2, 9, 1, 7, 3]\n"
                "data.append(8)\n"
                "print('list :', data)\n"
                "print('first 3 :', data[:3])     # slice\n"
                "print('last 2  :', data[-2:])\n"
                "print('every 2nd:', data[::2])\n"
                "print('reversed :', data[::-1])"},
            {"t": "h", "text": "Tuples: immutable, hashable records"},
            {"t": "p", "text":
                "Tuples cannot change after creation, which makes them safe to "
                "use as dict keys and as the fixed shapes and coordinates you "
                "see all over NumPy and PyTorch."},
            {"t": "code", "run": True, "caption": "Tuples as shapes and dict keys",
             "code":
                "shape = (32, 3, 224, 224)   # batch, channels, H, W\n"
                "batch, channels, h, w = shape   # unpacking\n"
                "print('channels =', channels, '| pixels =', h * w)\n"
                "cache = {(0, 0): 'origin', (1, 2): 'point'}  # tuple keys\n"
                "print(cache[(1, 2)])"},
            {"t": "h", "text": "Sets: membership and de-duplication"},
            {"t": "code", "run": True, "caption": "Building a vocabulary with a set",
             "code":
                "words = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']\n"
                "vocab = set(words)\n"
                "print('unique   :', sorted(vocab))\n"
                "print('size     :', len(vocab))\n"
                "print(\"'cat' in?  :\", 'cat' in vocab)   # O(1) lookup\n"
                "a, b = {1, 2, 3}, {2, 3, 4}\n"
                "print('overlap  :', a & b, '| union:', a | b)"},
            {"t": "h", "text": "Dicts: keyed lookup and the config pattern"},
            {"t": "p", "text":
                "Dicts map keys to values with O(1) lookup and preserve "
                "insertion order. They back token-to-id vocabularies, "
                "hyperparameter configs, and the `**kwargs` you pass into "
                "models."},
            {"t": "code", "run": True, "caption": "A config dict and safe access",
             "code":
                "config = {'lr': 3e-4, 'epochs': 10, 'model': 'resnet'}\n"
                "print('lr        :', config['lr'])\n"
                "print('dropout?  :', config.get('dropout', 0.0))  # default\n"
                "config['batch_size'] = 64          # add a key\n"
                "for key, value in config.items():\n"
                "    print(f'  {key:12} = {value}')\n"
                "stoi = {ch: i for i, ch in enumerate('abc')}  # char -> id\n"
                "print('token ids :', stoi)"},
            {"t": "note", "text":
                "Why it matters for AI: a token-to-id `dict`, a `set` "
                "vocabulary, a `list` batch and a `tuple` shape appear in "
                "essentially every data-loading and tokenisation pipeline. "
                "Picking the right container makes that code both faster and "
                "clearer."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Comprehensions and Generator Expressions",
        "blocks": [
            {"t": "p", "text":
                "Comprehensions are Python's signature feature for building a "
                "collection from an iterable in one readable line. In AI code "
                "they replace most manual `for` loops used to transform, filter "
                "and reshape data, and they are faster than the equivalent "
                "loop with `.append()`."},
            {"t": "h", "text": "List comprehensions: map and filter in one line"},
            {"t": "code", "run": True, "caption": "Transform and filter together",
             "code":
                "nums = [1, 2, 3, 4, 5, 6]\n"
                "squares = [x * x for x in nums]\n"
                "evens   = [x for x in nums if x % 2 == 0]\n"
                "labels  = ['pos' if x > 3 else 'neg' for x in nums]\n"
                "print('squares:', squares)\n"
                "print('evens  :', evens)\n"
                "print('labels :', labels)"},
            {"t": "h", "text": "Nested comprehensions: flattening batches"},
            {"t": "code", "run": True, "caption": "Flatten a list of sequences",
             "code":
                "batches = [[1, 2], [3, 4], [5, 6]]\n"
                "flat = [x for batch in batches for x in batch]\n"
                "print('flat  :', flat)\n"
                "# a 3x3 identity-like grid\n"
                "grid = [[1 if i == j else 0 for j in range(3)] for i in range(3)]\n"
                "for row in grid:\n"
                "    print(row)"},
            {"t": "h", "text": "Dict and set comprehensions"},
            {"t": "code", "run": True, "caption": "Invert a vocabulary; collect lengths",
             "code":
                "stoi = {'a': 0, 'b': 1, 'c': 2}\n"
                "itos = {i: ch for ch, i in stoi.items()}   # invert mapping\n"
                "print('id->char:', itos)\n"
                "words = ['hi', 'bye', 'hi', 'ok']\n"
                "lengths = {w: len(w) for w in set(words)}   # dict comp\n"
                "print('lengths :', lengths)"},
            {"t": "h", "text": "Generator expressions: lazy and memory-cheap"},
            {"t": "p", "text":
                "Swap the square brackets for parentheses and you get a "
                "**generator expression**: it produces items one at a time "
                "instead of building the whole list, so it uses almost no "
                "memory. This is how you stream over a dataset too large to fit "
                "in RAM, and it plugs straight into `sum`, `max` and `any`."},
            {"t": "code", "run": True, "caption": "Streaming aggregation without a list",
             "code":
                "import sys\n"
                "n = 1_000_000\n"
                "gen = (x * x for x in range(n))     # nothing computed yet\n"
                "total = sum(gen)                    # consumed lazily\n"
                "print('sum of squares:', total)\n"
                "lst = [x * x for x in range(n)]\n"
                "print('list  bytes ~', sys.getsizeof(lst))\n"
                "print('gen   bytes ~', sys.getsizeof(x*x for x in range(n)))"},
            {"t": "note", "text":
                "Why it matters for AI: comprehensions are the idiomatic way to "
                "preprocess datasets (`[clean(x) for x in raw]`), and generator "
                "expressions let you stream huge corpora with a flat memory "
                "footprint. You will read and write these constantly."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Control Flow, Truthiness, Unpacking and the Walrus",
        "blocks": [
            {"t": "p", "text":
                "Beyond the obvious `if`/`for`/`while`, a handful of small "
                "Python conveniences show up so often in AI code that they are "
                "worth calling out on their own: truthiness, iterable "
                "unpacking, `enumerate`/`zip`, and the walrus operator."},
            {"t": "h", "text": "Truthiness: empty things are False"},
            {"t": "p", "text":
                "Empty containers, `0`, and `None` are all *falsy*. This lets "
                "you write `if batch:` instead of `if len(batch) > 0:`. Note the "
                "important exception for arrays below."},
            {"t": "code", "run": True, "caption": "Falsy values in guards",
             "code":
                "for value in [[], [1], '', 'x', 0, 3, None]:\n"
                "    print(f'{str(value)!r:>6} -> {bool(value)}')\n"
                "batch = []\n"
                "if not batch:\n"
                "    print('skip: empty batch')"},
            {"t": "h", "text": "enumerate and zip: the loop duo"},
            {"t": "code", "run": True, "caption": "Indexing and pairing during iteration",
             "code":
                "names = ['loss', 'acc', 'lr']\n"
                "vals  = [0.31, 0.88, 3e-4]\n"
                "for i, (name, val) in enumerate(zip(names, vals)):\n"
                "    print(f'{i}: {name:5} = {val}')"},
            {"t": "h", "text": "Unpacking with * (star)"},
            {"t": "code", "run": True, "caption": "Splitting head/tail and merging",
             "code":
                "first, *rest = [10, 20, 30, 40]\n"
                "print('first:', first, '| rest:', rest)\n"
                "*init, last = [10, 20, 30, 40]\n"
                "print('init :', init, '| last:', last)\n"
                "a = [1, 2]; b = [3, 4]\n"
                "merged = [*a, *b]                 # splat into a new list\n"
                "cfg = {**{'lr': 1e-3}, 'lr': 3e-4}  # later key wins\n"
                "print('merged:', merged, '| cfg:', cfg)"},
            {"t": "h", "text": "The walrus operator := (assign inside an expression)"},
            {"t": "p", "text":
                "The walrus operator assigns and returns a value in the same "
                "expression, which avoids recomputing or duplicating a call, a "
                "common pattern in read-loops and early-stopping checks."},
            {"t": "code", "run": True, "caption": "Assign-and-test in one step",
             "code":
                "losses = [0.9, 0.6, 0.61, 0.60, 0.60]\n"
                "best = float('inf')\n"
                "for epoch, loss in enumerate(losses):\n"
                "    if (improvement := best - loss) > 0.02:\n"
                "        best = loss\n"
                "        print(f'epoch {epoch}: improved by {improvement:.2f}')\n"
                "    else:\n"
                "        print(f'epoch {epoch}: no significant gain')"},
            {"t": "note", "text":
                "Why it matters for AI: `enumerate`/`zip` drive training loops, "
                "`*`-unpacking merges configs and splits sequences, and the "
                "walrus keeps early-stopping and streaming reads concise. "
                "Caution: never use `if array:` on a NumPy/torch tensor with "
                "more than one element, it raises an ambiguity error, use "
                "`.any()`/`.all()` instead."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Structural Pattern Matching (match / case)",
        "blocks": [
            {"t": "p", "text":
                "Introduced in Python 3.10, `match`/`case` destructures data by "
                "*shape*, not just by value. It is a clean way to dispatch on "
                "the many differently-shaped messages that flow through AI "
                "systems: chat turns, tool-call payloads, config variants and "
                "streaming API events."},
            {"t": "h", "text": "Matching values and structures"},
            {"t": "code", "run": True, "caption": "Dispatch on a chat message shape",
             "code":
                "def render(msg):\n"
                "    match msg:\n"
                "        case {'role': 'system', 'content': c}:\n"
                "            return f'[system] {c}'\n"
                "        case {'role': 'user', 'content': c}:\n"
                "            return f'user> {c}'\n"
                "        case {'role': 'tool', 'name': n}:\n"
                "            return f'(tool call: {n})'\n"
                "        case _:\n"
                "            return '<unknown>'\n"
                "for m in [{'role': 'system', 'content': 'Be brief.'},\n"
                "          {'role': 'user', 'content': 'Hi'},\n"
                "          {'role': 'tool', 'name': 'search'}]:\n"
                "    print(render(m))"},
            {"t": "h", "text": "Capturing, guards and sequence patterns"},
            {"t": "code", "run": True, "caption": "Match on structure with an if-guard",
             "code":
                "def describe(point):\n"
                "    match point:\n"
                "        case (0, 0):\n"
                "            return 'origin'\n"
                "        case (x, 0):\n"
                "            return f'on x-axis at {x}'\n"
                "        case (x, y) if x == y:\n"
                "            return f'diagonal at {x}'\n"
                "        case (x, y):\n"
                "            return f'point ({x}, {y})'\n"
                "for p in [(0, 0), (5, 0), (3, 3), (1, 2)]:\n"
                "    print(p, '->', describe(p))"},
            {"t": "note", "text":
                "Why it matters for AI: LLM tool-calling and streaming responses "
                "arrive as tagged, differently-shaped dicts/events. `match` "
                "reads far better than a long `if/elif` chain when you branch on "
                "the *structure* of a payload."},
        ],
    },
]
