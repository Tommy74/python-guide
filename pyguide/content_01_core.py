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
            {"t": "h", "text": "Rounding, divmod and conversions"},
            {"t": "p", "text":
                "Python's `round` uses **banker's rounding** (round-half-to-"
                "even), which reduces bias when you round many numbers, exactly "
                "what you want for statistics. `divmod` returns the quotient and "
                "remainder together, handy for turning a flat step count into "
                "epochs and offsets. Conversions between `int` and `float` "
                "truncate toward zero, so use `math.floor`/`ceil` when you need "
                "a specific direction."},
            {"t": "code", "run": True, "caption": "Banker's Rounding and Divmod",
             "code":
                "print('round(0.5) =', round(0.5))   # -> 0, ties go to even\n"
                "print('round(1.5) =', round(1.5))   # -> 2, ties go to even\n"
                "print('round(2.675, 2) =', round(2.675, 2))  # float artifact\n"
                "epochs, offset = divmod(1003, 32)\n"
                "print('divmod(1003, 32) =', (epochs, offset))\n"
                "print('int(3.9) =', int(3.9), '| float(7) =', float(7))"},
            {"t": "h", "text": "Infinity, NaN, and complex numbers"},
            {"t": "p", "text":
                "`math.inf` is a useful sentinel for a starting best-loss. "
                "`math.nan` is special: it is **not equal to itself**, which is "
                "how you detect a training run whose loss has diverged to NaN. "
                "Python also has native `complex` numbers (written `2 + 3j`), "
                "which appear in signal processing and Fourier transforms."},
            {"t": "code", "run": True, "caption": "Inf, NaN and Complex",
             "code":
                "import math\n"
                "best = math.inf                     # nothing beats infinity\n"
                "print('0.9 < inf ?', 0.9 < best)\n"
                "nan = math.nan\n"
                "print('nan == nan ?', nan == nan)   # always False!\n"
                "print('isnan       ?', math.isnan(nan))\n"
                "z = 2 + 3j\n"
                "print('complex:', z, '| magnitude:', abs(z))"},
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
            {"t": "h", "text": "Rich format specs: alignment, separators, conversions"},
            {"t": "p", "text":
                "The format spec after `:` is a mini-language. Use `,` or `_` for "
                "thousands separators (readable parameter counts), `<`/`^`/`>` "
                "with a fill character for aligned tables, and the `!r`/`!s` "
                "conversions to force `repr()` or `str()`. A field width can "
                "itself be a variable via a nested `{}`."},
            {"t": "code", "run": True, "caption": "Separators, Fill and Conversions",
             "code":
                "params = 6_738_415_616\n"
                "print(f'params: {params:,}')        # 6,738,415,616\n"
                "print(f'params: {params/1e9:.2f}B')\n"
                "width = 10\n"
                "for name, val in [('loss', 0.31), ('acc', 0.88)]:\n"
                "    print(f'{name:<{width}}|{val:>8.3f}')  # nested width\n"
                "label = 'cat'\n"
                "print(f'raw={label!r} str={label!s}')  # repr vs str"},
            {"t": "h", "text": "str.format and format_map for prompt templates"},
            {"t": "p", "text":
                "When a template lives in a config file or database rather than "
                "in source code, you cannot use an f-string. `str.format` and "
                "`str.format_map` fill named placeholders from arguments or a "
                "dict, the classic way to store reusable prompt templates."},
            {"t": "code", "run": True, "caption": "Filling a Stored Prompt Template",
             "code":
                "template = 'You are a {role}. Answer about {topic}.'\n"
                "print(template.format(role='tutor', topic='calculus'))\n"
                "fields = {'role': 'critic', 'topic': 'poetry'}\n"
                "print(template.format_map(fields))  # fill from a dict"},
            {"t": "h", "text": "Bytes and encoding: what tokenizers really see"},
            {"t": "p", "text":
                "Text on disk and on the wire is **bytes**, not characters. "
                "`str.encode` turns a string into UTF-8 bytes and `bytes.decode` "
                "reverses it. Modern byte-level tokenizers (BPE) operate on "
                "these bytes directly, which is why a single emoji can cost "
                "several tokens."},
            {"t": "code", "run": True, "caption": "Encoding Text to UTF-8 Bytes",
             "code":
                "text = 'caf\\u00e9 \\U0001f600'   # 'cafe' + accent + emoji\n"
                "raw = text.encode('utf-8')\n"
                "print('chars:', len(text), '| bytes:', len(raw))\n"
                "print('utf-8:', raw)\n"
                "print('round-trip:', raw.decode('utf-8') == text)"},
            {"t": "h", "text": "translate, removeprefix and removesuffix"},
            {"t": "p", "text":
                "`str.translate` deletes or remaps many characters in one pass "
                "(fast punctuation stripping). `removeprefix`/`removesuffix` "
                "(3.9+) safely trim known markers such as a `<s>` sentinel or a "
                "file extension, unlike `strip`, which removes *any* of the "
                "given characters."},
            {"t": "code", "run": True, "caption": "Stripping Punctuation and Markers",
             "code":
                "import string\n"
                "drop = str.maketrans('', '', string.punctuation)\n"
                "print('Hello, world!!!'.translate(drop))  # no punctuation\n"
                "tag = '<s>hello</s>'\n"
                "print(tag.removeprefix('<s>').removesuffix('</s>'))"},
            {"t": "h", "text": "textwrap: fitting prompts to a width"},
            {"t": "code", "run": True, "caption": "Wrapping Long Prompt Text",
             "code":
                "import textwrap\n"
                "note = ('Large language models predict the next token given '\n"
                "        'all previous tokens in the context window.')\n"
                "for line in textwrap.wrap(note, width=40):\n"
                "    print(line)\n"
                "print('---')\n"
                "print(textwrap.shorten(note, width=45, placeholder=' ...'))"},
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
            {"t": "h", "text": "Dict tricks: setdefault, merging, sorting by value"},
            {"t": "p", "text":
                "`setdefault` inserts a key with a default only if it is absent, "
                "the one-liner for grouping. The `|` operator merges two dicts "
                "into a new one and `|=` updates in place (3.9+), the modern way "
                "to layer a base config with overrides. To rank a frequency map, "
                "sort its `.items()` by value with a `key` function."},
            {"t": "code", "run": True, "caption": "Merge, Group and Rank a Dict",
             "code":
                "base = {'lr': 1e-3, 'epochs': 10}\n"
                "override = {'lr': 3e-4, 'batch': 64}\n"
                "print('merged:', base | override)   # override wins\n"
                "groups = {}\n"
                "for label, x in [('a', 1), ('b', 2), ('a', 3)]:\n"
                "    groups.setdefault(label, []).append(x)\n"
                "print('groups:', groups)\n"
                "freq = {'the': 9, 'cat': 2, 'sat': 5}\n"
                "ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)\n"
                "print('ranked:', ranked)"},
            {"t": "h", "text": "sort vs sorted, and a tuple immutability caveat"},
            {"t": "p", "text":
                "`list.sort()` reorders in place and returns `None`; `sorted()` "
                "returns a new sorted list and works on any iterable. A subtle "
                "gotcha: a tuple is immutable, but if it *contains* a mutable "
                "object (a list), that inner object can still change, so such a "
                "tuple is no longer hashable and cannot be a dict/set key."},
            {"t": "code", "run": True, "caption": "In-place Sort vs Copy; Tuple Caveat",
             "code":
                "data = [5, 2, 9, 1]\n"
                "print('sorted() copy :', sorted(data), '| original:', data)\n"
                "data.sort(reverse=True)\n"
                "print('after .sort() :', data)\n"
                "t = (1, [2, 3])            # tuple holding a mutable list\n"
                "t[1].append(4)            # the inner list still mutates\n"
                "print('mutated tuple :', t)\n"
                "try:\n"
                "    hash(t)               # unhashable: contains a list\n"
                "except TypeError as e:\n"
                "    print('hash error  :', e)"},
            {"t": "h", "text": "Set algebra and frozenset keys"},
            {"t": "p", "text":
                "Beyond union and intersection, `difference` (`-`) and "
                "`symmetric_difference` (`^`) answer 'what is only in A' and "
                "'what differs between A and B', useful for comparing predicted "
                "vs true label sets. A `frozenset` is an immutable, hashable "
                "set, so it can serve as a dict key, for example caching a value "
                "per unordered feature combination."},
            {"t": "code", "run": True, "caption": "Difference, Symmetric Difference, Frozenset",
             "code":
                "pred = {'cat', 'dog', 'bird'}\n"
                "true = {'cat', 'dog', 'fish'}\n"
                "print('only predicted :', pred - true)\n"
                "print('disagreements  :', pred ^ true)\n"
                "cache = {frozenset({'a', 'b'}): 0.9}\n"
                "print('order-free key :', cache[frozenset({'b', 'a'})])"},
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
        "title": "The collections Module: Counter, defaultdict, deque",
        "blocks": [
            {"t": "p", "text":
                "The standard-library `collections` module adds three "
                "specialised containers that show up constantly in AI and NLP "
                "code: `Counter` for frequency counting, `defaultdict` for "
                "grouping without key checks, and `deque` for fixed-size rolling "
                "buffers. They are pure Python, fast, and dependency-free."},
            {"t": "h", "text": "Counter: token and word frequencies"},
            {"t": "p", "text":
                "`Counter` counts hashable items in one pass. `.most_common(k)` "
                "returns the top-k by frequency, the first step of building a "
                "vocabulary. Counters also support arithmetic, so you can add "
                "the word counts of two documents or subtract a stop-word "
                "profile."},
            {"t": "code", "run": True, "caption": "Counting Tokens With Counter",
             "code":
                "from collections import Counter\n"
                "text = 'the cat sat on the mat the cat ran'\n"
                "counts = Counter(text.split())\n"
                "print('counts     :', counts)\n"
                "print('top 2      :', counts.most_common(2))\n"
                "print('the -> ', counts['the'], '| missing ->', counts['dog'])\n"
                "doc_b = Counter('the dog ran'.split())\n"
                "print('combined   :', counts + doc_b)   # counter arithmetic"},
            {"t": "h", "text": "defaultdict: grouping and accumulating"},
            {"t": "p", "text":
                "A `defaultdict` supplies a default value the first time a key is "
                "touched, so you never write an `if key not in d` check. Use "
                "`defaultdict(list)` to bucket samples by label and "
                "`defaultdict(int)` to accumulate counts or running sums."},
            {"t": "code", "run": True, "caption": "Grouping Samples by Label",
             "code":
                "from collections import defaultdict\n"
                "samples = [('spam', 'buy now'), ('ham', 'hi mom'),\n"
                "           ('spam', 'free cash'), ('ham', 'call me')]\n"
                "by_label = defaultdict(list)\n"
                "for label, msg in samples:\n"
                "    by_label[label].append(msg)   # no key check needed\n"
                "for label, msgs in by_label.items():\n"
                "    print(f'{label}: {msgs}')\n"
                "totals = defaultdict(int)\n"
                "for label, _ in samples:\n"
                "    totals[label] += 1\n"
                "print('class counts:', dict(totals))"},
            {"t": "h", "text": "deque: sliding windows and rolling buffers"},
            {"t": "p", "text":
                "A `deque` (double-ended queue) appends and pops from both ends "
                "in O(1). With `maxlen` it becomes a fixed-size window: pushing a "
                "new item automatically drops the oldest. This is exactly a "
                "rolling loss buffer for smoothing, an n-gram window over tokens, "
                "or a replay buffer in reinforcement learning."},
            {"t": "code", "run": True, "caption": "A Rolling Loss Window With maxlen",
             "code":
                "from collections import deque\n"
                "window = deque(maxlen=3)          # keep only last 3\n"
                "stream = [0.9, 0.7, 0.6, 0.55, 0.5]\n"
                "for loss in stream:\n"
                "    window.append(loss)          # oldest drops off\n"
                "    avg = sum(window) / len(window)\n"
                "    print(f'loss={loss:.2f} window={list(window)} avg={avg:.3f}')"},
            {"t": "note", "text":
                "Why it matters for AI: `Counter` builds vocabularies and "
                "computes class balance, `defaultdict` groups data and tallies "
                "counts without boilerplate, and a `deque(maxlen=...)` is the "
                "canonical sliding window for smoothed metrics, n-grams and "
                "replay buffers. These three cover a huge share of everyday "
                "data-wrangling in ML pipelines."},
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
            {"t": "h", "text": "Dict comp from zip, and the walrus in a comprehension"},
            {"t": "p", "text":
                "Pairing two lists with `zip` inside a dict comprehension is the "
                "idiomatic way to build a mapping from parallel columns. A "
                "walrus (`:=`) lets you compute an expensive value once and both "
                "filter on it and keep it, avoiding a double call."},
            {"t": "code", "run": True, "caption": "Zip Into a Dict; Walrus Filter",
             "code":
                "names = ['loss', 'acc', 'f1']\n"
                "vals  = [0.31, 0.88, 0.79]\n"
                "metrics = {k: v for k, v in zip(names, vals)}\n"
                "print('metrics:', metrics)\n"
                "raw = ['  hi ', '', '  ok', '   ']\n"
                "# strip once, keep only non-empty results\n"
                "kept = [s for x in raw if (s := x.strip())]\n"
                "print('kept   :', kept)"},
            {"t": "h", "text": "Filtering vs conditional value (and not over-nesting)"},
            {"t": "p", "text":
                "Two very different `if`s appear in comprehensions. An `if` at "
                "the *end* **filters** which items survive. An `if/else` in the "
                "*value* position transforms every item but chooses between two "
                "outputs. You can combine them, but if a comprehension needs "
                "more than one level of nesting or a guard, a plain loop is "
                "usually clearer, readability beats cleverness."},
            {"t": "code", "run": True, "caption": "Filter vs Conditional Expression",
             "code":
                "nums = [-2, -1, 0, 1, 2, 3]\n"
                "positives = [x for x in nums if x > 0]        # filter\n"
                "signs = ['pos' if x > 0 else 'non-pos' for x in nums]  # map\n"
                "# combine: transform only the survivors\n"
                "inv = [round(1 / x, 2) for x in nums if x != 0]\n"
                "print('filter :', positives)\n"
                "print('map    :', signs)\n"
                "print('combo  :', inv)"},
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
            {"t": "h", "text": "for/else and while/else"},
            {"t": "p", "text":
                "A loop's `else` block runs only if the loop finished **without "
                "`break`**. This reads naturally as a search: 'scan the items; if "
                "none triggered the break, run the else'. It removes the usual "
                "`found = False` flag."},
            {"t": "code", "run": True, "caption": "Search With a Loop else",
             "code":
                "vocab = ['cat', 'dog', 'bird']\n"
                "target = 'fish'\n"
                "for word in vocab:\n"
                "    if word == target:\n"
                "        print('found:', word)\n"
                "        break\n"
                "else:\n"
                "    print(target, 'is out of vocabulary')"},
            {"t": "h", "text": "Chained comparisons and the ternary"},
            {"t": "p", "text":
                "Python lets you chain comparisons the way maths does: "
                "`0 <= x < 1` is one expression that evaluates `x` once, ideal "
                "for validating a probability or a learning rate. The ternary "
                "`a if cond else b` picks a value inline."},
            {"t": "code", "run": True, "caption": "Range Checks and Inline Choice",
             "code":
                "for p in [-0.1, 0.5, 1.0, 1.4]:\n"
                "    ok = 0.0 <= p <= 1.0          # chained comparison\n"
                "    status = 'valid' if ok else 'OUT OF RANGE'\n"
                "    print(f'p={p:>5} -> {status}')"},
            {"t": "h", "text": "zip(strict=True): catch silent length mismatches"},
            {"t": "p", "text":
                "By default `zip` stops at the shortest input, which can silently "
                "drop data when your features and labels drift out of sync. "
                "Passing `strict=True` (3.10+) raises a `ValueError` instead, a "
                "cheap guard against a subtle data bug."},
            {"t": "code", "run": True, "caption": "Strict Zip Guards Against Drift",
             "code":
                "features = [[1], [2], [3]]\n"
                "labels = [0, 1]              # one label missing!\n"
                "try:\n"
                "    pairs = list(zip(features, labels, strict=True))\n"
                "    print(pairs)\n"
                "except ValueError as e:\n"
                "    print('length mismatch:', e)"},
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
            {"t": "h", "text": "Class patterns with __match_args__"},
            {"t": "p", "text":
                "`match` can destructure objects by type and attribute. A "
                "dataclass supplies `__match_args__` automatically, so you can "
                "write positional class patterns, a clean way to dispatch on "
                "typed events or model configs."},
            {"t": "code", "run": True, "caption": "Matching Dataclass Instances",
             "code":
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class TextEvent:\n"
                "    content: str\n"
                "@dataclass\n"
                "class ToolCall:\n"
                "    name: str\n"
                "    args: dict\n"
                "def handle(ev):\n"
                "    match ev:\n"
                "        case TextEvent(content=c):\n"
                "            return f'text: {c}'\n"
                "        case ToolCall(name=n):\n"
                "            return f'call: {n}'\n"
                "for ev in [TextEvent('hi'), ToolCall('search', {'q': 'x'})]:\n"
                "    print(handle(ev))"},
            {"t": "h", "text": "OR-patterns, as-capture and mapping rest"},
            {"t": "p", "text":
                "The `|` operator merges alternatives in one case, `as` binds the "
                "matched value to a name, and `**rest` captures the leftover keys "
                "of a mapping, exactly what you need when an API payload carries "
                "extra fields you want to keep."},
            {"t": "code", "run": True, "caption": "OR-patterns and Capturing the Rest",
             "code":
                "def classify(event):\n"
                "    match event:\n"
                "        case {'type': ('start' | 'resume') as kind}:\n"
                "            return f'run event: {kind}'\n"
                "        case {'type': 'metric', 'name': n, **rest}:\n"
                "            return f'metric {n}, extra={rest}'\n"
                "        case _:\n"
                "            return 'unhandled'\n"
                "events = [{'type': 'resume'},\n"
                "          {'type': 'metric', 'name': 'loss', 'step': 5}]\n"
                "for e in events:\n"
                "    print(classify(e))"},
            {"t": "note", "text":
                "Why it matters for AI: LLM tool-calling and streaming responses "
                "arrive as tagged, differently-shaped dicts/events. `match` "
                "reads far better than a long `if/elif` chain when you branch on "
                "the *structure* of a payload."},
        ],
    },
]
