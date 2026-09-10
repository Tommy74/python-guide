"""Part IX - Capstone: A Tiny Transformer LLM.

Two hands-on chapters that build, train and run a character-level GPT-style
transformer small enough to train on a laptop CPU in about ten seconds. The
training corpus is a short passage taken from the WildFly Installation Guide
(https://docs.wildfly.org/41/Installation_Guide.html). These two chapters are
the only ones in the book that require PyTorch: `pip install torch`.
"""

PART = "Capstone: A Tiny Transformer LLM"

# The code blocks below are executed at build time exactly like every other
# runnable example, so the losses and generated text you see were produced by
# actually training the model. The training chapter saves a checkpoint to the
# system temp directory; the inference chapter loads it.

_TRAIN_CODE = '''\
import os, tempfile, time
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# 1. A tiny corpus taken from the WildFly Installation Guide.
CORPUS = (
    "Downloading the WildFly release zip and unzipping it is the "
    "traditional way to install a complete WildFly server with support "
    "for both standalone and managed domain operating modes. "
    "Galleon provisioning tooling allows you to construct a customized "
    "WildFly installation according to your application needs. "
    "This method suits those concerned about server size and memory "
    "footprint who only need specific Jakarta or MicroProfile APIs. "
    "The WildFly Maven Plugin enables developers building applications "
    "with Maven to provision customized installations directly within "
    "their build process. A bootable JAR contains both a customized "
    "WildFly server and your deployment, and can be run with a simple "
    "Java command, making it ideal for microservices architectures."
)

# 2. Character-level tokenizer: every character is a token.
chars = sorted(set(CORPUS))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(CORPUS), dtype=torch.long)
print(f"corpus chars: {len(CORPUS)} | vocab: {vocab_size}")

# 3. Tiny, laptop-friendly hyperparameters.
block_size = 64          # context length (chars the model can look back on)
batch_size = 32
n_embd = 96              # embedding width
n_head = 3               # attention heads
n_layer = 2             # transformer blocks
steps = 600
lr = 3e-3

def get_batch():
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y                       # y is x shifted by one (next char)

# 4. One transformer block: causal self-attention + a small MLP,
#    each wrapped in a residual connection with LayerNorm.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        T = x.size(1)
        # causal mask: position t may not attend to positions > t
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a                      # residual
        x = x + self.mlp(self.ln2(x))  # residual
        return x

# 5. The full model: token + position embeddings, blocks, and a head
#    that projects back to vocab-sized logits.
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

model = TinyGPT()
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. Training loop: predict the next character, minimise cross-entropy.
opt = torch.optim.AdamW(model.parameters(), lr=lr)
t0 = time.time()
for step in range(steps):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 150 == 0 or step == steps - 1:
        print(f"step {step:4d} | loss {loss.item():.3f}")
print(f"trained in {time.time() - t0:.1f}s")

# 7. Save a checkpoint the inference chapter will load.
ckpt = os.path.join(tempfile.gettempdir(), "tiny_wildfly_llm.pt")
torch.save({"model": model.state_dict(), "chars": chars,
            "config": dict(block_size=block_size, n_embd=n_embd,
                           n_head=n_head, n_layer=n_layer)}, ckpt)
print("saved checkpoint:", os.path.basename(ckpt))

# 8. Greedy sanity check: does it reproduce the guide's style?
model.eval()
idx = torch.tensor([encode("WildFly")], dtype=torch.long)
for _ in range(80):
    logits, _ = model(idx[:, -block_size:])
    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, nxt], dim=1)
print("sample:", repr(decode(idx[0].tolist())))
'''

_TOKENIZER_CODE = '''\
# A character tokenizer is the simplest possible tokenizer: the vocabulary
# is just the set of distinct characters, and each maps to an integer id.
text = "WildFly"
chars = sorted(set("Downloading WildFly and running it."))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
ids = [stoi[c] for c in text]
print("vocab size :", len(chars))
print("encode     :", ids)
print("decode back:", "".join(itos[i] for i in ids))
'''

_MASK_CODE = '''\
import torch
# The causal mask stops each position from attending to future tokens.
# True means "blocked". Row t can only see columns 0..t (the lower triangle).
T = 5
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
print(mask.int())
'''

_INFER_CODE = '''\
import os, tempfile
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(0)

# 1. Load the checkpoint saved by the training chapter.
ckpt_path = os.path.join(tempfile.gettempdir(), "tiny_wildfly_llm.pt")
if not os.path.exists(ckpt_path):
    print("no checkpoint found - run the training chapter first")
    raise SystemExit(0)

# weights_only=False uses pickle; safe here because WE wrote the file.
# (See Part VI: never unpickle a checkpoint from an untrusted source.)
ckpt = torch.load(ckpt_path, weights_only=False)
chars = ckpt["chars"]
cfg = ckpt["config"]
vocab_size = len(chars)
block_size = cfg["block_size"]
n_embd, n_head, n_layer = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# 2. Re-declare the same architecture so the weights can be loaded into it.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

model = TinyGPT()
model.load_state_dict(ckpt["model"])
model.eval()
print("loaded", sum(p.numel() for p in model.parameters()), "parameters")

# 3. Autoregressive generation: predict one char, append it, repeat.
@torch.no_grad()
def generate(prompt, n=90, temperature=1.0, top_k=None, greedy=False):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(n):
        logits = model(idx[:, -block_size:])[:, -1, :]   # last position
        if greedy:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature        # sharpen/flatten
            if top_k is not None:                # keep only top-k choices
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist())

print("\\n-- greedy (deterministic) --")
print(generate("Galleon", greedy=True))
print("\\n-- temperature=0.8 sampling --")
print(generate("Galleon", temperature=0.8))
print("\\n-- temperature=1.0, top_k=5 --")
print(generate("The WildFly", temperature=1.0, top_k=5))
'''


CHAPTERS = [
    # ------------------------------------------------------------------
    {
        "title": "Building and Training a Tiny Transformer LLM",
        "blocks": [
            {"t": "p", "text":
                "This capstone pulls the whole book together: classes and "
                "`__call__` (Part IV), tensors and matrix math (Part VIII) and "
                "a training loop all combine into a small **GPT-style "
                "transformer** you can train on a laptop CPU in about ten "
                "seconds. It is a *character-level language model*: it reads "
                "text one character at a time and learns to predict the next "
                "character. Train it on a passage and it learns that passage's "
                "style well enough to continue it."},
            {"t": "note", "text":
                "These two capstone chapters are the only ones that need "
                "PyTorch. Install it first with `pip install torch` (the CPU "
                "build is enough). Everything here runs on CPU; no GPU is "
                "required."},
            {"t": "h", "text": "Step 1 - a character tokenizer"},
            {"t": "p", "text":
                "A model works on numbers, not text, so first we map each "
                "character to an integer id and back. Real LLMs use subword "
                "tokenizers (BPE), but characters keep the vocabulary tiny and "
                "the idea identical: `encode` turns text into ids, `decode` "
                "turns ids back into text."},
            {"t": "code", "run": True, "caption": "A character-level tokenizer",
             "code": _TOKENIZER_CODE},
            {"t": "h", "text": "Step 2 - the causal mask"},
            {"t": "p", "text":
                "A language model must predict the next token from **only the "
                "tokens before it**, never from the future. Self-attention "
                "enforces this with a *causal mask* that blocks each position "
                "from attending to later positions, the upper triangle of the "
                "attention matrix."},
            {"t": "code", "run": True, "caption": "The look-ahead mask (1 = blocked)",
             "code": _MASK_CODE},
            {"t": "h", "text": "Step 3 - the model, in one screen"},
            {"t": "p", "text":
                "The transformer is small enough to read top to bottom. Each "
                "input id becomes a learned **token embedding**; a **position "
                "embedding** tells the model where each character sits. Two "
                "**blocks** follow, each doing masked multi-head "
                "self-attention (letting characters exchange information) then "
                "a small MLP, both wrapped in residual connections with "
                "`LayerNorm`. A final linear **head** turns the result into one "
                "score (logit) per vocabulary character."},
            {"t": "bullets", "items": [
                "`nn.Embedding` - lookup tables for tokens and positions.",
                "`nn.MultiheadAttention` - the attention mechanism, with our "
                "causal `attn_mask`.",
                "residual `x = x + sublayer(x)` - keeps gradients flowing "
                "through a deep stack.",
                "`F.cross_entropy` - the next-character prediction loss.",
            ]},
            {"t": "h", "text": "Step 4 - train it"},
            {"t": "p", "text":
                "Training draws random chunks of the corpus, asks the model to "
                "predict each chunk shifted by one character, and nudges the "
                "weights with `AdamW`. Watch the loss fall from ~3.8 (random "
                "guessing over the vocabulary) to well under 0.1 as the model "
                "memorises the passage. The full program below is what actually "
                "ran to produce the output panel."},
            {"t": "code", "run": True, "caption": "Train a character-level GPT on the WildFly guide",
             "code": _TRAIN_CODE},
            {"t": "p", "text":
                "The final line shows greedy continuation from the prompt "
                "`WildFly`: after a few seconds of training the model "
                "reproduces the guide's wording almost verbatim, which is "
                "exactly what we expect when a model with enough capacity "
                "overfits a very small corpus."},
            {"t": "note", "text":
                "Why it matters for AI: every large language model, from a "
                "few-million-parameter demo to a frontier model, is this same "
                "loop at scale, predict the next token, backpropagate the "
                "cross-entropy loss, repeat, on far more data, layers and "
                "compute. Understanding this 100-line version demystifies the "
                "rest."},
        ],
    },
    # ------------------------------------------------------------------
    {
        "title": "Running Inference with the Trained Transformer",
        "blocks": [
            {"t": "p", "text":
                "Training produced a checkpoint; **inference** is where we put "
                "it to work. Text generation is *autoregressive*: feed the "
                "prompt, take the model's prediction for the next character, "
                "append it, and feed the longer sequence back in, one "
                "character at a time. This chapter loads the saved model and "
                "generates text three different ways."},
            {"t": "h", "text": "Loading a saved model"},
            {"t": "p", "text":
                "A checkpoint stores the learned weights (`state_dict`), not "
                "the code. To use it you re-create the same architecture and "
                "load the weights into it, then switch to `eval()` mode. We "
                "also saved the vocabulary and config so the tokenizer and "
                "model shapes match exactly."},
            {"t": "h", "text": "Greedy, temperature and top-k sampling"},
            {"t": "p", "text":
                "The model outputs a score for every possible next character. "
                "How you pick from those scores controls the output:"},
            {"t": "bullets", "items": [
                "**Greedy** - always take the highest-scoring character. "
                "Deterministic, but repetitive.",
                "**Temperature** - divide the logits by a temperature before "
                "sampling. Below 1.0 makes the model more confident and "
                "focused; above 1.0 makes it more random and creative.",
                "**Top-k** - sample only from the k most likely characters, "
                "cutting off the unlikely long tail that produces gibberish.",
            ]},
            {"t": "p", "text":
                "These are the very same knobs (`temperature`, `top_k`, "
                "`top_p`) you set on a real LLM API. The program below loads "
                "the checkpoint and shows all three strategies."},
            {"t": "code", "run": True, "caption": "Generate text from the trained model",
             "code": _INFER_CODE},
            {"t": "p", "text":
                "Greedy decoding reproduces the guide's text deterministically; "
                "the sampled variants stay on-topic but vary their wording. On "
                "a corpus this small the model mostly recites what it "
                "memorised, exactly how a large model, trained on far more "
                "text, instead learns general language patterns it can "
                "recombine into novel sentences."},
            {"t": "note", "text":
                "Why it matters for AI: serving a model is this load-then-"
                "generate loop, and the sampling strategy is what separates a "
                "flat, repetitive assistant from a fluent one. The decoding "
                "knobs you just implemented by hand are precisely the "
                "`temperature` and `top_k`/`top_p` parameters exposed by every "
                "production LLM API."},
        ],
    },
]
