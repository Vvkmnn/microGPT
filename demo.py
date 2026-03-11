"""microGPT Live Training Demo — Full-Screen TUI"""

import os
import math
import random
import time
import argparse
from threading import Event

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Header, Footer, ProgressBar
from textual.containers import Horizontal, Vertical, Center
from textual.message import Message
from textual import on, work
from textual.binding import Binding
from textual.reactive import reactive

from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree
from rich.columns import Columns
from rich.syntax import Syntax
from rich import box

# ---
# microGPT engine — Karpathy's code restructured into callable functions
# ---

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# Model hyperparameters
N_LAYER = 1
N_EMBD = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD
LEARNING_RATE = 0.01
BETA1, BETA2, EPS_ADAM = 0.85, 0.99, 1e-8
NUM_STEPS = 222


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_data(logits):
    """Softmax on plain float lists (not Value objects)."""
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def load_data():
    """Download names.txt if needed, return (docs, uchars, BOS, vocab_size)."""
    data_path = os.path.join(os.path.dirname(__file__) or '.', 'input.txt')
    if not os.path.exists(data_path):
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(url, data_path)
    docs = [line.strip() for line in open(data_path) if line.strip()]
    random.seed(42)
    random.shuffle(docs)
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return docs, uchars, BOS, vocab_size


def build_model(vocab_size):
    """Create weight matrices, return (state_dict, params)."""
    random.seed(42)
    matrix = lambda nout, nin, std=0.08: [
        [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
    ]
    sd = {
        'wte': matrix(vocab_size, N_EMBD),
        'wpe': matrix(BLOCK_SIZE, N_EMBD),
        'lm_head': matrix(vocab_size, N_EMBD),
    }
    for i in range(N_LAYER):
        sd[f'layer{i}.attn_wq'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wk'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wv'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wo'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc2'] = matrix(N_EMBD, 4 * N_EMBD)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params


def init_optimizer(params):
    """Create Adam buffers, return (m_buf, v_buf)."""
    return [0.0] * len(params), [0.0] * len(params)


def gpt_forward(token_id, pos_id, keys, values, sd):
    """Forward pass for one token, returns logits list."""
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs:hs+HEAD_DIM]
            k_h = [ki[hs:hs+HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs:hs+HEAD_DIM] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    return linear(x, sd['lm_head'])


def train_step(step, doc, sd, params, m_buf, v_buf, uchars, BOS):
    """One training step: forward + backward + Adam. Returns loss value."""
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt_forward(token_id, pos_id, keys, vals, sd)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    loss = (1 / n) * sum(losses)
    loss.backward()
    lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
    for i, p in enumerate(params):
        m_buf[i] = BETA1 * m_buf[i] + (1 - BETA1) * p.grad
        v_buf[i] = BETA2 * v_buf[i] + (1 - BETA2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - BETA1 ** (step + 1))
        v_hat = v_buf[i] / (1 - BETA2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
        p.grad = 0
    return loss.data


def generate_one(sd, uchars, BOS, vocab_size, temp=0.5, return_logits=False):
    """Sample one name from the model. Returns (name_str, steps_info)."""
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    token_id = BOS
    chars = []
    steps_info = []
    for pos_id in range(BLOCK_SIZE):
        logits = gpt_forward(token_id, pos_id, keys, vals, sd)
        raw_logit_data = [l.data for l in logits] if return_logits else None
        probs = softmax([l / temp for l in logits])
        prob_data = [p.data for p in probs]
        next_id = random.choices(range(vocab_size), weights=prob_data)[0]
        context = ''.join(chars) if chars else '\u27E8BOS\u27E9'
        if next_id == BOS:
            next_ch = '\u27E8BOS\u27E9'
        else:
            next_ch = uchars[next_id]
        steps_info.append({
            'pos': pos_id,
            'context': context,
            'next': next_ch,
            'prob': prob_data[next_id],
            'all_probs': list(zip(range(vocab_size), prob_data)),
            'raw_logits': raw_logit_data,
        })
        if next_id == BOS:
            break
        chars.append(uchars[next_id])
        token_id = next_id
    return ''.join(chars), steps_info


def generate_one_from(ch, sd, uchars, BOS, vocab_size, temp=0.5):
    """Generate a name starting with character ch. Feeds BOS then forces ch as first token."""
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    # Position 0: feed BOS
    gpt_forward(BOS, 0, keys, vals, sd)
    # Position 1: feed the forced first character
    tok = uchars.index(ch)
    chars = [ch]
    token_id = tok
    for pos_id in range(1, BLOCK_SIZE):
        logits = gpt_forward(token_id, pos_id, keys, vals, sd)
        probs = softmax([l / temp for l in logits])
        prob_data = [p.data for p in probs]
        next_id = random.choices(range(vocab_size), weights=prob_data)[0]
        if next_id == BOS:
            break
        chars.append(uchars[next_id])
        token_id = next_id
    return ''.join(chars)


def forward_explained(tokens, sd, uchars, BOS, pos_idx=None):
    """Instrumented forward pass capturing intermediates at pos_idx."""
    if pos_idx is None:
        pos_idx = min(2, len(tokens) - 2)
    token_id = tokens[pos_idx]
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    for p in range(pos_idx):
        gpt_forward(tokens[p], p, keys, vals, sd)
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_idx]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    info = {
        'token_id': token_id,
        'pos_id': pos_idx,
        'token_char': uchars[token_id] if token_id != BOS else '\u27E8BOS\u27E9',
        'tok_emb': [v.data for v in tok_emb[:4]],
        'pos_emb': [v.data for v in pos_emb[:4]],
    }
    x = rmsnorm(x)
    info['after_rmsnorm'] = [v.data for v in x[:4]]
    xn = rmsnorm(x)
    q = linear(xn, sd['layer0.attn_wq'])
    k = linear(xn, sd['layer0.attn_wk'])
    v = linear(xn, sd['layer0.attn_wv'])
    keys[0].append(k)
    vals[0].append(v)
    q_h = q[:HEAD_DIM]
    k_h = [ki[:HEAD_DIM] for ki in keys[0]]
    attn_logits = [
        sum(q_h[j].data * k_h[t][j].data for j in range(HEAD_DIM)) / HEAD_DIM**0.5
        for t in range(len(k_h))
    ]
    mx = max(attn_logits)
    exps = [math.exp(a - mx) for a in attn_logits]
    tot = sum(exps)
    attn_weights = [e / tot for e in exps]
    info['attn_scores'] = attn_logits
    info['attn_weights'] = attn_weights
    logits = gpt_forward(token_id, pos_idx, [[] for _ in range(N_LAYER)],
                         [[] for _ in range(N_LAYER)], sd)
    probs = softmax(logits)
    info['top_probs'] = sorted(
        [(i, p.data) for i, p in enumerate(probs)],
        key=lambda x: -x[1]
    )[:5]
    return info


def forward_stages(token_id, pos_id, sd):
    """Forward pass on plain floats, returning 16-dim vectors at each stage."""
    tok_emb = [v.data for v in sd['wte'][token_id]]
    pos_emb = [v.data for v in sd['wpe'][pos_id]]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    def rms(vec):
        ms = sum(v * v for v in vec) / len(vec)
        s = (ms + 1e-5) ** -0.5
        return [v * s for v in vec]

    def lin(x, w):
        return [sum(row[j].data * x[j] for j in range(len(x))) for row in w]

    x = rms(x)
    after_norm = list(x)

    xn = rms(x)
    v_vec = lin(xn, sd['layer0.attn_wv'])
    x_proj = lin(v_vec, sd['layer0.attn_wo'])
    x = [a + b for a, b in zip(x_proj, after_norm)]
    after_attn = list(x)

    xn = rms(x)
    fc1 = lin(xn, sd['layer0.mlp_fc1'])
    fc1_relu = [max(0.0, v) for v in fc1]
    fc2 = lin(fc1_relu, sd['layer0.mlp_fc2'])
    x = [a + b for a, b in zip(fc2, after_attn)]
    after_mlp = list(x)

    # Output projection: lm_head @ after_mlp -> logits (vocab_size-dim)
    logits = lin(after_mlp, sd['lm_head'])
    probs = softmax_data(logits)

    return [
        ('Token Embedding', tok_emb),
        ('Position Embedding', pos_emb),
        ('RMSNorm', after_norm),
        ('Attention', after_attn),
        ('MLP', after_mlp),
        ('Output (logits)', logits),
        ('Softmax', probs),
    ]


# ---
# TUI helpers
# ---

SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def prob_color(prob):
    """Color style based on probability confidence."""
    if prob > 0.4:
        return "bold green"
    if prob > 0.15:
        return "yellow"
    if prob > 0.05:
        return "dark_orange"
    return "bold red"


def make_bar(prob, width=20, max_val=None):
    """Horizontal probability bar, scaled to max_val (default 1.0)."""
    if max_val and max_val > 0:
        frac = prob / max_val
    else:
        frac = prob
    filled = int(frac * width)
    return "\u2588" * min(filled, width) + "\u2591" * max(0, width - filled)


def vec_to_sparkline(vec):
    """Convert a vector to a sparkline string using block chars."""
    if not vec:
        return ""
    mn, mx = min(vec), max(vec)
    if mx == mn:
        mx = mn + 1
    return "".join(SPARK[min(7, int((v - mn) / (mx - mn) * 7))] for v in vec)


def entropy_bits(probs):
    """Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probs if p > 1e-10)


# Tree structure for Screen 0 progressive reveal
TREE_OPS = [
    ('branch', 'val', "[bold]class Value[/]          [dim](autograd)[/]"),
    ('child', 'val', "[dim]__add__, __mul__, __pow__[/]"),
    ('child', 'val', "[dim]log, exp, relu[/]"),
    ('child', 'val', "[dim]backward()[/]"),
    ('root', None, ""),
    ('root', None, "[bold]linear()[/]             [dim](matmul)[/]"),
    ('root', None, "[bold]softmax()[/]            [dim](normalize)[/]"),
    ('root', None, "[bold]rmsnorm()[/]            [dim](stabilize)[/]"),
    ('root', None, ""),
    ('branch', 'gpt', "[bold]gpt()[/]                [dim](forward pass)[/]"),
    ('child', 'gpt', "[dim]token + position embedding[/]"),
    ('child', 'gpt', "[dim]multi-head attention[/]"),
    ('child', 'gpt', "[dim]MLP (16\u219264\u219216, ReLU)[/]"),
    ('child', 'gpt', "[dim]project to vocab logits[/]"),
    ('root', None, ""),
    ('root', None, "[bold]train loop[/]           [dim](Adam optimizer)[/]"),
    ('root', None, "[bold]inference[/]            [dim](sampling)[/]"),
]


def build_tree(count):
    """Build Rich Tree with first `count` nodes revealed."""
    tree = Tree("demo.py", guide_style="dim")
    branches = {}
    for i in range(min(count, len(TREE_OPS))):
        op, bid, label = TREE_OPS[i]
        if op == 'root':
            tree.add(label)
        elif op == 'branch':
            branches[bid] = tree.add(label)
        elif op == 'child' and bid in branches:
            branches[bid].add(label)
    return tree


# ---
# TUI — Screens and App
# ---

APP_CSS = """
Screen {
    background: $surface;
}

#outer {
    width: 100%;
    height: 100%;
    layout: vertical;
}

#screen-title {
    height: auto;
    max-height: 3;
    content-align: center middle;
    margin: 0 4;
}

#content-area {
    layout: horizontal;
    height: 1fr;
}

#left-pane {
    width: 60%;
    height: 100%;
    margin: 1;
    overflow-y: auto;
    border: round ansi_bright_blue;
    border-title-align: left;
    border-subtitle-align: center;
    padding: 1 1;
}

#right-pane {
    width: 40%;
    height: 100%;
    margin: 1;
    overflow-y: auto;
    border: round ansi_bright_blue;
    border-title-align: left;
    border-subtitle-align: center;
    padding: 1 1;
}

#single-pane {
    width: 100%;
    height: 1fr;
    margin: 1;
    overflow-y: auto;
}

#progress-area {
    height: auto;
    max-height: 3;
    margin: 0 2;
}

#chart-area {
    height: 1fr;
    min-height: 8;
    margin: 0 2;
}

#status-bar {
    dock: bottom;
    height: 1;
    background: $panel;
    color: $text-muted;
    padding: 0 2;
}

#help-overlay {
    display: none;
    layer: overlay;
    width: 100%;
    height: 100%;
    background: $surface 80%;
    align: center middle;
}

#help-overlay.visible {
    display: block;
}

#help-box {
    width: auto;
    max-width: 40;
    height: auto;
    background: $panel;
    border: solid $accent;
    padding: 1 2;
}
"""

HELP_TEXT = """\
[bold]Keys[/bold]

  Enter / Space   next step
  \u2192               new example
  \u2190               previous step
  j / k           scroll content
  ?               this help
  q               quit"""


class HelpOverlay(Static):
    """Centered help overlay, toggled by ?"""
    pass


# --- Custom messages for training progress ---
class TrainProgress(Message):
    def __init__(self, step: int, loss: float):
        super().__init__()
        self.step = step
        self.loss = loss

class TrainDone(Message):
    def __init__(self, elapsed: float):
        super().__init__()
        self.elapsed = elapsed


# ---
# Base screen with shared navigation and timer management
# ---
class DemoScreen(Screen):
    """Base screen with common key handling and timer management."""

    SCREEN_INDEX = 0
    TOTAL_SCREENS = 8
    SCREEN_TITLE = ""
    phase = reactive(0)
    max_phase = 1

    # --- Timer infrastructure ---
    # Note: can't use `_timers` — Textual uses that internally as a set.

    def _stop_all_timers(self):
        for t in getattr(self, '_anim_timers', []):
            try:
                t.stop()
            except Exception:
                pass
        self._anim_timers = []

    def _add_timer(self, timer):
        if not hasattr(self, '_anim_timers'):
            self._anim_timers = []
        self._anim_timers.append(timer)
        return timer

    # --- Screen title ---

    def _set_title(self, title):
        tp = Panel(
            Text(title, style="bold bright_white", justify="center"),
            box=box.HORIZONTALS,
            border_style="bright_cyan",
            padding=(0, 4),
        )
        try:
            self.query_one("#screen-title", Static).update(tp)
        except Exception:
            pass

    # --- Status bar ---

    def _status_text(self, extra=""):
        idx = self.SCREEN_INDEX
        # TitleScreen=0 → no step, DataScreen=1 → "Step 0", rest → Step 1-6
        if idx == 0:
            parts = []
        elif idx == 1:
            parts = ["Step 0/6"]
        else:
            parts = [f"Step {idx - 1}/6"]
        if self.max_phase > 1:
            parts.append(f"\u25cf {self.phase + 1}/{self.max_phase}")
        if extra:
            parts.append(extra)
        keys = ["\u2190/\u2192 prev/next"]
        if idx > 1:
            keys.append("r new example")
        keys.append("Enter advance")
        keys.append("q quit")
        return "  ".join(parts) + "    " + "  ".join(keys)

    def _update_status(self, extra=""):
        try:
            bar = self.query_one("#status-bar", Static)
            bar.update(self._status_text(extra))
        except Exception:
            pass

    def _set_pane(self, pane_id, content, title="", subtitle="", color=None):
        """Update a pane's content and border titles. Color overrides CSS default."""
        try:
            w = self.query_one(pane_id, Static)
            w.border_title = title
            w.border_subtitle = subtitle
            if color:
                w.styles.border = ("round", color)
            w.update(content)
        except Exception:
            pass

    # --- Help ---

    def _show_help(self):
        try:
            overlay = self.query_one("#help-overlay")
            overlay.toggle_class("visible")
        except Exception:
            pass

    # --- Key bindings ---

    def key_question_mark(self):
        self._show_help()

    def key_q(self):
        self.app.exit()

    def key_enter(self):
        self._advance()

    def key_space(self):
        self._advance()

    def key_right(self):
        self._advance()

    def key_left(self):
        self._go_back()

    def key_r(self):
        self._resample()

    def key_j(self):
        self.scroll_down()

    def key_k(self):
        self.scroll_up()

    # --- Navigation ---

    def _advance(self):
        self._stop_all_timers()
        if self.phase < self.max_phase - 1:
            self.phase += 1
            self._on_phase_change()
        else:
            self._next_screen()

    def _on_phase_change(self):
        pass

    def _resample(self):
        pass

    def _go_back(self):
        self._stop_all_timers()
        if self.phase > 0:
            self.phase -= 1
            self._on_phase_change()
        else:
            screens = self.app.screen_order
            idx = self.SCREEN_INDEX
            if idx > 0:
                prev = screens[idx - 1]()
                # Land on the last phase of the previous screen
                prev.phase = prev.max_phase - 1
                self.app.switch_screen(prev)

    def _next_screen(self):
        screens = self.app.screen_order
        idx = self.SCREEN_INDEX
        if idx < len(screens) - 1:
            self.app.switch_screen(screens[idx + 1]())

    # --- Compose & Mount ---

    def compose(self) -> ComposeResult:
        yield from self._compose_content()
        yield Static(id="help-overlay", classes="")
        yield Static(self._status_text(), id="status-bar")

    def on_mount(self):
        self._anim_timers = []
        try:
            overlay = self.query_one("#help-overlay")
            overlay.update(
                Panel(HELP_TEXT, title="Keys", border_style="bright_cyan",
                      box=box.ROUNDED, padding=(1, 2))
            )
        except Exception:
            pass
        self._update_status()
        self._on_screen_mount()

    def on_unmount(self):
        self._stop_all_timers()

    def _compose_content(self) -> ComposeResult:
        yield Static("Override _compose_content")

    def _on_screen_mount(self):
        pass


# ---
# Screen 0: Data — Cascading Name Grid
# ---
class DataScreen(DemoScreen):
    SCREEN_INDEX = 1
    SCREEN_TITLE = "The Data"

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._set_title(self.SCREEN_TITLE)
        docs = self.app.data[0]
        self._all_names = docs
        self._revealed = 0
        # How many names to show per tick (fast cascade)
        self._names_per_tick = max(1, len(docs) // 30)
        self._render_names()
        self._render_explanation()
        self._add_timer(self.set_interval(0.04, self._cascade_tick))

    def _cascade_tick(self):
        self._revealed = min(len(self._all_names),
                             self._revealed + self._names_per_tick)
        self._render_names()
        if self._revealed >= len(self._all_names):
            self._stop_all_timers()
            self._update_status()

    def _render_names(self):
        docs = self._all_names
        total = len(docs)
        shown = self._revealed

        data_text = Text()

        # Get available width for the name grid
        try:
            w = self.query_one("#left-pane", Static).size.width
            avail = max(40, w - 4)
        except Exception:
            avail = 60

        # Build dense grid of names — each name gets a fixed column width
        col_w = 10
        cols = max(1, avail // col_w)

        # Show a representative sample that fills the screen
        # Pick evenly spaced names from the revealed set
        try:
            h = self.query_one("#left-pane", Static).size.height
            max_rows = max(5, h - 4)
        except Exception:
            max_rows = 25

        max_display = cols * max_rows
        if shown <= max_display:
            display_names = docs[:shown]
        else:
            # Show evenly spaced sample from all revealed names
            step = max(1, shown // max_display)
            display_names = docs[:shown:step][:max_display]

        # Color gradient: dim → bright as we fill up
        fill_pct = shown / total if total > 0 else 0

        for i, name in enumerate(display_names):
            # Names cascade from dim to bright
            depth = i / max(1, len(display_names) - 1) if len(display_names) > 1 else 1.0
            if depth > 0.8:
                style = "bold bright_white"
            elif depth > 0.5:
                style = "bright_white"
            elif depth > 0.2:
                style = "white"
            else:
                style = "dim"

            truncated = name[:col_w - 1].ljust(col_w - 1)
            data_text.append(f" {truncated}", style=style)
            if (i + 1) % cols == 0:
                data_text.append("\n")

        # Counter at bottom
        data_text.append(f"\n\n")
        if shown < total:
            data_text.append(f"  Loading... {shown:,}/{total:,} names", style="bright_cyan")
        else:
            data_text.append(f"  {total:,} names loaded", style="bold bright_green")

        self._set_pane("#left-pane", data_text,
                       title=f"Training Data ({shown:,}/{total:,})",
                       subtitle="32,033 real human names from census data.")

    def _render_explanation(self):
        docs = self.app.data[0]
        total = len(docs)

        # Show some stats about the dataset
        lengths = [len(n) for n in docs]
        avg_len = sum(lengths) / len(lengths)
        first_chars = {}
        for n in docs:
            c = n[0].lower()
            first_chars[c] = first_chars.get(c, 0) + 1
        top_starts = sorted(first_chars.items(), key=lambda x: -x[1])[:5]

        text = Text()
        for line in [
            "This is what the model",
            "will learn from.",
            "",
            f"{total:,} real names — the same",
            "dataset Karpathy used in",
            "his makemore project.",
            "",
            "The model has never seen",
            "these names before. After",
            "training, it will learn",
            "the patterns: which letters",
            "follow which, how names",
            "start and end.",
            "",
            f"Average length: {avg_len:.1f} chars",
            "",
            "Most common first letters:",
        ]:
            text.append(line + "\n")
        max_pct = max(c / total for _, c in top_starts) if top_starts else 1
        for ch, count in top_starts:
            pct = count / total * 100
            bar = make_bar(pct / 100, width=15, max_val=max_pct)
            text.append(f"  {ch.upper()} ")
            text.append(f"{bar}", style="bright_cyan")
            text.append(f" {count:,} ({pct:.0f}%)\n")

        self._set_pane("#right-pane", text, title="What")

    def _status_text(self, extra=""):
        return "Step 0/6                                                    Enter begin  ? help"

    def _resample(self):
        # Restart cascade
        self._stop_all_timers()
        self._revealed = 0
        self._add_timer(self.set_interval(0.04, self._cascade_tick))


# ---
# Screen 1: Title — Progressive Tree Reveal + Counting Config
# ---
class TitleScreen(DemoScreen):
    SCREEN_INDEX = 0
    SCREEN_TITLE = "microGPT"

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._set_title(self.SCREEN_TITLE)
        self._reveal_idx = 0
        self._counter_progress = 0.0
        app = self.app
        self._target_params = len(app.model[1]) if app.model else 0
        self._target_docs = len(app.data[0]) if app.data else 0
        self._render_tree()
        self._render_config()
        self._add_timer(self.set_interval(0.15, self._reveal_next_tree))
        self._add_timer(self.set_interval(0.04, self._count_up_config))

    def _reveal_next_tree(self):
        if self._reveal_idx >= len(TREE_OPS):
            # tree fully revealed, stop only tree timer
            if self._anim_timers:
                self._anim_timers[0].stop()
            return
        self._reveal_idx += 1
        self._render_tree()

    def _count_up_config(self):
        self._counter_progress = min(1.0, self._counter_progress + 0.04)
        self._render_config()
        if self._counter_progress >= 1.0 and len(self._anim_timers) > 1:
            self._anim_timers[1].stop()

    def _render_tree(self):
        tree = build_tree(self._reveal_idx)
        self._set_pane("#left-pane", tree,
                       title="Structure",
                       subtitle="Pure Python, no dependencies.")

    def _render_config(self):
        p = self._counter_progress

        def anim(target):
            return int(target * p)

        n_params = anim(self._target_params)
        n_docs = anim(self._target_docs)
        config_text = Text()
        config_lines = [
            f"  {n_params:,} parameters",
            f"  {anim(27)} token vocabulary",
            f"  {anim(N_EMBD)}-dim embeddings",
            f"  {anim(N_HEAD)} attention heads",
            f"  {anim(N_LAYER)} transformer layer",
            f"  {anim(BLOCK_SIZE)} context window",
            f"",
            f"  {n_docs:,} training names",
            f"  {anim(NUM_STEPS)} training steps",
            f"",
            f"  0 dependencies",
            f"  pure Python scalars",
            f"",
            f"  Built by Andrej Karpathy",
            f"  (@karpathy)",
            f"",
            f"  Visualized by Vivek Menon",
            f"  (@vvkmnn)",
        ]
        for line in config_lines:
            config_text.append(line + "\n")
        self._set_pane("#right-pane", config_text,
                       title="Config",
                       subtitle="Complete GPT. Same architecture as GPT-2/3/4.")

    def _status_text(self, extra=""):
        return "                                                              Enter begin  ? help"


# ---
# Screen 1: Tokenization — Animated Character-by-Character + Auto-Cycle
# ---
class TokenizationScreen(DemoScreen):
    SCREEN_INDEX = 2
    SCREEN_TITLE = "Step 1: Tokenization"

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._set_title(self.SCREEN_TITLE)
        self._render_explanation()
        self._start_animation()

    def _render_explanation(self):
        text = Text()
        for line in [
            "The model can't read text",
            "directly \u2014 it only works",
            "with numbers. So we give",
            "each character an ID.",
            "",
            "a=0, b=1, ... z=25.",
            "Plus one special marker",
            "(BOS=26) that signals",
            '"start" and "end" of a name.',
            "",
            "That gives us 27 tokens.",
            "",
            "GPT-4 uses ~100K tokens",
            "for word pieces. We use",
            "the simplest version:",
            "one character = one number.",
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="What")

    def _start_animation(self):
        docs = self.app.data[0]
        self._name = random.choice(docs)
        self._char_idx = 0
        self._add_timer(self.set_interval(0.15, self._reveal_next_char))

    def _reveal_next_char(self):
        if self._char_idx > len(self._name):
            self._stop_all_timers()
            self._add_timer(self.set_timer(1.5, self._restart_animation))
            return
        self._render_partial_tokens()
        self._char_idx += 1

    def _restart_animation(self):
        self._stop_all_timers()
        self._start_animation()

    def _render_partial_tokens(self):
        docs, uchars, BOS, _ = self.app.data
        name = self._name
        idx = self._char_idx
        tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]

        data_text = Text()
        data_text.append(f"  {len(docs):,} names loaded\n\n")
        samples = docs[:5]
        data_text.append(f"  Samples:\n    {', '.join(samples)}\n\n")

        # Vocabulary grid
        data_text.append("  Vocabulary (27 tokens):\n\n")
        for row_start in range(0, len(uchars), 6):
            row_items = []
            for i in range(row_start, min(row_start + 6, len(uchars))):
                row_items.append(f"{i:2d}:{uchars[i]}")
            data_text.append("   " + "  ".join(row_items) + "\n")
        data_text.append(f"  {BOS:2d}:\u27E8BOS\u27E9\n\n")

        # Animated tokenization: show name with highlighting
        # Use 3-char columns so token IDs (up to 2 digits) align under chars
        data_text.append('  "')
        for i, ch in enumerate(name):
            if i < idx:
                data_text.append(f"{ch}  ", style="bold bright_green")
            elif i == idx:
                data_text.append(f"{ch}  ", style="bold reverse")
            else:
                data_text.append(f"{ch}  ", style="dim")
        data_text.append('"\n')

        # Token numbers below revealed chars
        data_text.append("   ")
        for i, ch in enumerate(name):
            if i < idx:
                tok = uchars.index(ch)
                data_text.append(f"{tok:<3d}", style="bright_green")
            elif i == idx:
                tok = uchars.index(ch)
                data_text.append(f"{tok:<3d}", style="bold reverse")
            else:
                data_text.append("   ", style="dim")
        data_text.append("\n")

        # Show full token sequence so far
        if idx > 0:
            revealed = tokens[:idx + 1]
            data_text.append(f"\n  [{', '.join(str(t) for t in revealed)}, ...]\n")

        self._set_pane("#left-pane", data_text,
                       title=f'Tokenizing: "{name}"',
                       subtitle="Each name \u2192 numbers, wrapped with start/end markers.")

    def _resample(self):
        self._stop_all_timers()
        self._start_animation()


# ---
# Screen 2: Prediction (2 phases) — Animated Bars + Pipeline
# ---
class PredictionScreen(DemoScreen):
    SCREEN_INDEX = 3
    max_phase = 2

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._pick_name()
        self._show_phase_a()

    def _pick_name(self):
        docs, uchars, BOS, _ = self.app.data
        self._name = random.choice(docs)
        self._tokens = [BOS] + [uchars.index(ch) for ch in self._name] + [BOS]

    # --- Phase A: Untrained name generation with animated table ---

    def _show_phase_a(self):
        self._set_title("Step 2a: Prediction")
        self._stop_all_timers()
        sd = self.app.model[0]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]

        name, steps = generate_one(sd, uchars, BOS, vocab_size, temp=0.8)
        self._gen_name = name or "???"
        self._gen_steps = steps
        self._row_idx_a = 0

        # Save to app for comparison screen
        if not hasattr(self.app, 'before_names'):
            self.app.before_names = []
        if len(self.app.before_names) < 8:
            self.app.before_names.append(self._gen_name)

        # Explanation (right pane)
        text = Text()
        for line in [
            "Watch what an untrained",
            "model produces. Every",
            "letter is random \u2014 roughly",
            "1-in-27 chance for each.",
            "",
            "It doesn't pick one letter",
            "\u2014 it gives a percentage",
            "to every possible letter.",
            "",
            "With no training, all",
            "letters are roughly equal.",
            "The result? Gibberish.",
            "",
        ]:
            text.append(line + "\n")
        text.append("  >40% ", style="bold green")
        text.append("sure\n")
        text.append("  15-40% ", style="yellow")
        text.append("likely\n")
        text.append("  5-15% ", style="dark_orange")
        text.append("uncertain\n")
        text.append("  <5% ", style="bold red")
        text.append("surprised\n")
        self._set_pane("#right-pane", text, title="What")

        self._render_prediction_table()
        self._add_timer(self.set_interval(0.3, self._add_row_a))
        self._update_status()

    def _add_row_a(self):
        if self._row_idx_a >= len(self._gen_steps):
            self._stop_all_timers()
            self._render_prediction_table()
            self._add_timer(self.set_timer(3.0, self._restart_phase_a))
            return
        self._row_idx_a += 1
        self._render_prediction_table()

    def _restart_phase_a(self):
        self._stop_all_timers()
        self._show_phase_a()

    def _render_prediction_table(self):
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        steps = self._gen_steps
        name = self._gen_name
        row_count = self._row_idx_a

        data_text = Text()

        # Live top-5 histogram — shows current step's distribution
        if row_count > 0 and row_count <= len(steps):
            current = steps[row_count - 1]
            all_probs = current['all_probs']
            sorted_p = sorted(all_probs, key=lambda x: -x[1])[:5]
            top_prob = sorted_p[0][1] if sorted_p else 1
            ctx_label = current['context'][:8] if current['context'] != '\u27E8BOS\u27E9' else '\u27E8BOS\u27E9'
            data_text.append(f'  After "{ctx_label}", top predictions:\n')
            for tok_id, prob in sorted_p:
                if tok_id == BOS:
                    label = "\u27E8B\u27E9"
                else:
                    label = f" {uchars[tok_id]} "
                bar = make_bar(prob, width=25, max_val=top_prob)
                style = prob_color(prob)
                data_text.append(f"   {label} ")
                data_text.append(f"{bar}", style=style)
                data_text.append(f" {prob*100:5.1f}%\n")
            data_text.append("\n")

        # Building name with confidence colors
        data_text.append("  Building: ")
        for i, s in enumerate(steps[:row_count]):
            ch = s['next']
            if ch == '\u27E8BOS\u27E9':
                break
            data_text.append(ch, style=prob_color(s['prob']))
        if row_count < len(steps):
            data_text.append("\u2588", style="bright_cyan")
        data_text.append("\n\n")

        # Table header
        data_text.append("  Pos  Context    \u2192 Next   Prob\n")
        data_text.append("  \u2500\u2500\u2500  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500  \u2500\u2500\u2500\u2500\u2500\u2500  \u2500\u2500\u2500\u2500\u2500\u2500\n")
        for i, s in enumerate(steps[:row_count]):
            ctx = s['context'][:8].ljust(8)
            nxt = s['next'][:6].ljust(6)
            style = prob_color(s['prob'])
            data_text.append(f"  {s['pos']:3d}   {ctx}  \u2192 ")
            data_text.append(f"{nxt}", style=style)
            data_text.append(f" {s['prob']*100:5.1f}%\n")

        self._set_pane("#left-pane", data_text,
                       title=f'Untrained: "{name}"',
                       subtitle="Pure guessing \u2014 no patterns yet.")

    # --- Phase B: Forward Pass with animated pipeline ---

    def _show_phase_b(self):
        self._set_title("Step 2b: Forward Pass")
        self._stop_all_timers()
        docs, uchars, BOS, vocab_size = self.app.data
        sd = self.app.model[0]
        tokens = self._tokens
        name = self._name

        # Pre-compute all stage vectors
        token_id = tokens[min(2, len(tokens) - 2)]
        pos_id = min(2, len(tokens) - 2)
        self._stages = forward_stages(token_id, pos_id, sd)
        self._stage_char = uchars[token_id] if token_id != BOS else '\u27E8BOS\u27E9'
        self._stage_pos = pos_id
        self._stage_idx = 0
        self._bar_fill = 0

        # Explanation (right pane)
        ch = self._stage_char
        text = Text()
        for line in [
            f"Here's what happens inside",
            f'when the model sees "{ch}" at',
            f"position {pos_id}.",
            "",
            "1. Look up what the letter",
            "   means (token embedding)",
            "2. Add where it is in the",
            "   name (position embedding)",
            "3. Look at earlier letters",
            "   for context (attention)",
            "4. Transform the signal",
            "   (MLP layer)",
            "",
            "Watch the 16 numbers change",
            "shape at each stage.",
            "",
            "These are real numbers from",
            "this model running right now.",
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="What")

        self._render_pipeline()
        self._add_timer(self.set_interval(0.06, self._animate_pipeline))
        self._update_status()

    def _animate_pipeline(self):
        if self._stage_idx >= len(self._stages):
            self._stop_all_timers()
            self._render_pipeline()
            self._add_timer(self.set_timer(3.0, self._restart_phase_b))
            return
        stage_len = len(self._stages[self._stage_idx][1])
        if self._bar_fill <= stage_len:
            self._render_pipeline()
            self._bar_fill += 3
        else:
            self._stage_idx += 1
            self._bar_fill = 0

    def _restart_phase_b(self):
        self._stop_all_timers()
        self._pick_name()
        self._show_phase_b()

    def _render_pipeline(self):
        data_text = Text()
        ch = self._stage_char
        docs, uchars, BOS, vocab_size = self.app.data
        data_text.append(f'  Input: "{ch}" at pos {self._stage_pos}\n\n')

        # Fixed-width sparkline: always N_EMBD display bars, downsampling longer vectors
        display_w = N_EMBD

        def wide_sparkline(vec, fill_frac=None):
            """Render vec as display_w double-width spark chars. fill_frac 0..1 controls partial reveal."""
            if not vec:
                return "", 0
            # Downsample to display_w bins
            n = len(vec)
            bins = []
            for b in range(display_w):
                lo = b * n // display_w
                hi = (b + 1) * n // display_w
                if hi <= lo:
                    hi = lo + 1
                bins.append(sum(vec[lo:hi]) / (hi - lo))
            mn, mx = min(bins), max(bins)
            if mx == mn:
                mx = mn + 1
            filled = display_w if fill_frac is None else int(fill_frac * display_w)
            filled = max(0, min(display_w, filled))
            chars = "".join(
                SPARK[min(7, int((v - mn) / (mx - mn) * 7))] * 2
                for v in bins[:filled]
            )
            padding = "··" * (display_w - filled)
            return chars + padding, filled

        last = len(self._stages) - 1
        for i, (label, vec) in enumerate(self._stages):
            n = len(vec)
            connector = " +" if label == 'Token Embedding' else " \u21b4"
            if i > self._stage_idx:
                data_text.append(f"  {label}\n", style="dim")
                data_text.append(f"   {'··' * display_w}", style="dim")
                if i <= last:
                    data_text.append(connector, style="dim")
                data_text.append("\n\n")
            elif i == self._stage_idx:
                data_text.append(f"  {label}\n", style="bold bright_cyan")
                fill_frac = self._bar_fill / n if n > 0 else 0
                bars, filled = wide_sparkline(vec, fill_frac)
                data_text.append(f"   {bars}", style="bright_cyan")
                if i <= last:
                    data_text.append(connector, style="dim")
                data_text.append("\n")
                # Show numeric values for active stage
                shown = vec[:min(4, self._bar_fill)]
                if shown:
                    vals_str = "   [" + ", ".join(f"{v:+.3f}" for v in shown) + ", ...]"
                    data_text.append(f"{vals_str}\n", style="dim")
                data_text.append("\n")
            else:
                data_text.append(f"  {label}\n", style="bright_green")
                bars, _ = wide_sparkline(vec)
                data_text.append(f"   {bars}", style="green")
                if i <= last:
                    data_text.append(connector, style="green")
                data_text.append("\n\n")

        done = self._stage_idx >= len(self._stages)
        if done:
            probs = self._stages[-1][1]
            pick_id = probs.index(max(probs))
            pick_ch = uchars[pick_id] if pick_id != BOS else '\u27E8end\u27E9'
            pick_pct = probs[pick_id] * 100
            data_text.append(f"  Output: ", style="dim")
            data_text.append(f" {pick_ch} ", style="bold reverse bright_green")
            data_text.append(f" ({pick_pct:.0f}% confidence)\n", style="bright_green")
        else:
            data_text.append(f"  Output: ?\n", style="dim")

        self._set_pane("#left-pane", data_text,
                       title="Pipeline",
                       subtitle="All numbers are live from this model.")

    # --- Phase switching ---

    def _on_phase_change(self):
        self._stop_all_timers()
        self._pick_name()
        if self.phase == 0:
            self._show_phase_a()
        else:
            self._show_phase_b()

    def _resample(self):
        self._on_phase_change()


# ---
# Screen 3: Training (auto-start, progress, chart, before/after)
# ---
class TrainingScreen(DemoScreen):
    SCREEN_INDEX = 4
    max_phase = 3
    SCREEN_TITLE = "Step 3: Training"
    _training_done = False
    _skip_requested = False

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")
            yield Static(id="progress-area")
            yield Static(id="chart-area")

    def on_resize(self, event):
        """Re-render chart when terminal is resized."""
        if self.app.losses:
            self._update_chart()

    def _on_screen_mount(self):
        self._set_title(self.SCREEN_TITLE)
        if self.app.losses:
            # Training already done — show results, don't re-train
            self._training_done = True
            self._skip_requested = False
            self._show_phase_a()
            self._show_phase_c()
        else:
            self._training_done = False
            self._skip_requested = False
            self._show_phase_a()
            self._start_training()

    def _show_phase_a(self):
        """Before training: explanation + before names."""
        # Explanation (right pane)
        text = Text()
        for line in [
            "Training teaches the model",
            "by showing it real names",
            "and adjusting its numbers",
            "to make better guesses.",
            "",
            "Each step:",
            "1. Show it a name",
            "2. It guesses each next",
            "   letter",
            "3. Measure how wrong it was",
            "4. Nudge all parameters",
            "   slightly to be less wrong",
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="What")

        # Generate "before" names (untrained model)
        sd = self.app.model[0]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]

        before_names = []
        for _ in range(3):
            name, _ = generate_one(sd, uchars, BOS, vocab_size, temp=0.8)
            before_names.append(name or "???")
        self._before_names = before_names

        before_panel_text = Text()
        for n in before_names:
            before_panel_text.append(f" {n[:12]}\n", style="bold red")

        after_panel_text = Text()
        after_panel_text.append(" \u25cc training...\n", style="dim")

        before_panel = Panel(before_panel_text, title="Before",
                           border_style="red", box=box.ROUNDED, padding=(0, 0))
        after_panel = Panel(after_panel_text, title="After",
                          border_style="dim", box=box.ROUNDED, padding=(0, 0))

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(before_panel, after_panel)
        self._set_pane("#left-pane", grid, title="Names")

        try:
            bar_w = max(20, self.query_one("#progress-area", Static).size.width - 30)
        except Exception:
            bar_w = 50
        self.query_one("#progress-area", Static).update(
            f"  {'\u2591' * bar_w}  0/{NUM_STEPS}   loss: ----"
        )
        self.query_one("#chart-area", Static).update("")
        self._update_status("Training...")

    def _show_phase_c(self):
        """After training: results + after names with typing effect."""
        sd = self.app.model[0]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]
        losses = self.app.losses
        elapsed = self.app.train_elapsed

        # Generate "after" names (retry if empty)
        self._after_names = []
        for _ in range(3):
            for _attempt in range(5):
                name, _ = generate_one(sd, uchars, BOS, vocab_size, temp=0.5)
                if name:
                    break
            self._after_names.append(name.capitalize() if name else "\u2014")
        self._typing_idx = 0
        self._typing_char_idx = 0

        # Results panel (right pane)
        text = Text()
        start_loss = losses[0] if losses else 0
        final_loss = losses[-1] if losses else 0
        actual = getattr(self.app, 'train_steps_done', NUM_STEPS)
        for line in [
            f"{actual} steps in {elapsed:.1f}s",
            "",
            f"Start loss: {start_loss:.2f}",
            f"Final loss: {final_loss:.2f}",
            "",
            f"{start_loss:.2f} is what you get from",
            "pure random guessing.",
            "",
            f"{final_loss:.2f} means the model has",
            "learned common letter",
            "patterns from real names.",
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="Results", color="green")

        # Final progress bar (dynamic width)
        try:
            bar_w = max(20, self.query_one("#progress-area", Static).size.width - 30)
        except Exception:
            bar_w = 50
        frac = actual / NUM_STEPS
        filled = int(frac * bar_w)
        bar = '\u2588' * filled + '\u2591' * (bar_w - filled)
        self.query_one("#progress-area", Static).update(
            f"  {bar} {actual}/{NUM_STEPS}  loss: {final_loss:.2f}  \u2713"
        )
        self._update_status(f"\u2713 {elapsed:.1f}s")
        self._training_done = True
        self.phase = 2

        # Start typing effect for after names
        self._add_timer(self.set_interval(0.08, self._type_after_names))

    def _type_after_names(self):
        """Reveal after-training names character by character."""
        if self._typing_idx >= len(self._after_names):
            self._stop_all_timers()
            return
        name = self._after_names[self._typing_idx]
        self._typing_char_idx += 1
        if self._typing_char_idx > len(name):
            self._typing_idx += 1
            self._typing_char_idx = 0
        self._render_after_panel()

    def _render_after_panel(self):
        before_panel_text = Text()
        for n in self._before_names:
            before_panel_text.append(f" {n[:12]}\n", style="bold red")

        after_panel_text = Text()
        for i, name in enumerate(self._after_names):
            if i < self._typing_idx:
                after_panel_text.append(f" {name}\n", style="green bold")
            elif i == self._typing_idx:
                revealed = name[:self._typing_char_idx]
                after_panel_text.append(f" {revealed}", style="green bold")
                after_panel_text.append("\u2588\n", style="bright_green")
            else:
                after_panel_text.append(f" \n", style="dim")

        before_panel = Panel(before_panel_text, title="Before",
                           border_style="red", box=box.ROUNDED, padding=(0, 0))
        actual = getattr(self.app, 'train_steps_done', NUM_STEPS)
        after_panel = Panel(after_panel_text, title=f"After {actual} Steps",
                          border_style="green", box=box.ROUNDED, padding=(0, 0))
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(before_panel, after_panel)
        self._set_pane("#left-pane", grid, title="Names")

    @work(thread=True)
    def _start_training(self):
        """Run training in background thread."""
        app = self.app
        docs = app.data[0]
        uchars = app.data[1]
        BOS = app.data[2]
        sd, params = app.model
        m_buf, v_buf = app.optimizer

        t0 = time.time()
        actual_steps = 0
        for step in range(NUM_STEPS):
            doc = docs[step % len(docs)]
            loss_val = train_step(step, doc, sd, params, m_buf, v_buf, uchars, BOS)
            app.losses.append(loss_val)
            actual_steps = step + 1
            self.post_message(TrainProgress(step + 1, loss_val))
            if self._skip_requested:
                break
        elapsed = time.time() - t0
        app.train_elapsed = elapsed
        app.train_steps_done = actual_steps
        self.post_message(TrainDone(elapsed))

    def on_train_progress(self, msg: TrainProgress):
        step = msg.step
        frac = step / NUM_STEPS
        try:
            bar_w = max(20, self.query_one("#progress-area", Static).size.width - 30)
        except Exception:
            bar_w = 50
        filled = int(frac * bar_w)
        bar = '\u2588' * filled + '\u2591' * (bar_w - filled)
        try:
            self.query_one("#progress-area", Static).update(
                f"  {bar} {step}/{NUM_STEPS}  loss: {msg.loss:.2f}"
            )
        except Exception:
            pass
        if step % 10 == 0 or step == NUM_STEPS:
            self._update_chart()
        self._update_status(f"\u25cf {step}/{NUM_STEPS}")

    def on_train_done(self, msg: TrainDone):
        self._update_chart()
        self._show_phase_c()

    def _update_chart(self):
        losses = self.app.losses
        if not losses:
            return

        # Query actual widget dimensions so chart fills the container
        try:
            w = self.query_one("#chart-area", Static)
            avail_w = w.size.width
            avail_h = w.size.height
        except Exception:
            avail_w, avail_h = 80, 20

        # Axis label "  3.3│" = 6 chars, leave 1 char margin right
        width = max(10, avail_w - 7)
        # Top label + bottom label = 2 lines
        height = max(5, avail_h - 2)

        # Resample losses to fit chart width
        n = len(losses)
        step = max(1, n // width)
        sampled = losses[::step]
        if not sampled:
            return
        # Stretch sampled to fill width exactly
        if len(sampled) < width:
            width = len(sampled)

        max_loss = max(sampled)
        min_loss = min(sampled)
        if max_loss == min_loss:
            max_loss = min_loss + 1
        chars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

        chart_lines = []
        chart_lines.append(f"  {max_loss:.1f}\u2502")
        for row in range(height - 1, -1, -1):
            line = "     \u2502"
            for col in range(width):
                val = sampled[min(col, len(sampled) - 1)]
                normalized = (val - min_loss) / (max_loss - min_loss)
                level = int(normalized * (height - 1))
                if level == row:
                    ci = int((normalized * (height - 1) - row) * (len(chars) - 1))
                    line += chars[min(ci, len(chars) - 1)]
                elif level > row:
                    line += "\u2588"
                else:
                    line += " "
            chart_lines.append(line)
        chart_lines.append(f"  {min_loss:.1f}\u2502" + "\u2500" * width)

        chart_text = "\n".join(chart_lines)
        try:
            w.update(chart_text)
        except Exception:
            pass

    def _advance(self):
        if not self._training_done:
            self._skip_requested = True
            return
        self._stop_all_timers()
        self._next_screen()

    def _resample(self):
        if self._training_done:
            self._stop_all_timers()
            sd = self.app.model[0]
            uchars = self.app.data[1]
            BOS = self.app.data[2]
            vocab_size = self.app.data[3]
            self._after_names = []
            for _ in range(3):
                name, _ = generate_one(sd, uchars, BOS, vocab_size, temp=0.5)
                self._after_names.append(name.capitalize() if name else "???")
            self._typing_idx = 0
            self._typing_char_idx = 0
            self._add_timer(self.set_interval(0.08, self._type_after_names))


# ---
# Screen 4: Comparison — Before/After Training Side-by-Side
# ---
class ComparisonScreen(DemoScreen):
    SCREEN_INDEX = 5

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._set_title("Step 4: Learning")
        self._context_chars = list(self.app.untrained_logits.keys())
        self._ctx_idx = 0
        self._bar_progress = 0.0

        self._generate_names_for_ctx()
        self._start_bar_animation()

        # Explanation (right pane)
        text = Text()
        for line in [
            "Same model, same code.",
            "The only difference:",
            f"{NUM_STEPS} steps of training.",
            "",
            "Before: flat distribution,",
            "every letter equally likely.",
            "Entropy near maximum.",
            "",
            "After: peaked distribution,",
            "model learned which letters",
            "follow which. Lower entropy",
            "= more confident predictions.",
            "",
            "This is what learning",
            "looks like: going from",
            '"I have no idea" to',
            '"I see patterns."',
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="What")
        self._update_status()

    def _generate_names_for_ctx(self):
        """Generate before/after names starting with the current context char."""
        ch = self._context_chars[self._ctx_idx]
        sd = self.app.model[0]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]

        # Generate "before" names using untrained weights
        # Save trained .data, swap in untrained, generate, swap back
        saved = {
            k: [[p.data for p in row] for row in mat]
            for k, mat in sd.items()
        }
        for k, mat in sd.items():
            for r, row in enumerate(mat):
                for c, p in enumerate(row):
                    p.data = self.app.untrained_weights[k][r][c]
        self._before_names = []
        for _ in range(3):
            n = generate_one_from(ch, sd, uchars, BOS, vocab_size, temp=0.8)
            self._before_names.append(n or "???")
        # Restore trained weights
        for k, mat in sd.items():
            for r, row in enumerate(mat):
                for c, p in enumerate(row):
                    p.data = saved[k][r][c]

        # Generate "after" names using current (trained) weights
        self._after_names = []
        for _ in range(3):
            n = generate_one_from(ch, sd, uchars, BOS, vocab_size, temp=0.5)
            self._after_names.append(n.capitalize() if n else "???")

    def _start_bar_animation(self):
        self._bar_progress = 0.0
        self._render_comparison()
        self._add_timer(self.set_interval(0.04, self._animate_bars))

    def _animate_bars(self):
        self._bar_progress = min(1.0, self._bar_progress + 0.04)
        self._render_comparison()
        if self._bar_progress >= 1.0:
            self._stop_all_timers()
            # Pause, then cycle to next context and re-animate
            self._add_timer(self.set_timer(3.0, self._next_context))

    def _next_context(self):
        self._stop_all_timers()
        self._ctx_idx = (self._ctx_idx + 1) % len(self._context_chars)
        self._generate_names_for_ctx()
        self._start_bar_animation()

    def _render_comparison(self):
        ch = self._context_chars[self._ctx_idx]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]
        sd = self.app.model[0]

        # Untrained distribution (saved at startup)
        untrained_logits = self.app.untrained_logits[ch]
        before_probs = softmax_data(untrained_logits)

        # Trained distribution (current model)
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        gpt_forward(BOS, 0, keys, vals, sd)
        tok = uchars.index(ch)
        logits = gpt_forward(tok, 1, keys, vals, sd)
        after_probs = softmax_data([l.data for l in logits])

        before_ent = entropy_bits(before_probs)
        after_ent = entropy_bits(after_probs)
        max_ent = math.log2(vocab_size)

        p = getattr(self, '_bar_progress', 1.0)
        # First half of animation fills BEFORE, second half fills AFTER
        before_p = min(1.0, p * 2.0)
        after_p = max(0.0, (p - 0.5) * 2.0)

        data_text = Text()

        # --- BEFORE ---
        data_text.append(f'  BEFORE \u2014 next letter after "{ch}":\n\n', style="bold red")

        before_top = sorted(enumerate(before_probs), key=lambda x: -x[1])[:8]
        before_max = before_top[0][1] if before_top else 1
        for tok_id, prob in before_top:
            lbl = "\u27E8B\u27E9" if tok_id == BOS else f" {uchars[tok_id]} "
            bar = make_bar(prob * before_p, width=15, max_val=before_max)
            data_text.append(f"  {lbl} ", style="red")
            data_text.append(f"{bar}", style="red")
            if before_p > 0.5:
                data_text.append(f" {prob*100:.1f}%\n")
            else:
                data_text.append(f"\n")

        data_text.append(f"  entropy: {before_ent:.1f}/{max_ent:.1f} bits ", style="dim")
        data_text.append(f'"random guessing"\n', style="dim")
        if before_p >= 1.0:
            data_text.append(f'  "{ch}" names: ', style="dim")
            for n in self._before_names:
                data_text.append(f"{n}  ", style="bold red")
            data_text.append("\n")

        # --- AFTER ---
        data_text.append(f'\n  AFTER {NUM_STEPS} steps \u2014 next letter after "{ch}":\n\n', style="bold green")

        after_top = sorted(enumerate(after_probs), key=lambda x: -x[1])[:8]
        after_max = after_top[0][1] if after_top else 1
        for tok_id, prob in after_top:
            lbl = "\u27E8B\u27E9" if tok_id == BOS else f" {uchars[tok_id]} "
            bar = make_bar(prob * after_p, width=15, max_val=after_max)
            data_text.append(f"  {lbl} ", style="green")
            data_text.append(f"{bar}", style="green")
            if after_p > 0.5:
                data_text.append(f" {prob*100:.1f}%\n")
            else:
                data_text.append(f"\n")

        data_text.append(f"  entropy: {after_ent:.1f}/{max_ent:.1f} bits ", style="dim")
        data_text.append(f'"has preferences"\n', style="dim")
        if after_p >= 1.0:
            data_text.append(f"  Names: ", style="dim")
            for n in self._after_names:
                data_text.append(f"{n}  ", style="bold green")
            data_text.append("\n")

        self._set_pane("#left-pane", data_text,
                       title=f'Context: "{ch}" \u2014 Before vs After',
                       subtitle=f"[{self._ctx_idx+1}/{len(self._context_chars)}] [\u2192] cycle contexts")

    def _resample(self):
        self._stop_all_timers()
        self._ctx_idx = (self._ctx_idx + 1) % len(self._context_chars)
        self._generate_names_for_ctx()
        self._start_bar_animation()


# ---
# Screen 5: Inference (2 phases) — Animated Table + Temperature Comparison
# ---
class InferenceScreen(DemoScreen):
    SCREEN_INDEX = 6
    max_phase = 3

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._show_phase_a()

    # --- Phase A: Trained forward pass pipeline ---

    def _show_phase_a(self):
        self._set_title("Step 5a: Forward Pass (Trained)")
        self._stop_all_timers()
        docs, uchars, BOS, vocab_size = self.app.data
        sd = self.app.model[0]

        # Pick a random name to trace through
        name = random.choice(docs)
        tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
        token_id = tokens[min(2, len(tokens) - 2)]
        pos_id = min(2, len(tokens) - 2)
        self._stages = forward_stages(token_id, pos_id, sd)
        self._stage_char = uchars[token_id] if token_id != BOS else '\u27E8BOS\u27E9'
        self._stage_pos = pos_id
        self._stage_idx = 0
        self._bar_fill = 0

        ch = self._stage_char
        text = Text()
        for line in [
            "Same pipeline as Step 2b,",
            "but now the model is trained.",
            "",
            f'Seeing "{ch}" at position {pos_id}.',
            "",
            "Compare the numbers to before:",
            "- Embeddings are sharper",
            "- Attention is more focused",
            "- MLP activations are larger",
            "",
            "The architecture didn't change.",
            f"Only the {len(self.app.model[1]):,} parameters",
            "were adjusted by training.",
            "",
            "These are the real trained",
            "numbers running right now.",
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="What")

        self._render_trained_pipeline()
        self._add_timer(self.set_interval(0.06, self._animate_trained_pipeline))
        self._update_status()

    def _animate_trained_pipeline(self):
        if self._stage_idx >= len(self._stages):
            self._stop_all_timers()
            self._render_trained_pipeline()
            self._add_timer(self.set_timer(3.0, self._restart_trained))
            return
        stage_len = len(self._stages[self._stage_idx][1])
        if self._bar_fill <= stage_len:
            self._render_trained_pipeline()
            self._bar_fill += 3
        else:
            self._stage_idx += 1
            self._bar_fill = 0

    def _restart_trained(self):
        self._stop_all_timers()
        self._show_phase_a()

    def _render_trained_pipeline(self):
        data_text = Text()
        ch = self._stage_char
        docs, uchars, BOS, vocab_size = self.app.data
        data_text.append(f'  Input: "{ch}" at pos {self._stage_pos}\n\n')

        display_w = N_EMBD

        def wide_sparkline(vec, fill_frac=None):
            if not vec:
                return "", 0
            n = len(vec)
            bins = []
            for b in range(display_w):
                lo = b * n // display_w
                hi = (b + 1) * n // display_w
                if hi <= lo:
                    hi = lo + 1
                bins.append(sum(vec[lo:hi]) / (hi - lo))
            mn, mx = min(bins), max(bins)
            if mx == mn:
                mx = mn + 1
            filled = display_w if fill_frac is None else int(fill_frac * display_w)
            filled = max(0, min(display_w, filled))
            chars = "".join(
                SPARK[min(7, int((v - mn) / (mx - mn) * 7))] * 2
                for v in bins[:filled]
            )
            padding = "··" * (display_w - filled)
            return chars + padding, filled

        last = len(self._stages) - 1
        for i, (label, vec) in enumerate(self._stages):
            n = len(vec)
            connector = " +" if label == 'Token Embedding' else " \u21b4"
            if i > self._stage_idx:
                data_text.append(f"  {label}\n", style="dim")
                data_text.append(f"   {'··' * display_w}", style="dim")
                if i <= last:
                    data_text.append(connector, style="dim")
                data_text.append("\n\n")
            elif i == self._stage_idx:
                data_text.append(f"  {label}\n", style="bold bright_cyan")
                fill_frac = self._bar_fill / n if n > 0 else 0
                bars, filled = wide_sparkline(vec, fill_frac)
                data_text.append(f"   {bars}", style="bright_cyan")
                if i <= last:
                    data_text.append(connector, style="dim")
                data_text.append("\n")
                shown = vec[:min(4, self._bar_fill)]
                if shown:
                    vals_str = "   [" + ", ".join(f"{v:+.3f}" for v in shown) + ", ...]"
                    data_text.append(f"{vals_str}\n", style="dim")
                data_text.append("\n")
            else:
                data_text.append(f"  {label}\n", style="bright_green")
                bars, _ = wide_sparkline(vec)
                data_text.append(f"   {bars}", style="green")
                if i <= last:
                    data_text.append(connector, style="green")
                data_text.append("\n\n")

        done = self._stage_idx >= len(self._stages)
        if done:
            probs = self._stages[-1][1]
            pick_id = probs.index(max(probs))
            pick_ch = uchars[pick_id] if pick_id != BOS else '\u27E8end\u27E9'
            pick_pct = probs[pick_id] * 100
            data_text.append(f"  Output: ", style="dim")
            data_text.append(f" {pick_ch} ", style="bold reverse bright_green")
            data_text.append(f" ({pick_pct:.0f}% confidence)\n", style="bright_green")
        else:
            data_text.append(f"  Output: ?\n", style="dim")

        self._set_pane("#left-pane", data_text,
                       title="Pipeline (Trained)",
                       subtitle="Same architecture, trained parameters.")

    # --- Phase B: Token-by-token generation with animated table ---

    def _show_phase_b(self):
        self._set_title("Step 5b: Inference")
        self._stop_all_timers()
        sd = self.app.model[0]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]

        name, steps = generate_one(sd, uchars, BOS, vocab_size, temp=0.5)
        self._last_name = name
        self._last_steps = steps
        self._row_idx = 0

        # Explanation (right pane)
        text = Text()
        for line in [
            "Now we use the trained model",
            "to invent new names.",
            "",
            "One letter at a time:",
            "1. Ask the model: what",
            "   letter comes next?",
            "2. Pick one based on the",
            "   confidence percentages",
            "3. Stop when it signals",
            '   "end of name"',
            "",
            "Watch the name build",
            "letter by letter. Colors",
            "show confidence:",
            "",
        ]:
            text.append(line + "\n")
        text.append("  >40% ", style="bold green")
        text.append("sure\n")
        text.append("  15-40% ", style="yellow")
        text.append("likely\n")
        text.append("  5-15% ", style="dark_orange")
        text.append("uncertain\n")
        text.append("  <5% ", style="bold red")
        text.append("surprised\n")
        self._set_pane("#right-pane", text, title="What")

        self._render_inference_table()
        self._add_timer(self.set_interval(0.3, self._add_row))
        self._update_status()

    def _add_row(self):
        if self._row_idx >= len(self._last_steps):
            self._stop_all_timers()
            self._render_inference_table()  # final render with top-5
            self._add_timer(self.set_timer(3.0, self._restart_phase_b))
            return
        self._row_idx += 1
        self._render_inference_table()

    def _restart_phase_b(self):
        self._stop_all_timers()
        self._show_phase_b()

    def _render_inference_table(self):
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        steps = self._last_steps
        name = self._last_name
        row_count = self._row_idx

        data_text = Text()

        # Building name with confidence colors
        data_text.append("  Building: ")
        for i, s in enumerate(steps[:row_count]):
            ch = s['next']
            if ch == '\u27E8BOS\u27E9':
                break
            data_text.append(ch, style=prob_color(s['prob']))
        if row_count < len(steps):
            data_text.append("\u2588", style="bright_cyan")
        data_text.append("\n\n")

        # Table header
        data_text.append("  Pos  Context    \u2192 Next   Prob\n")
        data_text.append("  \u2500\u2500\u2500  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500  \u2500\u2500\u2500\u2500\u2500\u2500  \u2500\u2500\u2500\u2500\u2500\u2500\n")
        for i, s in enumerate(steps[:row_count]):
            ctx = s['context'][:8].ljust(8)
            nxt = s['next'][:6].ljust(6)
            style = prob_color(s['prob'])
            data_text.append(f"  {s['pos']:3d}   {ctx}  \u2192 ")
            data_text.append(f"{nxt}", style=style)
            data_text.append(f" {s['prob']*100:5.1f}%\n")

        # Show top-5 bars after table is complete
        if row_count >= len(steps) and len(steps) >= 2:
            last_step = steps[-2] if steps[-1]['next'] == '\u27E8BOS\u27E9' else steps[-1]
            all_probs = last_step['all_probs']
            sorted_p = sorted(all_probs, key=lambda x: -x[1])[:5]
            top_prob = sorted_p[0][1] if sorted_p else 1
            data_text.append(f"\n  Top-5 at pos {last_step['pos']}:\n")
            for tok_id, prob in sorted_p:
                if tok_id == BOS:
                    label = "\u27E8BOS\u27E9"
                else:
                    label = f"  {uchars[tok_id]}  "
                bar = make_bar(prob, max_val=top_prob)
                data_text.append(f"   {label} {bar} {prob*100:5.1f}%\n")

        self._set_pane("#left-pane", data_text,
                       title=f'Generating: "{name}"',
                       subtitle="Each step: score every letter, pick one.")

    # --- Phase C: Temperature comparison with histograms ---

    def _show_phase_c(self):
        self._set_title("Step 5c: Temperature")
        self._stop_all_timers()
        sd = self.app.model[0]
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]

        # Explanation (right pane)
        text = Text()
        for line in [
            'Temperature controls how',
            '"creative" the output is.',
            "",
            "Low temperature: plays it",
            "safe. Picks the most common",
            "letters. Names look like",
            "ones you've heard before.",
            "",
            "High temperature: takes",
            "more risks. Picks unusual",
            "letters. Names get exotic",
            "and sometimes strange.",
            "",
            "The model doesn't change \u2014",
            "only how we pick from its",
            "suggestions.",
            "",
            "Entropy measures how spread",
            "out the distribution is.",
            f"Max = {math.log2(vocab_size):.1f} bits (uniform).",
        ]:
            text.append(line + "\n")
        self._set_pane("#right-pane", text, title="What")

        # Pre-generate names at three temperatures with logits
        self._temps = [0.3, 0.8, 1.5]
        self._temp_labels = ["conservative", "balanced", "creative"]
        self._temp_colors = ["green", "yellow", "red"]
        self._gen_data = []
        for temp in self._temps:
            name, steps = generate_one(sd, uchars, BOS, vocab_size,
                                       temp=temp, return_logits=True)
            self._gen_data.append({'name': name, 'steps': steps})

        self._temp_step = 0
        # Max steps across all three generations
        self._max_steps = max(len(g['steps']) for g in self._gen_data)

        self._render_temperature()
        self._add_timer(self.set_interval(0.4, self._animate_temperature))
        self._update_status()

    def _animate_temperature(self):
        self._temp_step += 1
        if self._temp_step >= self._max_steps:
            self._stop_all_timers()
            self._render_temperature()
            self._add_timer(self.set_timer(3.0, self._restart_phase_c))
            return
        self._render_temperature()

    def _restart_phase_c(self):
        self._stop_all_timers()
        self._show_phase_c()

    def _render_temperature(self):
        uchars = self.app.data[1]
        BOS = self.app.data[2]
        vocab_size = self.app.data[3]
        step_idx = self._temp_step

        data_text = Text()

        for ti, (temp, label, color) in enumerate(
                zip(self._temps, self._temp_labels, self._temp_colors)):
            gen = self._gen_data[ti]
            steps = gen['steps']
            name = gen['name']

            data_text.append(f"  T={temp}  ", style=f"bold {color}")
            data_text.append(f"({label})\n", style="dim")

            # Show histogram for current step if available
            if step_idx < len(steps) and steps[step_idx].get('raw_logits'):
                raw = steps[step_idx]['raw_logits']
                scaled = [l / temp for l in raw]
                probs = softmax_data(scaled)

                # Top-5 horizontal bars
                indexed = list(enumerate(probs))
                top5 = sorted(indexed, key=lambda x: -x[1])[:5]
                top_prob = top5[0][1] if top5 else 1
                for tok_id, prob in top5:
                    if tok_id == BOS:
                        lbl = "\u27E8B\u27E9"
                    else:
                        lbl = f" {uchars[tok_id]} "
                    bar = make_bar(prob, width=25, max_val=top_prob)
                    data_text.append(f"   {lbl} {bar} {prob*100:4.1f}%\n")

                # Entropy
                ent = entropy_bits(probs)
                max_ent = math.log2(vocab_size)
                ent_bar = make_bar(ent / max_ent, width=15)
                ent_color = "green" if ent < 1.5 else ("yellow" if ent < 3.0 else "red")
                data_text.append(f"   entropy: {ent:.1f} bits ")
                data_text.append(f"{ent_bar}\n", style=ent_color)
            else:
                data_text.append("   (complete)\n", style="dim")

            # Building name with confidence colors
            data_text.append("   ")
            visible_steps = min(step_idx + 1, len(steps))
            for i in range(visible_steps):
                s = steps[i]
                ch = s['next']
                if ch == '\u27E8BOS\u27E9':
                    break
                data_text.append(ch, style=prob_color(s['prob']))
            if visible_steps < len(steps):
                data_text.append("\u2588", style=f"bright_{color}")
            data_text.append("\n\n")

        self._set_pane("#left-pane", data_text,
                       title="Temperature Comparison",
                       subtitle="Same model, different temperature \u2014 controls creativity.")

    # --- Phase switching ---

    def _on_phase_change(self):
        self._stop_all_timers()
        if self.phase == 0:
            self._show_phase_a()
        elif self.phase == 1:
            self._show_phase_b()
        else:
            self._show_phase_c()

    def _resample(self):
        self._on_phase_change()


# ---
# Screen 5: Closing — Auto-Scrolling Source + Scale Comparison
# ---
class ClosingScreen(DemoScreen):
    SCREEN_INDEX = 7
    SCREEN_TITLE = "Summary"

    def _compose_content(self) -> ComposeResult:
        with Vertical(id="outer"):
            yield Static(id="screen-title")
            with Horizontal(id="content-area"):
                yield Static(id="left-pane")
                yield Static(id="right-pane")

    def _on_screen_mount(self):
        self._set_title(self.SCREEN_TITLE)

        # Load source code
        source_path = os.path.abspath(__file__)
        try:
            source = open(source_path).read()
        except Exception:
            source = "# Could not read source file"

        start_marker = "# microGPT"
        end_marker = "# TUI"
        start_idx = source.find(start_marker)
        end_idx = source.find(end_marker)
        if start_idx >= 0 and end_idx > start_idx:
            engine_source = source[start_idx:end_idx].rstrip()
        elif end_idx > 0:
            engine_source = source[:end_idx].rstrip()
        else:
            engine_source = source[:3000]

        self._source_lines = engine_source.split('\n')
        self._scroll_offset = 0
        self._scroll_speed = 1
        self._scroll_paused = False
        self._visible_height = 25

        self._render_code_window()
        self._render_scale()
        self._add_timer(self.set_interval(0.5, self._scroll_code))

    def _scroll_code(self):
        if self._scroll_paused:
            return
        self._scroll_offset += self._scroll_speed
        max_offset = max(0, len(self._source_lines) - self._visible_height)
        if self._scroll_offset > max_offset:
            self._scroll_offset = 0
        if self._scroll_offset < 0:
            self._scroll_offset = max_offset
        self._render_code_window()

    def _render_code_window(self):
        # Query actual pane height for visible lines (subtract 2 for border)
        try:
            h = self.query_one("#left-pane", Static).size.height
            if h > 4:
                self._visible_height = h - 2
        except Exception:
            pass
        lines = self._source_lines
        offset = max(0, self._scroll_offset)
        visible = lines[offset:offset + self._visible_height]
        window_text = '\n'.join(visible)

        syntax = Syntax(window_text, "python", theme="monokai",
                       line_numbers=True, start_line=offset + 1,
                       word_wrap=True)

        # Scroll position indicator
        total = len(lines)
        if total > self._visible_height:
            pct = offset / (total - self._visible_height) * 100
            pos_info = f"line {offset+1}/{total} ({pct:.0f}%)"
        else:
            pos_info = f"line 1/{total}"

        speed_label = f"speed: {self._scroll_speed}"
        if self._scroll_paused:
            speed_label = "PAUSED"

        self._set_pane("#left-pane", syntax,
                       title=f"Source  |  {pos_info}  |  {speed_label}",
                       subtitle="j/k speed \u00b7 Space pause")

    def _render_scale(self):
        n_params = len(self.app.model[1]) if self.app.model else "?"
        elapsed = self.app.train_elapsed
        docs = self.app.data[0]
        n_names = len(docs)

        text = Text()
        text.append("  This demo\n", style="bold bright_white")
        text.append(f"   {n_params:,} parameters\n")
        text.append(f"   {n_names:,} training names\n")
        text.append(f"   {NUM_STEPS} training steps\n")
        text.append(f"   {elapsed:.1f}s training time\n")
        text.append("   scalar Python\n")
        text.append("\n")
        text.append("  Production scale\n", style="bold bright_white")
        text.append("\n")
        text.append("   Model      Params\n", style="dim")
        text.append("   GPT-2      117M\n")
        text.append("   GPT-3      175B     (1.5M\u00d7)\n")
        text.append("   GPT-4     ~1.8T    (15M\u00d7)\n")
        text.append("\n")
        text.append("  Speed multipliers\n", style="bold bright_white")
        text.append("   (same algorithm, different tools)\n", style="dim")
        text.append("\n")
        text.append("   numpy        250\u00d7\n")
        text.append("   C/C++        400\u00d7\n")
        text.append("   GPU       10,000\u00d7\n")
        text.append("\n")
        text.append("  The algorithm is identical.\n", style="dim")
        text.append("  Everything else is efficiency.\n", style="dim")
        self._set_pane("#right-pane", text,
                       title="Scale",
                       subtitle="Same math, bigger numbers.")

    # Override keys for scroll control
    def key_j(self):
        self._scroll_speed = min(5, self._scroll_speed + 1)
        self._render_code_window()

    def key_k(self):
        self._scroll_speed = max(-2, self._scroll_speed - 1)
        self._render_code_window()

    def key_space(self):
        self._scroll_paused = not self._scroll_paused
        self._render_code_window()

    def _advance(self):
        pass  # Last screen

    def _resample(self):
        pass

    def _status_text(self, extra=""):
        return "Step 6/6                          j/k speed \u00b7 Space pause     ? help  q quit"


# ---
# App
# ---
class MicroGPTApp(App):
    CSS = APP_CSS
    TITLE = "microGPT"

    SCREENS = {}

    def __init__(self, explain: bool = False):
        super().__init__()
        self.explain_mode = explain
        self.data = None
        self.model = None
        self.optimizer = None
        self.losses = []
        self.train_elapsed = 0.0
        self.screen_order = [
            TitleScreen, DataScreen, TokenizationScreen, PredictionScreen,
            TrainingScreen, ComparisonScreen, InferenceScreen, ClosingScreen,
        ]

    def on_mount(self) -> None:
        self.data = load_data()
        self.model = build_model(self.data[3])
        self.optimizer = init_optimizer(self.model[1])

        # Snapshot untrained model for comparison screen
        sd = self.model[0]
        uchars, BOS, vocab_size = self.data[1], self.data[2], self.data[3]

        # Save untrained logits for several context chars
        self.untrained_logits = {}
        for ch in ['a', 'e', 'r', 's', 'm']:
            keys = [[] for _ in range(N_LAYER)]
            vals = [[] for _ in range(N_LAYER)]
            gpt_forward(BOS, 0, keys, vals, sd)
            tok = uchars.index(ch)
            logits = gpt_forward(tok, 1, keys, vals, sd)
            self.untrained_logits[ch] = [l.data for l in logits]

        # Snapshot untrained weights for generating "before" names later
        self.untrained_weights = {
            k: [[p.data for p in row] for row in mat]
            for k, mat in sd.items()
        }

        # Generate names from untrained model (for PredictionScreen)
        self.before_names = []
        for _ in range(5):
            name, _ = generate_one(sd, uchars, BOS, vocab_size, temp=0.8)
            self.before_names.append(name or "???")

        self.push_screen(TitleScreen())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="microGPT Live Training Demo")
    parser.add_argument("--explain", action="store_true",
                       help="Wait for Enter between phases")
    args = parser.parse_args()
    MicroGPTApp(explain=args.explain).run()
