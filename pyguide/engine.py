"""
Rendering engine for "Python for AI: A Practical Feature Guide".

Turns a list of chapter dicts (a tiny block DSL) into a Kindle-friendly
6x9-inch PDF. Every code block flagged run=True is executed in a fresh
Python subprocess at build time and its REAL output is embedded, so the
examples in the book are verified, not merely claimed.

Block DSL (each block is a dict with key "t"):
  {"t":"p",       "text": "prose, may contain `inline code`"}
  {"t":"h",       "text": "sub-heading inside a chapter"}
  {"t":"bullets", "items": ["item `code`", ...]}
  {"t":"note",    "text": "callout box (e.g. 'Why it matters for AI')"}
  {"t":"code",    "code": "python source", "run": True/False,
                  "caption": "optional label", "expect_error": False}

A chapter is:
  {"title": str, "part": str, "blocks": [block, ...]}
"""

import io
import re
import os
import subprocess
import sys
import textwrap

from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# ---- Page geometry (inches -> mm). 6x9 is the classic trade-paperback /
# ---- Kindle-friendly size. ----------------------------------------------
IN = 25.4
PAGE_W = 6 * IN
PAGE_H = 9 * IN
MARGIN = 0.6 * IN

# ---- Colours -------------------------------------------------------------
INK = (28, 28, 30)
MUTED = (110, 110, 115)
ACCENT = (11, 83, 148)        # headings / links
CODE_BG = (244, 245, 247)
CODE_BORDER = (223, 226, 230)
OUT_BG = (235, 244, 236)      # program output tint
OUT_BORDER = (200, 224, 204)
NOTE_BG = (255, 248, 227)
NOTE_BORDER = (240, 219, 150)
RULE = (210, 212, 216)


# =========================================================================
#  Running examples at build time
# =========================================================================
def run_example(code):
    """Execute a snippet in a fresh interpreter; return captured output."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout
    if proc.stderr:
        out += ("" if out.endswith("\n") or not out else "\n") + proc.stderr
    return out.rstrip("\n"), proc.returncode


# =========================================================================
#  PDF document
# =========================================================================
class Book(FPDF):
    def __init__(self, title, subtitle):
        super().__init__(orientation="P", unit="mm", format=(PAGE_W, PAGE_H))
        self.title_txt = title
        self.subtitle_txt = subtitle
        self.set_auto_page_break(True, margin=MARGIN)
        self.set_margins(MARGIN, MARGIN, MARGIN)

        # Fonts
        self.add_font("body", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
        self.add_font("body", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
        self.add_font("body", "I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf"))
        self.add_font("mono", "", os.path.join(FONT_DIR, "DejaVuSansMono.ttf"))
        self.add_font("mono", "B", os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"))

        self.cur_chapter = ""
        self.in_frontmatter = True
        self._on_part = False
        self.toc_entries = []   # (title, part, page_label)

    # ---- running header / footer ----
    def header(self):
        if self.in_frontmatter or self._on_part:
            return
        self.set_y(MARGIN - 6)
        self.set_font("body", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, self.cur_chapter, align="R")
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        y = MARGIN - 1.5
        self.line(MARGIN, y, PAGE_W - MARGIN, y)
        self.set_y(MARGIN)

    def footer(self):
        if self.in_frontmatter:
            return
        self.set_y(-MARGIN + 3)
        self.set_font("body", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, str(self.page_no()), align="C")

    @property
    def content_w(self):
        return PAGE_W - 2 * MARGIN

    # ---- inline `code` formatting inside prose ----
    def _write_rich(self, text, size=10.5, lh=5.4, color=INK):
        """Write a paragraph, rendering `code`, **bold** and *italic* spans."""
        self.set_text_color(*color)
        token = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
        for part in token.split(text):
            if not part:
                continue
            if part.startswith("`") and part.endswith("`"):
                self.set_font("mono", "", size - 0.5)
                self.set_text_color(*ACCENT)
                self.write(lh, part[1:-1])
                self.set_text_color(*color)
            elif part.startswith("**") and part.endswith("**"):
                self.set_font("body", "B", size)
                self.write(lh, part[2:-2])
            elif part.startswith("*") and part.endswith("*"):
                self.set_font("body", "I", size)
                self.write(lh, part[1:-1])
            else:
                self.set_font("body", "", size)
                self.write(lh, part)
        self.ln(lh)

    # ---- blocks ----
    def para(self, text):
        self.ln(1.5)
        self._write_rich(text)

    def subhead(self, text):
        self.ln(4)
        if self.will_page_break(11):
            self.add_page()
        self.set_font("body", "B", 13)
        self.set_text_color(*ACCENT)
        self.multi_cell(0, 6.5, text, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullets(self, items):
        self.ln(1.5)
        for it in items:
            x0 = self.get_x()
            self.set_font("body", "B", 10.5)
            self.set_text_color(*ACCENT)
            self.cell(5, 5.4, "•")
            self.set_x(x0 + 5)
            # simple hanging indent via a temporary left margin
            old = self.l_margin
            self.set_left_margin(x0 + 5)
            self.set_x(x0 + 5)
            self._write_rich(it)
            self.set_left_margin(old)

    def note(self, text):
        self.ln(2.5)
        self.set_font("body", "", 10)
        # measure height
        pad = 3
        # draw box with an estimate: use multi_cell in a dry run
        start_y = self.get_y()
        # render text inside a tinted box
        self.set_fill_color(*NOTE_BG)
        self.set_draw_color(*NOTE_BORDER)
        x = self.l_margin
        w = self.content_w
        # compute wrapped lines
        self.set_font("body", "", 10)
        lines = self.multi_cell(w - 2 * pad, 5.2, text, dry_run=True, output="LINES")
        h = len(lines) * 5.2 + 2 * pad
        if self.will_page_break(h):
            self.add_page()
            start_y = self.get_y()
        self.set_line_width(0.3)
        self.rect(x, start_y, w, h, style="DF")
        # accent left bar
        self.set_fill_color(*NOTE_BORDER)
        self.rect(x, start_y, 1.4, h, style="F")
        self.set_xy(x + pad, start_y + pad)
        self.set_left_margin(x + pad)
        self.set_text_color(*INK)
        self._write_rich(text, size=10, lh=5.2)
        self.set_left_margin(x)
        self.set_xy(x, start_y + h)
        self.ln(1.5)

    def _code_lines(self, code, mono_size):
        """Wrap code to page width, preserving indentation, no reflow of tokens
        unless a single line is too long (then hard-wrap with continuation)."""
        avail = self.content_w - 2 * 3  # padding
        self.set_font("mono", "", mono_size)
        char_w = self.get_string_width("0")
        max_chars = max(20, int(avail / char_w))
        out = []
        for raw in code.split("\n"):
            if self.get_string_width(raw) <= avail:
                out.append(raw)
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            cont = " " * indent + "    "
            cur = raw
            first = True
            while self.get_string_width(cur) > avail:
                # find break point
                cut = max_chars
                while cut > 1 and self.get_string_width((cur if first else "")[:cut]) > avail:
                    cut -= 1
                out.append(cur[:cut] + " ↵")
                cur = (cont + cur[cut:])
                first = False
            out.append(cur)
        return out

    def code_block(self, code, output=None, caption=None):
        mono_size = 8.6
        line_h = 4.2
        pad = 3
        code = code.rstrip("\n")
        code_lines = self._code_lines(code, mono_size)

        def render_box(lines, bg, border, textcolor, bold_first=False):
            h = len(lines) * line_h + 2 * pad
            if self.will_page_break(h + 4):
                self.add_page()
            x = self.l_margin
            y = self.get_y()
            w = self.content_w
            self.set_fill_color(*bg)
            self.set_draw_color(*border)
            self.set_line_width(0.3)
            self.rect(x, y, w, h, style="DF")
            self.set_xy(x + pad, y + pad)
            self.set_font("mono", "", mono_size)
            self.set_text_color(*textcolor)
            for ln_txt in lines:
                self.set_x(x + pad)
                self.cell(w - 2 * pad, line_h, ln_txt)
                self.ln(line_h)
            self.set_y(y + h)

        self.ln(2)
        if caption:
            self.set_font("body", "B", 8.5)
            self.set_text_color(*MUTED)
            self.cell(0, 4, caption.upper(), new_x="LMARGIN", new_y="NEXT")
            self.ln(0.5)
        render_box(code_lines, CODE_BG, CODE_BORDER, INK)

        if output is not None and output.strip() != "":
            out_lines = self._code_lines(output, mono_size)
            # cap extremely long outputs
            if len(out_lines) > 40:
                out_lines = out_lines[:40] + ["... (output truncated)"]
            self.ln(0.5)
            self.set_font("body", "B", 8.5)
            self.set_text_color(*MUTED)
            self.cell(0, 4, "OUTPUT", new_x="LMARGIN", new_y="NEXT")
            self.ln(0.5)
            render_box(out_lines, OUT_BG, OUT_BORDER, (30, 90, 40))
        self.ln(2)

    # ---- structural ----
    def start_part(self, name, number):
        self._on_part = True
        self.add_page()
        self.start_section(f"Part {number}: {name}", level=0)
        self.set_y(PAGE_H * 0.33)
        self.set_font("body", "", 12)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"PART {number}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("body", "B", 22)
        self.set_text_color(*ACCENT)
        self.multi_cell(0, 10, name, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        cx = PAGE_W / 2
        self.line(cx - 15, self.get_y() + 4, cx + 15, self.get_y() + 4)

    def start_chapter(self, num, title, part):
        self._on_part = False
        self.add_page()
        self.cur_chapter = f"{num}.  {title}"
        self.start_section(f"{num}. {title}", level=1)
        self.set_font("body", "", 10)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"CHAPTER {num}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_font("body", "B", 19)
        self.set_text_color(*INK)
        self.multi_cell(0, 8.5, title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y() + 2, self.l_margin + 22, self.get_y() + 2)
        self.ln(6)
