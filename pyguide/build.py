"""
Build "Python for AI: A Practical Feature Guide" into a Kindle-friendly PDF.

Usage:  python3 build.py
Output: python-for-ai-guide.pdf
"""
import importlib
import sys
import time

from engine import (Book, run_example, ACCENT, INK, MUTED, PAGE_W, PAGE_H,
                    MARGIN, RULE)

TITLE = "Python for AI"
SUBTITLE = "A Practical Feature Guide, Every Example Runnable"
AUTHOR = "Tommaso Borgato"

# Content modules, in order. Each defines PART (str) and CHAPTERS (list).
CONTENT_MODULES = [
    "content_00_project_setup",
    "content_01_core",
    "content_02_functional",
    "content_03_iteration",
    "content_04_objects",
    "content_05_typing",
    "content_06_data",
    "content_07_concurrency",
    "content_08_numeric",
    "content_09_transformer",
]


def load_content():
    parts = []  # [(part_name, [chapter, ...]), ...]
    for mod_name in CONTENT_MODULES:
        mod = importlib.import_module(mod_name)
        parts.append((mod.PART, mod.CHAPTERS))
    return parts


def render_toc(pdf, outline):
    pdf.in_frontmatter = True
    first_page = pdf.page_no()
    pdf.set_xy(MARGIN, MARGIN)
    pdf.set_font("body", "B", 20)
    pdf.set_text_color(*INK)
    pdf.cell(0, 12, "Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for section in outline:
        label = section.name
        page = section.page_number
        if section.level == 0:      # part
            pdf.ln(3)
            pdf.set_font("body", "B", 11.5)
            pdf.set_text_color(*ACCENT)
            pdf.multi_cell(0, 6, label, new_x="LMARGIN", new_y="NEXT")
        else:                        # chapter
            pdf.set_font("body", "", 9.7)
            pdf.set_text_color(*INK)
            y = pdf.get_y()
            page_str = str(page)
            page_w = pdf.get_string_width(page_str) + 2
            # leave a little room for at least a short dot-leader run
            max_title = pdf.content_w - page_w - 4 - 8
            name = label
            if pdf.get_string_width(name) > max_title:
                while pdf.get_string_width(name + "…") > max_title and len(name) > 4:
                    name = name[:-1]
                name = name.rstrip() + "…"
            pdf.set_x(pdf.l_margin + 4)
            pdf.cell(pdf.get_string_width(name) + 1, 5.6, name)
            # dot leaders
            dots_x = pdf.get_x()
            dots_end = pdf.l_margin + pdf.content_w - page_w
            pdf.set_text_color(*RULE)
            dot = "."
            dots = ""
            while pdf.get_string_width(dots + dot) < (dots_end - dots_x):
                dots += " ."
            pdf.cell(dots_end - dots_x, 5.6, dots)
            pdf.set_text_color(*MUTED)
            pdf.set_x(dots_end)
            pdf.cell(page_w, 5.6, page_str, align="R", new_x="LMARGIN", new_y="NEXT")
    # Pad so the ToC spans EXACTLY the reserved number of pages.
    reserved = getattr(pdf, "_toc_reserved", 1)
    while (pdf.page_no() - first_page + 1) < reserved:
        pdf.add_page()
    pdf.in_frontmatter = False


def cover(pdf):
    pdf.in_frontmatter = True
    pdf.add_page()
    # top rule band
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 0, PAGE_W, 6, style="F")
    pdf.set_y(PAGE_H * 0.30)
    pdf.set_font("body", "B", 34)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 15, TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.8)
    cx = PAGE_W / 2
    pdf.line(cx - 30, pdf.get_y(), cx + 30, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("body", "", 13)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 7, SUBTITLE, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(PAGE_H - MARGIN - 30)
    pdf.set_font("body", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        0, 5.5,
        "The Python language features that matter most for machine learning\n"
        "and AI engineering, each with a short, self-contained example whose\n"
        "output was captured by actually running the code.",
        align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("body", "I", 9)
    pdf.cell(0, 5, AUTHOR, align="C")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, PAGE_H - 6, PAGE_W, 6, style="F")


def preface(pdf):
    pdf.in_frontmatter = True
    pdf.add_page()
    pdf.set_font("body", "B", 20)
    pdf.set_text_color(*INK)
    pdf.cell(0, 12, "How to read this book", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for para in [
        "This guide is organised around Python *language features* rather than "
        "around any single AI library, because the features are what transfer "
        "across PyTorch, TensorFlow, JAX, scikit-learn, Hugging Face and the "
        "LLM API clients you will actually use. For every feature you get a "
        "one- or two-line explanation of what it is, why it shows up constantly "
        "in AI code, and a small program you can run.",
        "Every code block marked with an OUTPUT panel was executed while this "
        "PDF was being generated. The green panel is the real captured output "
        "of the grey code above it, so if it prints here, it runs on your "
        "machine too (Python 3.10+ recommended; the numeric chapters use "
        "NumPy and pandas).",
        "The examples are deliberately tiny and dependency-light. Where a line "
        "shows how a real framework uses a feature (for example a PyTorch "
        "`nn.Module` or an async LLM call), it is clearly marked as "
        "illustrative and is not executed.",
    ]:
        pdf.para(para)
    pdf.ln(4)
    pdf.note("Tip for Kindle: use the built-in table of contents (the ≡ "
             "navigation menu) to jump between chapters. Every chapter and part "
             "is a bookmark. Rotating to landscape enlarges the code panels.")
    pdf.in_frontmatter = False


def render_block(pdf, blk, stats):
    t = blk["t"]
    if t == "p":
        pdf.para(blk["text"])
    elif t == "h":
        pdf.subhead(blk["text"])
    elif t == "bullets":
        pdf.bullets(blk["items"])
    elif t == "note":
        pdf.note(blk["text"])
    elif t == "code":
        output = None
        if blk.get("run"):
            out, rc = run_example(blk["code"])
            output = out
            stats["run"] += 1
            expect_err = blk.get("expect_error", False)
            if (rc != 0) != expect_err:
                stats["fail"].append((pdf.cur_chapter, blk.get("caption", ""),
                                      rc, out[:400]))
        pdf.code_block(blk["code"], output=output, caption=blk.get("caption"))
    else:
        raise ValueError(f"unknown block type: {t}")


def main():
    t0 = time.time()
    parts = load_content()
    pdf = Book(TITLE, SUBTITLE)
    pdf.set_title(f"{TITLE}: {SUBTITLE}")
    pdf.set_author(AUTHOR)
    pdf.set_lang("en")

    cover(pdf)
    preface(pdf)

    pdf.in_frontmatter = True
    pdf.add_page()                       # ToC gets its own page(s)
    pdf._toc_reserved = 3
    pdf.insert_toc_placeholder(render_toc, pages=pdf._toc_reserved)
    pdf.in_frontmatter = False

    stats = {"run": 0, "fail": [], "chapters": 0, "examples": 0}
    chapter_no = 0
    for pnum, (part_name, chapters) in enumerate(parts, start=1):
        pdf.start_part(part_name, _roman(pnum))
        for ch in chapters:
            chapter_no += 1
            stats["chapters"] += 1
            pdf.start_chapter(chapter_no, ch["title"], part_name)
            for blk in ch["blocks"]:
                if blk["t"] == "code":
                    stats["examples"] += 1
                render_block(pdf, blk, stats)

    out_path = "python-for-ai-guide.pdf"
    pdf.output(out_path)
    dt = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Built {out_path}")
    print(f"  chapters       : {stats['chapters']}")
    print(f"  code examples  : {stats['examples']}")
    print(f"  examples run   : {stats['run']}")
    print(f"  pages          : {pdf.page_no()}")
    print(f"  build time     : {dt:.1f}s")
    if stats["fail"]:
        print(f"\n  !! {len(stats['fail'])} UNEXPECTED FAILURES:")
        for ch, cap, rc, out in stats["fail"]:
            print(f"   - [{ch}] {cap} (rc={rc})")
            print(f"       {out.splitlines()[-1] if out else ''}")
        sys.exit(1)
    else:
        print("  all runnable examples executed cleanly ✔")
    print("="*60)


def _roman(n):
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    res = ""
    for val, sym in numerals:
        while n >= val:
            res += sym
            n -= val
    return res


if __name__ == "__main__":
    main()
