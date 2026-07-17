"""Render the Markdown findings to PDF.

No pandoc/LaTeX/wkhtmltopdf on this machine, so: markdown -> HTML -> PyMuPDF's Story API, which
ships with the MuPDF layout engine. Pure Python, no external binaries, reproducible anywhere the
repo's requirements install.

Usage:  python src/make_pdf.py docs/FINDINGS.md docs/FINDINGS.pdf "Title"
"""
import sys, re, html
import markdown
import fitz

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #111; margin: 0 0 2pt 0; }
h2 { font-size: 13pt; color: #111; margin: 16pt 0 4pt 0; border-bottom: 1px solid #bbb; }
h3 { font-size: 11pt; color: #222; margin: 11pt 0 3pt 0; }
h4 { font-size: 10pt; color: #333; margin: 8pt 0 2pt 0; }
p { margin: 0 0 6pt 0; }
li { margin: 0 0 3pt 0; }
code { font-family: monospace; font-size: 8.5pt; background: #f0f0f0; color: #a02020; }
pre { font-family: monospace; font-size: 8pt; background: #f6f6f6; padding: 5pt;
      margin: 4pt 0 8pt 0; line-height: 1.3; }
pre code { background: none; color: #1a1a1a; }
table { margin: 4pt 0 9pt 0; font-size: 8.5pt; }
th { background: #e8e8e8; font-weight: bold; text-align: left; padding: 3pt 6pt;
     border-bottom: 1px solid #888; }
td { padding: 2.5pt 6pt; border-bottom: 1px solid #ddd; }
blockquote { color: #444; margin: 4pt 0 8pt 10pt; }
hr { margin: 10pt 0; }
strong { font-weight: bold; }
a { color: #1a4f8a; }
"""


def md_to_html(md_text):
    # Story's CSS subset has no ::marker, so bullets are rendered literally.
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    # Story chokes on <br />-in-<td> and on nested <p> inside <li>; flatten them.
    body = body.replace("<br />", " ").replace("<br>", " ")
    body = re.sub(r"<li>\s*<p>(.*?)</p>\s*</li>", r"<li>\1</li>", body, flags=re.S)
    return f"<html><head><style>{CSS}</style></head><body>{body}</body></html>"


def render(md_path, pdf_path, title):
    md_text = open(md_path, encoding="utf-8").read()
    doc_html = md_to_html(md_text)

    story = fitz.Story(html=doc_html, user_css=CSS)
    writer = fitz.DocumentWriter(pdf_path)
    page_w, page_h = fitz.paper_size("a4")
    margin = 44
    frame = fitz.Rect(margin, margin, page_w - margin, page_h - margin)

    pages = 0
    more = 1
    while more:
        dev = writer.begin_page(fitz.Rect(0, 0, page_w, page_h))
        more, _ = story.place(frame)
        story.draw(dev)
        writer.end_page()
        pages += 1
        if pages > 200:
            raise RuntimeError("runaway pagination")
    writer.close()

    # stamp metadata + page numbers
    d = fitz.open(pdf_path)
    d.set_metadata({"title": title, "author": "Multimodal Emotion Detection — reproduction study",
                    "subject": "Actor-independent re-evaluation of Emotion Unlocked (SSRN 5274911)"})
    for i, page in enumerate(d):
        page.insert_text((page_w - margin - 46, page_h - 24),
                         f"{i + 1} / {d.page_count}", fontsize=7.5, color=(0.45, 0.45, 0.45))
    d.saveIncr()
    d.close()
    return pdf_path, pages


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/FINDINGS.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else "docs/FINDINGS.pdf"
    ttl = sys.argv[3] if len(sys.argv) > 3 else "Findings"
    p, n = render(src, dst, ttl)
    import os
    print(f"wrote {p}  ({n} pages, {os.path.getsize(p) / 1024:.0f} KB)")
