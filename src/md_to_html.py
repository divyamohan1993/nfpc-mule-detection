"""Convert markdown report to professional PDF-ready HTML.
Times New Roman 12pt, 1.5 line height, justified, 1-inch margins,
page numbers, headers/footers, proper section formatting.
"""
import markdown2
import base64
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MD_PATH = os.path.join(ROOT_DIR, "reports", "NFPC_Phase1_EDA_Report.md")
HTML_PATH = os.path.join(ROOT_DIR, "reports", "NFPC_Phase1_EDA_Report.html")
BASE_DIR = os.path.join(ROOT_DIR, "reports")

with open(MD_PATH, 'r', encoding='utf-8') as f:
    md_content = f.read()

html_body = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'header-ids', 'toc'])

# Embed images as base64
def embed_image(match):
    img_path = match.group(1)
    full_path = os.path.join(BASE_DIR, img_path)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = os.path.splitext(full_path)[1].lower()
        mime = 'image/png' if ext == '.png' else 'image/jpeg'
        return f'src="data:{mime};base64,{data}"'
    return match.group(0)

html_body = re.sub(r'src="([^"]+)"', embed_image, html_body)

# Convert remaining markdown images (if any) to HTML img tags
def embed_md_image(match):
    alt = match.group(1)
    path = match.group(2)
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{data}" alt="{alt}"/>'
    return match.group(0)

html_body = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed_md_image, html_body)

# Post-process: wrap <img> + following italic caption into <figure>
# Pattern: <p><img .../></p> followed by <p><em>Figure X.Y: ...</em></p>
def wrap_figures(html):
    pattern = r'<p>(<img[^>]+/>)</p>\s*<p><em>(Figure\s+\d+\.\d+:\s*[^<]+)</em></p>'
    def make_figure(m):
        img_tag = m.group(1)
        caption = m.group(2)
        return f'<figure>{img_tag}<figcaption>{caption}</figcaption></figure>'
    return re.sub(pattern, make_figure, html)

html_body = wrap_figures(html_body)

# Also handle images without our specific caption (wrap in figure without caption)
def wrap_bare_images(html):
    pattern = r'<p>(<img[^>]+/>)</p>'
    def make_figure(m):
        img_tag = m.group(1)
        return f'<figure>{img_tag}</figure>'
    return re.sub(pattern, make_figure, html)

html_body = wrap_bare_images(html_body)

# Post-process: style table captions (italic "Table X.Y: ..." paragraphs)
def style_table_captions(html):
    pattern = r'<p><em>(Table\s+\d+\.\d+:\s*[^<]+)</em></p>'
    def make_caption(m):
        caption = m.group(1)
        return f'<p class="table-caption">{caption}</p>'
    return re.sub(pattern, make_caption, html)

html_body = style_table_captions(html_body)

# Wrap Table of Contents in a dedicated page div
toc_pattern = r'(<h2[^>]*id="table-of-contents"[^>]*>.*?</h2>\s*<ol>.*?</ol>)'
toc_match = re.search(toc_pattern, html_body, re.DOTALL)
if toc_match:
    toc_html = toc_match.group(0)
    html_body = html_body.replace(toc_html, f'<div class="toc-page">{toc_html}</div>')

# Also handle if ToC uses <ul> instead of <ol>
toc_pattern2 = r'(<h2[^>]*id="table-of-contents"[^>]*>.*?</h2>\s*<ul>.*?</ul>)'
toc_match2 = re.search(toc_pattern2, html_body, re.DOTALL)
if toc_match2 and 'toc-page' not in html_body[:html_body.find('id="table-of-contents"')+200]:
    toc_html2 = toc_match2.group(0)
    html_body = html_body.replace(toc_html2, f'<div class="toc-page">{toc_html2}</div>')

# Wrap in professional HTML
html_full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>NFPC Phase 1 EDA Report - dmj.one</title>
<style>
/* ========== PAGE SETUP ========== */
@page {{
    size: A4;
    margin: 1in;

    @top-center {{
        content: "National Fraud Prevention Challenge - Phase 1 EDA Report";
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        color: #666;
        padding-bottom: 4pt;
    }}

    @bottom-left {{
        content: "Team: dmj.one";
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        color: #666;
    }}

    @bottom-center {{
        content: "Confidential";
        font-family: 'Times New Roman', Times, serif;
        font-size: 8pt;
        color: #999;
        font-style: italic;
    }}

    @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        color: #666;
    }}
}}

@page :first {{
    @top-center {{ content: none; }}
    @bottom-left {{ content: none; }}
    @bottom-center {{ content: none; }}
    @bottom-right {{ content: none; }}
}}

/* ========== TYPOGRAPHY ========== */
body {{
    font-family: 'Times New Roman', Times, 'DejaVu Serif', Georgia, serif;
    font-size: 12pt;
    line-height: 1.5;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
    max-width: 100%;
    margin: 0;
    padding: 0;
}}

p {{
    margin: 0 0 6pt 0;
    orphans: 3;
    widows: 3;
    text-align: justify;
}}

/* ========== HEADINGS ========== */
h1 {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 22pt;
    font-weight: bold;
    color: #1a1a1a;
    text-align: center;
    margin: 0 0 6pt 0;
    padding: 0;
    border: none;
    page-break-before: auto;
    line-height: 1.3;
}}

h2 {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 16pt;
    font-weight: bold;
    color: #1a1a1a;
    margin: 18pt 0 8pt 0;
    page-break-after: avoid;
    line-height: 1.3;
}}

h3 {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 13pt;
    font-weight: bold;
    color: #333;
    margin: 14pt 0 6pt 0;
    page-break-after: avoid;
    line-height: 1.3;
}}

h4 {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    font-weight: bold;
    font-style: italic;
    color: #333;
    margin: 10pt 0 5pt 0;
    page-break-after: avoid;
}}

/* ========== TITLE BLOCK ========== */
h1 + p {{
    text-align: center;
    margin-bottom: 4pt;
}}

h1 + p + p {{
    text-align: center;
    margin-bottom: 4pt;
}}

h1 + p + p + p {{
    text-align: center;
    margin-bottom: 16pt;
}}

/* ========== TABLES ========== */
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0 10pt 0;
    font-size: 10pt;
    line-height: 1.4;
    page-break-inside: auto;
}}

thead th {{
    background-color: #2c3e50;
    color: #ffffff;
    font-weight: bold;
    padding: 5pt 6pt;
    text-align: left;
    font-size: 10pt;
    border: 0.5pt solid #2c3e50;
}}

tbody td {{
    padding: 4pt 6pt;
    border: 0.5pt solid #bdc3c7;
    vertical-align: top;
}}

tbody tr:nth-child(even) {{
    background-color: #f7f9fb;
}}

tbody tr:nth-child(odd) {{
    background-color: #ffffff;
}}

/* ========== FIGURES ========== */
figure {{
    margin: 10pt 0;
    text-align: center;
    page-break-inside: avoid;
}}

figure img {{
    max-width: 95%;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 0.5pt solid #ccc;
}}

figcaption {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    color: #333;
    font-style: italic;
    margin-top: 4pt;
    text-align: center;
    line-height: 1.3;
}}

img {{
    max-width: 95%;
    height: auto;
    display: block;
    margin: 8pt auto;
    page-break-inside: avoid;
}}

/* ========== CODE ========== */
code {{
    font-family: 'Courier New', Courier, monospace;
    font-size: 10pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
    border: 0.5pt solid #ddd;
    border-radius: 2pt;
}}

pre {{
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    background-color: #f8f8f8;
    border: 0.5pt solid #ddd;
    padding: 8pt;
    margin: 8pt 0;
    overflow-x: auto;
    line-height: 1.3;
    page-break-inside: avoid;
    text-align: left;
}}

pre code {{
    background: none;
    border: none;
    padding: 0;
}}

/* ========== BLOCKQUOTES ========== */
blockquote {{
    margin: 8pt 0 8pt 16pt;
    padding: 6pt 10pt;
    border-left: 3pt solid #2c3e50;
    background-color: #f9f9f9;
    font-style: italic;
    color: #444;
    page-break-inside: avoid;
}}

blockquote p {{
    margin: 3pt 0;
}}

/* ========== LISTS ========== */
ul, ol {{
    margin: 4pt 0 8pt 0;
    padding-left: 22pt;
    text-align: left;
}}

li {{
    margin-bottom: 3pt;
    line-height: 1.5;
}}

/* ========== HORIZONTAL RULES ========== */
hr {{
    display: none;
}}

/* ========== TABLE CAPTIONS ========== */
.table-caption {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    color: #333;
    font-style: italic;
    text-align: center;
    margin: 2pt 0 10pt 0;
    line-height: 1.3;
}}

/* ========== EMPHASIS ========== */
strong {{
    font-weight: bold;
    color: #1a1a1a;
}}

em {{
    font-style: italic;
}}

/* ========== LINKS ========== */
a {{
    color: #2c3e50;
    text-decoration: none;
}}

a:hover {{
    text-decoration: underline;
}}

/* ========== PAGE BREAK HELPERS ========== */
h2 {{
    page-break-before: always;
}}

h2:first-of-type {{
    page-break-before: auto;
}}

/* Subtopics flow continuously - no page breaks */
h3, h4 {{
    page-break-before: auto;
    page-break-after: avoid;
}}

/* Keep table header row with first data row */
thead {{
    display: table-header-group;
}}

/* ========== TABLE OF CONTENTS ========== */
.toc-page {{
    page-break-before: always;
    page-break-after: always;
}}

.toc-page h2 {{
    page-break-before: auto;
    text-align: center;
    font-size: 18pt;
    margin-top: 36pt;
}}

.toc-page ol, .toc-page ul {{
    font-size: 12pt;
    line-height: 2.0;
    list-style-type: none;
    padding-left: 0;
    max-width: 85%;
    margin: 16pt auto;
}}

.toc-page li {{
    padding: 3pt 0;
    border-bottom: 0.5pt dotted #ccc;
}}

.toc-page li a {{
    color: #2c3e50;
    text-decoration: none;
    font-size: 12pt;
}}

.toc-page li a:hover {{
    text-decoration: underline;
}}

/* ========== COVER PAGE ========== */
.cover-page {{
    text-align: center;
    padding-top: 2.5in;
    page-break-after: always;
}}

.cover-page h1 {{
    font-size: 28pt;
    margin-bottom: 12pt;
    border: none;
}}

.cover-page .subtitle {{
    font-size: 16pt;
    color: #555;
    margin-bottom: 30pt;
}}

.cover-page .meta {{
    font-size: 12pt;
    color: #333;
    line-height: 2;
}}

.cover-page .meta strong {{
    color: #333;
}}

.cover-page .logo-line {{
    margin-top: 40pt;
    padding-top: 20pt;
    border-top: 1pt solid #ccc;
    font-size: 10pt;
    color: #888;
}}

/* ========== PRINT TWEAKS ========== */
@media print {{
    body {{
        font-size: 12pt;
    }}
    table, figure, img {{
        page-break-inside: avoid;
    }}
    .toc-page {{
        page-break-after: always;
    }}
    pre {{
        white-space: pre-wrap;
        word-wrap: break-word;
    }}
}}
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-page">
    <h1>National Fraud Prevention Challenge</h1>
    <div class="subtitle">Phase 1 - Exploratory Data Analysis Report</div>
    <hr style="width: 40%; margin: 20pt auto; border-top: 2pt solid #2c3e50;"/>
    <div class="meta">
        <strong>Team:</strong> dmj.one<br/>
        <strong>Date:</strong> February 24, 2026<br/>
        <strong>Dataset:</strong> NFPC Phase 1 - 20% Representative Sample<br/>
        <strong>Challenge Host:</strong> Reserve Bank Innovation Hub (RBIH)
    </div>
    <div class="logo-line">
        Submitted as part of the IITD TRYST Hackathon<br/>
        in association with Reserve Bank Innovation Hub
    </div>
</div>

<!-- REPORT BODY -->
{html_body}

</body>
</html>"""

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html_full)

print(f"HTML report saved to: {HTML_PATH}")
print(f"File size: {os.path.getsize(HTML_PATH) / 1024 / 1024:.1f} MB")
