#!/usr/bin/env python3
"""Render a Markdown report to one self-contained HTML file.

Why a generator instead of a hand-written page: the Markdown is the single
source of truth (it is what gets diffed and reviewed). A hand-maintained HTML
file drifts from it, and the pretty copy then wins by default -- exactly
backwards for a report whose subject is measurement discipline.

What it does
  1. python-markdown (tables / fenced_code / sane_lists / attr_list / toc);
  2. two-level sticky sidebar TOC built from toc_tokens;
  3. wraps wide tables in a horizontal-scroll container;
  4. colours verdict words inside table cells (通过/成立/证伪/不可识别/...);
  5. wraps the result in one HTML shell with screen + print CSS.

No figures are inlined: the report has none. If one is added later, the builder
refuses rather than shipping a page with broken image links.

Usage
    python tools/build_report_html.py --src notes/REPORT.md --out notes/REPORT.html
    python tools/build_report_html.py --src /abs/REPORT.md --out /abs/REPORT.html \
        --eyebrow "..." --subtitle "..." --chip "10 run" --chip "94.7 GPU-h"

Requires: pip install markdown   (not a project dependency; the error below is
explicit instead of a bare ImportError so a fresh clone knows what to install).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
:root{
  --ink:#1b1c1e; --ink-2:#4a4d52; --ink-3:#74797f;
  --bg:#f4f4f2; --card:#ffffff; --line:#e2e1dd; --line-2:#cfcec9;
  --accent:#2f5d8a; --accent-soft:#eaf1f8;
  --ok:#1f6b45; --ok-bg:#e8f3ec;
  --bad:#9b2c2c; --bad-bg:#fbeceb;
  --warn:#8a5a1a; --warn-bg:#fbf1e2;
  --explore:#5b4b8a; --explore-bg:#efecf7;
  --mono:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  --sans:"Inter","Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans SC",system-ui,sans-serif;
  --serif:"Source Han Serif SC","Songti SC","Georgia","Noto Serif SC",serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.78;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1280px;margin:0 auto;padding:0 28px 96px;}
.layout{display:grid;grid-template-columns:206px minmax(0,1fr);gap:48px;align-items:start;}
main{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:44px 54px 58px;box-shadow:0 1px 2px rgba(0,0,0,.03);min-width:0;}

/* ---------- header ---------- */
header.masthead{padding:54px 0 28px;}
.eyebrow{font-size:12.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3);margin-bottom:14px}
h1.title{font-family:var(--serif);font-size:33px;line-height:1.32;margin:0 0 10px;letter-spacing:-.01em;font-weight:700}
.subtitle{font-size:15.5px;color:var(--ink-2);margin:0 0 20px;max-width:80ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:4px}
.chip{font-size:12.5px;color:var(--ink-2);background:#fff;border:1px solid var(--line);
      border-radius:999px;padding:3px 11px;white-space:nowrap}
.chip b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:16px;font-size:12.5px;color:var(--ink-3)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block}

/* ---------- toc ---------- */
nav.toc{position:sticky;top:24px;max-height:calc(100vh - 48px);overflow:auto;
        font-size:13px;padding:8px 4px 8px 0;}
nav.toc .toc-h{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;
               color:var(--ink-3);margin:0 0 12px;padding-left:10px}
nav.toc ul{list-style:none;margin:0;padding:0}
nav.toc li{margin:0}
nav.toc a{display:block;padding:5px 10px;color:var(--ink-2);text-decoration:none;
          border-left:2px solid transparent;line-height:1.45}
nav.toc a:hover{color:var(--accent);background:var(--accent-soft)}
nav.toc a.active{color:var(--accent);border-left-color:var(--accent);
                 background:var(--accent-soft);font-weight:600}
nav.toc ul ul a{padding-left:22px;font-size:12.3px;color:var(--ink-3)}

/* ---------- typography ---------- */
main h2{font-family:var(--serif);font-size:24px;font-weight:700;margin:52px 0 16px;
        padding-top:22px;border-top:1px solid var(--line);scroll-margin-top:20px}
main h2:first-of-type{border-top:0;padding-top:0;margin-top:8px}
main h3{font-size:17.5px;font-weight:650;margin:32px 0 10px;color:#26282b;scroll-margin-top:20px}
main h4{font-size:15px;margin:22px 0 8px;scroll-margin-top:20px}
main p{margin:0 0 15px}
main a{color:var(--accent);text-decoration:none;border-bottom:1px solid #c9d8e8}
main a:hover{border-bottom-color:var(--accent)}
strong{font-weight:650;color:#111}
em{font-style:normal;background:#fdf6d8;padding:0 2px}
hr{border:0;border-top:1px solid var(--line);margin:36px 0}
ul,ol{margin:0 0 16px;padding-left:24px}
li{margin:5px 0}
code{font-family:var(--mono);font-size:.86em;background:#f2f2ef;border:1px solid #e6e5e1;
     border-radius:4px;padding:1px 5px;color:#3d4046;word-break:break-word}
pre{background:#1f2226;color:#e8e8e6;padding:16px 18px;border-radius:8px;overflow:auto;
    font-size:12.8px;line-height:1.62;margin:0 0 18px}
pre code{background:none;border:0;color:inherit;padding:0;font-size:12.8px}

/* ---------- blockquote ---------- */
blockquote{margin:0 0 18px;padding:14px 18px;background:#faf9f6;border-left:3px solid var(--line-2);
           color:var(--ink-2);border-radius:0 6px 6px 0;font-size:15px}
blockquote p:last-child{margin-bottom:0}
blockquote.lead{background:var(--accent-soft);border-left-color:var(--accent);color:#2b3d4f}

/* ---------- tables ---------- */
.tw{overflow-x:auto;margin:0 0 22px;border:1px solid var(--line);border-radius:8px;
    background:#fff}
table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.58;background:#fff}
thead th{background:#f7f7f4;text-align:left;font-weight:650;color:#33363a;
         padding:8px 11px;border-bottom:1px solid var(--line-2);white-space:nowrap;
         position:sticky;top:0}
tbody td{padding:7px 11px;border-bottom:1px solid #eeedea;vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:#fbfbf9}
td:not(:first-child){font-variant-numeric:tabular-nums}

/* ---------- verdict chips inside cells ---------- */
.v{display:inline-block;font-size:11.5px;font-weight:650;line-height:1.5;
   padding:1px 7px;border-radius:4px;border:1px solid transparent;white-space:nowrap}
.v-ok{color:var(--ok);background:var(--ok-bg);border-color:#cfe4d7}
.v-bad{color:var(--bad);background:var(--bad-bg);border-color:#f0d3d0}
.v-warn{color:var(--warn);background:var(--warn-bg);border-color:#f0dfc4}
.v-explore{color:var(--explore);background:var(--explore-bg);border-color:#ddd6ef}

/* ---------- footer ---------- */
footer.foot{border-top:1px solid var(--line);margin-top:54px;padding-top:16px;
            font-size:12.5px;color:var(--ink-3);display:flex;flex-wrap:wrap;gap:14px;
            justify-content:space-between}
footer.foot code{font-size:.92em}

/* ---------- responsive / print ---------- */
@media (max-width:1080px){
  .layout{grid-template-columns:1fr}
  nav.toc{display:none}
  main{padding:30px 22px 40px}
  .wrap{padding:0 14px 60px}
}
@media print{
  body{background:#fff;font-size:10.3pt}
  nav.toc,.legend,footer.foot .build{display:none}
  .wrap{max-width:none;padding:0}
  .layout{display:block}
  main{border:0;box-shadow:none;padding:0}
  main h2{border-top:0;page-break-after:avoid}
  main h3,main h4{page-break-after:avoid}
  .tw{overflow:visible;border:0}
  table{font-size:9.2pt}
  thead th{position:static}
  tr,blockquote,pre{page-break-inside:avoid}
}
"""

JS = """
(function(){
  var links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a'));
  var targets = links.map(function(a){
    return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
  }).filter(Boolean);
  if(!targets.length) return;
  function setActive(id){
    links.forEach(function(a){
      var on = decodeURIComponent(a.getAttribute('href').slice(1)) === id;
      a.classList.toggle('active', on);
    });
  }
  var seen = {};
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting) seen[e.target.id] = e.boundingClientRect.top;
      else delete seen[e.target.id];
    });
    var ids = Object.keys(seen);
    if(ids.length){
      ids.sort(function(a,b){ return seen[a] - seen[b]; });
      setActive(ids[0]);
    }
  }, {rootMargin:'-10% 0px -72% 0px', threshold:[0,1]});
  targets.forEach(function(t){ io.observe(t); });
})();
"""

VERDICTS = {
    "通过": "ok", "成立": "ok", "达成": "ok", "已验证": "ok",
    "证伪": "bad", "不成立": "bad", "未复现": "bad", "失效": "bad", "相反": "bad",
    "部分": "warn", "未检验": "warn", "不可识别": "warn", "不确定": "warn",
    "未达": "warn", "待定": "warn", "不可判定": "warn", "限定": "warn",
    "探索性": "explore", "未预注册": "explore", "事后": "explore",
}

IMG_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$", re.M)


def wrap_tables(body: str) -> str:
    return body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")


def chipify_cells(body: str) -> str:
    """Colour whole-cell verdict words (<strong>通过</strong>) as small chips."""

    def per_cell(cm: re.Match) -> str:
        def repl(m: re.Match) -> str:
            word = m.group(1)
            cls = VERDICTS.get(word)
            if not cls:
                return m.group(0)
            return f'<strong class="v v-{cls}">{word}</strong>'

        return re.sub(r"<strong>([^<]{1,8})</strong>", repl, cm.group(0))

    return re.sub(r"<td[^>]*>.*?</td>", per_cell, body, flags=re.S)


def mark_lead_blockquote(body: str) -> str:
    """The first blockquote run gets the accent style (it is the report header)."""
    first = body.find("<blockquote>")
    if first == -1:
        return body
    return body[:first] + '<blockquote class="lead">' + body[first + len("<blockquote>"):]


def render_toc(tokens: list[dict]) -> str:
    out = ['<nav class="toc" aria-label="目录"><p class="toc-h">目录</p><ul>']
    for t in tokens:
        if t["level"] > 2:
            continue
        out.append(f'<li><a href="#sec-{t["id"]}">{html.escape(t["name"])}</a>')
        kids = [c for c in t.get("children", []) if c["level"] == 3]
        if kids:
            out.append("<ul>")
            for c in kids:
                out.append(f'<li><a href="#sec-{c["id"]}">{html.escape(c["name"])}</a></li>')
            out.append("</ul>")
        out.append("</li>")
    out.append("</ul></nav>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="REPORT.md")
    ap.add_argument("--out", default="REPORT.html")
    ap.add_argument("--eyebrow", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--chip", action="append", default=[],
                    help="Pill in the masthead. Repeatable.")
    ap.add_argument("--legend", action="store_true",
                    help="Show the verdict-colour legend under the chips.")
    args = ap.parse_args()

    try:
        import markdown  # noqa: PLC0415  (deliberate: one clear error message)
    except ImportError:
        print("[fatal] the 'markdown' package is required:\n"
              "        pip install markdown\n"
              "        (kept out of pyproject: only the report build needs it)",
              file=sys.stderr)
        return 2

    src_path = Path(args.src)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    src_path = src_path.resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path = out_path.resolve()

    src = src_path.read_text(encoding="utf-8")
    stray = IMG_RE.search(src)
    if stray:
        raise SystemExit(f"[fatal] image reference found but this builder inlines no "
                         f"figures: {stray.group(0)!r}\n"
                         f"        either drop the figure or extend the builder")
    md_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]

    md = markdown.Markdown(
        # nl2br: these reports are written with one fact per source line inside a
        # paragraph (日期 / 代码基准 / 执行者标注 …). Without it those separate
        # lines would be joined into one run-on sentence.
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc",
                    "md_in_html", "nl2br"],
        extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
    )
    body = md.convert(src)
    body = re.sub(r'<(h[1-6]) id="([^"]+)"', r'<\1 id="sec-\2"', body)
    body = wrap_tables(body)
    body = chipify_cells(body)

    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    title = re.sub(r"<[^>]+>", "", h1.group(1)).strip() if h1 else src_path.stem
    if h1:
        body = body[: h1.start()] + body[h1.end():]
    body = mark_lead_blockquote(body)

    toc_html = render_toc(md.toc_tokens or [])

    def walk(nodes: list[dict]):
        for n in nodes:
            yield n
            yield from walk(n.get("children", []))

    flat = list(walk(md.toc_tokens or []))   # h3 is nested under h2, not top-level
    n_h2 = sum(1 for t in flat if t["level"] == 2)
    n_h3 = sum(1 for t in flat if t["level"] == 3)
    chips = "".join(f'<span class="chip">{c}</span>' for c in args.chip)
    legend = ('<div class="legend">'
              '<span><i class="dot" style="background:#cfe4d7"></i>通过 / 成立</span>'
              '<span><i class="dot" style="background:#f0d3d0"></i>证伪 / 不成立</span>'
              '<span><i class="dot" style="background:#f0dfc4"></i>部分 / 不确定 / 不可识别</span>'
              '<span><i class="dot" style="background:#ddd6ef"></i>探索性 / 事后登记</span>'
              '</div>') if args.legend else ""
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="masthead">
  <div class="eyebrow">{args.eyebrow or "Experiment report"}</div>
  <h1 class="title">{html.escape(title)}</h1>
  <p class="subtitle">{args.subtitle}</p>
  <div class="chips">{chips}</div>
  {legend}
</header>

<div class="layout">
{toc_html}
<main>
{body}
<footer class="foot">
  <div>事实源：<code>{html.escape(src_path.name)}</code> · sha256:{md_sha} · 由 <code>tools/build_report_html.py</code> 生成，请勿手改本文件</div>
  <div class="build">{n_h2} 节 / {n_h3} 小节 · built {built}</div>
</footer>
</main>
</div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" on purpose: on Windows the default would write CRLF and every
    # regenerated file would then differ from the committed one in every line.
    out_path.write_text(doc, encoding="utf-8", newline="\n")
    print(f"[ok] {out_path}  {len(doc) / 1024:.0f} KB")
    print(f"     source {src_path.name} sha256:{md_sha}  {len(src.splitlines())} lines")
    print(f"     toc: {n_h2} h2 / {n_h3} h3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
