"""
Step 4 (HSRI adaptation of 04_Datamapplot_Visualization.ipynb)
==================================================================

Builds the interactive research map from the clustered/named HSRI projects.
Single clustering layer (see step 3 notes on why the original 5-layer
hierarchy isn't reproduced). Tooltip rebuilt around HSRI fields (PI, research
design, policy/academic/social/economic impact dimensions) instead of the
original paper fields (authors/journal/DOI). Deliberately excludes grant
budget amounts and implementation obstacles/challenges.

Input:  data/hsri/hsri_clustered_named.pkl (from step 3)
Output: hsri_research_map.html (repo root; copy over index.html to deploy)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import datamapplot

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "hsri" / "hsri_clustered_named.pkl"
TRANSLATIONS_TH_FILE = ROOT / "data" / "hsri" / "hsri_translations_th.json"
OUT_FILE = ROOT / "hsri_research_map.html"

IMPACT_DIM_TH = {"Policy": "นโยบาย", "Academic": "วิชาการ", "Social": "สังคม", "Economic": "เศรษฐกิจ"}


def translate_impact_dims(impact_dims_en):
    if not impact_dims_en or impact_dims_en == "Not reported":
        return "ไม่ได้รายงาน"
    parts = [p.strip() for p in impact_dims_en.split(",")]
    return ", ".join(IMPACT_DIM_TH.get(p, p) for p in parts)

matplotlib.rcParams["figure.dpi"] = 72


def format_tags_as_html(tags, color):
    if not isinstance(tags, list) or not tags:
        return ""
    items = "".join(
        f'<div style="background-color:{color};color:#fff;border-radius:6px;'
        f'padding:2px 10px;font-size:7pt;font-weight:600;display:inline-block;">{t}</div>'
        for t in tags
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:5px;">{items}</div>'


def format_yn_dims(row):
    dims = []
    for key, label in [
        ("policy_dimension_yn", "Policy"),
        ("academic_dimension_yn", "Academic"),
        ("social_dimension_yn", "Social"),
        ("economic_dimension_yn", "Economic"),
    ]:
        if str(row.get(key, "")).strip().lower() == "yes":
            dims.append(label)
    return ", ".join(dims) if dims else "Not reported"


TOOLTIP_CSS = """
    display: none !important;
"""
# deck.gl's native hover-driven tooltip (this .deck-tooltip class) is permanently
# hidden. It's wired to getTooltip, which gets re-evaluated on every native
# mousemove/mouseout the browser fires -- including the synthetic ones iOS
# generates around a touch tap, which raced with any tap-triggered show and either
# hid it immediately or, once that race was patched, broke pan/zoom and tap
# recognition entirely (each attempt documented in git log). Rather than continue
# patching around that fragile hover machinery, the tooltip is now driven entirely
# by CUSTOM_TOOLTIP_TEMPLATE + ON_CLICK_JS below, via deck.gl's native onClick prop
# -- a one-shot callback, never re-evaluated by hover state, so there's nothing for
# stray touch-emulation events to race with. Explicit close button instead of
# double-tap, for the same reliability reason.

# UI labels for each language. Translatable data fields are suffixed "_th" in
# extra_point_data for Thai (e.g. summary_short_th); pi_name and year are shared
# (pi_name is already Thai in the source data -- a person's name isn't
# translated -- and year is just a number).
UI_LABELS = {
    "en": {
        "pi": "PI", "year": "Year", "summary": "Summary",
        "simple": "Simple Explanation", "findings": "Key Findings",
        "methodology": "Methodology", "techniques": "Techniques & Tools",
        "concepts": "Key Concepts", "impact": "Impact Dimensions",
    },
    "th": {
        "pi": "หัวหน้าโครงการ", "year": "ปี", "summary": "สรุป",
        "simple": "คำอธิบายอย่างง่าย", "findings": "ผลการวิจัยที่สำคัญ",
        "methodology": "วิธีการวิจัย", "techniques": "เทคนิคและเครื่องมือ",
        "concepts": "แนวคิดหลัก", "impact": "มิติผลกระทบ",
    },
}


def build_tooltip_template(lang):
    """lang: 'en' or 'th'. Field placeholders get a _th suffix for Thai (matching
    the extra_point_data columns produced by 06_build_bilingual.py); pi_name/year
    are unsuffixed and shared between languages."""
    labels = UI_LABELS[lang]
    suffix = "" if lang == "en" else f"_{lang}"
    f = lambda col: f"{{{col}{suffix}}}"  # e.g. f("summary_short") -> "{summary_short_th}"

    return f"""
<div style="
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    width: min(750px, 92vw);
    max-height: min(80vh, 600px);
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    color: #0f172a;
    text-align: left;
    display: flex;
    align-items: stretch;
    pointer-events: auto;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 10000;
">
    <button onclick="document.getElementById('custom-tooltip-root').style.display='none'" style="
        position: absolute;
        top: 8px;
        right: 8px;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: none;
        background: #f1f5f9;
        color: #475569;
        font-size: 16px;
        line-height: 1;
        cursor: pointer;
        z-index: 1;
    ">&times;</button>
    <div style="
        width: 58%;
        flex-shrink: 0;
        padding: 20px;
        overflow-y: auto;
        box-sizing: border-box;
    ">
        <h2 style="margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #1e293b; line-height: 1.3;">
            {f("hover_text")}
        </h2>

        <div style="font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 6px;">
            {labels["pi"]}: {{pi_name}} | {labels["year"]}: {{year}}
        </div>

        <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px;">
            <div style="background-color: #2a5982; color: #fff; border-radius: 6px; padding: 2px 10px; font-size: 7pt; font-weight: 600;">
                {f("domain")}
            </div>
            <div style="background-color: #d6ac4b; color: #fff; border-radius: 6px; padding: 2px 10px; font-size: 7pt; font-weight: 600;">
                {f("data_type")}
            </div>
        </div>

        <div style="font-size: 11px; font-weight: 700; color: #2a5982; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
            {labels["summary"]}
        </div>
        <p style="margin: 0; font-size: 12.5px; line-height: 1.5; color: #334155; text-align: justify;">
            {f("summary_short")}
        </p>
    </div>

    <div style="
        width: 42%;
        flex-shrink: 0;
        padding: 20px;
        overflow-y: auto;
        background-color: #f8fafc;
        border-left: 1px solid #e2e8f0;
        box-sizing: border-box;
    ">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 6px;">{labels["simple"]}</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{f("summary_simple")}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 6px;">{labels["findings"]}</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{f("main_findings")}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 6px;">{labels["methodology"]}</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{f("methodology")}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 5px;">{labels["techniques"]}</div>
            {f("techniques_html")}
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #d6ac4b; margin-bottom: 5px;">{labels["concepts"]}</div>
            {f("concepts_html")}
        </div>

        <div>
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 5px;">{labels["impact"]}</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{f("impact_dims")}</p>
        </div>
    </div>
</div>
"""


TOOLTIP_TEMPLATE_EN = build_tooltip_template("en")
TOOLTIP_TEMPLATE_TH = build_tooltip_template("th")

# Both language variants are rendered into JS vars on every click (cheap -- just
# string building, both branches use the same {index}-driven hoverData lookups),
# then the currently-active one is shown and BOTH are cached on the root element's
# dataset so the language-toggle button can swap between them instantly without
# needing to re-click. Wrapped in backticks (JS template literals) exactly like
# datamapplot's own hover_text_html_template mechanism does internally -- proven
# to survive whatever escaping happens downstream.
ON_CLICK_JS = (
    "var __en = `" + TOOLTIP_TEMPLATE_EN + "`;\n"
    "var __th = `" + TOOLTIP_TEMPLATE_TH + "`;\n"
    "var __root = document.getElementById('custom-tooltip-root');\n"
    "__root.dataset.en = __en;\n"
    "__root.dataset.th = __th;\n"
    "__root.innerHTML = (window.__hsriLang === 'th') ? __th : __en;\n"
    "__root.style.display = 'block';\n"
    # datamapplot wraps this whole snippet as `({index, picked, layer}, event)
    # => { if (picked) { ...this... } }` (see prepare_hover_data in
    # datamapplot's interactive_helpers.py), so `index` is in scope here.
    # Recorded so the hover wiring below can tell "is this the same point
    # that's already shown" without needing to intercept datamap's own
    # onClick at all -- see the note in BILINGUAL_UI_SCRIPT for why.
    "__root.dataset.shownIndex = String(index);"
)


VENDOR_DIR = ROOT / "scripts" / "hsri" / "vendor"


def inline_vendor_scripts(filename):
    """datamapplot hardcodes unpkg.com <script src> tags (one with an unpinned
    '@latest' version). Repointing to another CDN (jsdelivr) still hangs in some
    sandboxed HTML viewers that block ALL external script loading, not just
    unpkg specifically -- the page waits forever for globals that never arrive.
    Make the file fully self-contained instead: inline deck.gl/apache-arrow/d3
    directly as <script> content, sourced from scripts/hsri/vendor/ (populate
    that dir first, e.g. via curl from jsdelivr -- see task.md)."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    replacements = [
        ("https://unpkg.com/deck.gl@9.1/dist.min.js", "deck.gl.min.js"),
        ("https://unpkg.com/apache-arrow@latest/Arrow.es2015.min.js", "Arrow.es2015.min.js"),
        ("https://unpkg.com/d3@latest/dist/d3.min.js", "d3.min.js"),
    ]
    changes = 0
    for url, vendor_filename in replacements:
        vendor_path = VENDOR_DIR / vendor_filename
        tag = f'<script src="{url}">'
        if tag not in content:
            print(f"  ! script tag not found (datamapplot version may have changed): {url}")
            continue
        if not vendor_path.exists():
            print(f"  ! vendor file missing, leaving as CDN reference: {vendor_path}")
            continue
        js = vendor_path.read_text(encoding="utf-8")
        assert "</script" not in js, f"{vendor_filename} contains a literal </script — would break inlining"
        content = content.replace(tag, f"<script>\n{js}\n")
        changes += 1

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Inlined {changes}/{len(replacements)} vendor scripts (fully self-contained, no CDN)")


TITLE_TH = "แผนที่ทุนวิจัย สวรส."
SUB_TITLE_TH = "รายงานปิดโครงการทุน วช. RG4 (2565–2568)"

BILINGUAL_UI_SCRIPT = f"""
<div id="custom-tooltip-root" style="display:none;"></div>
<div id="lang-toggle-container" style="
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 100;
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    padding: 4px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
">
    <button id="lang-toggle-btn" onclick="hsriToggleLang()" style="
        border: none;
        background: #2a5982;
        color: #fff;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
    ">TH</button>
</div>
<script>
(function () {{
  window.__hsriLang = (function () {{
    try {{ return localStorage.getItem('hsriLang') || 'en'; }} catch (e) {{ return 'en'; }}
  }})();

  var TITLE_EN = "HSRI Research Grant Map", TITLE_TH = {TITLE_TH!r};
  var SUBTITLE_EN = "HSRI RG4 Grant Close-out Reports (2022–2025)", SUBTITLE_TH = {SUB_TITLE_TH!r};

  // Cluster/group labels on the map stay English-only in both languages (deck.gl's
  // WebGL TextLayer glyph renderer doesn't apply Thai's combining-mark shaping
  // correctly regardless of font -- verified empirically -- and the HTML-overlay
  // workaround for it was reverted per user request). Only the title/subtitle and
  // the point tooltips are bilingual.

  window.hsriApplyLang = function () {{
    var isTh = window.__hsriLang === 'th';
    var btn = document.getElementById('lang-toggle-btn');
    if (btn) btn.textContent = isTh ? 'EN' : 'TH';

    var spans = document.querySelectorAll('#title-container span');
    if (spans.length >= 2) {{
      spans[0].textContent = isTh ? TITLE_TH : TITLE_EN;
      spans[1].textContent = isTh ? SUBTITLE_TH : SUBTITLE_EN;
    }}

    var root = document.getElementById('custom-tooltip-root');
    if (root && root.style.display !== 'none' && root.dataset.en) {{
      root.innerHTML = isTh ? root.dataset.th : root.dataset.en;
    }}
  }};

  window.hsriToggleLang = function () {{
    window.__hsriLang = window.__hsriLang === 'en' ? 'th' : 'en';
    try {{ localStorage.setItem('hsriLang', window.__hsriLang); }} catch (e) {{}}
    window.hsriApplyLang();
  }};

  window.hsriApplyLang();

  // Hover-to-show for mouse users, on top of the existing tap/click-to-show
  // (kept as-is for touch -- iPad has no real hover, and tap-to-show is what
  // this session spent the most rounds getting right; see git log). Both
  // paths call the SAME datamap.deckgl onClick function datamapplot already
  // compiled from ON_CLICK_JS/build_tooltip_template, so there's only one
  // render code path to keep correct.
  //
  // getTooltip is deck.gl's own picking hook (re-evaluated internally on
  // every native pointer move) -- not app-level mousemove/mouseout listeners,
  // which is the machinery that broke pan/zoom three times earlier this
  // session. Returning null here keeps deck's native .deck-tooltip div (hidden
  // anyway via TOOLTIP_CSS) out of the picture entirely; #custom-tooltip-root
  // does all the rendering, same centered/clamped box as the click path.
  var __hoverIndex = -1;
  var __hideTimer = null;
  var __isTouch = false;
  window.addEventListener('pointerdown', function (e) {{
    __isTouch = e.pointerType !== 'mouse';
  }}, {{passive: true}});

  function hsriWireHoverTooltip() {{
    if (typeof datamap === 'undefined' || !datamap.deckgl) return false;
    var onClickFn = datamap.deckgl.props.onClick;
    if (!onClickFn) return false;

    // getTooltip turns out NOT to be pointer-move-driven in this deck.gl
    // bundle -- confirmed by testing (real hover never invoked it, zero
    // calls even landing exactly on a point's projected pixel) and by
    // datamapplot's own dynamic_tooltip.js, which explicitly sets
    // `getTooltip: null` and wires `onHover` instead for its own live
    // hover feature. onHover is the real one: same one-shot-per-pick-change
    // callback shape as onClick, just re-evaluated on pointer move.
    function hsriScheduleHide() {{
      if (__hideTimer) clearTimeout(__hideTimer);
      __hideTimer = setTimeout(function () {{
        var root = document.getElementById('custom-tooltip-root');
        if (root) root.style.display = 'none';
        __hoverIndex = -1;
        __hideTimer = null;
      }}, 150);
    }}

    datamap.deckgl.setProps({{
      getTooltip: null,
      onHover: function (info) {{
        // Touch: hover path fully disabled (not just hide -- the whole thing),
        // so a tap is handled ONLY by onClick, unchanged from before. Gating
        // it here, before touching __hoverIndex at all, keeps the two paths'
        // state fully separate -- a hide skipped for touch must never get
        // "recorded" as done, or a later real mouse hide can end up deduped
        // against it and silently no-op (found via testing: a stray touch
        // mid-session could otherwise wedge mouse hover-out forever).
        if (__isTouch) return;

        var picked = info && info.picked;
        var index = picked ? info.index : -1;

        if (picked) {{
          // Click (deck's native onClick, wired independently of this
          // handler -- see the top of this function's containing block, and
          // deliberately left untouched here rather than re-wrapped: the
          // getTooltip prop turned out not to be reliably re-assignable via
          // setProps in this exact bundle, so onClick -- the one path that's
          // survived every regression this session -- isn't a place to take
          // that same risk) opens the SAME box but interactive (its
          // template's own inline style sets pointer-events:auto). If the
          // cursor is still resting on that point -- likely, since they just
          // clicked it -- this handler fires again right after with the
          // same index. __hoverIndex alone can't detect that: it's private
          // to this closure and click doesn't update it. So read which
          // index is ACTUALLY shown from the DOM instead (dataset.shownIndex,
          // set by ON_CLICK_JS itself on every render, click or hover) and
          // leave it alone when it matches and is still interactive.
          // Without this, hover would immediately downgrade the just-opened
          // interactive box back to click-through, breaking the close
          // button and pane scrolling a frame after the click.
          var root = document.getElementById('custom-tooltip-root');
          var box = document.querySelector('#custom-tooltip-root > *');
          var shownIndex = root.dataset.shownIndex;
          var isShowingThis = root.style.display !== 'none' && shownIndex === String(index);
          var clickPinned = isShowingThis && box && box.style.pointerEvents !== 'none';
          if (clickPinned) {{ __hoverIndex = index; return; }}
          if (isShowingThis && index === __hoverIndex) return; // already previewing this exact point via hover

          __hoverIndex = index;
          // A pending hide (from having *just* wobbled off a point) is
          // cancelled by any new pick.
          if (__hideTimer) {{ clearTimeout(__hideTimer); __hideTimer = null; }}
          onClickFn(info);
          // The tooltip box is centered and can be exactly where the cursor
          // already is for any point near mid-screen -- the instant it
          // renders, IT becomes the topmost element under the pointer, so
          // the canvas would stop receiving that cursor's events entirely
          // (confirmed by testing: elementFromPoint at a shown point's own
          // screen coords resolved to the tooltip div, not the canvas; a
          // mouseenter/mouseleave keep-alive on the tooltip was tried first
          // but depends on the browser firing a *move* event right as the
          // box appears under an already-stationary cursor, which it does
          // not reliably do -- so points "around the textbox" kept failing).
          // Making the rendered box click-through for the HOVER path only
          // removes the problem instead of chasing it: the canvas keeps
          // seeing every real pointer move straight through the tooltip, so
          // it keeps reporting picked:true for as long as the cursor
          // actually rests on the point, with nothing to debounce around.
          // Re-query -- onClickFn just replaced the box via innerHTML, so
          // the reference read above is already stale.
          box = document.querySelector('#custom-tooltip-root > *');
          if (box) box.style.pointerEvents = 'none';
        }} else {{
          if (index === __hoverIndex) return;
          __hoverIndex = index;
          // Still debounced (not instant): protects against genuine
          // pick-radius edge jitter, separate from the self-covering issue
          // above. Any re-pick within the window cancels it (see above).
          hsriScheduleHide();
        }}
      }},
    }});
    return true;
  }}

  if (!hsriWireHoverTooltip()) {{
    // datamap/deckgl not ready yet on first paint; retry shortly.
    var __hoverWireTries = 0;
    var __hoverWireTimer = setInterval(function () {{
      __hoverWireTries++;
      if (hsriWireHoverTooltip() || __hoverWireTries > 40) clearInterval(__hoverWireTimer);
    }}, 250);
  }}
}})();
</script>
"""


def inject_bilingual_ui(filename):
    """Custom tooltip root (see the note above build_tooltip_template for why this
    replaces the old hover/touch-remap approach entirely -- three iterations of
    patching deck.gl's native hover machinery each surfaced a new failure mode,
    see git log) plus the EN/TH toggle button and title-swap logic."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    if "</body>" not in content:
        print("  ! </body> not found, could not inject bilingual UI")
        return
    content = content.replace("</body>", BILINGUAL_UI_SCRIPT + "</body>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print("Injected bilingual UI (tooltip root + language toggle)")


def main():
    df = pd.read_pickle(IN_FILE)
    print(f"Loaded {len(df)} projects")

    coordinates = df[["umap_x", "umap_y"]].fillna(0).to_numpy()
    cluster_labels_arr = df["cluster_name"].fillna("Unclustered").to_numpy()

    df["techniques_html"] = df["techniques_tools"].apply(lambda x: format_tags_as_html(x, "#2a5982"))
    df["concepts_html"] = df["key_concepts"].apply(lambda x: format_tags_as_html(x, "#d6ac4b"))
    df["impact_dims"] = df.apply(format_yn_dims, axis=1)
    df["hover_title"] = df["title_en"].fillna(df["project_title_th"]).fillna("Untitled project")

    # Thai translations (from 05_translate.py), keyed by positional project_id --
    # must match the row order this df was in when 05_translate.py extracted its
    # source data, since that's how the translations were positionally indexed.
    df = df.reset_index(drop=True)
    df["project_id"] = df.index
    translations = []
    if TRANSLATIONS_TH_FILE.exists():
        with open(TRANSLATIONS_TH_FILE, "r", encoding="utf-8") as f:
            translations = json.load(f)
    th_by_id = {t["project_id"]: t for t in translations}
    print(f"Loaded {len(th_by_id)} Thai translations")

    def th_field(pid, field, default=""):
        t = th_by_id.get(pid)
        if not t:
            return default
        val = t.get(field, default)
        return default if val is None else val

    df["hover_text_th"] = df["project_id"].apply(lambda pid: th_field(pid, "hover_text_th"))
    df["domain_th"] = df["project_id"].apply(lambda pid: th_field(pid, "domain_th"))
    df["data_type_th"] = df["project_id"].apply(lambda pid: th_field(pid, "data_type_th"))
    df["summary_short_th"] = df["project_id"].apply(lambda pid: th_field(pid, "summary_short_th"))
    df["summary_simple_th"] = df["project_id"].apply(lambda pid: th_field(pid, "summary_simple_th"))
    df["main_findings_th"] = df["project_id"].apply(lambda pid: th_field(pid, "main_findings_th"))
    df["methodology_th"] = df["project_id"].apply(lambda pid: th_field(pid, "methodology_th"))
    df["techniques_html_th"] = df["project_id"].apply(
        lambda pid: format_tags_as_html(th_field(pid, "techniques_th", []), "#2a5982")
    )
    df["concepts_html_th"] = df["project_id"].apply(
        lambda pid: format_tags_as_html(th_field(pid, "concepts_th", []), "#d6ac4b")
    )
    df["impact_dims_th"] = df["impact_dims"].apply(translate_impact_dims)

    # PI name and impact dimensions are allowed; grant budget and implementation
    # obstacles/challenges are not -- neither is shown here, and no budget/challenge
    # content leaks through the summary/findings text (verified separately).
    # pi_name and year are shared across languages (see build_tooltip_template).
    tooltip_columns = [
        "pi_name", "year", "domain", "data_type",
        "summary_short", "summary_simple", "main_findings", "methodology",
        "techniques_html", "concepts_html", "impact_dims",
        "hover_text_th", "domain_th", "data_type_th",
        "summary_short_th", "summary_simple_th", "main_findings_th", "methodology_th",
        "techniques_html_th", "concepts_html_th", "impact_dims_th",
    ]

    print("Creating interactive research map...")
    interactive_plot = datamapplot.create_interactive_plot(
        coordinates,
        cluster_labels_arr,
        # "hover_text" is the reserved placeholder name datamapplot fills from this
        # array; all other {placeholders} in the template map to extra_point_data columns.
        hover_text=df["hover_title"].fillna(""),
        noise_label="Unclustered",
        cluster_boundary_line_width=6,
        color_label_text=False,
        # Only 318 points total (vs. the much larger corpus datamapplot's auto-sizing
        # heuristic assumes) -- points read as near-invisible dust at the default size.
        # point_size_scale's world-space semantics turned out to shrink points rather
        # than grow them (verified empirically), so left at its data-driven default;
        # these pixel-space clamps are the reliable, unambiguous lever.
        point_radius_min_pixels=4,
        point_radius_max_pixels=32,
        title="HSRI Research Grant Map",
        sub_title="HSRI RG4 Grant Close-out Reports (2022–2025)",
        enable_search=True,
        extra_point_data=df[tooltip_columns].fillna(""),
        tooltip_css=TOOLTIP_CSS,  # permanently hides the native hover tooltip
        on_click=ON_CLICK_JS,
        # "-07-01", not just the bare year: datamapplot bins this with pd.cut's
        # default right-closed/left-open intervals, e.g. (2022-01-01, 2023-01-01].
        # A bare-year timestamp is exactly midnight Jan 1 -- precisely a bin
        # EDGE -- so every year's points silently landed in the PRECEDING
        # year's bar (2023 counted as 2022, 2024 as 2023, ...), and the true
        # final year's bar was always empty. Confirmed by reproducing pd.cut
        # directly on the real per-year counts: bare-year timestamps gave
        # [80, 132, 74] (+32 rescued into bin 0 by the library's own min-edge
        # fallback, landing 112 there) instead of the correct [32, 80, 132, 74].
        # Mid-year timestamps sit safely inside their bin, not on its edge.
        histogram_data=pd.to_datetime(df["year"].astype("Int64").astype(str) + "-07-01", format="%Y-%m-%d", errors="coerce"),
        histogram_group_datetime_by="year",
    )

    interactive_plot.save(str(OUT_FILE))
    print(f"Saved {OUT_FILE}")

    inline_vendor_scripts(str(OUT_FILE))
    inject_bilingual_ui(str(OUT_FILE))


if __name__ == "__main__":
    main()
