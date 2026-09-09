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
    "__root.style.display = 'block';"
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
        histogram_data=pd.to_datetime(df["year"].astype("Int64").astype(str), format="%Y", errors="coerce"),
        histogram_group_datetime_by="year",
    )

    interactive_plot.save(str(OUT_FILE))
    print(f"Saved {OUT_FILE}")

    inline_vendor_scripts(str(OUT_FILE))
    inject_bilingual_ui(str(OUT_FILE))


if __name__ == "__main__":
    main()
