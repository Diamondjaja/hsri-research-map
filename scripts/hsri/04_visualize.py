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
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import datamapplot

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "hsri" / "hsri_clustered_named.pkl"
OUT_FILE = ROOT / "hsri_research_map.html"

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

CUSTOM_TOOLTIP_TEMPLATE = """
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
            {hover_text}
        </h2>

        <div style="font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 6px;">
            PI: {pi_name} | Year: {year}
        </div>

        <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px;">
            <div style="background-color: #2a5982; color: #fff; border-radius: 6px; padding: 2px 10px; font-size: 7pt; font-weight: 600;">
                {domain}
            </div>
            <div style="background-color: #d6ac4b; color: #fff; border-radius: 6px; padding: 2px 10px; font-size: 7pt; font-weight: 600;">
                {data_type}
            </div>
        </div>

        <div style="font-size: 11px; font-weight: 700; color: #2a5982; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
            Summary
        </div>
        <p style="margin: 0; font-size: 12.5px; line-height: 1.5; color: #334155; text-align: justify;">
            {summary_short}
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
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 6px;">Simple Explanation</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{summary_simple}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 6px;">Key Findings</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{main_findings}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 6px;">Methodology</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{methodology}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 5px;">Techniques & Tools</div>
            {techniques_html}
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #d6ac4b; margin-bottom: 5px;">Key Concepts</div>
            {concepts_html}
        </div>

        <div>
            <div style="font-size: 12px; font-weight: 700; color: #2a5982; margin-bottom: 5px;">Impact Dimensions</div>
            <p style="margin: 0; font-size: 12.5px; line-height: 1.45; color: #0f172a;">{impact_dims}</p>
        </div>
    </div>
</div>
"""

# Wrapped in backticks (a JS template literal) exactly like datamapplot's own
# hover_text_html_template mechanism does internally -- proven to survive
# whatever escaping happens downstream, since that's the same structure the
# working hover-based template used before this redesign.
ON_CLICK_JS = (
    "document.getElementById('custom-tooltip-root').innerHTML = `"
    + CUSTOM_TOOLTIP_TEMPLATE
    + "`;\ndocument.getElementById('custom-tooltip-root').style.display = 'block';"
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


CUSTOM_TOOLTIP_ROOT_SCRIPT = """
<div id="custom-tooltip-root" style="display:none;"></div>
"""


def inject_custom_tooltip_root(filename):
    """Empty container that ON_CLICK_JS populates and shows. See the note above
    CUSTOM_TOOLTIP_TEMPLATE for why this replaces the old hover/touch-remap approach
    entirely (three iterations of patching deck.gl's native hover machinery each
    surfaced a new failure mode -- see git log)."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    if "</body>" not in content:
        print("  ! </body> not found, could not inject custom tooltip root")
        return
    content = content.replace("</body>", CUSTOM_TOOLTIP_ROOT_SCRIPT + "</body>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print("Injected custom tooltip root container")


def main():
    df = pd.read_pickle(IN_FILE)
    print(f"Loaded {len(df)} projects")

    coordinates = df[["umap_x", "umap_y"]].fillna(0).to_numpy()
    cluster_labels_arr = df["cluster_name"].fillna("Unclustered").to_numpy()

    df["techniques_html"] = df["techniques_tools"].apply(lambda x: format_tags_as_html(x, "#2a5982"))
    df["concepts_html"] = df["key_concepts"].apply(lambda x: format_tags_as_html(x, "#d6ac4b"))
    df["impact_dims"] = df.apply(format_yn_dims, axis=1)
    df["hover_title"] = df["title_en"].fillna(df["project_title_th"]).fillna("Untitled project")

    # PI name and impact dimensions are allowed; grant budget and implementation
    # obstacles/challenges are not -- neither is shown here, and no budget/challenge
    # content leaks through the summary/findings text (verified separately).
    tooltip_columns = [
        "pi_name", "year", "domain", "data_type",
        "summary_short", "summary_simple", "main_findings", "methodology",
        "techniques_html", "concepts_html", "impact_dims",
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
    inject_custom_tooltip_root(str(OUT_FILE))


if __name__ == "__main__":
    main()
