"""
Step 4 (HSRI adaptation of 04_Datamapplot_Visualization.ipynb)
==================================================================

Builds the interactive research map from the clustered/named HSRI projects.
Single clustering layer (see step 3 notes on why the original 5-layer
hierarchy isn't reproduced). Tooltip rebuilt around HSRI fields (PI,
institution, budget, research design, policy/academic/social/economic
dimensions) instead of the original paper fields (authors/journal/DOI).

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


def format_budget(v):
    if pd.isna(v):
        return "Not reported"
    try:
        return f"{float(v):,.0f} THB"
    except (TypeError, ValueError):
        return "Not reported"


TOOLTIP_CSS = """
    max-width: none !important;
    width: auto !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    border: none !important;
    pointer-events: none;
"""

TOOLTIP_TEMPLATE = """
<div style="
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    width: 750px;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    overflow: hidden;
    color: #0f172a;
    text-align: left;
    position: relative;
">
    <div style="
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 58%;
        padding: 20px;
        overflow-y: auto;
        box-sizing: border-box;
    ">
        <h2 style="margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #1e293b; line-height: 1.3;">
            {hover_text}
        </h2>

        <div style="font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 6px;">
            PI: {pi_name} | Year: {year} | Budget: {budget_fmt}
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
        margin-left: 58%;
        width: 42%;
        padding: 20px;
        background-color: #f8fafc;
        border-left: 1px solid #e2e8f0;
        box-sizing: border-box;
        height: fit-content;
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


def pin_cdn_urls(filename):
    """datamapplot hardcodes unpkg.com CDN script tags, some with an unpinned
    '@latest' version. unpkg is blocked by some sandboxed HTML viewers (silent hang,
    page loads forever waiting for globals that never arrive) and '@latest' is a
    reproducibility risk. Repoint to jsdelivr with pinned versions."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    replacements = [
        ("https://unpkg.com/deck.gl@9.1/dist.min.js",
         "https://cdn.jsdelivr.net/npm/deck.gl@9.1/dist.min.js"),
        ("https://unpkg.com/apache-arrow@latest/Arrow.es2015.min.js",
         "https://cdn.jsdelivr.net/npm/apache-arrow@21.2.0/Arrow.es2015.min.js"),
        ("https://unpkg.com/d3@latest/dist/d3.min.js",
         "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"),
    ]
    changes = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes += 1
        else:
            print(f"  ! CDN pattern not found (datamapplot version may have changed): {old}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Pinned {changes}/{len(replacements)} CDN URLs to jsdelivr")


def optimize_for_touch_devices(filename):
    """Make the map usable on touch devices (iPad etc.): they don't fire mousemove /
    mouseleave, which the hover tooltip depends on, so remap tap -> hover-show and
    double-tap -> hover-hide. Ported as-is from the original notebook."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    modifications = [
        (".on('click',e=>this.#handleClick(e))", ".on('disabled_click',e=>this.#handleClick(e))"),
        (".on('mousemove',e=>this.#handleMouseMove(e))", ".on('click',e=>this.#handleMouseMove(e))"),
        (".on('mouseleave',e=>this.#handleMouseLeave(e))", ".on('dblclick',e=>this.#handleMouseLeave(e))"),
    ]

    changes = 0
    for search, replace in modifications:
        if search in content:
            content = content.replace(search, replace)
            changes += 1
        else:
            print(f"  ! touch-optimization pattern not found: {search[:40]}...")

    if changes:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Applied {changes}/{len(modifications)} touch optimizations")


def main():
    df = pd.read_pickle(IN_FILE)
    print(f"Loaded {len(df)} projects")

    coordinates = df[["umap_x", "umap_y"]].fillna(0).to_numpy()
    cluster_labels_arr = df["cluster_name"].fillna("Unclustered").to_numpy()

    df["techniques_html"] = df["techniques_tools"].apply(lambda x: format_tags_as_html(x, "#2a5982"))
    df["concepts_html"] = df["key_concepts"].apply(lambda x: format_tags_as_html(x, "#d6ac4b"))
    df["impact_dims"] = df.apply(format_yn_dims, axis=1)
    df["budget_fmt"] = df["approved_budget_thb"].apply(format_budget)
    df["hover_title"] = df["title_en"].fillna(df["project_title_th"]).fillna("Untitled project")

    tooltip_columns = [
        "pi_name", "year", "budget_fmt", "domain", "data_type",
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
        hover_text_html_template=TOOLTIP_TEMPLATE,
        tooltip_css=TOOLTIP_CSS,
        histogram_data=pd.to_datetime(df["year"].astype("Int64").astype(str), format="%Y", errors="coerce"),
        histogram_group_datetime_by="year",
    )

    interactive_plot.save(str(OUT_FILE))
    print(f"Saved {OUT_FILE}")

    pin_cdn_urls(str(OUT_FILE))
    optimize_for_touch_devices(str(OUT_FILE))


if __name__ == "__main__":
    main()
