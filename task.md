# Task: Adapt murex-map pipeline to a new dataset (OpenRouter + Cohere)

Reference doc for the in-progress work on this repo. Update as decisions/steps change.

## Goal

Swap in a new dataset (different format from the original Scopus export) and re-run the
pipeline to produce an updated `index.html` research map, while switching the LLM provider
from OpenAI to OpenRouter for the two LLM-driven steps.

## Decisions made so far

- **Embeddings**: keep **Cohere** (`embed-english-v3.0`) for step 02 — OpenRouter is a
  chat/completions router only and doesn't serve embeddings.
- **LLM provider**: switch to **OpenRouter** for:
  - Step 01 (paper summarization)
  - Step 03 (cluster naming)
- **Model**: `google/gemini-2.5-flash` (via OpenRouter) for both of the above.
- **New dataset**: `HSRI_RG4_All_Years.xlsx`, found at
  `C:\Users\Diamond\Desktop\research_hsri\processed_data\rg4\HSRI_RG4_All_Years.xlsx`
  (not yet copied into the murex-map repo — the path the user gave is relative to a
  *different* project folder, `research_hsri`, not this repo). See "New dataset — actual
  contents" below for what it contains; it is **not** a Scopus-shaped export and changes
  the pipeline more than originally scoped.

## New dataset — actual contents (inspected 2026-09-09)

`HSRI_RG4_All_Years.xlsx` — Thai Health Systems Research Institute (สวรส./HSRI) grant
close-out reports ("แบบ รง.4"), **not** academic papers. Very different shape from the
original Scopus dataset:

- **3 sheets**:
  - `Dummy_Table` — 318 rows × 98 columns, one row per research project. Main sheet.
  - `Utilization_Long` — 844 rows × 11 columns, long-format: multiple "output/utilization
    item" rows per project (policy/academic/social/economic dimension text, target
    audience, OKR code).
  - `OKR_Long` — 106 rows × 5 columns, long-format: KR (key result) codes/milestones per
    project.
- **Language**: mostly **Thai** (titles, objectives, results, challenges, policy-use text).
  A handful of fields are already English (e.g. `mapping_focus_text` reads as an
  English one-paragraph summary of the project — looks LLM-generated already).
- **Already LLM-processed**: this file was itself produced by an OCR + LLM extraction
  pipeline — it already has `llm_model_name` (= `google/gemini-2.5-flash`, interesting
  coincidence with our chosen model), `llm_confidence_overall`, `evidence_json` (page/quote
  citations), `missingness_json`, `extraction_notes` per row. So `results_summary_text`,
  `key_numeric_findings_text`, `objectives_text`, `policy_use_examples_text`,
  `mapping_focus_text`, `mapping_implications_text` etc. are **already-extracted structured
  text**, not raw abstracts needing a first LLM pass.
- **Other useful `Dummy_Table` columns**: `project_id`, `agreement_no_raw`, `fiscal_year_be`
  (Thai Buddhist year), `project_title_th`, `pi_name`, `pi_institution`,
  `approved_budget_thb`, `project_type_primary`, `research_design`, `geographic_scope`,
  `sample_size_n`, boolean dimension flags (`policy_dimension_yn`,
  `academic_dimension_yn`, `social_dimension_yn`, `economic_dimension_yn`),
  `challenge_category`, `beneficiary_organizations_text`, various `out_*_n` output counts.
- No `Title`/`Abstract`/`EID`/`Document Type`/`Year`/`Authors` columns at all — the entire
  step-01 cleaning logic (built for Scopus CSVs) does not apply.

### Implications for the plan

- Step 01 probably needs to become "compose a per-project text blob from existing
  structured fields + optionally one OpenRouter call to normalize/translate into a
  consistent English summary+tags schema" rather than "summarize a raw abstract" — most of
  the summarization work is already done in this file.
- Step 02's embedding model choice matters: original used Cohere `embed-english-v3.0`, but
  most source text here is Thai. Cohere's `embed-multilingual-v3.0` (or embedding the
  already-English `mapping_focus_text`/translated fields) should be considered instead.
- Tooltip/map content (step 04) will look nothing like the paper tooltips (journal/DOI/
  authors) — more like PI, institution, fiscal year, budget, research design, policy/
  academic/social/economic dimensions, key findings.
- `Utilization_Long` / `OKR_Long` are optional enrichment (join back to `project_id`) —
  not required for a first pass; decide whether to fold their text into the embedded
  document or ignore them initially.
- File needs to actually be copied (or read directly) into/from the murex-map repo — it
  currently lives entirely outside this repo under a sibling project folder.

## Decisions on the new dataset (confirmed with user 2026-09-09)

- **Embedding language**: use **both** Thai and English. Compose each project's embedding
  text from the original Thai fields *plus* an English summary (produced in the step-01
  light pass, or existing `mapping_focus_text` where present), then embed the combined
  blob with Cohere's multilingual model (`embed-multilingual-v3.0`) so both halves are
  captured in one vector.
- **Step 01 scope ("light pass: compose + tag only")**: no re-extraction from scratch.
  One OpenRouter (`google/gemini-2.5-flash`) call per project that:
  - merges the existing structured fields (`objectives_text`, `results_summary_text`,
    `key_numeric_findings_text`, `policy_use_examples_text`, `implementation_challenges_text`,
    etc., plus folded-in utilization text — see below) into one clean bilingual-ready
    summary blob,
  - and generates tags/keywords for clustering (analogous to the original pipeline's
    `tags` field).
  This replaces "summarize a raw abstract" with "consolidate already-extracted fields."
- **Side tables**: **fold `Utilization_Long` in now** — join its rows back onto each
  project by `project_id` (concatenate `output_item` + dimension text + `target_audience`
  per project) and include that joined text as part of what step 01 consolidates.
  `OKR_Long` was not explicitly decided — default to leaving it out for the first pass
  unless it turns out useful (revisit if needed).

## Repo pipeline (as-is, before changes)

| File | Role |
|---|---|
| `scripts/01_LLM_Prompting.ipynb` | Cleans raw Scopus CSV export, calls OpenAI (gpt-4o-mini) per paper → structured summaries (domain, methodology, findings, tags, etc.) → `*_paper_summaries_final.json` |
| `scripts/02_Embedding_and_UMAP.ipynb` | Embeds paper text via Cohere (`embed-english-v3.0`), reduces to 2D (viz) + 10D (clustering) with UMAP, first-pass HDBSCAN → `research_with_topics_umap.pkl` / `.csv` |
| `scripts/03_Clustering_and_Name_Assignment.ipynb` | Refines clustering (HDBSCAN + kNN noise reassignment), computes density classes, uses GPT-4o-mini to name/describe clusters hierarchically (`5L_Layer_1..5`) → `final_clustered_papers_with_names.csv/.pkl` |
| `scripts/04_Datamapplot_Visualization.ipynb` | Merges UMAP coords + LLM summaries, builds tooltip HTML, calls `datamapplot.create_interactive_plot(...)` → `murex_research_map.html` |
| `index.html` | Checked-in copy of the step-04 output, served via GitHub Pages (`.nojekyll`, no build step) |

## Known issues to reconcile while adapting

- Notebooks were run in Google Colab with **hardcoded local/Drive paths**
  (`/content/drive/MyDrive/...`, `/content/MUSearch/...`) and **inline API key placeholders** —
  not turnkey re-runnable as-is.
- **Column/filename mismatches between steps**, e.g.:
  - Step 03 expects `5L_Layer_5_Final` and `umap_3d_x/y/z` columns that step 02 doesn't
    actually produce (hierarchical layer columns look like they came from an ad hoc Colab
    step not captured in these cleaned notebooks).
  - Step 04 expects `umap_cluster_2d.csv` + `final_paper_summaries.json`, whose names don't
    exactly match steps 01–03's actual outputs (manual renaming was happening between runs).
- **Cost**: one OpenRouter call per project for the step-01 light pass, one Cohere embedding
  call per project, one (or a handful, per cluster) OpenRouter call for step 03 naming — 318
  projects, so cheap in absolute terms this round, but keep the pattern reusable.
- Step 01's cleaning logic is built entirely around Scopus-export columns (`Title`,
  `Abstract`, `EID`, `Document Type`, `Year`, `Authors`, ...) — none of that applies to the
  HSRI dataset; it needs a new ingestion path, not a patch.

## Plan / next steps

1. **Get the dataset into this repo** — copy (or symlink) `HSRI_RG4_All_Years.xlsx` from
   `research_hsri\processed_data\rg4\` into `murex-map` (e.g. under `data/` or
   `processed_data/rg4/` to match the path the user referenced), or confirm reading it
   in place from the sibling folder.
2. **Notebook 01 (rewritten as HSRI ingestion + light OpenRouter pass)**:
   - Load `Dummy_Table` from the xlsx.
   - Join `Utilization_Long` rows back onto each project by `project_id` (concatenate
     `output_item` + dimension columns + `target_audience` text).
   - One OpenRouter (`google/gemini-2.5-flash`) call per project: consolidate the existing
     structured fields (+ folded-in utilization text) into a clean bilingual (Thai +
     English) summary blob, and generate tags/keywords.
   - Output: a per-project JSON/DataFrame analogous to `*_paper_summaries_final.json`.
3. **Notebook 02**: switch Cohere model to `embed-multilingual-v3.0`; embed the bilingual
   summary blob from step 01; keep UMAP (2D + 10D) + first-pass HDBSCAN as-is otherwise.
4. **Notebook 03**: replace OpenAI client with OpenRouter client (`google/gemini-2.5-flash`)
   for cluster naming; reconcile missing `5L_Layer_*` / `umap_3d_*` column expectations
   against what 02 actually emits (may need to add the hierarchical-layer step explicitly).
5. **Notebook 04**: rebuild the tooltip HTML around HSRI fields (PI, institution, fiscal
   year, budget, research design, policy/academic/social/economic dimension flags, key
   findings) instead of the old paper fields (authors/journal/DOI); reconcile input
   filenames against 01–03's actual outputs; regenerate `hsri_research_map.html`.
6. Copy result over `index.html`, commit, push to `main` → GitHub Pages auto-updates.
7. Env/config: `OPENROUTER_API_KEY` + `COHERE_API_KEY` in `.env` (confirmed gitignored,
   see below) — not committed.

## Results (2026-09-09) — first full run complete

Pipeline built and run end-to-end as `scripts/hsri/01_ingest_and_summarize.py` →
`02_embed_and_umap.py` → `03_cluster_and_name.py` → `04_visualize.py` (plain scripts,
not notebooks — kept the original 4 notebooks untouched as historical record).

- **Step 1**: 318/318 projects consolidated via OpenRouter (`google/gemini-2.5-flash`).
  309 `VALID`, 9 `INSUFFICIENT_INFO`. No failed retries. → `data/hsri/hsri_project_summaries.json`
- **Step 2**: Cohere `embed-multilingual-v3.0` bilingual embeddings (English-first blob,
  `truncate="END"` delegated to the API). UMAP 10D (clustering) + 2D (viz), same params as
  original. HDBSCAN swept `min_cluster_size` ∈ {5,8,10,11,12,13,14,15,16,18,20} —
  **11** chosen (12 clusters, 21% pre-refinement noise) as the best fit for this dataset's
  size. → `data/hsri/hsri_with_embeddings.pkl`
- **Step 3**: HDBSCAN + kNN noise refinement (single layer, per the advisor's call not to
  reconstruct the unreproducible external 5-layer file) + local density scoring, cluster
  names generated via OpenRouter. **12 named clusters**, 12–48 projects each (e.g.
  "Genomic Medicine for Rare and Complex Diseases in Thailand" (48), "Chronic Disease
  Management & Value-Based Care" (42), "Migrant and Border Health Systems Research" (34),
  ... down to "Enhancing Pediatric and Public Oral Healthcare in Thailand" (12)).
  → `data/hsri/hsri_clustered_named.pkl`
- **Step 4**: `datamapplot` interactive map, HSRI-specific tooltip (PI, year, budget,
  domain, summary, findings, methodology, impact dimensions). Fiscal year (BE) converted
  to CE for the timeline histogram. → `hsri_research_map.html` (repo root, same as
  original convention; **not yet copied over `index.html`** — pending user review).

Verified in-browser (via local HTTP server + Chrome): title/subtitle/cluster labels
render, timeline histogram correctly spans Jul 2022–Jul 2025 (confirms BE→CE conversion),
search correctly filters to matching projects. Also verified by decoding the page's
embedded gzip+base64 point-data blob directly: `hover_text`, `pi_name`, `year`,
`budget_fmt`, `domain`, `summary_short` etc. all present and correct for a spot-checked
project (65-030, dolutegravir/TB study).

Bugs caught and fixed during the build: pandas `NaN` serializing as invalid literal `NaN`
in JSON (step 1), HDBSCAN `min_samples` mismatch between the step-2 sweep and step-3's
actual clustering call (collapsed 12 clusters down to 3 until fixed).

**Not yet done**: copying `hsri_research_map.html` over `index.html`, committing, or
pushing — outward-facing/hard-to-reverse, needs explicit go-ahead.

## Open items / things to confirm with user

- Exact destination path for the dataset inside `murex-map` (or: read it in place from
  `research_hsri`?).
- Whether `OKR_Long` should be folded in too (deferred for now, per decision above).
- Whether to keep the OpenAI-era summary schema shape (domain, methodology, findings, tags)
  for the consolidated blob, or design a new schema suited to HSRI fields.
- `.env` confirmed gitignored (`.env` + `.envrc` both listed in `.gitignore`) — safe to put
  `OPENROUTER_API_KEY` / `COHERE_API_KEY` there.
