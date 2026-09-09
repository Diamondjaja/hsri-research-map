"""
Step 1 (HSRI adaptation of 01_LLM_Prompting.ipynb)
===================================================

Ingests the HSRI RG4 grant close-out reports (Dummy_Table + Utilization_Long),
folds bounded utilization text into each project, then makes one OpenRouter
(google/gemini-2.5-flash) call per project to CONSOLIDATE the already-extracted
structured fields into a clean summary + tags, in the same JSON-schema *shape*
as the original Scopus pipeline (so notebook 04's tooltip needs minimal surgery).

Unlike the original notebook, this does not summarize from a raw abstract --
the source xlsx was already produced by its own OCR+LLM extraction pass, so
this step's job is "compose + tag", not "extract from scratch".

Output: data/hsri/hsri_project_summaries.json (list of dicts, one per project)
        data/hsri/hsri_summaries_cache.jsonl (resumable stream cache)
"""
import json
import os
import time
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

XLSX_PATH = ROOT / "processed_data" / "rg4" / "HSRI_RG4_All_Years.xlsx"
OUT_DIR = ROOT / "data" / "hsri"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STREAM_FILE = OUT_DIR / "hsri_summaries_cache.jsonl"
FINAL_FILE = OUT_DIR / "hsri_project_summaries.json"

MODEL = "google/gemini-2.5-flash"
UTILIZATION_CHAR_CAP = 600  # bound per-project utilization text (see advisor note)
SLEEP_TIME = 0.3
MAX_RETRIES = 4

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY not found in environment / .env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def be_to_ce(fiscal_year_be):
    """Convert Thai Buddhist Era fiscal year to CE. 2565 BE -> 2022 CE."""
    if pd.isna(fiscal_year_be):
        return None
    return int(fiscal_year_be) - 543


def safe(val):
    """Convert pandas/numpy NaN to None so it serializes as JSON null, not bare NaN."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "item"):  # numpy scalar -> native Python type
        return val.item()
    return val


def clip(text, n):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def build_utilization_blob(project_id, util_df, cap=UTILIZATION_CHAR_CAP):
    rows = util_df[util_df["project_id"] == project_id]
    if rows.empty:
        return ""
    parts = []
    for _, r in rows.iterrows():
        item = clip(r.get("output_item", ""), 200)
        audience = clip(r.get("target_audience", ""), 100)
        if item:
            parts.append(item + (f" (audience: {audience})" if audience else ""))
    blob = " | ".join(parts)
    return clip(blob, cap)


def build_source_context(row, utilization_blob):
    """Assemble the raw (Thai-heavy) source material for one project."""
    fields = {
        "Project title (Thai)": row.get("project_title_th", ""),
        "PI": row.get("pi_name", ""),
        "Institution": row.get("pi_institution", ""),
        "Project type": row.get("project_type_primary", ""),
        "Research design": row.get("research_design", ""),
        "Design detail": row.get("design_detail_text", ""),
        "Geographic scope": row.get("geographic_scope", ""),
        "Sample size": row.get("sample_size_n", ""),
        "Objectives": row.get("objectives_text", ""),
        "Results summary": row.get("results_summary_text", ""),
        "Key numeric findings": row.get("key_numeric_findings_text", ""),
        "Implementation challenges": row.get("implementation_challenges_text", ""),
        "Policy use examples": row.get("policy_use_examples_text", ""),
        "Beneficiary organizations": row.get("beneficiary_organizations_text", ""),
        "English focus summary (existing)": row.get("mapping_focus_text", ""),
        "English implications (existing)": row.get("mapping_implications_text", ""),
        "Utilization / outputs": utilization_blob,
    }
    lines = [f"{k}: {v}" for k, v in fields.items() if isinstance(v, str) and v.strip()]
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a research-policy analyst. You are given ALREADY-EXTRACTED structured "
    "information about a Thai health-systems research grant project (source text is a "
    "mix of Thai and English -- some fields are OCR'd Thai, some are prior English "
    "translations). Your job is to CONSOLIDATE this into a clean, structured English "
    "summary. Do not invent facts not present in the source. Output JSON only."
)

USER_PROMPT_TEMPLATE = """
Consolidate the following project information into EXACT JSON format:
{{
  "project_id": "{project_id}",
  "title_en": "English translation of the project title (concise, formal)",
  "domain": "Primary research field / health area (e.g. Infectious disease, Genomics, Health policy)",
  "problem": "Context & Question: background or knowledge gap addressed (1-2 sentences, English)",
  "methodology": "Research approach: design, subjects, key analytical steps (1-2 sentences, English)",
  "data_type": "Type of research design (e.g. Clinical trial, Survey, Qualitative, Health services research, Tool/instrument development)",
  "techniques_tools": ["2-5 key methods, technologies, or analytical tools actually used"],
  "key_concepts": ["3-6 core subject-matter tags/keywords for this project"],
  "main_findings": "Observed results, stated objectively (1-2 sentences, English)",
  "summary_short": "A cohesive formal English paragraph balancing methodology and findings, ending with policy/practical relevance.",
  "summary_simple": "Plain-language English explanation for a general audience. Explain jargon in parentheses.",
  "practical_relevance": "How this project's results have been or could be used (policy, clinical practice, further research).",
  "data_quality_flag": "VALID or INSUFFICIENT_INFO"
}}

Strict rules:
1. Use ONLY information present in the source fields below. Do not guess beyond them.
2. All output text must be in ENGLISH, even though source fields are partly Thai.
3. If a field cannot be determined from the source, use "Not mentioned".
4. Keep summary_short and summary_simple each to 2-4 sentences.
5. Output valid JSON only, no markdown code fences, no commentary.

Source project information:
{source_context}
"""


def call_openrouter(project_id, source_context):
    prompt = USER_PROMPT_TEMPLATE.format(project_id=project_id, source_context=source_context)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            return json.loads(raw)
        except Exception as e:
            last_err = e
            wait = 3 * (attempt + 1)
            print(f"  ! {project_id} attempt {attempt+1} failed: {e} (retrying in {wait}s)")
            time.sleep(wait)
    print(f"  x {project_id} failed after {MAX_RETRIES} attempts: {last_err}")
    return None


def main(limit=None):
    df = pd.read_excel(XLSX_PATH, sheet_name="Dummy_Table")
    util_df = pd.read_excel(XLSX_PATH, sheet_name="Utilization_Long")
    print(f"Loaded {len(df)} projects, {len(util_df)} utilization rows")

    done_ids = set()
    if STREAM_FILE.exists():
        with open(STREAM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_ids.add(str(d.get("project_id")))
                except Exception:
                    continue
    print(f"Already done: {len(done_ids)}")

    rows = df.to_dict("records") if limit is None else df.head(limit).to_dict("records")

    for row in tqdm(rows, desc="Consolidating projects"):
        project_id = str(row["project_id"])
        if project_id in done_ids:
            continue

        utilization_blob = build_utilization_blob(project_id, util_df)
        source_context = build_source_context(row, utilization_blob)

        result = call_openrouter(project_id, source_context)
        if result is None:
            continue

        result["project_id"] = project_id
        result["fiscal_year_be"] = safe(row.get("fiscal_year_be"))
        result["year"] = be_to_ce(row.get("fiscal_year_be"))
        result["pi_name"] = safe(row.get("pi_name"))
        result["pi_institution"] = safe(row.get("pi_institution"))
        result["project_title_th"] = safe(row.get("project_title_th"))
        result["approved_budget_thb"] = safe(row.get("approved_budget_thb"))
        result["policy_dimension_yn"] = safe(row.get("policy_dimension_yn"))
        result["academic_dimension_yn"] = safe(row.get("academic_dimension_yn"))
        result["social_dimension_yn"] = safe(row.get("social_dimension_yn"))
        result["economic_dimension_yn"] = safe(row.get("economic_dimension_yn"))
        result["utilization_blob"] = utilization_blob
        result["source_context"] = source_context  # kept for step-02 bilingual embedding text

        with open(STREAM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        time.sleep(SLEEP_TIME)

    # compile final file
    final_list = []
    if STREAM_FILE.exists():
        with open(STREAM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    final_list.append(json.loads(line))
                except Exception:
                    continue

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(final_list)} projects consolidated -> {FINAL_FILE}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=n)
