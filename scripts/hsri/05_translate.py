"""
Step 5: Thai translations for the language switch
======================================================

Extracts the current English tooltip content directly from the already-built
hsri_research_map.html (same extraction technique used for the tooltip-fix
recovery scripts -- see git log), then makes one OpenRouter
(google/gemini-2.5-flash) call per project to translate the translatable
fields to Thai. Runs on 5 projects first for a spot check, then all 318.

pi_name is NOT translated -- it's already Thai (the PI's real name from the
source data), never English to begin with. year/impact_dims category labels
are handled separately (impact_dims via a fixed small vocabulary, not an API
call -- see 06_build_bilingual.py).

Output: data/hsri/hsri_translations_th.jsonl (resumable stream cache)
        data/hsri/hsri_translations_th.json (final, list of dicts)
"""
import json
import os
import re
import base64
import gzip
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SRC_HTML = ROOT / "hsri_research_map.html"
OUT_DIR = ROOT / "data" / "hsri"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STREAM_FILE = OUT_DIR / "hsri_translations_th.jsonl"
FINAL_FILE = OUT_DIR / "hsri_translations_th.json"

MODEL = "google/gemini-2.5-flash"
SLEEP_TIME = 0.3
MAX_RETRIES = 4

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY not found in environment / .env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def extract_current_data():
    """Pull the 318-project English tooltip data straight out of the already-built
    HTML's embedded gzip+base64 JSON blob -- no need for the (deleted)
    intermediate pipeline pickle."""
    with open(SRC_HTML, "r", encoding="utf-8") as f:
        content = f.read()
    blobs = re.findall(r"[A-Za-z0-9+/=]{500,}", content)
    biggest = max(blobs, key=len)
    data = json.loads(gzip.decompress(base64.b64decode(biggest)))
    df = pd.DataFrame(data)
    df["project_id"] = range(len(df))  # positional id; stable within this build
    return df


def extract_tag_texts(tags_html):
    """Pull plain tag text back out of format_tags_as_html's <div>tag</div> output."""
    if not isinstance(tags_html, str) or not tags_html.strip():
        return []
    return re.findall(r">([^<]+)</div>", tags_html)


SYSTEM_PROMPT = (
    "You are a professional Thai translator specializing in health/medical research. "
    "Translate the given English research-project summary fields into natural, formal "
    "Thai (ภาษาไทยทางการ) suitable for a research database. Preserve all technical terms, "
    "drug names, and numeric findings exactly (transliterate technical/drug names into Thai "
    "script where that is the natural convention, but keep numbers, units, and abbreviations "
    "like HIV, TB, DNA as-is). Output JSON only."
)

USER_PROMPT_TEMPLATE = """
Translate these fields to Thai. Return EXACT JSON:
{{
  "hover_text_th": "Thai translation of the title",
  "domain_th": "Thai translation of the research domain",
  "data_type_th": "Thai translation of the research type",
  "summary_short_th": "Thai translation of the summary",
  "summary_simple_th": "Thai translation of the simple explanation",
  "main_findings_th": "Thai translation of the key findings",
  "methodology_th": "Thai translation of the methodology",
  "techniques_th": ["Thai translation of each technique/tool, same order"],
  "concepts_th": ["Thai translation of each key concept, same order"]
}}

Source (English):
Title: {hover_text}
Domain: {domain}
Type: {data_type}
Summary: {summary_short}
Simple explanation: {summary_simple}
Key findings: {main_findings}
Methodology: {methodology}
Techniques/tools: {techniques}
Key concepts: {concepts}
"""


def call_openrouter(project_id, row):
    techniques = extract_tag_texts(row.get("techniques_html", ""))
    concepts = extract_tag_texts(row.get("concepts_html", ""))
    prompt = USER_PROMPT_TEMPLATE.format(
        hover_text=row.get("hover_text", ""),
        domain=row.get("domain", ""),
        data_type=row.get("data_type", ""),
        summary_short=row.get("summary_short", ""),
        summary_simple=row.get("summary_simple", ""),
        main_findings=row.get("main_findings", ""),
        methodology=row.get("methodology", ""),
        techniques=", ".join(techniques) if techniques else "None",
        concepts=", ".join(concepts) if concepts else "None",
    )
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
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
            result = json.loads(raw)
            result["project_id"] = project_id
            return result
        except Exception as e:
            last_err = e
            wait = 3 * (attempt + 1)
            print(f"  ! project {project_id} attempt {attempt+1} failed: {e} (retrying in {wait}s)")
            time.sleep(wait)
    print(f"  x project {project_id} failed after {MAX_RETRIES} attempts: {last_err}")
    return None


def main(limit=None):
    df = extract_current_data()
    print(f"Loaded {len(df)} projects from {SRC_HTML}")

    done_ids = set()
    if STREAM_FILE.exists():
        with open(STREAM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_ids.add(d.get("project_id"))
                except Exception:
                    continue
    print(f"Already done: {len(done_ids)}")

    rows = df.to_dict("records") if limit is None else df.head(limit).to_dict("records")

    for row in tqdm(rows, desc="Translating"):
        pid = row["project_id"]
        if pid in done_ids:
            continue
        result = call_openrouter(pid, row)
        if result is None:
            continue
        with open(STREAM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        time.sleep(SLEEP_TIME)

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

    print(f"\nDone. {len(final_list)} translations -> {FINAL_FILE}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=n)
