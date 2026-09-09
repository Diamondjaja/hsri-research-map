"""
Step 6: Thai translations for the 12 cluster names
======================================================

Only 12 short strings, so one batch OpenRouter call (not per-cluster). Output
feeds the client-side label-layer swap in 04_visualize.py's BILINGUAL_UI_SCRIPT.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OUT_FILE = ROOT / "data" / "hsri" / "hsri_cluster_names_th.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MODEL = "google/gemini-2.5-flash"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY not found in environment / .env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

CLUSTER_NAMES_EN = [
    "Advanced Medical Technologies and Diagnostics for Thailand",
    "Child and Adolescent Health and Education Interventions",
    "Chronic Disease Management & Value-Based Care",
    "Decentralized Primary Healthcare System Management",
    "Economic Evaluation and Health Policy for Thailand",
    "Enhancing Pediatric and Public Oral Healthcare in Thailand",
    "Genomic Medicine for Rare and Complex Diseases in Thailand",
    "Infectious Disease Diagnostics, Genomics, and Therapeutics",
    "Migrant and Border Health Systems Research",
    "Strengthening Thailand's Health Research Ecosystem and Ethics",
    "Thai Health System Performance and Policy Development",
    "Thai Pharmaceutical System and Rational Drug Use",
]

SYSTEM_PROMPT = (
    "You are a professional Thai translator specializing in health/medical research. "
    "Translate these short research-cluster names into natural, formal Thai suitable "
    "for labels on a research map. Output JSON only."
)

USER_PROMPT = """
Translate each of these cluster names to Thai. Return a JSON array of strings,
same order, same length ({n} items):
{names}
""".format(
    n=len(CLUSTER_NAMES_EN),
    names="\n".join(f"{i+1}. {name}" for i, name in enumerate(CLUSTER_NAMES_EN)),
)


def main():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    names_th = json.loads(raw)
    assert len(names_th) == len(CLUSTER_NAMES_EN), (
        f"expected {len(CLUSTER_NAMES_EN)} translations, got {len(names_th)}"
    )

    mapping = dict(zip(CLUSTER_NAMES_EN, names_th))
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    for en, th in mapping.items():
        print(f"{en}\n  -> {th}\n")

    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
