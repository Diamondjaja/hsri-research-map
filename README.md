# HSRI Research Grant Map

An interactive visualization exploring the research landscape funded by the Health Systems
Research Institute (HSRI / สวรส.) of Thailand, built from RG4 grant close-out reports
covering fiscal years 2022–2025.

[🗺️ Explore the visualization](https://biodatlab.github.io/murex-map) (recommended using Google Chrome on desktop)

<img src="assets/map_screenshot.png" alt="HSRI Research Grant Map" width="800">

### Overview

This project visualizes the structure of 318 HSRI-funded health systems research projects,
revealing thematic clusters — from genomic medicine and infectious disease to health policy,
pharmaceutical systems, and primary care — and how they relate to each other. Source project
reports are in Thai; the visualization surfaces English summaries, key findings, and impact
dimensions (policy / academic / social / economic) for each project. The visualization employs
dimensionality reduction and clustering techniques inspired by
[The Illustrated NeurIPS 2025](https://newsletter.languagemodels.co/p/the-illustrated-neurips-2025-a-visual).

### Methodology

- **Consolidate project data**: Each project's already-extracted structured fields (objectives,
  results, findings, policy use, utilization) are consolidated into a clean bilingual summary
  and tags using an LLM ([`google/gemini-2.5-flash`](https://openrouter.ai/google/gemini-2.5-flash)
  via [OpenRouter](https://openrouter.ai/))
- **Embed projects**: Generate bilingual (Thai + English) [semantic embeddings](https://docs.cohere.com/docs/embeddings)
  using Cohere's `embed-multilingual-v3.0`
- **Dimensionality reduction**: Apply [UMAP](https://github.com/lmcinnes/umap) to project
  high-dimensional embeddings into 2D space
- **Clustering**: Identify research clusters using [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html),
  with names generated per cluster via LLM
- **Interactive visualization**: Build an explorable interface using [`datamapplot`](https://datamapplot.readthedocs.io/en/latest/)

Pipeline code lives in [`scripts/hsri/`](scripts/hsri/); see [`task.md`](task.md) for detailed
notes on the adaptation from the original MUREX (Scopus paper) pipeline.
