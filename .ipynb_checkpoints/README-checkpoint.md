# Citation Map (OpenAlex-based)

This mini-project does two things:

1. **Build a citation table** for *your publications*, listing **one row per citing paper**.
2. **Visualize the citing-paper countries** as a world map and a (sorted) bar plot.

The pipeline is designed to avoid Selenium/Google Scholar scraping by default and relies on **OpenAlex** (plus **Crossref** only when you need DOI completion for ORCID/Scholar inputs).

---

## Files in this repo

- `citation_fetcher.py`
  - Main data builder.
  - Finds *your publications* via one of:
    - `--openalex_id` (recommended)
    - `--orcid`
    - `--csv` (a DOI list you provide)
    - `--scholar_id` (optional; often blocked)
  - Uses OpenAlex to fetch all citing works.
  - Writes `citation_info.csv` (one row per citing paper).

- `create_citation_map.py`
  - Plotting utilities:
    - `create_citation_map(...)` for the world map
    - `plot_citations_by_country_bar(...)` for a sorted country bar plot

- `Citation_map_Lixu.ipynb`
  - Example notebook showing a full, end-to-end run (install → fetch → map → bar plot).

- `requirements.txt`
  - Python dependencies.

---

## Installation

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Notes
- `geopandas` can require extra system libraries on some machines. If `pip install geopandas` fails, consider conda/mamba.

---

## Step 1 — Generate the citation table

Pick **exactly one** input source per run.

### Option A (recommended): OpenAlex Author ID

```bash
python3 citation_fetcher.py \
  --openalex_id A1234567890 \
  --email your_email@domain.edu \
  --output ./data_created/citation_info.csv
```

### Option B: ORCID

```bash
python3 citation_fetcher.py \
  --orcid 0000-0000-0000-0000 \
  --email your_email@domain.edu \
  --output ./data_created/citation_info.csv
```

### Option C: CSV of your DOIs

Your CSV must contain a `DOI` or `doi` column.

```bash
python3 citation_fetcher.py \
  --csv my_dois.csv \
  --email your_email@domain.edu \
  --output ./data_created/citation_info.csv
```

### Option D (optional): Google Scholar ID

This uses the `scholarly` library and may be blocked on shared IPs / Colab / HPC.

```bash
python3 citation_fetcher.py \
  --scholar_id Q_sLQJIAAAAJ \
  --email your_email@domain.edu \
  --output ./data_created/citation_info.csv
```

---

## Output: `citation_info.csv`

`citation_fetcher.py` outputs **one row per citing paper** with these key columns:

- `my_publication` — title of your publication
- `cited_by_title` — title of the citing paper
- `first_author` — first author name (OpenAlex authorships)
- `cited_by_year` — publication year
- `cited_by_doi` — citing paper DOI (if available)
- `cited_by_country` — best-effort ISO A2 country code inferred from the first author’s first institution
- `cited_by_journal` — citing paper venue/journal name (OpenAlex `host_venue.display_name`)
- `cited_by_jcr` — **optional** JCR metric/value attached from your mapping CSV (see below)
- `cited_by_id` — OpenAlex work ID (or fallback identifier)

Important limitations
- OpenAlex does not reliably expose corresponding authors; this project uses **first author** consistently.
- `cited_by_country` is a practical proxy based on OpenAlex affiliation metadata.

---

## Adding “journal JCR” (Impact Factor / quartile)

JCR (Journal Citation Reports) values are **licensed/proprietary** (Clarivate). This script **does not scrape JCR**.

Instead, you can **attach JCR values** by providing a CSV mapping via `--jcr_csv`.

### 1) Create a mapping CSV

Make a CSV like this (column names are case-insensitive):

```csv
issn_l,journal,jcr
0378-5955,Environmental Science & Technology,"11.4 (2024, Q1)"
0098-5765,Atmospheric Environment,"5.6 (2024, Q2)"
```

Rules
- You need **at least** a `jcr` column.
- Best matching is by **`issn_l`** (recommended).
- If `issn_l` is missing, it falls back to matching by lowercased `journal` name.
- `jcr` can be a number or a text string (e.g., `"11.4 (2024, Q1)"`).

### 2) Run with `--jcr_csv`

```bash
python3 citation_fetcher.py \
  --openalex_id A1234567890 \
  --email your_email@domain.edu \
  --jcr_csv jcr_mapping.csv \
  --output ./data_created/citation_info.csv
```

---

## Step 2 — Make a world citation map

`create_citation_map.py` provides `create_citation_map(...)`.

Minimal example:

```python
from create_citation_map import create_citation_map

create_citation_map(
    "citation_info.csv",
    output_filename="./figs/citation_map.png",
)
```

Example with styling:

```python
from create_citation_map import create_citation_map

create_citation_map(
    "citation_info.csv",
    output_filename="./figs/heatmap_with_labels.png",
    scale="log_rank",          # linear | log | rank | log_rank
    fill_mode="heatmap",       # heatmap | alpha | simple
    fill_cmap="Blues",
    show_labels=True,
    show_counts=True,
    show_legend=True,
    label_top_n=25,
    adjust_labels=True,
    show_pins=False,
)
```

If you save figures to a folder, create it first:

```bash
mkdir -p figs
```

---

## Step 3 — Country bar plot (sorted high → low)

`create_citation_map.py` also provides `plot_citations_by_country_bar(...)`:

```python
from create_citation_map import plot_citations_by_country_bar

plot_citations_by_country_bar(
    "./data_created/citation_info.csv",
    output_filename="./figs/citations_by_country_bar.png",
    top_n=30,               # optional
    show_country_names=True,
    log_y=False,
)
```

---

## Notebook (example end-to-end run)

Open the notebook to reproduce the workflow:

```bash
jupyter lab
# or
jupyter notebook
```

Then open `Citation_map_Lixu.ipynb`.

---

## Troubleshooting

### A paper is missing from “my publications”

That paper likely never entered Phase 1 (publication list). Most robust workaround:
- add its DOI into a DOI CSV and run with `--csv`, or
- ensure the DOI exists in your ORCID record and run with `--orcid`.

### Rate limits / transient API errors

`citation_fetcher.py` retries on HTTP 429 and 5xx with backoff. If you still see failures, re-run.

### `geopandas` install issues

If `pip install -r requirements.txt` fails due to `geopandas`, try installing with conda/mamba.

---

## Data sources

- OpenAlex API (works, citations, host venue metadata)
- Crossref API (only used to fill missing DOIs when needed)
