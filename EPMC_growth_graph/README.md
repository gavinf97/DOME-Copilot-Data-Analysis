# Europe PMC Publication Growth Analysis

Analyses the growth of publications mentioning **"machine learning"** and
**"artificial intelligence"** (exact-phrase search) in the Europe PMC database
from 2000 to 2025, with proper PMID-level deduplication of the combined figure.

## Methodology

### Search strategy

| Aspect | Detail |
|---|---|
| **Database** | [Europe PMC](https://europepmc.org/) via REST API |
| **Terms** | `"machine learning"`, `"artificial intelligence"` (quoted = exact phrase) |
| **Date field** | `FIRST_PDATE` — the date a record first appeared in any source database |
| **Date range** | 2000-01-01 to 2025-12-31 |
| **Pagination** | `cursorMark` deep pagination (fetches **every** result, not just hitCount) |
| **Dedup key** | PMID (preferred) → PMCID → `source:id` composite |

### Deduplication of the combined figure

A publication that mentions **both** "machine learning" and
"artificial intelligence" would be counted in each individual term's total.
Naïvely summing the two totals therefore **double-counts** such publications.

To produce a scientifically correct combined figure, the script:

1. Fetches **every publication identifier** for each term using `cursorMark`
   deep pagination (no 1 000-result cap).
2. Stores per-term identifiers in CSV files for full transparency.
3. Computes the **set union** of identifiers per year — each publication is
   counted exactly once regardless of how many terms it matches.

In the current dataset, the overlap is **~17–19 %** of the naïve sum for
recent years (2020–2025), meaning roughly one in five publications mentions
both terms.

### Why do numbers differ from the EPMC website UI?

Two factors cause differences between the counts this script returns and what
you see when searching directly on europepmc.org:

#### 1. Exact-phrase vs. bag-of-words search

| Query type | Example | 2024 AI count |
|---|---|---|
| **Unquoted** (EPMC UI default) | `artificial intelligence` | ~65,424 |
| **Quoted** (this script) | `"artificial intelligence"` | ~63,951 |

The EPMC search bar performs an **unquoted** search by default: it matches any
publication containing both words *anywhere* in the record, even if they are
not adjacent (e.g. "… artificial … intelligence …" across different sentences).

This script uses **quoted** (exact-phrase) search, which requires the words to
appear as a contiguous phrase.  This is the scientifically appropriate choice
for tracking usage of a specific term.

#### 2. `PUB_YEAR` vs. `FIRST_PDATE`

| Field | Meaning | 2024 AI count (quoted) |
|---|---|---|
| `PUB_YEAR` | Journal-assigned publication year | ~62,292 |
| `FIRST_PDATE` | Date the record first appeared online | ~63,951 |

A paper published online in December 2023 but assigned to a 2024 journal issue
would have `FIRST_PDATE` in 2023 but `PUB_YEAR` = 2024, or vice versa.

This script uses `FIRST_PDATE` to match the EPMC website's default date filter.

## Files

| File | Description |
|---|---|
| `fetch_epmc_growth_data.py` | Main Python script |
| `epmc_publication_growth.png` | Growth visualisation (300 dpi) |
| `epmc_publication_data.csv` | Yearly summary: ML, AI, combined (dedup), overlap |
| `epmc_ml_publication_ids.csv` | All ML publication identifiers (~478 k rows) |
| `epmc_ai_publication_ids.csv` | All AI publication identifiers (~356 k rows) |

### CSV schemas

**`epmc_publication_data.csv`**

| Column | Description |
|---|---|
| `Year` | Publication year |
| `ML_Collected` | Machine learning IDs collected via pagination |
| `ML_HitCount` | EPMC-reported hitCount for validation |
| `AI_Collected` | Artificial intelligence IDs collected |
| `AI_HitCount` | EPMC-reported hitCount for validation |
| `Combined_Deduplicated` | Union of ML ∪ AI identifier sets |
| `Naive_Sum` | ML + AI (without deduplication) |
| `Overlap` | Publications matching both terms (Naive − Combined) |

**`epmc_ml_publication_ids.csv`** / **`epmc_ai_publication_ids.csv`**

| Column | Description |
|---|---|
| `uid` | Unique deduplication key (PMID or PMCID or source:id) |
| `pmcid` | PubMed Central ID (if available) |
| `pmid` | PubMed ID (if available) |
| `year` | Year from `FIRST_PDATE` |

## Requirements

```bash
pip install requests matplotlib
```

## Usage

```bash
python fetch_epmc_growth_data.py
```

> **Note:** The full run takes ~30 minutes because it paginates through every
> result in EPMC to collect individual publication identifiers for deduplication.
> There are 800 000+ publications to fetch across both terms.

## Output graph

The graph shows:

- **Black dashed line (top of legend):** Combined (AI ∪ ML), deduplicated —
  each publication counted once.
- **Blue line:** Machine Learning (exact phrase).
- **Orange line:** Artificial Intelligence (exact phrase).
- Value labels on every combined data point.
- X-axis hard-bounded to 2000–2025.

## Europe PMC API

- **Endpoint:** `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
- **Pagination:** `cursorMark` for deep pagination beyond 1 000 results
- **No authentication required**
- **Rate limiting:** The script uses 4 concurrent workers with retries
