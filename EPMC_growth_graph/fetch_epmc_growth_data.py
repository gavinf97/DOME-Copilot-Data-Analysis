#!/usr/bin/env python3
"""
Europe PMC Publication Growth Analysis — Machine Learning & Artificial Intelligence
==================================================================================

Fetches ALL publication identifiers from Europe PMC for "machine learning" and
"artificial intelligence" (exact-phrase, quoted), 2000–2025, using cursorMark
deep pagination.  Deduplicates at the individual-publication level using a
unique identifier (PMID → PMCID → source:id fallback) so that publications
mentioning BOTH terms are counted only once in the combined figure.

Date field
----------
Uses ``FIRST_PDATE`` (the date a record first appeared in any source database),
which matches the default date filter shown on the Europe PMC website UI.

Note on EPMC UI number discrepancy
-----------------------------------
The EPMC website search bar performs an **unquoted** (bag-of-words) search by
default:  ``artificial intelligence``  matches any publication containing both
words *anywhere*, even if they are not adjacent (e.g. "artificial … intelligence"
in separate sentences).

This script uses **exact-phrase** (quoted) search:  ``"artificial intelligence"``
matches only publications where the two words appear as a contiguous phrase.

This is the scientifically correct approach for tracking usage of a specific
term, and explains why our counts are slightly lower than the raw EPMC UI count.

Outputs
-------
- ``epmc_ml_publication_ids.csv``   — every ML publication UID per year
- ``epmc_ai_publication_ids.csv``   — every AI publication UID per year
- ``epmc_publication_data.csv``     — yearly summary counts (ML, AI, combined dedup)
- ``epmc_publication_growth.png``   — publication growth graph
"""

import csv
import os
import sys
import time
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Configuration ────────────────────────────────────────────────────────────
EPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
YEAR_FROM = 2000
YEAR_TO = 2025
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = 4  # concurrent year-fetches (be polite to the API)


# ── Session ──────────────────────────────────────────────────────────────────

def create_session():
    """Requests session with automatic retries and connection pooling."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ── Deep-pagination ID fetcher ───────────────────────────────────────────────

def _make_uid(result):
    """
    Derive a unique publication identifier from an EPMC result dict.

    Priority: PMID (most universal) → PMCID → source:id composite.
    """
    pmid = result.get("pmid", "")
    pmcid = result.get("pmcid", "")
    source = result.get("source", "")
    rid = result.get("id", "")
    if pmid:
        return pmid, pmcid, pmid
    if pmcid:
        return pmcid, pmcid, pmid
    return f"{source}:{rid}", pmcid, pmid


def fetch_ids_for_year(session, term, year):
    """
    Fetch ALL publication UIDs for *term* in *year* using cursorMark pagination.

    Returns
    -------
    hit_count : int
        The hitCount reported by EPMC (for validation).
    records : list of dict
        Each dict has keys: uid, pmcid, pmid, year.
    """
    query = f'"{term}" AND (FIRST_PDATE:[{year}-01-01 TO {year}-12-31])'
    cursor = "*"
    records = []
    hit_count = 0

    while True:
        params = {
            "query": query,
            "pageSize": 1000,
            "cursorMark": cursor,
            "format": "json",
            "resultType": "lite",
        }
        try:
            resp = session.get(EPMC_API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"    ⚠  Error {term}/{year}: {exc}", file=sys.stderr)
            break

        if hit_count == 0:
            hit_count = data.get("hitCount", 0)

        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        for r in results:
            uid, pmcid, pmid = _make_uid(r)
            records.append({
                "uid": uid,
                "pmcid": pmcid,
                "pmid": pmid,
                "year": year,
            })

        next_cursor = data.get("nextCursorMark", "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return hit_count, records


def fetch_all_ids(term, year_from=YEAR_FROM, year_to=YEAR_TO):
    """
    Fetch IDs for *term* across all years, concurrently by year.

    Returns
    -------
    yearly_uids : dict  {year: set of uid}
    yearly_hitcounts : dict  {year: int}
    all_records : list of dict  (for CSV export)
    """
    print(f'\n{"─"*60}')
    print(f'  Fetching all IDs for: "{term}"')
    print(f'{"─"*60}')

    session = create_session()
    yearly_uids = defaultdict(set)
    yearly_hitcounts = {}
    all_records = []

    def _worker(y):
        return y, fetch_ids_for_year(session, term, y)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, y): y for y in range(year_from, year_to + 1)}
        for future in as_completed(futures):
            y, (hc, recs) = future.result()
            yearly_hitcounts[y] = hc
            for rec in recs:
                yearly_uids[y].add(rec["uid"])
            all_records.extend(recs)
            collected = len(yearly_uids[y])
            flag = " ✓" if collected == hc else f" (hitCount={hc:,})"
            print(f"  {y}: {collected:>8,} IDs collected{flag}")

    session.close()

    # Print sorted summary
    total_collected = sum(len(v) for v in yearly_uids.values())
    total_hitcount = sum(yearly_hitcounts.values())
    print(f"  TOTAL: {total_collected:,} IDs collected  "
          f"(hitCount sum: {total_hitcount:,})")

    return dict(yearly_uids), yearly_hitcounts, all_records


# ── CSV export: per-term IDs ────────────────────────────────────────────────

def save_ids_csv(records, filename):
    """Save a list of publication-ID dicts to CSV."""
    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["uid", "pmcid", "pmid", "year"])
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: (r["year"], r["uid"])))
    print(f"  📄 {len(records):,} records → {filename}")


# ── CSV export: summary table ───────────────────────────────────────────────

def save_summary_csv(ml_uids, ai_uids, ml_hc, ai_hc,
                     filename="epmc_publication_data.csv"):
    """
    Save yearly summary CSV with columns:
      Year, ML_Collected, ML_HitCount, AI_Collected, AI_HitCount,
      Combined_Deduplicated, Naive_Sum, Overlap
    """
    filepath = os.path.join(SCRIPT_DIR, filename)
    all_years = sorted(set(ml_uids.keys()) | set(ai_uids.keys()))

    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Year",
            "ML_Collected", "ML_HitCount",
            "AI_Collected", "AI_HitCount",
            "Combined_Deduplicated", "Naive_Sum", "Overlap",
        ])
        for y in all_years:
            ml_set = ml_uids.get(y, set())
            ai_set = ai_uids.get(y, set())
            combined = len(ml_set | ai_set)
            naive = len(ml_set) + len(ai_set)
            overlap = naive - combined
            writer.writerow([
                y,
                len(ml_set), ml_hc.get(y, 0),
                len(ai_set), ai_hc.get(y, 0),
                combined, naive, overlap,
            ])
    print(f"  📄 Summary → {filename}")


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(ml_uids, ai_uids,
                 output_file="epmc_publication_growth.png"):
    """
    Publication-quality growth graph with:
      - Combined (deduplicated) line on top, with value labels
      - Individual term lines
      - Legend in visual stacking order (combined first)
      - Hard x-axis bounds at 2000–2025
      - Comma-formatted y-axis
    """
    all_years = sorted(set(ml_uids.keys()) | set(ai_uids.keys()))

    ml_counts = [len(ml_uids.get(y, set())) for y in all_years]
    ai_counts = [len(ai_uids.get(y, set())) for y in all_years]
    combined_counts = [len(ml_uids.get(y, set()) | ai_uids.get(y, set()))
                       for y in all_years]

    fig, ax = plt.subplots(figsize=(16, 9))

    # ── Combined line (plotted first → top of legend) ──
    ax.plot(all_years, combined_counts,
            marker="s", linewidth=2.5, linestyle="--",
            label="Combined (AI ∪ ML) — deduplicated",
            color="black", markersize=7, zorder=3)

    # ── Value labels on combined line ──
    max_count = max(combined_counts) if combined_counts else 1
    offset = max_count * 0.025  # 2.5% of y-range above the point

    for x, y_val in zip(all_years, combined_counts):
        ax.annotate(
            f"{y_val:,}",
            xy=(x, y_val),
            xytext=(0, 10),           # 10 points above the marker
            textcoords="offset points",
            fontsize=6.5,
            fontweight="bold",
            ha="center", va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="white", edgecolor="none", alpha=0.75),
        )

    # ── Individual lines ──
    ax.plot(all_years, ml_counts,
            marker="o", linewidth=2, label="Machine Learning",
            color="#1f77b4", markersize=5, zorder=2)
    ax.plot(all_years, ai_counts,
            marker="o", linewidth=2, label="Artificial Intelligence",
            color="#ff7f0e", markersize=5, zorder=2)

    # ── Axes ──
    ax.set_xlim(YEAR_FROM - 0.5, YEAR_TO + 0.5)
    ax.set_xticks(range(YEAR_FROM, YEAR_TO + 1))
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Publications", fontsize=13, fontweight="bold")
    ax.set_title(
        "Publication Growth: Machine Learning & Artificial Intelligence\n"
        "Europe PMC — exact-phrase search, FIRST_PDATE, PMID-deduplicated (2000–2025)",
        fontsize=14, fontweight="bold",
    )

    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(SCRIPT_DIR, output_file)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"\n  ✅ Plot saved to: {output_file}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 70)
    print("  Europe PMC Publication Growth Analysis")
    print("  Exact-phrase search · FIRST_PDATE · PMID-level deduplication")
    print("=" * 70)

    # ── Phase 1: Fetch ALL IDs ──
    ml_uids, ml_hc, ml_records = fetch_all_ids("machine learning")
    ai_uids, ai_hc, ai_records = fetch_all_ids("artificial intelligence")

    # ── Phase 2: Save per-term ID CSVs ──
    print(f'\n{"─"*60}')
    print("  Saving publication-ID CSVs")
    print(f'{"─"*60}')
    save_ids_csv(ml_records, "epmc_ml_publication_ids.csv")
    save_ids_csv(ai_records, "epmc_ai_publication_ids.csv")

    # ── Phase 3: Deduplication & summary ──
    print(f'\n{"─"*60}')
    print("  Deduplication Summary")
    print(f'{"─"*60}')

    all_years = sorted(set(ml_uids.keys()) | set(ai_uids.keys()))
    total_ml = sum(len(s) for s in ml_uids.values())
    total_ai = sum(len(s) for s in ai_uids.values())
    total_combined = sum(
        len(ml_uids.get(y, set()) | ai_uids.get(y, set()))
        for y in all_years
    )
    total_naive = total_ml + total_ai
    total_overlap = total_naive - total_combined

    print(f"\n  Machine Learning:        {total_ml:>10,} publications")
    print(f"  Artificial Intelligence: {total_ai:>10,} publications")
    print(f"  Naïve sum (ML + AI):     {total_naive:>10,}")
    print(f"  Combined (deduplicated): {total_combined:>10,}")
    print(f"  Overlap removed:         {total_overlap:>10,}  "
          f"({total_overlap/total_naive*100:.1f}% of naïve sum)")

    # Per-year breakdown
    print(f"\n  {'Year':>6}  {'ML':>8}  {'AI':>8}  {'Naive':>8}  "
          f"{'Combined':>8}  {'Overlap':>8}  {'Overlap%':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for y in all_years:
        ml = len(ml_uids.get(y, set()))
        ai = len(ai_uids.get(y, set()))
        naive = ml + ai
        comb = len(ml_uids.get(y, set()) | ai_uids.get(y, set()))
        ov = naive - comb
        ov_pct = (ov / naive * 100) if naive > 0 else 0
        print(f"  {y:>6}  {ml:>8,}  {ai:>8,}  {naive:>8,}  "
              f"{comb:>8,}  {ov:>8,}  {ov_pct:>7.1f}%")

    # ── Phase 4: Save summary CSV ──
    save_summary_csv(ml_uids, ai_uids, ml_hc, ai_hc)

    # ── Phase 5: Plot ──
    plot_results(ml_uids, ai_uids)

    # ── Cleanup old files ──
    for old in ["epmc_publication_data.json"]:
        old_path = os.path.join(SCRIPT_DIR, old)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"  🗑  Removed old file: {old}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Analysis complete!  ({elapsed:.0f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
