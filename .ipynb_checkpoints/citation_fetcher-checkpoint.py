#!/usr/bin/env python3
"""
citation_fetcher.py

Goal
----
Create a "citation table" for YOUR publications, listing the papers that cite them.

This script supports several input sources for "your publications":
  1) --openalex_id  : Best & most stable (no Google/Selenium)
  2) --orcid        : Stable; DOIs from ORCID then fill missing via Crossref
  3) --csv          : You provide a DOI column
  4) --scholar_id   : Uses `scholarly` (often blocked on Colab/HPC), kept as optional

Output (ONE ROW PER CITING PAPER)
---------------------------------
For each of your publications, we output rows like:
  my_publication, cited_by_title, first_author, cited_by_year, cited_by_doi, cited_by_id

Important Notes
---------------
- OpenAlex does NOT reliably provide "corresponding author". We only provide first author.
- OpenAlex citation counts differ from Google Scholar sometimes (coverage differences).
- Using OpenAlex avoids Selenium/Chrome entirely.

Usage Examples
--------------
# Preferred: OpenAlex author id
python3 citation_fetcher.py --openalex_id A1234567890 --email you@domain.edu --output ./data_created/citation_info.csv

# ORCID
python3 citation_fetcher.py --orcid 0000-0000-0000-0000 --email you@domain.edu --output ./data_created/citation_info.csv

# CSV (must include DOI or doi column)
python3 citation_fetcher.py --csv my_dois.csv --email you@domain.edu --output ./data_created/citation_info.csv

# Scholar (often blocked on shared IPs)
python3 citation_fetcher.py --scholar_id Q_sLQJIAAAAJ --email you@domain.edu --output ./data_created/citation_info.csv
"""

import argparse
import concurrent.futures
import csv
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Google Scholar is optional and often blocked; keep import but handle failures gracefully.
try:
    from scholarly import scholarly  # type: ignore
except Exception:
    scholarly = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENALEX_API_URL = "https://api.openalex.org"
CROSSREF_API_URL = "https://api.crossref.org/works"

# Crossref can rate-limit; keep conservative on shared IPs (Colab/HPC).
MAX_WORKERS = 5

# OpenAlex politeness: keep below 10 req/sec; we add a small delay.
OPENALEX_SLEEP_S = 0.12


class CitationFetcher:
    """
    Fetcher class that:
    1) determines "your publications" (OpenAlex works objects) from an input source
    2) fetches the works that cite each publication via OpenAlex
    3) exports "one row per citing paper" with first author
    """

    def __init__(self, email: Optional[str] = None):
        self.session = requests.Session()

        # IMPORTANT FIX:
        # Always initialize self.session.params to avoid AttributeError
        # when email=None and later we do dict(getattr(self.session,'params',...)).
        self.session.params = {}
        if email:
            # OpenAlex recommends mailto for politeness
            self.session.params["mailto"] = email

        self.email = email

    # -----------------------------------------------------------------------
    # Utility helpers (retry/backoff, normalization, title similarity)
    # -----------------------------------------------------------------------

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        """Exponential backoff + jitter."""
        time.sleep((2**attempt) + random.random())

    def _get_with_retries(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_tries: int = 6,
    ) -> requests.Response:
        """
        Robust GET with retry for rate-limit/transient errors.
        Retries: HTTP 429, 5xx, plus request exceptions.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(max_tries):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=timeout)

                # Rate limit or transient server errors
                if r.status_code in (429, 500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue

                r.raise_for_status()
                return r

            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt == max_tries - 1:
                    raise
                self._sleep_backoff(attempt)

        raise RuntimeError(f"Unreachable: failed GET {url}") from last_exc

    @staticmethod
    def _normalize_doi(doi_str: str) -> str:
        """
        Normalize DOI for consistent matching:
          - remove https://doi.org/ or http://doi.org/
          - remove leading 'doi:'
          - lower-case + strip whitespace
        """
        if not doi_str:
            return ""
        s = doi_str.strip().lower()
        s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
        s = s.replace("doi:", "").strip()
        return s

    @staticmethod
    def _title_similarity_ok(a: str, b: str, threshold: float = 0.60) -> bool:
        """
        Quick heuristic to reduce wrong Crossref DOI matches.
        Returns True if:
          - one title contains the other, OR
          - token overlap fraction >= threshold
        """
        if not a or not b:
            return False
        a_clean = re.sub(r"\W+", " ", a.lower()).strip()
        b_clean = re.sub(r"\W+", " ", b.lower()).strip()
        if not a_clean or not b_clean:
            return False
        if a_clean in b_clean or b_clean in a_clean:
            return True
        a_set = set(a_clean.split())
        b_set = set(b_clean.split())
        overlap = len(a_set & b_set) / max(1, len(a_set))
        return overlap >= threshold

    # -----------------------------------------------------------------------
    # OpenAlex: pagination + DOI batching
    # -----------------------------------------------------------------------

    def _get_paginated_results(
        self,
        endpoint_or_url: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch ALL pages of results from OpenAlex using cursor pagination.

        endpoint_or_url can be:
          - a full URL (e.g., works_api_url from OpenAlex), OR
          - an endpoint path starting with '/' (e.g., '/works')
        """
        all_results: List[Dict[str, Any]] = []

        # Start with session params (e.g., mailto), then add OpenAlex defaults.
        params: Dict[str, Any] = dict(getattr(self.session, "params", {}) or {})
        params.update({"per_page": 200, "cursor": "*"})


        if extra_params:
            params.update(extra_params)

        # Build a full URL if needed
        if endpoint_or_url.startswith("http"):
            url = endpoint_or_url
        else:
            url = f"{OPENALEX_API_URL}{endpoint_or_url}"

        while params.get("cursor"):
            try:
                r = self._get_with_retries(url, params=params, timeout=30)
                data = r.json()

                all_results.extend(data.get("results", []))
                params["cursor"] = data.get("meta", {}).get("next_cursor")

                # Politeness delay
                time.sleep(OPENALEX_SLEEP_S)

            except requests.exceptions.RequestException as e:
                resolved = getattr(locals().get("r", None), "url", url)
                print(f"[Error] OpenAlex request failed: {e} (URL: {resolved})")
                break

        return all_results

    def _fetch_works_by_doi_batch(self, dois: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch OpenAlex works objects from a DOI list, in batches, using:
          /works?filter=doi:doi1|doi2|doi3

        IMPORTANT:
        - We pass filter via params (so requests URL-encodes it correctly).
        - We normalize + deduplicate DOIs first.
        """
        BATCH_SIZE = 50
        all_works: List[Dict[str, Any]] = []

        # Normalize + deduplicate
        valid_dois: List[str] = []
        for d in dois:
            if isinstance(d, str) and d.strip():
                nd = self._normalize_doi(d)
                if nd:
                    valid_dois.append(nd)
        valid_dois = sorted(set(valid_dois))

        print(f"Fetching metadata for {len(valid_dois)} DOIs from OpenAlex...")

        for i in range(0, len(valid_dois), BATCH_SIZE):
            batch = valid_dois[i : i + BATCH_SIZE]
            doi_filter = "|".join(batch)

            # Use params encoding (robust)
            results = self._get_paginated_results(
                "/works",
                extra_params={"filter": f"doi:{doi_filter}"},
            )
            all_works.extend(results)

        return all_works

    # -----------------------------------------------------------------------
    # Publication acquisition: ORCID, Scholar, Crossref, OpenAlex author id
    # -----------------------------------------------------------------------

    def _get_doi_info_from_crossref(self, title: str) -> Tuple[str, str]:
        """
        Query Crossref to find a DOI for a title.
        Returns (doi, matched_title). May return ("","") if no match.
        """
        if not title or len(title) < 5:
            return "", ""

        params: Dict[str, Any] = {"query.title": title, "rows": 1}
        if self.email:
            # Crossref supports mailto for polite usage
            params["mailto"] = self.email

        try:
            r = self._get_with_retries(CROSSREF_API_URL, params=params, timeout=20)
            items = r.json().get("message", {}).get("items", [])
            if not items:
                return "", ""
            item = items[0]
            doi = item.get("DOI", "") or ""
            titles = item.get("title", []) or []
            found_title = titles[0] if titles else ""
            return doi, found_title
        except Exception:
            return "", ""

    def _fetch_orcid_data(self, orcid_id: str) -> List[List[str]]:
        """
        Fetch ORCID works list.
        Returns rows: [title, doi, 'ORCID'].
        """
        print(f"Fetching data from ORCID ID: {orcid_id}...")
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
        headers = {"Accept": "application/json"}

        results: List[List[str]] = []
        try:
            r = self._get_with_retries(url, headers=headers, timeout=20)
            data = r.json()

            for group in data.get("group", []):
                summary = group["work-summary"][0]
                title = summary.get("title", {}).get("title", {}).get("value", "Unknown Title")

                doi = ""
                for eid in summary.get("external-ids", {}).get("external-id", []):
                    if eid.get("external-id-type") == "doi":
                        doi = eid.get("external-id-value", "") or ""
                        break

                results.append([title, doi, "ORCID"])

        except Exception as e:
            print(f"[Error] accessing ORCID: {e}")

        return results

    def _fetch_scholar_data(self, scholar_id: str) -> List[List[str]]:
        """
        Fetch publication list from Google Scholar via `scholarly`.
        Returns rows: [title, '', ''].
        NOTE: Often blocked on Colab/HPC.
        """
        print(f"Fetching publication list from Google Scholar ID: {scholar_id}...")

        if scholarly is None:
            print("[Error] scholarly package not available in this environment.")
            return []

        try:
            author = scholarly.search_author_id(scholar_id)
            scholarly.fill(author, sections=["publications"])
            pubs = author.get("publications", [])
            return [[p.get("bib", {}).get("title", ""), "", ""] for p in pubs]
        except Exception as e:
            print(f"[Error] accessing Google Scholar: {e}")
            return []

    def _resolve_missing_dois(self, data: List[List[str]]) -> List[List[str]]:
        """
        Fill missing DOIs in-place via Crossref.
        We use a title similarity heuristic to reduce wrong matches.
        """
        missing_indices = [i for i, row in enumerate(data) if not (row[1] or "").strip()]
        total_missing = len(missing_indices)

        if total_missing == 0:
            return data

        print(f"Resolving {total_missing} missing DOIs via Crossref...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(self._get_doi_info_from_crossref, data[i][0]): i
                for i in missing_indices
            }

            done = 0
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    found_doi, found_title = future.result()
                    if found_doi and self._title_similarity_ok(data[idx][0], found_title):
                        data[idx][1] = found_doi
                        data[idx][2] = f"Crossref ({found_title})"
                except Exception:
                    pass

                done += 1
                print(f"\rProgress: {done}/{total_missing}", end="", flush=True)

        print("\nResolution complete.")
        return data

    # -----------------------------------------------------------------------
    # Main workflow
    # -----------------------------------------------------------------------

    def run(self, source_type: str, source_value: str, output_csv: str) -> None:
        """
        Full pipeline:
          Phase 1: Identify your publications (OpenAlex works objects)
          Phase 2: For each work, fetch citing works and export 1 row per citing paper
        """
        my_publications: List[Dict[str, Any]] = []

        # Intermediate publication list output
        pub_list_filename = "./data_created/publications_with_doi.csv"
        if source_type == "orcid":
            pub_list_filename = "./data_created/publications_with_doi_orcid.csv"
        elif source_type == "scholar":
            pub_list_filename = "./data_created/publications_with_doi_scholar.csv"
        elif source_type == "openalex":
            pub_list_filename = "./data_created/publications_with_doi_openalex.csv"

        # -----------------------------
        # PHASE 1: Acquire publications
        # -----------------------------
        if source_type in ["orcid", "scholar"]:
            # A) Get raw publication list with missing DOIs allowed
            if source_type == "orcid":
                if not re.match(r"^\d{4}-\d{4}-\d{4}-[\dX]{4}$", source_value):
                    print("Warning: ID does not look like a valid ORCID format.")
                data = self._fetch_orcid_data(source_value)
            else:
                data = self._fetch_scholar_data(source_value)

            if not data:
                print("No data found from source. Exiting.")
                return

            # B) Fill missing DOIs via Crossref
            data = self._resolve_missing_dois(data)
            data.sort(key=lambda x: str(x[0]).lower())

            # C) Save intermediate list
            try:
                with open(pub_list_filename, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["my_publication", "DOI", "DOI Source"])
                    writer.writerows(data)
                print(f"\nPublication list saved to {pub_list_filename}\n")
                print("Proceeding to fetch citations using DOIs from this list...")
            except IOError as e:
                print(f"[Error] File I/O Error: {e}")
                return

            # D) Use DOIs to fetch OpenAlex works objects
            input_doi_list = [row[1] for row in data if row[1]]
            print(f"Found {len(input_doi_list)} DOIs. Querying OpenAlex...")
            my_publications = self._fetch_works_by_doi_batch(input_doi_list)

        elif source_type == "csv":
            input_source = source_value
            if not os.path.exists(input_source):
                print(f"[Error] CSV file not found: {input_source}")
                return

            print(f"Reading CSV: {input_source}")
            df = pd.read_csv(input_source)

            doi_col = next((c for c in df.columns if c.lower() == "doi"), None)
            if not doi_col:
                print("[Error] CSV must contain a 'DOI' or 'doi' column.")
                return

            # Normalize/deduplicate input DOIs
            raw_dois = df[doi_col].astype(str).tolist()
            input_doi_map: Dict[str, str] = {}
            for d in raw_dois:
                if str(d).strip():
                    norm = self._normalize_doi(str(d))
                    if norm:
                        input_doi_map[norm] = str(d).strip()

            print(f"Total unique DOIs provided in CSV: {len(input_doi_map)}")
            my_publications = self._fetch_works_by_doi_batch(list(input_doi_map.values()))

            # Report missing DOIs (not found in OpenAlex)
            found_norm = set()
            for pub in my_publications:
                pub_doi = pub.get("doi")
                if pub_doi:
                    found_norm.add(self._normalize_doi(pub_doi))
            missing = set(input_doi_map.keys()) - found_norm

            if missing:
                print(f"\n[Warning] {len(missing)} DOIs were NOT found in OpenAlex:")
                for m in sorted(missing):
                    print(f" - {input_doi_map[m]}")
                time.sleep(2)
            else:
                print(f"All {len(input_doi_map)} provided DOIs were successfully found.")

        elif source_type == "openalex":
            author_id = source_value.strip()
            print(f"\nFetching author details for ID: {author_id}")

            author_url = f"{OPENALEX_API_URL}/authors/{author_id}"
            try:
                r = self._get_with_retries(author_url, timeout=30)
                author_data = r.json()

                print(f"Author found: {author_data.get('display_name', 'Unknown')}")
                works_api_url = author_data.get("works_api_url")
                if not works_api_url:
                    print(f"[Error] Could not find 'works_api_url' for author ID: {author_id}")
                    return

                print(f"Fetching works from: {works_api_url}")
                my_publications = self._get_paginated_results(works_api_url)

                # Save publication list for reference
                if my_publications:
                    print(f"Found {len(my_publications)} works. Saving publication list...")
                    oa_csv_data = []
                    for pub in my_publications:
                        title = pub.get("title", "Unknown Title")
                        raw_doi = pub.get("doi", "") or ""
                        clean_doi = self._normalize_doi(raw_doi) if raw_doi else ""
                        oa_csv_data.append([title, clean_doi, "OpenAlex"])

                    try:
                        with open(pub_list_filename, "w", newline="", encoding="utf-8-sig") as f:
                            writer = csv.writer(f)
                            writer.writerow(["my_publication", "DOI", "DOI Source"])
                            writer.writerows(oa_csv_data)
                        print(f"\nPublication list saved to {pub_list_filename}")
                    except IOError as e:
                        print(f"\n[Warning] Could not save publication list CSV: {e}\n")

            except Exception as e:
                print(f"[Error] fetching OpenAlex Author: {e}")
                return

        else:
            print(f"[Error] Unknown source_type: {source_type}")
            return

        # -----------------------------
        # PHASE 2: Fetch citations
        # -----------------------------
        print(f"\nProcessing citations for {len(my_publications)} publications...")
        all_rows: List[Dict[str, Any]] = []

        for i, my_pub in enumerate(my_publications):
            my_pub_title = my_pub.get("title", "N/A")
            cited_by_count = int(my_pub.get("cited_by_count", 0) or 0)

            print(f"\n--- Processing my publication {i+1}/{len(my_publications)} ---")
            print(f"Title: {my_pub_title}")
            print(f"Citations (OpenAlex): {cited_by_count}")

            if cited_by_count == 0:
                print("No citations. Skipping.")
                continue

            my_pub_full_id = my_pub.get("id")
            if not my_pub_full_id:
                print("Warning: Publication ID missing. Skipping.")
                continue

            # OpenAlex work ID is like https://openalex.org/W##########
            work_id = my_pub_full_id.split("/")[-1]

            # Fetch all works that cite this work
            citing_papers = self._get_paginated_results(
                "/works",
                extra_params={"filter": f"cites:{work_id}"},
            )

            # ONE ROW PER CITING PAPER
            for citing_paper in citing_papers:
                citing_title = citing_paper.get("title", "N/A")
                citing_year  = citing_paper.get("publication_year", "N/A")
                citing_doi   = citing_paper.get("doi", "") or ""
                
                citing_id_full = citing_paper.get("id") or ""
                citing_id = (
                    citing_id_full.split("/")[-1]
                    if citing_id_full
                    else (self._normalize_doi(citing_doi) or f"{citing_year}:{citing_title}")

                )

                # First author (reliable)
                authorships = citing_paper.get("authorships", []) or []
                
                # First author name
                first_author = (
                    authorships[0].get("author", {}).get("display_name", "N/A")
                    if authorships else "N/A"
                )


                # Best-effort single country assignment:
                # use the first author's first institution country code (e.g., "US")
                cited_by_country = "N/A"
                for a in authorships:
                    insts = a.get("institutions", []) or []
                    for inst in insts:
                        cc = inst.get("country_code")
                        if cc:
                            cited_by_country = cc
                            break
                    if cited_by_country != "N/A":
                        break

                
                all_rows.append({
                    "my_publication": my_pub_title,
                    "cited_by_id": citing_id,
                    "cited_by_title": citing_title,
                    "first_author": first_author,
                    "cited_by_year": citing_year,
                    "cited_by_doi": citing_doi,
                    "cited_by_country": cited_by_country,   # <-- ADD THIS
                })

        if not all_rows:
            print("No citation data found.")
            return

        out_df = pd.DataFrame(all_rows)

        # Deduplicate to guarantee ONE ROW per (my_publication, citing_work)
        out_df = out_df.drop_duplicates(subset=["my_publication", "cited_by_id"])

        # Column order
        cols = [
            "my_publication",
            "cited_by_title",
            "first_author",
            "cited_by_year",
            "cited_by_doi",
            "cited_by_country",   # <-- ADD THIS
            "cited_by_id",
        ]
        out_df = out_df[[c for c in cols if c in out_df.columns]]

        # Save
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n[Success] Generated {len(out_df)} rows (unique citing papers).")
        print(f"Citation info saved to: {output_csv}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch academic citations using OpenAlex/Crossref/ORCID.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--openalex_id", help="Your OpenAlex Author ID (e.g., A5023888391)")
    group.add_argument("--orcid", help="Your ORCID iD (e.g., 0000-0000-0000-0000)")
    group.add_argument("--scholar_id", help="Your Google Scholar ID (often blocked on Colab/HPC)")
    group.add_argument("--csv", help="Path to a CSV file containing a 'DOI' or 'doi' column")

    parser.add_argument("--output", default="./data_created/citation_info.csv", help="Output CSV filename")
    parser.add_argument("--email", help="Your email for API politeness (recommended)")

    args = parser.parse_args()
    fetcher = CitationFetcher(email=args.email)

    if args.openalex_id:
        fetcher.run("openalex", args.openalex_id, args.output)
    elif args.orcid:
        fetcher.run("orcid", args.orcid, args.output)
    elif args.scholar_id:
        fetcher.run("scholar", args.scholar_id, args.output)
    elif args.csv:
        fetcher.run("csv", args.csv, args.output)


if __name__ == "__main__":
    main()
