import requests
import pandas as pd
import time
import os
import sys
import re
import csv
import argparse
import concurrent.futures
from typing import Optional, List, Dict, Any, Tuple
from scholarly import scholarly

# --- Configuration & Constants ---
OPENALEX_API_URL = "https://api.openalex.org"
CROSSREF_API_URL = "https://api.crossref.org/works"
MAX_WORKERS = 10 # For Crossref multithreading

class CitationFetcher:
    def __init__(self, email: Optional[str] = None):
        self.session = requests.Session()
        if email:
            self.session.params = {'mailto': email}
        self.email = email

    # =========================================================================
    # MODULE 1: Functions from fetch_citation_info.py (OpenAlex & Processing)
    # =========================================================================

    def _get_paginated_results(self, url: str) -> List[Dict[str, Any]]:
        """
        A helper function to handle OpenAlex API cursor pagination.
        It fetches all pages of results for a given URL.
        """
        all_results = []
        params = self.session.params.copy()
        params.update({'per_page': 200, 'cursor': '*'})
        
        while params['cursor']:
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()  # Raise an exception for bad responses
                data = response.json()
                
                all_results.extend(data.get('results', []))
                
                # Get the next_cursor. If None, the loop will stop.
                params['cursor'] = data.get('meta', {}).get('next_cursor')
                
                # Comply with the OpenAlex politeness policy (10 requests/sec)
                time.sleep(0.1) 
                
            except requests.exceptions.RequestException as e:
                print(f"[Error] API request failed: {e} (URL: {url})")
                break
                
        return all_results

    def _fetch_works_by_doi_batch(self, dois: List[str]) -> List[Dict[str, Any]]:
        """
        Helper: Fetch works using a list of DOIs via the OpenAlex 'filter' parameter.
        Batches requests to avoid URL length limits.
        """
        all_works = []
        BATCH_SIZE = 50 # Safe batch size for URL length
        
        # Remove empty or non-string values
        valid_dois = [d.strip().lower() for d in dois if isinstance(d, str) and d.strip()]
        
        print(f"Fetching metadata for {len(valid_dois)} DOIs from OpenAlex...")
        for i in range(0, len(valid_dois), BATCH_SIZE):
            batch = valid_dois[i:i + BATCH_SIZE]
            # Construct filter: filter=doi:url1|url2|url3
            doi_filter = "|".join(batch)
            url = f"{OPENALEX_API_URL}/works?filter=doi:{doi_filter}"
            
            # print(f"Fetching DOI batch {i//BATCH_SIZE + 1}...")
            results = self._get_paginated_results(url)
            all_works.extend(results)
            
        return all_works


    # =========================================================================
    # MODULE 2: Functions from fetch_pubs.py (ORCID, Scholar, Crossref)
    # =========================================================================

    def _get_doi_info_from_crossref(self, title: str) -> Tuple[str, str]:
        """
        Retrieve DOI and the matched title from Crossref API.
        Returns: (doi, crossref_title)
        """
        if not title or len(title) < 5: return "", ""
        
        params = {'query.title': title, 'rows': 1}
        if self.email: params['mailto'] = self.email

        try:
            response = requests.get(CROSSREF_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                items = response.json().get('message', {}).get('items', [])
                if items:
                    item = items[0]
                    doi = item.get('DOI', '')
                    # Crossref returns titles as a list
                    titles = item.get('title', [])
                    found_title = titles[0] if titles else ""
                    return doi, found_title
        except Exception:
            pass
        return "", ""

    def _normalize_doi_for_comparison(self, doi_str: str) -> str:
        """
        Helper: Remove 'https://doi.org/' prefix to compare DOIs reliably.
        """
        if not doi_str:
            return ""
        return doi_str.strip().lower().replace("https://doi.org/", "").replace("http://doi.org/", "")


    def _fetch_orcid_data(self, orcid_id):
        """Retrieve data from ORCID API."""
        print(f"Fetching data from ORCID ID: {orcid_id}...")
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
        headers = {"Accept": "application/json"}
        
        results = []
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: 
                print(f"[Error] ORCID API returned status {response.status_code}")
                return []
                
            data = response.json()
            for group in data.get("group", []):
                summary = group["work-summary"][0]
                title = summary.get("title", {}).get("title", {}).get("value", "Unknown Title")
                doi = ""
                # Attempt to extract DOI from ORCID metadata
                for eid in summary.get("external-ids", {}).get("external-id", []):
                    if eid.get("external-id-type") == "doi":
                        doi = eid.get("external-id-value")
                        break
                # Row format: [Original Title, DOI, ORCID source]
                results.append([title, doi, "ORCID"])
                
        except Exception as e:
            print(f"[Error] accessing ORCID: {e}")
        
        return results

    def _fetch_scholar_data(self, scholar_id):
        """Retrieve data from Google Scholar."""
        print(f"Fetching publication list from Google Scholar ID: {scholar_id}...")
        try:
            author = scholarly.search_author_id(scholar_id)
            scholarly.fill(author, sections=['publications'])
            pubs = author['publications']
            
            # Initialize with empty DOIs and empty DOI Source
            # Row format: [Original Title, DOI, DOI Source]
            return [[p['bib'].get('title'), "", ""] for p in pubs]
            
        except Exception as e:
            print(f"[Error] accessing Google Scholar: {e}")
            return []

    def _resolve_missing_dois(self, data):
        """
        Scans data for missing DOIs and fetches them via Crossref.
        Updates data in-place with found DOI and the Crossref title for verification.
        """
        # Indices where DOI is missing (index 1 is DOI)
        missing_indices = [i for i, row in enumerate(data) if not row[1]]
        total_missing = len(missing_indices)

        if total_missing == 0:
            return data

        print(f"Resolving {total_missing} missing DOIs via Crossref...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map future to the row index
            future_to_index = {
                executor.submit(self._get_doi_info_from_crossref, data[i][0]): i 
                for i in missing_indices
            }
            
            for count, future in enumerate(concurrent.futures.as_completed(future_to_index)):
                index = future_to_index[future]
                try:
                    found_doi, found_title = future.result()
                    if found_doi:
                        data[index][1] = found_doi
                        data[index][2] = f"Crossref ({found_title})" # Store source
                except Exception:
                    pass
                print(f"\rProgress: {count + 1}/{total_missing}", end='', flush=True)
        
        print("\nResolution complete.")
        return data


    # =========================================================================
    # MODULE 3: Integrated Workflow
    # =========================================================================

    def run(self, source_type: str, source_value: str, output_csv: str):
        """
        Main execution logic combining fetch_pubs and fetch_citation_info flows.
        """
        my_publications = [] # This will store OpenAlex work objects
        
        # Determine intermediate filename based on source
        pub_list_filename = "publications_with_doi.csv" # Default fallback
        if source_type == 'orcid':
            pub_list_filename = "publications_with_doi_orcid.csv"
        elif source_type == 'scholar':
            pub_list_filename = "publications_with_doi_scholar.csv"
        elif source_type == 'openalex':
            pub_list_filename = "publications_with_doi_openalex.csv"
        
        # ---------------------------------------------------------------------
        # PHASE 1: Acquire Publication Data (ID -> List of DOIs)
        # ---------------------------------------------------------------------
        
        # Case A: ORCID or Google Scholar
        if source_type in ['orcid', 'scholar']:
            # 1. Fetch raw list
            if source_type == 'orcid':
                # Check format
                if not re.match(r'^\d{4}-\d{4}-\d{4}-[\dX]{4}$', source_value):
                    print("Warning: ID does not look like a valid ORCID format.")
                data = self._fetch_orcid_data(source_value)
            else:
                data = self._fetch_scholar_data(source_value)
            
            if not data:
                print("No data found from source. Exiting.")
                return

            # 2. Fill Missing DOIs and Crossref Titles
            data = self._resolve_missing_dois(data)
            data.sort(key=lambda x: str(x[0]).lower())

            # 3. Save intermediate file (as fetch_pubs did)
            try:
                with open(pub_list_filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["my_publication", "DOI", "DOI Source"])
                    writer.writerows(data)
                print(f"\nPublication list saved to {pub_list_filename}\n")
                print(f"Proceeding to fetch citations using DOIs from this list...")
            except IOError as e:
                print(f"[Error] File I/O Error: {e}")
                return

            # 4. Prepare data for OpenAlex step (simulate CSV input)
            # Extract only the DOIs that were found
            input_doi_list = [row[1] for row in data if row[1]]
            print(f"Found {len(input_doi_list)} DOIs. Querying OpenAlex...")
            
            # Fetch from API using the DOIs
            my_publications = self._fetch_works_by_doi_batch(input_doi_list)

        # Case B: Local CSV File
        elif source_type == 'csv':
            input_source = source_value
            if not os.path.exists(input_source):
                print(f"[Error] CSV file not found: {input_source}")
                return
            
            print(f"Reading CSV: {input_source}")
            df = pd.read_csv(input_source)
            
            # Find column named 'doi' or 'DOI' case-insensitive
            doi_col = next((c for c in df.columns if c.lower() == 'doi'), None)
            
            if not doi_col:
                print("[Error] CSV must contain a 'DOI' or 'doi' column.")
                return
                
            # Extract DOIs
            raw_doi_list = df[doi_col].astype(str).tolist()
            
            # Create a map for comparison later: { normalized_doi : original_input }
            input_doi_map = {}
            for d in raw_doi_list:
                if d.strip():
                     norm = self._normalize_doi_for_comparison(d)
                     input_doi_map[norm] = d.strip()

            print(f"Total unique DOIs provided in CSV: {len(input_doi_map)}")
            
            # Fetch from API
            my_publications = self._fetch_works_by_doi_batch(list(input_doi_map.values()))
            
            # --- Check which DOIs were NOT found ---
            found_dois_normalized = set()
            for pub in my_publications:
                pub_doi = pub.get('doi') # OpenAlex returns full URL usually
                if pub_doi:
                    found_dois_normalized.add(self._normalize_doi_for_comparison(pub_doi))
            
            missing_normalized = set(input_doi_map.keys()) - found_dois_normalized
            
            if missing_normalized:
                print(f"\n[Warning] {len(missing_normalized)} DOIs were NOT found in OpenAlex:")
                for missing in missing_normalized:
                    print(f" - {input_doi_map[missing]}")
                time.sleep(3) 
            else:
                print(f"All {len(input_doi_map)} provided DOIs were successfully found.")

        # Case C: OpenAlex Author ID
        elif source_type == 'openalex':
            author_id = source_value.strip()
            print(f"\nFetching author details for ID: {author_id}")
            
            author_url = f"{OPENALEX_API_URL}/authors/{author_id}"
            try:
                response = self.session.get(author_url)
                response.raise_for_status()
                author_data = response.json()
            
                print(f"Author found: {author_data.get('display_name', 'Unknown')}")
                
                works_api_url = author_data.get('works_api_url')
                if not works_api_url:
                    print(f"[Error] Could not find 'works_api_url' for author ID: {author_id}")
                    return

                print(f"Fetching works from: {works_api_url}")

                # --- Step 2: Get all publications for the author ---
                my_publications = self._get_paginated_results(works_api_url)
                
                # --- Save publication list to CSV for consistency ---
                if my_publications:
                    print(f"Found {len(my_publications)} works. Saving publication list...")
                    oa_csv_data = []
                    for pub in my_publications:
                        title = pub.get('title', 'Unknown Title')
                        # OpenAlex returns full URL (https://doi.org/...). 
                        # We strip it to look like a standard DOI for the CSV user.
                        raw_doi = pub.get('doi', '')
                        clean_doi = raw_doi.replace('https://doi.org/', '').replace('http://doi.org/', '') if raw_doi else ""
                        oa_csv_data.append([title, clean_doi, "OpenAlex"])
                    
                    try:
                        with open(pub_list_filename, 'w', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow(["my_publication", "DOI", "DOI Source"])
                            writer.writerows(oa_csv_data)
                        print(f"\nPublication list saved to {pub_list_filename}")
                    except IOError as e:
                        print(f"\n[Warning] Could not save publication list CSV: {e}\n")

            except Exception as e:
                print(f"[Error] fetching OpenAlex Author: {e}")
                return

        # ---------------------------------------------------------------------
        # PHASE 2: Fetch Citations (Common Logic)
        # ---------------------------------------------------------------------

        print(f"\nProcessing citations for {len(my_publications)} publications...")
        all_rows = []

        # --- Step 3: Iterate through each publication to get its citations ---
        for i, my_pub in enumerate(my_publications):
            my_pub_title = my_pub.get('title', 'N/A')
            cited_by_count = my_pub.get('cited_by_count', 0)
            
            print(f"\n--- Processing my publication {i+1}/{len(my_publications)} ---")
            print(f"Title: {my_pub_title}")
            print(f"Citations: {cited_by_count}")

            if cited_by_count == 0:
                print("No citations. Skipping.")
                continue

            # Construct URL using the work ID and filter
            my_pub_full_id = my_pub.get('id')
            if not my_pub_full_id:
                print("Warning: Publication ID missing. Skipping.")
                continue
                
            work_id = my_pub_full_id.split('/')[-1]
            target_url = f"{OPENALEX_API_URL}/works?filter=cites:{work_id}"

            print(f"Fetching citing works via: {target_url}")
            
            # Get all papers that cite this publication
            citing_papers = self._get_paginated_results(target_url)

            # --- Step 4: Extract details from each citing paper ---
            for citing_paper in citing_papers:
                # Internal key 'title'
                citing_title = citing_paper.get('title', 'N/A')
                authorships = citing_paper.get('authorships', [])
                
                # Helper to add row
                def add_row(author_name, inst_name, country_code):
                    all_rows.append({
                        'my_publication': my_pub_title,
                        'cited_by_title': citing_title,
                        'cited_by_author': author_name,
                        'cited_by_institution': inst_name,
                        'cited_by_country': country_code
                    })

                if not authorships:
                    add_row('N/A', 'N/A', 'N/A')
                    continue

                # Iterate through each author of the citing paper
                for authorship in authorships:
                    # Internal key 'author'
                    author_name = authorship.get('author', {}).get('display_name', 'N/A')
                    institutions = authorship.get('institutions', [])
                    
                    if not institutions:
                        add_row(author_name, 'N/A', 'N/A')
                        continue

                    # Iterate through each institution for the author
                    for inst in institutions:
                        add_row(author_name, inst.get('display_name', 'N/A'), inst.get('country_code', 'N/A'))

        # --- Step 5: Export all collected data to a CSV ---
        if not all_rows:
            print("No citation data found.")
            return
            
        
        
        # Create DataFrame using Pandas
        out_df = pd.DataFrame(all_rows)
        
        # Reorder columns to match the user's request
        cols = ['my_publication', 'cited_by_title', 'cited_by_author', 'cited_by_institution', 'cited_by_country']
        out_df = out_df[cols]
        
        try:
            # Use 'utf-8-sig' encoding to ensure Excel handles non-English characters correctly
            out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n[Success] Generated {len(out_df)} rows.")
            print(f"Citation info saved to: {output_csv}\n")
        except Exception as e:
            print(f"[Error] Saving CSV: {e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch academic citations.")
    
    # Mutually exclusive group for input source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--openalex_id", help="Your OpenAlex Author ID (e.g., A5023888391)")
    group.add_argument("--orcid", help="Your ORCID iD (e.g., 0000-0000-0000-0000)")
    group.add_argument("--scholar_id", help="Your Google Scholar ID")
    group.add_argument("--csv", help="Path to a CSV file containing a 'DOI' or 'doi' column")

    parser.add_argument("--output", default="citation_info.csv", help="Output CSV filename (default: citation_info.csv)")
    parser.add_argument("--email", help="Your email for API politeness (Recommended)")

    args = parser.parse_args()

    # Instantiate the fetcher
    fetcher = CitationFetcher(email=args.email)

    # Determine source type and value
    if args.openalex_id:
        fetcher.run(source_type='openalex', source_value=args.openalex_id, output_csv=args.output)
    elif args.orcid:
        fetcher.run(source_type='orcid', source_value=args.orcid, output_csv=args.output)
    elif args.scholar_id:
        fetcher.run(source_type='scholar', source_value=args.scholar_id, output_csv=args.output)
    elif args.csv:
        fetcher.run(source_type='csv', source_value=args.csv, output_csv=args.output)