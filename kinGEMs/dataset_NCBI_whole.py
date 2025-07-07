import pandas as pd
import logging
import urllib.error
import urllib.parse
import urllib.request
import os
import gzip
from bs4 import BeautifulSoup
from Bio import SeqIO
import time # Import the time module for time tracking

logger = logging.getLogger(__name__) # Initialize logger for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_fasta_whole(URL: str, target_filename: str, output_directory: str = "downloaded_fasta_files"):
    """
    Accesses a URL displaying a directory index, finds a specific file link,
    and downloads that file using urllib.

    Args:
        URL (str): The full URL of the directory listing page.
                             (e.g., "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/...")
        target_filename (str): The exact name of the file to find and download
                                from the directory listing.
                                (e.g., "GCF_000401375.1_HelPylUM034_1.0_protein.faa.gz")
        output_directory (str): The local directory path where the downloaded file will be saved.
                                (Defaults to "downloaded_fasta_files" in the script's directory)
    Returns:
        str: The full local path to the downloaded file if successful, None otherwise.
    """
    logger.info(f"Starting download process for '{target_filename}' from '{URL}'")

    # --- Part 2.1: Access the directory listing URL ---
    html_content = None
    try:
        logger.info(f"Attempting to fetch directory listing from: {URL}")
        with urllib.request.urlopen(URL, timeout=10) as response:
            html_content = response.read().decode('utf8')
        logger.info("Successfully fetched the directory listing.")
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error ({e.code}) when accessing directory '{URL}': {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.error(f"URL error when accessing directory '{URL}': {e.reason}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while accessing directory '{URL}': {e}")
        return None

    if not html_content:
        logger.error("No HTML content received. Exiting download process.")
        return None

    # --- Part 2.2: Parse the HTML content to find the specific download link ---
    logger.info(f"Parsing HTML content to find link for '{target_filename}'...")
    soup = BeautifulSoup(html_content, 'html.parser')
    download_link_href = None

    for link_tag in soup.find_all('a'):
        # Check if the exact filename is in the link's text or href (case-insensitive for robustness)
        if link_tag.get_text() == target_filename or (link_tag.get('href') and target_filename in link_tag.get('href')):
            download_link_href = link_tag.get('href')
            logger.info(f"Found link tag for '{target_filename}'. Href: {download_link_href}")
            break

    if not download_link_href:
        logger.error(f"Error: Could not find a link for '{target_filename}' on the page '{URL}'.")
        logger.info("Listing available files/directories on the page for debugging:")
        for link_tag in soup.find_all('a'):
            href = link_tag.get('href')
            text = link_tag.get_text().strip()
            if text and href and not text.startswith("Parent Directory"):
                logger.info(f"  - Found: '{text}' (Link fragment: '{href}')")
        return None

    # --- Part 2.3: Construct the full download URL ---
    full_download_url = urllib.parse.urljoin(URL, download_link_href)
    logger.info(f"Constructed full download URL: {full_download_url}")

    # --- Part 2.4: Download the file ---
    os.makedirs(output_directory, exist_ok=True)
    output_filepath = os.path.join(output_directory, target_filename)

    try:
        logger.info(f"Starting download of '{target_filename}' to '{output_filepath}'...")
        urllib.request.urlretrieve(full_download_url, output_filepath)
        logger.info(f"Successfully downloaded '{target_filename}' to '{output_filepath}'")
        return output_filepath
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error ({e.code}) downloading '{target_filename}' from '{full_download_url}': {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.error(f"URL error downloading '{target_filename}' from '{full_download_url}': {e.reason}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading '{target_filename}': {e}")
        return None


def get_fasta_from_NCBI_whole(database: str):
    """
    Function to retrieve protein sequences from NCBI or AGORA2 database for ALL entries,
    saving the parsed data incrementally to a CSV file to optimize memory usage.
    It also includes a check to skip already processed entries, making it resumable.

    Parameters:
    database (str): The database to use (currently only 'AGORA2' is handled).

    Returns:
    pd.DataFrame: DataFrame summarizing download/parsing errors
                  ('microbe_id', 'url', 'error_type' columns).
                  The main protein data is saved to 'all_proteins_output.csv'.
    """

    logger = logging.getLogger(__name__)

    if not isinstance(database, str):
        raise ValueError("The 'database' parameter must be a string.")

    if database != 'AGORA2':
        logger.warning(f"Database '{database}' is not supported. Only 'AGORA2' is currently implemented.")
        return pd.DataFrame() # Return empty DataFrame for unsupported database (only error_df will be returned)

    file_path = '/Users/niol/Desktop/AGORA2_data_organized.xlsx' # Path to your Excel file
    output_csv_path = "all_proteins_output.csv" # Define the path for the output CSV file

    df_ncbi = pd.DataFrame()
    df_other = pd.DataFrame()

    try:
        df_ncbi = pd.read_excel(file_path, sheet_name='NCBI')
        logger.info("Successfully loaded 'NCBI' sheet.")
    except FileNotFoundError:
        logger.error(f"Excel file not found at: {file_path}")
        return pd.DataFrame() # Return empty error_df as no data could be processed
    except Exception as e:
        logger.error(f"Error reading Excel file or sheet 'NCBI': {e}")
        # Continue to try loading 'other' sheet even if 'NCBI' fails

    try:
        df_other = pd.read_excel(file_path, sheet_name='other')
        logger.info("Successfully loaded 'other' sheet.")
    except Exception as e:
        logger.warning(f"Could not load 'other' sheet (may not exist or error reading): {e}. Proceeding with 'NCBI' data only if available.")

    # Concatenate the dataframes
    if df_ncbi.empty and df_other.empty:
        logger.warning("Both 'NCBI' and 'other' sheets are empty or failed to load. No data to process.")
        return pd.DataFrame() # Return empty error_df
    elif df_ncbi.empty:
        df_combined = df_other
        logger.info("Proceeding with data from 'other' sheet only.")
    elif df_other.empty:
        df_combined = df_ncbi
        logger.info("Proceeding with data from 'NCBI' sheet only.")
    else:
        # Ensure consistent columns before concatenation if they differ in order or presence
        common_cols = list(set(df_ncbi.columns) & set(df_other.columns))
        df_combined = pd.concat([df_ncbi[common_cols], df_other[common_cols]], ignore_index=True)
        logger.info(f"Combined data from 'NCBI' ({len(df_ncbi)} rows) and 'other' ({len(df_other)} rows) sheets, resulting in {len(df_combined)} total entries.")

    # Initialize error records
    error_records = []

    total_items = len(df_combined)
    start_time_total = time.time()
    processed_count = 0

    # Load existing processed microbe IDs from the output CSV for resumability
    processed_microbe_ids = set()
    if os.path.exists(output_csv_path):
        try:
            # Read only the 'microbe_id' column to save memory
            existing_data = pd.read_csv(output_csv_path, usecols=['microbe_id'])
            processed_microbe_ids = set(existing_data['microbe_id'].unique())
            logger.info(f"Found {len(processed_microbe_ids)} previously processed microbe IDs in '{output_csv_path}'.")
        except Exception as e:
            logger.warning(f"Could not read existing '{output_csv_path}' for resumability: {e}. Starting fresh.")
            # If reading fails, it implies the file might be corrupt or malformed, so we start fresh.
            if os.path.exists(output_csv_path):
                os.remove(output_csv_path)
                logger.info(f"Removed existing output file due to read error: {output_csv_path}")

    # Write header to output CSV if it's a new file
    if not os.path.exists(output_csv_path):
        # Create an empty DataFrame with the expected columns and write header
        pd.DataFrame(columns=['microbe_id', 'gene', 'protein_sequence']).to_csv(output_csv_path, mode='w', header=True, index=False)
        logger.info(f"Created new output CSV file with header: {output_csv_path}")


    # Iterate through each row in the combined DataFrame to process all entries
    for index, row in df_combined.iterrows():
        item_start_time = time.time() # Start time for current item
        
        microbe_id = row.get('MicrobeID', f"UnknownMicrobe_{index}") # Use .get() for robustness
        genome_link = row.get('Genome link')

        # Check if this microbe_id has already been processed
        if microbe_id in processed_microbe_ids:
            logger.info(f"Skipping already processed item: MicrobeID '{microbe_id}' ({processed_count + 1}/{total_items})")
            processed_count += 1 # Increment processed count even for skipped items for accurate progress
            continue # Skip to the next microbe

        processed_count += 1 # Increment only for newly processed items

        logger.info(f"Processing item {processed_count}/{total_items}: MicrobeID '{microbe_id}'")

        if not isinstance(genome_link, str) or not genome_link.startswith(('http://', 'https://')):
            logger.error(f"Skipping row {index} for MicrobeID '{microbe_id}': Invalid or missing 'Genome link': {genome_link}")
            error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': 'Invalid URL'})
            continue

        # Ensure the URL ends with a slash if it's a directory
        if not genome_link.endswith('/'):
            genome_link += '/'

        extracted_organism_id = None
        try:
            parsed_url_path = urllib.parse.urlparse(genome_link).path
            # Split by '/', remove empty strings from split, and get the last non-empty part
            path_parts = [part for part in parsed_url_path.strip('/').split('/') if part]
            if path_parts:
                extracted_organism_id = path_parts[-1]
            else:
                logger.error(f"Could not extract organism ID from URL: {genome_link}. Path is empty.")
                error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': 'Cannot extract organism ID'})
                continue
            target_filename = f"{extracted_organism_id}_protein.faa.gz"
            logger.info(f"Constructed target filename for {microbe_id}: {target_filename}")
        except Exception as e:
            logger.error(f"Error processing URL '{genome_link}' for filename extraction for '{microbe_id}': {e}")
            error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': f'Filename extraction error: {e}'})
            continue

        # Calling the renamed download_fasta_whole function
        output_dir_for_microbe = os.path.join("downloaded_fasta_files", microbe_id) # Create a sub-directory for each microbe
        downloaded_fasta_path = download_fasta_whole(genome_link, target_filename, output_dir_for_microbe)

        if downloaded_fasta_path:
            logger.info(f"Successfully downloaded FASTA file for {microbe_id}: {downloaded_fasta_path}")
            try:
                if not os.path.exists(downloaded_fasta_path):
                    logger.error(f"Downloaded file not found at expected path: {downloaded_fasta_path} for {microbe_id}")
                    error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': 'Download path failure (file not found after download)'})
                    # Skip parsing and continue to next microbe
                    continue

                # Temporary lists for current file's data
                current_file_data = []
                with gzip.open(downloaded_fasta_path, "rt") as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        current_file_data.append({
                            'microbe_id': microbe_id,
                            'gene': record.id,
                            'protein_sequence': str(record.seq)
                        })

                # Create a small DataFrame for the current file's data
                temp_df = pd.DataFrame(current_file_data)

                # Append to the output CSV file instead of accumulating in memory
                # Write header only if the file does not exist (first write)
                temp_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
                logger.info(f"Appended {len(temp_df)} sequences for {microbe_id} to {output_csv_path}")
                # Add the microbe ID to the set of processed IDs
                processed_microbe_ids.add(microbe_id)

            except FileNotFoundError:
                logger.error(f"File not found during FASTA parsing: {downloaded_fasta_path} for {microbe_id}")
                error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': 'File not found during parsing'})
            except Exception as e:
                logger.error(f"Error parsing FASTA file '{downloaded_fasta_path}' for {microbe_id}: {e}")
                error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': f'Error parsing FASTA: {e}'})
        else:
            logger.error(f"Failed to download the FASTA file for {microbe_id} from {genome_link}. Cannot parse sequences.")
            error_records.append({'microbe_id': microbe_id, 'url': genome_link, 'error_type': 'Download failure'})

        # Calculate and display estimated time remaining
        # Ensure we don't divide by zero if processed_count is 0 (should not happen after first item)
        if processed_count > 0:
            avg_time_per_item = (time.time() - start_time_total) / processed_count
            items_remaining = total_items - processed_count
            estimated_time_remaining = avg_time_per_item * items_remaining

            m, s = divmod(estimated_time_remaining, 60)
            h, m = divmod(m, 60)
            logger.info(f"Progress: {processed_count}/{total_items} items processed. Estimated time remaining: {int(h)}h {int(m)}m {int(s)}s")
        else:
            logger.info(f"Progress: {processed_count}/{total_items} items processed. Estimating time after first successful item.")


    # Create error_df from collected error records
    error_df = pd.DataFrame(error_records)

    logger.info(f"Finished processing all entries. All parsed protein sequences have been saved to '{output_csv_path}'.")
    logger.info(f"Encountered {len(error_df)} errors during the process.")

    return error_df # Now only returns the error DataFrame
