import pandas as pd
import logging
import urllib.error
import urllib.parse
import urllib.request
import os
import gzip
import time
from bs4 import BeautifulSoup
from Bio import SeqIO 

logger = logging.getLogger(__name__) # Initialize logger for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_fasta(URL: str, target_filename: str, output_directory: str = "downloaded_fasta_files"):
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
        if link_tag.get_text() == target_filename:
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


def get_fasta_from_NCBI(name, database): 
    """
    Function to retrieve protein sequences from NCBI or AGORA2 database.
    Parameters:
    name (str): Name of the protein (MicrobeID).
    database (str): The database to use (currently only 'AGORA2' is handled).
    Returns:
    pd.DataFrame: DataFrame containing protein sequences ('gene' and 'protein_sequence' columns).
                  Returns an empty DataFrame if no data or download/parsing fails.
    """

    logger = logging.getLogger(__name__) 

    if not isinstance(name, str):
        raise ValueError("The 'name' parameter must be a string.")
    if not isinstance(database, str):
        raise ValueError("The 'database' parameter must be a string.")
    
    if database != 'AGORA2':
        logger.warning(f"Database '{database}' is not supported. Only 'AGORA2' is currently implemented.")
        return pd.DataFrame() # Return empty DataFrame for unsupported database
    
    elif database == 'AGORA2': 
        file_path = '/Users/niol/Desktop/AGORA2_data_organized.xlsx' # Path to your Excel file
        
        try:
            df_ncbi = pd.read_excel(file_path, sheet_name='NCBI') # MODIFIED: Renamed df to df_ncbi for clarity
        except FileNotFoundError:
            logger.error(f"Excel file not found at: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading Excel file or sheet 'NCBI': {e}")
            return pd.DataFrame()

        search_df = df_ncbi[df_ncbi['MicrobeID'].str.contains(name, case=False, na=False)]
        url_list = []
        if len(search_df) == 0:
            logger.error(f"No data found for MicrobeID: {name} in the 'NCBI' sheet.")
            return pd.DataFrame() # MODIFIED: Return empty DataFrame instead of raising ValueError
        else: 
            # 'url' here is the base_dir_url that needs to be passed to download_fasta
            for link in search_df['Genome link']:
                url = search_df['Genome link'].iloc[0] 
                if not url.endswith('/'):
                    url += '/'
                url_list.append(url)
                logger.info(f"URL retrieved for the organism '{name}': {url}")
    gene_ids = []
    protein_sequences = []
    download_path_failure = []
    file_not_found = []
    error_parsing = []
    
    try:
        parsed_url_path = urllib.parse.urlparse(url).path
        # Split by '/', remove empty strings from split, and get the last part
        extracted_organism_id = [part for part in parsed_url_path.strip('/').split('/') if part][-1]
        target_filename = f"{extracted_organism_id}_protein.faa.gz"
        logger.info(f"Constructed target filename: {target_filename}")
    except IndexError:
        logger.error(f"Could not extract organism ID from URL: {url}. URL format might be unexpected.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing URL '{url}' for filename extraction: {e}")
        return pd.DataFrame()
    
    # Loop through each URL in the list and download the corresponding FASTA file
    for url in url_list: 
        downloaded_fasta_path = download_fasta(url, target_filename, f"{name}_fasta_files")

    # Logic to parse the downloaded FASTA file and create a DataFrame
        if downloaded_fasta_path:
            logger.info(f"Successfully downloaded FASTA file: {downloaded_fasta_path}")
            try:
                # Check if the downloaded file exists before attempting to open
                if not os.path.exists(downloaded_fasta_path):
                    logger.error(f"Downloaded file not found at expected path: {downloaded_fasta_path}")
                    download_path_failure.append(url)

                # Open the gzipped FASTA file in read text mode ('rt') for Biopython
                with gzip.open(downloaded_fasta_path, "rt") as handle:
                    # Use Biopython's SeqIO.parse to read each FASTA record
                    for record in SeqIO.parse(handle, "fasta"):
                        gene_ids.append(record.id)          # Extracts the ID (e.g., WP_003792170.1)
                        protein_sequences.append(str(record.seq)) # Extracts the sequence as a string
            except FileNotFoundError: # Catches if gzip.open somehow still can't find it
                logger.error(f"File not found during FASTA parsing: {downloaded_fasta_path}")
                file_not_found.append(url)
            except Exception as e:
                logger.error(f"Error parsing FASTA file '{downloaded_fasta_path}': {e}")
                error_parsing.append(url)
        else:
            logger.error("Failed to download the FASTA file. Cannot parse sequences.")
            result = 'Download failure' # Return an empty DataFrame if download failed

    # Create a pandas DataFrame from the extracted data
    protein_df = pd.DataFrame({
        'gene': gene_ids,
        'protein_sequence': protein_sequences
    })
    error_df = pd.DataFrame({
        'Download path failure': download_path_failure,
        'File not found': file_not_found,
        'Error parsing': error_parsing
    })
    # Log the number of errors encountered
    print (f"# of Download path failure: {len(download_path_failure)}")
    print (f"# of File not found: {len(file_not_found)}")
    print (f"# of Error parsing: {len(error_parsing)}")
    logger.info(f"Successfully parsed {len(protein_df)} protein sequences into a DataFrame.")
    return protein_df, error_df # Return the DataFrame with sequences

def handle_merging(merged_data_df: pd.DataFrame, sequences_folder_path: str) -> pd.DataFrame:
    print(f"Starting merging process. Loading sequences from: {sequences_folder_path}")
    start_time = time.time() # Start global timer

    if 'Single_gene' not in merged_data_df.columns:
        print("Error: 'merged_data_df' must contain a 'Single_gene' column. 'SEQ' column will be left blank.")
        merged_data_df['SEQ'] = '' # Initialize SEQ column as blank
        return merged_data_df # Return early as merging based on this column is impossible

    all_sequences_lookup_data = []
    
    # Count total Excel files to be processed for time estimation
    excel_files_to_process = [f for f in os.listdir(sequences_folder_path) if f.endswith(".xlsx") and not f.startswith("~")]
    total_excel_files = len(excel_files_to_process)
    files_processed_count = 0

    for filename in excel_files_to_process: # Iterate only over identified excel files
        excel_file_path = os.path.join(sequences_folder_path, filename)
        print(f"Processing Excel file: {filename}")

        try:
            xls = pd.ExcelFile(excel_file_path)
            for sheet_name in xls.sheet_names:
                df_sheet = pd.read_excel(xls, sheet_name=sheet_name,
                                         usecols=['gene', 'protein_sequence'],
                                         dtype={'gene': str, 'protein_sequence': str})
                all_sequences_lookup_data.append(df_sheet)
                print(f"  Loaded {len(df_sheet)} rows from sheet '{sheet_name}'.")
            files_processed_count += 1
        except Exception as e:
            print(f"Error loading Excel file '{filename}' or its sheets: {e}")
            continue

        # Estimate time remaining after each file
        elapsed_time = time.time() - start_time
        if files_processed_count > 0:
            avg_time_per_file = elapsed_time / files_processed_count
            files_remaining = total_excel_files - files_processed_count
            estimated_time_remaining = avg_time_per_file * files_remaining

            m, s = divmod(estimated_time_remaining, 60)
            h, m = divmod(m, 60)
            print(f"Progress: {files_processed_count}/{total_excel_files} Excel files loaded. Estimated time remaining: {int(h)}h {int(m)}m {int(s)}s")

    if not all_sequences_lookup_data:
        print("Warning: No sequence data found in the specified folder. 'SEQ' column will be empty.")
        merged_data_df['SEQ'] = '' # Initialize SEQ column as blank


    all_sequences_lookup_df = pd.concat(all_sequences_lookup_data, ignore_index=True)
    
    print(f"Initial rows in sequence lookup table: {len(all_sequences_lookup_df)}")
    all_sequences_lookup_df.drop_duplicates(subset=['gene'], inplace=True)
    print(f"Total unique gene-sequence pairs loaded for lookup: {len(all_sequences_lookup_df)}")

    merged_data_df = merged_data_df.merge(
        all_sequences_lookup_df[['gene', 'protein_sequence']],
        left_on='Single_gene',
        right_on='gene',
        how='left'
    )

    merged_data_df.rename(columns={'protein_sequence': 'SEQ'}, inplace=True)
    
    if 'gene' in merged_data_df.columns:
        merged_data_df.drop(columns=['gene'], inplace=True)

    end_time = time.time()
    total_duration = end_time - start_time
    m, s = divmod(total_duration, 60)
    h, m = divmod(m, 60)
    print(f"Merging complete. 'SEQ' column added to 'merged_data_df'. Total processing time: {int(h)}h {int(m)}m {int(s)}s")
    return merged_data_df

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function: Download FASTQ File ---
def download_fastq_file(url: str, output_filepath: str) -> str:
    """
    Downloads a FASTQ.gz file directly from a given URL.

    Args:
        url (str): The direct URL of the FASTQ.gz file.
        output_filepath (str): The local path where the downloaded file will be saved.

    Returns:
        str: The full local path to the downloaded file if successful, None otherwise.
    """
    output_directory = os.path.dirname(output_filepath)
    os.makedirs(output_directory, exist_ok=True) # Ensure output directory exists

    if os.path.exists(output_filepath):
        logger.info(f"File already exists: {output_filepath}. Skipping download.")
        return output_filepath

    try:
        logger.info(f"Starting download of '{os.path.basename(output_filepath)}' from '{url}'...")
        urllib.request.urlretrieve(url, output_filepath)
        logger.info(f"Successfully downloaded '{os.path.basename(output_filepath)}' to '{output_filepath}'")
        return output_filepath
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error ({e.code}) downloading from '{url}': {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.error(f"URL error downloading from '{url}': {e.reason}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading '{url}': {e}")
        return None

# --- Main Function: Process EBI FASTQ Data ---
def process_EBI_fastq_data(names: list[str], excel_config_path: str, agora2_db_base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes EBI data to download FASTQ files, parse DNA/RNA sequencing reads,
    and save them to CSV files for multiple MicrobeIDs.

    IMPORTANT: This function downloads and parses raw DNA/RNA sequencing reads.
    It does NOT produce protein sequences. If protein sequences are desired,
    a separate, complex bioinformatics pipeline (assembly, gene prediction, translation)
    is required.

    Args:
        names (list[str]): A list of MicrobeIDs or organism names to search for in the 'EBI' sheet.
        excel_config_path (str): Path to the Excel file containing the 'EBI' sheet
                                 (e.g., '/Users/niol/Desktop/AGORA2_data_organized.xlsx').
        agora2_db_base_path (str): The base path to your 'AGORA2_Database' folder.
                                   Downloaded FASTQ files will be stored in
                                   'AGORA2_Database/fastq_downloads/<MicrobeID>/'.
                                   Parsed reads will be saved to
                                   'AGORA2_Database/parsed_fastq_reads/reads_<MicrobeID>.csv'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: A DataFrame summarizing the FASTQ download status for all processed MicrobeIDs.
              Columns: 'MicrobeID', 'Fastq_URL_1', 'Fastq_URL_2', 'Local_Path_1', 'Local_Path_2', 'Download_Status_1', 'Download_Status_2'.
            - pd.DataFrame: A DataFrame summarizing any parsing errors encountered for all processed MicrobeIDs.
              Columns: 'MicrobeID', 'Fastq_File', 'Error_Type', 'Error_Message'.
    """
    logger.info(f"Starting EBI FASTQ data processing for {len(names)} MicrobeIDs.")
    start_time_total = time.time() # Global timer for the entire list of names

    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("The 'names' parameter must be a list of strings.")
    if not isinstance(excel_config_path, str) or not os.path.exists(excel_config_path):
        logger.error(f"Excel config file not found or invalid path: {excel_config_path}")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    df_ebi = pd.DataFrame()
    try:
        df_ebi = pd.read_excel(excel_config_path, sheet_name='EBI')
        logger.info(f"Successfully loaded 'EBI' sheet from {excel_config_path}.")
    except FileNotFoundError:
        logger.error(f"Excel config file not found at: {excel_config_path}")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs
    except Exception as e:
        logger.error(f"Error reading Excel file or sheet 'EBI': {e}")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    # Prepare lists to store aggregated download and parsing summary for all microbe IDs
    all_download_summary_data = []
    all_parsing_error_data = []

    total_names_to_process = len(names)
    names_processed_count = 0

    for name in names: # Loop through the list of microbe IDs
        logger.info(f"\nProcessing MicrobeID: '{name}' ({names_processed_count + 1}/{total_names_to_process})")
        
        search_df = df_ebi[df_ebi['MicrobeID'].str.contains(name, case=False, na=False)]

        if len(search_df) == 0:
            logger.warning(f"No data found for MicrobeID: '{name}' in the 'EBI' sheet. Skipping.")
            names_processed_count += 1
            # Update time estimate even for skipped items
            elapsed_time_so_far = time.time() - start_time_total
            if names_processed_count > 0:
                avg_time_per_name = elapsed_time_so_far / names_processed_count
                names_remaining = total_names_to_process - names_processed_count
                estimated_time_remaining = avg_time_per_name * names_remaining
                m, s = divmod(estimated_time_remaining, 60)
                h, m = divmod(m, 60)
                logger.info(f"Overall Progress: {names_processed_count}/{total_names_to_process} MicrobeIDs processed. Estimated total time remaining: {int(h)}h {int(m)}m {int(s)}s")
            continue

        # Prepare lists for current microbe ID's summary
        current_microbe_download_summary = []
        current_microbe_parsing_errors = []

        # Iterate through each matching entry (assumes one MicrobeID entry per row in config Excel)
        for index, row in search_df.iterrows():
            microbe_id = row.get('MicrobeID', f"UnknownMicrobe_{index}") # Using the name from the input list
            genome_link_str = row.get('genome_link') # Assuming the column is named 'genome_link'

            fastq_url1, fastq_url2 = None, None
            local_path1, local_path2 = None, None
            download_status1, download_status2 = 'NOT_ATTEMPTED', 'NOT_ATTEMPTED'
            parsed_reads_csv_path = None

            if not isinstance(genome_link_str, str) or ';' not in genome_link_str:
                logger.error(f"Skipping MicrobeID '{microbe_id}': Invalid or missing 'genome_link' format (expected two URLs separated by ';'). Link: {genome_link_str}")
                download_status1 = download_status2 = 'Invalid Link Format'
            else:
                urls = [u.strip() for u in genome_link_str.split(';') if u.strip()]
                if len(urls) != 2:
                    logger.error(f"Skipping MicrobeID '{microbe_id}': Expected exactly two URLs in 'genome_link'. Found {len(urls)}. Link: {genome_link_str}")
                    download_status1 = download_status2 = 'Link Count Mismatch'
                    fastq_url1 = urls[0] if len(urls) > 0 else None
                    fastq_url2 = urls[1] if len(urls) > 1 else None
                else:
                    fastq_url1, fastq_url2 = urls[0], urls[1]

                    # Define output directory for FASTQ files within AGORA2_Database
                    # Creates a structure like AGORA2_Database/fastq_downloads/MicrobeID/
                    fastq_output_dir = os.path.join(agora2_db_base_path, "fastq_downloads", microbe_id)
                    
                    # Determine local file paths
                    local_path1 = os.path.join(fastq_output_dir, os.path.basename(fastq_url1))
                    local_path2 = os.path.join(fastq_output_dir, os.path.basename(fastq_url2))

                    # Download the FASTQ files
                    downloaded_path1 = download_fastq_file(fastq_url1, local_path1)
                    downloaded_path2 = download_fastq_file(fastq_url2, local_path2)

                    download_status1 = 'SUCCESS' if downloaded_path1 else 'FAILED'
                    download_status2 = 'SUCCESS' if downloaded_path2 else 'FAILED'

                    # --- NEW: Parse FASTQ files and save reads to CSV ---
                    if downloaded_path1 and downloaded_path2:
                        parsed_reads_output_dir = os.path.join(agora2_db_base_path, "parsed_fastq_reads")
                        os.makedirs(parsed_reads_output_dir, exist_ok=True)
                        parsed_reads_csv_path = os.path.join(parsed_reads_output_dir, f"reads_{microbe_id}.csv")
                        
                        reads_data = []
                        
                        for fastq_file_to_parse in [downloaded_path1, downloaded_path2]:
                            try:
                                logger.info(f"Parsing reads from '{os.path.basename(fastq_file_to_parse)}' for '{microbe_id}'...")
                                with gzip.open(fastq_file_to_parse, "rt") as handle:
                                    for record in SeqIO.parse(handle, "fastq"):
                                        reads_data.append({
                                            'MicrobeID': microbe_id,
                                            'Read_ID': record.id,
                                            'Sequence': str(record.seq)
                                            # Quality scores can also be added: 'Quality_Scores': record.qual
                                        })
                                logger.info(f"Successfully parsed {len(reads_data)} reads from '{os.path.basename(fastq_file_to_parse)}'.")
                            except FileNotFoundError:
                                logger.error(f"FASTQ file not found during parsing: {fastq_file_to_parse}")
                                current_microbe_parsing_errors.append({ # Append to current microbe's errors
                                    'MicrobeID': microbe_id, 'Fastq_File': os.path.basename(fastq_file_to_parse),
                                    'Error_Type': 'FileNotFound', 'Error_Message': 'File not found during parsing.'
                                })
                            except Exception as e:
                                logger.error(f"Error parsing FASTQ file '{fastq_file_to_parse}' for '{microbe_id}': {e}")
                                current_microbe_parsing_errors.append({ # Append to current microbe's errors
                                    'MicrobeID': microbe_id, 'Fastq_File': os.path.basename(fastq_file_to_parse),
                                    'Error_Type': 'ParsingError', 'Error_Message': str(e)
                                })
                        
                        if reads_data:
                            try:
                                reads_df = pd.DataFrame(reads_data)
                                # Append to CSV if file exists, otherwise write with header
                                mode = 'a' if os.path.exists(parsed_reads_csv_path) else 'w'
                                header = not os.path.exists(parsed_reads_csv_path)
                                reads_df.to_csv(parsed_reads_csv_path, mode=mode, header=header, index=False)
                                logger.info(f"Appended {len(reads_df)} reads for '{microbe_id}' to '{parsed_reads_csv_path}'.")
                            except Exception as e:
                                logger.error(f"Error saving parsed reads for '{microbe_id}' to CSV: {e}")
                                current_microbe_parsing_errors.append({ # Append to current microbe's errors
                                    'MicrobeID': microbe_id, 'Fastq_File': 'N/A',
                                    'Error_Type': 'SaveToCSVError', 'Error_Message': str(e)
                                })
                        else:
                            logger.warning(f"No reads parsed from FASTQ files for '{microbe_id}'. Skipping saving reads CSV.")
                    else: # This 'else' aligns with 'if downloaded_path1 and downloaded_path2'
                        logger.warning(f"FASTQ files not downloaded successfully for '{microbe_id}'. Skipping parsing.")

            # Add download summary for the current microbe_id to the current microbe's list
            current_microbe_download_summary.append({
                'MicrobeID': microbe_id,
                'Fastq_URL_1': fastq_url1,
                'Fastq_URL_2': fastq_url2,
                'Local_Path_1': local_path1,
                'Local_Path_2': local_path2,
                'Download_Status_1': download_status1,
                'Download_Status_2': download_status2,
                'Parsed_Reads_CSV': parsed_reads_csv_path if parsed_reads_csv_path and os.path.exists(parsed_reads_csv_path) else 'NOT_SAVED'
            })
        
        # Extend the global lists with data from the current microbe ID
        all_download_summary_data.extend(current_microbe_download_summary)
        all_parsing_error_data.extend(current_microbe_parsing_errors)

        names_processed_count += 1
        elapsed_time_so_far = time.time() - start_time_total
        if names_processed_count > 0:
            avg_time_per_name = elapsed_time_so_far / names_processed_count
            names_remaining = total_names_to_process - names_processed_count
            estimated_time_remaining = avg_time_per_name * names_remaining
            m, s = divmod(estimated_time_remaining, 60)
            h, m = divmod(m, 60)
            logger.info(f"Overall Progress: {names_processed_count}/{total_names_to_process} MicrobeIDs processed. Estimated total time remaining: {int(h)}h {int(m)}m {int(s)}s")


    # Convert collected data into DataFrames
    final_download_summary_df = pd.DataFrame(all_download_summary_data)
    final_parsing_error_df = pd.DataFrame(all_parsing_error_data)
    
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    m, s = divmod(total_duration, 60)
    h, m = divmod(m, 60)
    logger.info(f"EBI FASTQ processing complete for all {total_names_to_process} MicrobeIDs. Total duration: {int(h)}h {int(m)}m {int(s)}s")

    return final_download_summary_df, final_parsing_error_df # Now returns two aggregated DataFrames

# --- Main Execution Logic to run for all MicrobeIDs in EBI sheet ---
if __name__ == "__main__":
    logger.info("--- Script started: Running EBI FASTQ Processor for all MicrobeIDs ---")

    # 1. Define your Excel config file path (where the 'EBI' sheet is)
    # This path assumes the Excel file is inside "AGORA2 paper database" on your Desktop.
    excel_config_file = os.path.join(os.path.expanduser("~"), "Desktop", "AGORA2 paper database", "AGORA2_data_organized.xlsx")

    # 2. Define the base path to your AGORA2_Database folder.
    # This is where downloaded FASTQ files and parsed reads CSVs will be saved.
    # This assumes AGORA2_Database is directly on your Desktop.
    agora2_db_root_path = os.path.join(os.path.expanduser("~"), "Desktop", "AGORA2_Database")

    # --- Extract all MicrobeIDs from the EBI sheet ---
    microbe_ids_to_process = []
    try:
        # Load only the 'MicrobeID' column from the 'EBI' sheet for efficiency
        ebi_df = pd.read_excel(excel_config_file, sheet_name='EBI', usecols=['MicrobeID'])
        # Get unique MicrobeIDs and convert to a list, dropping any NaN values
        microbe_ids_to_process = ebi_df['MicrobeID'].dropna().unique().tolist()
        logger.info(f"Successfully loaded {len(microbe_ids_to_process)} unique MicrobeIDs from '{excel_config_file}' 'EBI' sheet.")
    except FileNotFoundError:
        logger.error(f"Excel config file not found at: {excel_config_file}. Cannot load MicrobeIDs.")
    except KeyError:
        logger.error(f"Column 'MicrobeID' not found in the 'EBI' sheet of {excel_config_file}.")
    except Exception as e:
        logger.error(f"Could not load MicrobeIDs from EBI sheet in {excel_config_file}: {e}")

    if not microbe_ids_to_process:
        logger.error("No MicrobeIDs to process. Exiting.")
    else:
        # --- Run the process_EBI_fastq_data function for all MicrobeIDs ---
        logger.info("Starting batch processing of EBI FASTQ data for all MicrobeIDs...")
        download_results_df, parsing_errors_df = process_EBI_fastq_data(
            names=microbe_ids_to_process,
            excel_config_path=excel_config_file,
            agora2_db_base_path=agora2_db_root_path
        )

        # --- Save Results (Optional, but recommended for large outputs) ---
        results_output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EBI_Processing_Results")
        os.makedirs(results_output_dir, exist_ok=True)

        download_summary_file = os.path.join(results_output_dir, "ebi_fastq_download_summary.csv")
        parsing_errors_file = os.path.join(results_output_dir, "ebi_fastq_parsing_errors.csv")

        try:
            download_results_df.to_csv(download_summary_file, index=False)
            logger.info(f"Download summary saved to: {download_summary_file}")
        except Exception as e:
            logger.error(f"Failed to save download summary to CSV: {e}")

        try:
            parsing_errors_df.to_csv(parsing_errors_file, index=False)
            logger.info(f"Parsing errors saved to: {parsing_errors_file}")
        except Exception as e:
            logger.error(f"Failed to save parsing errors to CSV: {e}")

        logger.info("Batch processing finished. Check the output files for details.")

        # You can also print the head of the DataFrames for quick inspection
        print("\n--- Aggregated EBI Download Results (first 5 rows) ---")
        print(download_results_df.head())
        print("\n--- Aggregated FASTQ Parsing Errors (first 5 rows) ---")
        print(parsing_errors_df.head())
        print(f"\nTotal download summary entries: {len(download_results_df)}")
        print(f"Total parsing error entries: {len(parsing_errors_df)}")
