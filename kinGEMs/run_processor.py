# --- TEMPORARY DEBUGGING PRINTS AND LOGGING SETUP ---
# These lines are for immediate feedback when you start the script.
# They can be removed once the script is consistently running as expected.
print("--- Script started. Checking logging setup. ---")
import logging
# Configure logging at the beginning of the run script
# This ensures that all log messages from dataset_NCBI_whole.py are captured and displayed here.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging configured. Starting main process.")
# --- END TEMPORARY DEBUGGING PRINTS ---

import pandas as pd
import os
import sys # Import sys to add current directory to path if needed

# Add the directory containing your dataset_NCBI_whole.py to the Python path
# This helps with imports when running the script directly.
# Assuming this run script is in /Users/niol/kinGEMs/kinGEMs/, and dataset_NCBI_whole.py is also there.
# If dataset_NCBI_whole.py is inside a 'module_folder' within kinGEMs/kinGEMs, adjust accordingly.
script_dir = os.path.dirname(os.path.abspath(__file__))
# If dataset_NCBI_whole.py is directly in the same folder as this run script:
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now, import your functions from dataset_NCBI_whole.py
# If dataset_NCBI_whole.py is directly in 'kinGEMs/kinGEMs', you import it like this:
try:
    from dataset_NCBI_whole import download_fasta_whole, get_fasta_from_NCBI_whole
    logging.info("Successfully imported functions from dataset_NCBI_whole.py")
except ImportError as e:
    logging.error(f"Failed to import functions from dataset_NCBI_whole.py: {e}")
    logging.error("Please ensure 'dataset_NCBI_whole.py' is in the same directory as this run script,")
    logging.error("or adjust the import path (e.g., 'from your_module_folder.dataset_NCBI_whole import ...').")
    sys.exit(1) # Exit if essential functions can't be imported

# --- Main Execution Logic ---
logging.info("Calling get_fasta_from_NCBI_whole function...")

# Call the main function. It will save the protein sequences directly to
# multiple 'all_proteins_output_part_X.csv' files in the script's execution directory.
# It only returns the error_df.
# You can adjust num_output_files and sleep_time_seconds here if desired,
# e.g., get_fasta_from_NCBI_whole('AGORA2', num_output_files=4, sleep_time_seconds=1.0)
error_df = get_fasta_from_NCBI_whole('AGORA2')

# Define where you want to save the error log Excel file
# This will save the error log to your Desktop.
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
error_file_name = "protein_download_errors.xlsx"
file_path_error = os.path.join(desktop_path, error_file_name)

# Save the error DataFrame to Excel
try:
    error_df.to_excel(file_path_error, index=False)
    logging.info(f"Errors saved to {file_path_error}")
except Exception as e:
    logging.error(f"Failed to save error_df to Excel: {e}")

# Confirm where the main protein data is saved
logging.info(f"Main protein sequence data saved incrementally to 'all_proteins_output_part_X.csv' files ")
logging.info(f"These files are located in the script's execution directory: {script_dir}")

logging.info("Script finished execution.")
