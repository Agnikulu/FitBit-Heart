import os
import pickle
import pandas as pd
import synapseclient
from dotenv import load_dotenv

from src.data.importer import Importer
from src.config.synapse import SYNAPSE_TABLES, TABLES_EMBEDDED_DATA_COLUMN


class Downloader:
    """
    Class that downloads data from Synapse using their Python client.

    Attributes:
        syn (synapseclient.Synapse): The Synapse instance.
    """

    def __init__(self, user_email: str, user_token: str):
        """
        Initializes the Downloader with Synapse login.

        Args:
            user_email (str): Synapse user email.
            user_token (str): Synapse authentication token.
        """
        self.syn = synapseclient.Synapse()
        self.syn.login(email=user_email, authToken=user_token)

    def download_synapse_table_as_df(self, entity_id: str) -> pd.DataFrame:
        """
        Downloads a Synapse table as a DataFrame.

        Args:
            entity_id (str): The Synapse entity ID.

        Returns:
            pd.DataFrame: The table's data as a DataFrame.
        """
        query = f"SELECT * FROM {entity_id}"
        entity = self.syn.tableQuery(query)
        return entity.asDataFrame()

    def download_embedded_data_files(self, embedded_column: str, entity_id: str) -> dict:
        """
        Downloads all embedded files for a Synapse table.

        The files are saved to the default Synapse cache directory.

        Args:
            embedded_column (str): The column that holds embedded files.
            entity_id (str): The Synapse entity ID.

        Returns:
            dict: A dictionary with file IDs as keys and file paths as values.
        """
        self.syn.table_query_timeout = 9999999999999999
        query = f"SELECT * FROM {entity_id}"
        query_results = self.syn.tableQuery(query)
        return self.syn.downloadTableColumns(query_results, [embedded_column])


if __name__ == '__main__':
    load_dotenv()

    # Fetch credentials from environment variables
    user_email = os.getenv('SYNAPSE_EMAIL')
    user_token = os.getenv('SYNAPSE_TOKEN')

    # Initialize Downloader and Importer
    downloader = Downloader(user_email, user_token)
    importer = Importer()

    # Create separate directories
    raw_dir = os.path.join('data', 'raw')
    embedded_dir = os.path.join(raw_dir, 'embedded')
    os.makedirs(raw_dir, exist_ok=True) 
    os.makedirs(embedded_dir, exist_ok=True) 

    # Download and import tables
    for table, entity_id in SYNAPSE_TABLES.items():
        df = downloader.download_synapse_table_as_df(entity_id)
        importer.import_syn_table(dataframe=df, collection_name=table)

    # Download and process embedded data
    for table_name, column in TABLES_EMBEDDED_DATA_COLUMN.items():
        entity_id = SYNAPSE_TABLES[table_name]

        # Download embedded files
        file_handles = downloader.download_embedded_data_files(
            embedded_column=column,
            entity_id=entity_id
        )

        # Save file handles to CSV
        df_files = pd.DataFrame.from_dict(file_handles, orient='index')
        df_files.to_csv(f'{raw_dir}/file_handles_{table_name}.csv')

        # Import embedded data and handle failures
        failed_ids = importer.import_embedded_data(
            dataframe=df_files,
            collection_name=table_name  # Keep original name
        )

        # Save failed file IDs to pickle
        with open(f"{raw_dir}/failed_ids_{table_name}.pkl", 'wb') as f:
            pickle.dump(failed_ids, f)
            
    raw_dir = os.path.join('data', 'raw')
    embedded_dir = os.path.join(raw_dir, 'embedded')

    importer = Importer()
    importer.merge_original_and_embedded_data(
        original_file=f'{raw_dir}/healthkit_data.csv',
        embedded_file=f'{raw_dir}/file_handles_healthkit_data.csv',  
        collection_name=f'healthkit_stepscount_singles'
    )

    importer = Importer()
    importer.merge_original_and_embedded_data(
        original_file=f'{raw_dir}/healthkit_data.csv',
        embedded_file=f'{raw_dir}/file_handles_healthkit_data.csv',  
        collection_name=f'healthkit_heartrate_singles',
        activity_to_keep="HKQuantityTypeIdentifierHeartRate"
    )
    