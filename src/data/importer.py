import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class Importer:
    """
    Class to import Synapse data and save it to local files.
    """

    def __init__(self, output_dir: str = './data/raw/myheartcounts'):
        """
        Initializes the Importer with the specified output directory.

        Args:
            output_dir (str): The directory where files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_to_file(self, data: list, filename: str):
        """
        Saves the given data to a CSV file locally.

        Args:
            data (list of dict): The data to be saved.
            filename (str): The name of the output file.
        """
        output_file = os.path.join(self.output_dir, f'{filename}.csv')
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    def import_syn_table(self, dataframe: pd.DataFrame, collection_name: str):
        """
        Saves the Synapse table to a CSV file locally.

        Args:
            dataframe (pd.DataFrame): The Synapse table in DataFrame format.
            collection_name (str): The name of the output file.
        """
        docs = dataframe.to_dict(orient='records')
        self.save_to_file(docs, collection_name)

    def import_embedded_data(self, dataframe: pd.DataFrame, collection_name: str) -> list:
        """
        Saves embedded data from file handle ids to local CSV files.

        Args:
            dataframe (pd.DataFrame): The file handle ids in DataFrame format.
            collection_name (str): The name of the output file.

        Returns:
            list: A list of tuples that weren't successfully processed.
        """
        failed_ids = []
        embedded_dir = os.path.join(self.output_dir, 'embedded')  # Set subdirectory for embedded files
        os.makedirs(embedded_dir, exist_ok=True)  # Ensure the directory exists

        for index, row in dataframe.iterrows():
            try:
                print(f"Importing data id: {index}")

                data_id = index
                filepath = row.values[0]

                df_inner = pd.read_csv(filepath)

                # Also save the data_id to know which record it refers to
                df_inner['data_id'] = data_id

                # Handle empty records
                if not df_inner.empty:
                    docs = df_inner.to_dict(orient='records')
                else:
                    docs = [{'data_id': data_id, 'record': None}]

                # Save to the embedded subdirectory
                self.save_to_file(docs, os.path.join('embedded', f"{collection_name}_{data_id}"))
            except Exception as e:
                failed_ids.append((index, str(e)))

        return failed_ids

    def merge_original_and_embedded_data(
        self, 
        original_file: str, 
        embedded_file: str, 
        collection_name: str, 
        activity_to_keep: str = 'HKQuantityTypeIdentifierStepCount'
    ) -> list:
        """
        Joins original and embedded files, and saves them locally in a flattened structure.

        Args:
            original_file (str): The CSV file path of the original table data.
            embedded_file (str): The CSV file path with the embedded data.
            collection_name (str): The name of the output file.
            activity_to_keep (str): Filter out all activities except the specified one.
                Defaults to step count activity.

        Returns:
            list: A list of tuples that weren't successfully processed.
        """
        os.makedirs(os.path.join(self.output_dir, collection_name))
        hk_data = pd.read_csv(original_file, index_col=0).reset_index(drop=True)
        file_ids = pd.read_csv(embedded_file, header=0, names=['data_id', 'cache_filepath'])

        # Get all unique users
        all_users = hk_data['healthCode'].unique()
        failed_files = []

        for user in tqdm(all_users):
            user_df = hk_data[hk_data['healthCode'] == user]
            df_merged = user_df.merge(file_ids, left_on='data.csv', right_on='data_id', how='inner')
            
            # Convert 'createdOn' to datetime
            df_merged['createdOn'] = df_merged['createdOn'].apply(lambda x: datetime.fromtimestamp(x / 1000))

            for index, row in df_merged.iterrows():
                cache_file = row['cache_filepath']

                try:
                    embedded_df = pd.read_csv(cache_file)
                except UnicodeDecodeError:
                    print(f"Opening file with unicode encoding: {cache_file}")
                    embedded_df = pd.read_csv(cache_file, encoding='unicode_escape')
                except Exception as e:
                    print(f"Exception raised. Corrupted file: {e}")
                    failed_files.append((cache_file, str(e)))
                    continue

                # Keep only the specified activity
                embedded_df = embedded_df[embedded_df['type'] == activity_to_keep]

                # Convert times to UTC and handle missing values
                embedded_df['startTime'] = pd.to_datetime(embedded_df['startTime'], errors='coerce', utc=True)
                embedded_df['endTime'] = pd.to_datetime(embedded_df['endTime'], errors='coerce', utc=True)
                embedded_df.replace({pd.NaT: None}, inplace=True)

                if embedded_df.empty:
                    continue
                else:
                    # Populate user's info into embedded data
                    for key, value in row.items():
                        embedded_df[key] = value

                    # Drop unnecessary columns before saving
                    embedded_df.drop(['data_id', 'cache_filepath'], axis=1, inplace=True)

                    final_docs = embedded_df.to_dict(orient='records')
                    self.save_to_file(final_docs, os.path.join(collection_name, f"{user}"))
                    
        return failed_files