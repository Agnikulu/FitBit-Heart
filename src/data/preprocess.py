import os
import pandas as pd
from datetime import datetime

def process_healthkit_folder(heartrate_folder, stepcount_folder, par_q_file, time_granularity='hour'):
    # Function to read all CSV files from a folder into a single DataFrame
    def read_all_csvs_from_folder(folder_path):
        all_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path, dtype={'healthCode': str})  # Ensure healthCode is treated as a string
                all_files.append(df)
        return pd.concat(all_files, ignore_index=True) if all_files else pd.DataFrame()

    # Load all heart rate and step count data
    hr_df = read_all_csvs_from_folder(heartrate_folder)
    steps_df = read_all_csvs_from_folder(stepcount_folder)

    # Load the PAR-Q file with healthCode and heartCondition
    par_q_df = pd.read_csv(par_q_file, dtype={'healthCode': str})[['healthCode', 'heartCondition']]

    # Function to extract date and time based on granularity
    def extract_date_time(df, time_col, granularity):
        df['datetime'] = pd.to_datetime(df[time_col])
        df['date'] = df['datetime'].dt.date
        if granularity == 'hour':
            df['hour'] = df['datetime'].dt.hour  # Extract hour (0-23)
        elif granularity == 'minute':
            df['minute'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute  # Convert to minute of the day (0-1439)
        return df

    # Process heart rate data (convert from count/s to bpm)
    hr_df = extract_date_time(hr_df, 'startTime', time_granularity)
    hr_df['bpm'] = hr_df['value'] * 60  # Convert count/s to bpm

    # Group by based on granularity
    group_cols = ['healthCode', 'date']
    if time_granularity == 'hour':
        group_cols.append('hour')
    elif time_granularity == 'minute':
        group_cols.append('minute')

    hr_grouped = hr_df.groupby(group_cols).agg({'bpm': 'mean'}).reset_index()

    # Process step count data
    steps_df = extract_date_time(steps_df, 'startTime', time_granularity)
    steps_grouped = steps_df.groupby(group_cols).agg({'value': 'sum'}).reset_index().rename(columns={'value': 'steps'})

    # Merge heart rate and step count data on healthCode, date, and time
    merged_df = pd.merge(hr_grouped, steps_grouped, on=group_cols, how='inner')

    # Filter users with at least 8 overlapping time entries
    user_entry_count = merged_df.groupby('healthCode').size()
    valid_users = user_entry_count[user_entry_count >= 8].index
    filtered_df = merged_df[merged_df['healthCode'].isin(valid_users)]

    # Merge with the PAR-Q heartCondition data on healthCode
    final_merged_df = pd.merge(filtered_df, par_q_df, on='healthCode', how='left')

    # Drop users with no heart condition label
    final_merged_df = final_merged_df[final_merged_df['heartCondition'].notna()]

    # Create a unique ID column based on healthCode
    final_merged_df['id'] = final_merged_df['healthCode']

    # Select final columns based on granularity
    if time_granularity == 'hour':
        final_columns = ['id', 'date', 'hour', 'bpm', 'steps', 'heartCondition']
    elif time_granularity == 'minute':
        final_columns = ['id', 'date', 'minute', 'bpm', 'steps', 'heartCondition']

    final_df = final_merged_df[final_columns]

    return final_df

if __name__ == '__main__':
    # Define input folders and files
    heartrate_folder = 'data/myheartcounts/healthkit_heartrate_singles'
    stepscount_folder = 'data/myheartcounts/healthkit_stepscount_singles'
    par_q_file = 'data/myheartcounts/par_q.csv'

    # Process the health kit data with default granularity as hour
    result_df = process_healthkit_folder(heartrate_folder, stepscount_folder, par_q_file)
    print(result_df)

    # Save the result to a new CSV file
    result_df.to_csv('data/myheartcounts.csv', index=False)
