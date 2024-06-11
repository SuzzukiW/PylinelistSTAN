# linelistSTAN/utils.py

import pandas as pd

def create_caseCounts(sample_dates, sample_location, sample_cases):
    # Create a DataFrame from the input data
    data = {
        'date': sample_dates,
        'location': sample_location,
        'cases': sample_cases
    }
    df = pd.DataFrame(data)
    
    # Convert the 'date' column to datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Group the data by date and location, and sum the cases
    case_counts = df.groupby(['date', 'location'])['cases'].sum().reset_index()
    
    # Rename columns to match the expected format
    case_counts.columns = ['report_date', 'location', 'cases']
    
    # Calculate the onset date assuming a constant incubation period of 7 days
    case_counts['onset_date'] = case_counts['report_date'] - pd.Timedelta(days=7)
    
    # Calculate the delay interval in days between onset and report
    case_counts['delay_int'] = (case_counts['report_date'] - case_counts['onset_date']).dt.days
    
    return case_counts