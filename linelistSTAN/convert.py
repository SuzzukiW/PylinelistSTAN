# linelistSTAN/convert.py

import pandas as pd
import numpy as np
from datetime import timedelta

def create_linelist(report_dates, onset_dates):
    # Check types
    assert all(pd.notnull(pd.to_datetime(report_dates))), "Invalid report dates"
    assert any(pd.isnull(pd.to_datetime(onset_dates))), "Some onset dates should be NA"
    
    # Length equal
    assert len(report_dates) == len(onset_dates), "Length of report_dates and onset_dates must be equal"
    
    # NO NA report dates
    assert not any(pd.isnull(report_dates)), "No NA values allowed in report_dates"
    
    # Some NA onset dates
    which_na = pd.isnull(onset_dates)
    if not any(which_na):
        raise ValueError("No missing onset_dates - what will you impute??")
    if all(which_na):
        raise ValueError("All onset_dates missing")
    
    # All onset must be <= report
    onset_dates = pd.to_datetime(onset_dates)
    report_dates = pd.to_datetime(report_dates)
    valid_dates = ~pd.isnull(onset_dates)
    if any(onset_dates[valid_dates] > report_dates[valid_dates]):
        raise ValueError("Some onset dates are after report dates")
    
    # Create dataframe
    d = pd.DataFrame({'report_date': report_dates, 'onset_date': onset_dates})
    d = d.sort_values('report_date')
    
    # Calculate delay
    d['delay_int'] = (d['report_date'] - d['onset_date']).dt.days
    
    # Is weekend
    d['is_weekend'] = d['report_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Calculate report_int and week_int
    min_day = d['report_date'].min()
    d['report_int'] = (d['report_date'] - min_day).dt.days + 1
    d['week_int'] = np.ceil(d['report_int'] / 7)
    
    # Convert types
    d['delay_int'] = d['delay_int'].astype('Int64')  # nullable integer
    d['is_weekend'] = d['is_weekend'].astype(int)
    d['report_int'] = d['report_int'].astype(int)
    d['week_int'] = d['week_int'].astype(int)
    
    # Reorder columns
    d = d[["report_date", "delay_int", "onset_date", "is_weekend", "report_int", "week_int"]]
    
    # Set class (in Python, we'll use a custom attribute)
    d.attrs['class'] = 'lineList'
    
    return d

def convert_to_linelist(caseCounts, reportF_missP=0.5):
    dates = pd.to_datetime(caseCounts['date'])
    cases = caseCounts['cases']
    
    all_onset_dates = []
    all_report_dates = []
    
    for date, count in zip(dates, cases):
        report_dates = [date] * count
        onset_dates = [date - timedelta(days=max(0, int(np.random.negative_binomial(3, 0.25)))) for _ in range(count)]
        
        # Introduce missing data
        mask = np.random.random(count) < reportF_missP
        onset_dates = [d if not m else pd.NaT for d, m in zip(onset_dates, mask)]
        
        all_onset_dates.extend(onset_dates)
        all_report_dates.extend(report_dates)
    
    return create_linelist(all_report_dates, all_onset_dates)