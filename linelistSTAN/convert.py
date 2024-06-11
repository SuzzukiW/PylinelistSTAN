# linelistSTAN/convert.py

import pandas as pd

def convert_to_linelist(caseCounts, reportF_missP=0.01):
    linelist = pd.DataFrame(caseCounts)
    linelist = linelist[['report_date', 'onset_date', 'delay_int']]
    linelist = linelist.dropna(subset=['delay_int'])
    
    linelist['value'] = 1
    linelist['id'] = range(1, len(linelist) + 1)
    
    # Calculate start_dt based on report_date
    linelist['start_dt'] = (linelist['report_date'] - linelist['report_date'].min()).dt.days
    
    # Create actual_onset and actual_report columns
    linelist['actual_onset'] = linelist['start_dt']
    linelist['actual_report'] = linelist['start_dt'] + linelist['delay_int']
    
    # Calculate week_int based on start_dt
    linelist['week_int'] = (linelist['start_dt'] // 7) + 1
    
    # Create is_weekend column
    linelist['is_weekend'] = linelist['onset_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    return linelist