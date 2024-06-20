# linelistSTAN/convert.py

import pandas as pd
import numpy as np

def convert_to_linelist(caseCounts, reportF_missP=0.01):
    linelist = caseCounts.copy()
    linelist = linelist[['report_date', 'onset_date', 'delay_int']]
    linelist = linelist.dropna(subset=['delay_int'])
    
    linelist['value'] = 1
    linelist['id'] = range(1, len(linelist) + 1)
    
    linelist['start_dt'] = (linelist['report_date'] - linelist['report_date'].min()).dt.days
    linelist['actual_onset'] = linelist['start_dt']
    linelist['actual_report'] = linelist['start_dt'] + linelist['delay_int']
    
    linelist['week_int'] = (linelist['start_dt'] // 7) + 1
    linelist['is_weekend'] = linelist['onset_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Add more features
    linelist['log_delay'] = np.log1p(linelist['delay_int'])
    linelist['day_of_week'] = linelist['report_date'].dt.dayofweek
    linelist['month'] = linelist['report_date'].dt.month
    
    return linelist