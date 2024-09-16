# linelistSTAN/utils.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt

def create_caseCounts(date_vec, location_vec, cases_vec):
    # Check types
    assert all(pd.notnull(pd.to_datetime(date_vec))), "Invalid dates"
    assert all(isinstance(loc, str) for loc in location_vec), "Location must be strings"
    
    # Check if cases are numeric (including numpy integer types)
    if not all(isinstance(case, (int, float, np.integer, np.floating)) for case in cases_vec):
        raise ValueError(f"Cases must be numeric. Received types: {[type(case) for case in cases_vec]}")
    
    # No negative case numbers
    assert all(case >= 0 for case in cases_vec), "Negative case numbers not allowed"
    
    # Check for multiple locations
    if len(set(location_vec)) > 1:
        print('Warning: More than 1 location')
    
    # Check lengths
    assert len(date_vec) == len(location_vec) == len(cases_vec), "All vectors must have the same length"
    
    # Create caseCounts
    caseCounts = pd.DataFrame({
        'date': pd.to_datetime(date_vec),
        'cases': [int(case) for case in cases_vec],  # Convert numpy integers to Python integers
        'location': location_vec
    })
    
    # Check for NA values
    assert not caseCounts[['date', 'cases', 'location']].isna().any().any(), "NA values not allowed"
    
    # Add class attribute (Python doesn't have direct equivalents, so we'll use a custom attribute)
    caseCounts.attrs['class'] = 'caseCounts'
    
    return caseCounts

def si(ndays, shape, rate, leading0=False):
    x = np.arange(1, ndays + 1)
    prob = stats.gamma.pdf(x, a=shape, scale=1/rate)
    prob = prob / np.sum(prob)
    if leading0:
        prob = np.concatenate(([0], prob))
    return prob

def plot_si(sip):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sip)), sip, 'b-')
    plt.title('Serial Interval Distribution')
    plt.xlabel('Days')
    plt.ylabel('Probability')
    plt.show()

def plot_caseCounts(x, loc=None):
    if len(x['location'].unique()) > 1 and loc is None:
        raise ValueError('Specify loc="..." if more than one location exists')
    
    if loc is not None:
        x = x[x['location'] == loc]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x['date'], x['cases'])
    plt.xlabel('Date')
    plt.ylabel('N. Cases')
    plt.title('Case Counts Over Time')
    
    x_pos = int(len(x) * 0.05)
    y_pos = sorted(x['cases'], reverse=True)[1]
    plt.text(sorted(x['date'])[x_pos], y_pos, x['location'].iloc[0])
    
    plt.show()

def load_sample_data():
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(80)]
    
    sample_dates = dates
    sample_location = ['Tatooine'] * 80
    sample_cases = [int(case) for case in np.random.poisson(lam=20, size=80)]  # Convert to Python integers
    
    sample_report_dates = dates * 3  # Assuming 3 cases per day on average
    sample_onset_dates = [
        report_date - timedelta(days=np.random.randint(0, 14))
        for report_date in sample_report_dates
    ]
    
    # Introduce some missing onset dates
    missing_indices = np.random.choice(len(sample_onset_dates), size=int(len(sample_onset_dates) * 0.2), replace=False)
    for idx in missing_indices:
        sample_onset_dates[idx] = pd.NaT
    
    sample_multi_site = pd.DataFrame({
        'date': dates,
        'Tatooine': [int(case) for case in np.random.poisson(lam=20, size=80)],
        'Hoth': [int(case) for case in np.random.poisson(lam=15, size=80)]
    })
    
    transfer_matrix = pd.DataFrame({
        'Tatooine': [0.8, 0.2] * 80,
        'Hoth': [0.2, 0.8] * 80
    }, index=[f"{d.strftime('%Y-%m-%d')}:{loc}" for d in dates for loc in ['Tatooine', 'Hoth']])
    
    return {
        'sample_cases': sample_cases,
        'sample_dates': sample_dates,
        'sample_location': sample_location,
        'sample_onset_dates': sample_onset_dates,
        'sample_report_dates': sample_report_dates,
        'sample_multi_site': sample_multi_site,
        'transfer_matrix': transfer_matrix
    }