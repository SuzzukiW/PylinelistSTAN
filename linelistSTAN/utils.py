# linelistSTAN/utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

def create_caseCounts(sample_dates, sample_location, sample_cases):
    df = pd.DataFrame({
        'date': pd.to_datetime(sample_dates),
        'location': sample_location,
        'cases': sample_cases
    })
    
    case_counts = df.groupby(['date', 'location'])['cases'].sum().reset_index()
    case_counts.columns = ['report_date', 'location', 'cases']
    
    case_counts['onset_date'] = case_counts['report_date'] - pd.Timedelta(days=7)
    case_counts['delay_int'] = (case_counts['report_date'] - case_counts['onset_date']).dt.days
    
    return case_counts

def plot_predictions(predicted_onset):
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_onset['pred_onset'], predicted_onset['ex'], label='Predicted Median')
    plt.fill_between(predicted_onset['pred_onset'], predicted_onset['lb'], predicted_onset['ub'], alpha=0.3, label='95% CI')
    plt.xlabel('Predicted Onset Date')
    plt.ylabel('Number of Cases')
    plt.title('Predicted COVID-19 Case Onsets')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_trace(trace):
    az.plot_trace(trace)
    plt.tight_layout()
    plt.show()

def plot_posterior(trace):
    az.plot_posterior(trace)
    plt.tight_layout()
    plt.show()