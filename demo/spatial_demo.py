import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linelistSTAN import spatialRt, summarize_spatial_results, si

# Sample data
def generate_sample_data():
    dates = pd.date_range(start='2020-01-01', end='2020-03-20')
    site1 = np.random.poisson(lam=20, size=len(dates))
    site2 = np.random.poisson(lam=15, size=len(dates))
    
    sample_multi_site = pd.DataFrame({
        'date': dates,
        'Tatooine': site1,
        'Hoth': site2
    })
    
    transfer_matrix = np.array([[0.9, 0.1],
                                [0.1, 0.9]])
    
    return sample_multi_site, transfer_matrix

def main():
    # Generate sample data
    sample_multi_site, transfer_matrix = generate_sample_data()
    
    # Prepare data for spatialRt
    report_dates = sample_multi_site['date']
    case_matrix = sample_multi_site[['Tatooine', 'Hoth']].values
    site_names = ['Tatooine', 'Hoth']
    
    # Define serial interval
    sip = si(14, 4.29, 1.18, leading0=False)
    
    # Run spatialRt
    results = spatialRt(report_dates, case_matrix, transfer_matrix, sip, chains=1)
    
    # Summarize results
    data_all = summarize_spatial_results(results, report_dates, site_names)
    
    # Plot expected cases
    plt.figure(figsize=(10, 6))
    for region in site_names:
        region_data = data_all[data_all['region'] == region]
        plt.plot(region_data['x'], region_data['y'], label=region)
        plt.fill_between(region_data['x'], region_data['yl'], region_data['yh'], alpha=0.3)
    plt.xlabel('Days')
    plt.ylabel('Cases')
    plt.title('Expected Cases')
    plt.legend()
    plt.show()
    
    # Plot R(t)
    plt.figure(figsize=(10, 6))
    for region in site_names:
        region_data = data_all[data_all['region'] == region]
        plt.plot(region_data['x'], region_data['Rt'], label=region)
        plt.fill_between(region_data['x'], region_data['Rtl'], region_data['Rth'], alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('Reproduction Number')
    plt.title('R(t)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()