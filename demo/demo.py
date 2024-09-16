# demo/demo.py

from linelistSTAN import (
    create_linelist, run_backnow, si, load_sample_data, plot_results,
    create_caseCounts, plot_caseCounts, convert_to_linelist
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def main():
    print("Loading sample data...")
    data = load_sample_data()
    if data is None:
        print("Error: Failed to load sample data. Make sure load_sample_data() is implemented correctly.")
        return

    print("Creating case counts...")
    try:
        case_counts = create_caseCounts(
            date_vec=data['sample_dates'],
            location_vec=data['sample_location'],
            cases_vec=data['sample_cases']
        )
    except Exception as e:
        print(f"Error creating case counts: {e}")
        return

    print("Plotting case counts...")
    try:
        plot_caseCounts(case_counts)
    except Exception as e:
        print(f"Error plotting case counts: {e}")

    print("Converting case counts to linelist...")
    try:
        linelist_from_counts = convert_to_linelist(case_counts, reportF_missP=0.5)
    except Exception as e:
        print(f"Error converting case counts to linelist: {e}")
        return

    print("Creating linelist from report and onset dates...")
    try:
        my_linelist = create_linelist(data['sample_report_dates'], data['sample_onset_dates'])
    except Exception as e:
        print(f"Error creating linelist: {e}")
        return

    print("Defining serial interval...")
    sip = si(14, 4.29, 1.18, leading0=False)  # Set leading0 to False
    print(f"Length of sip: {len(sip)}")  # Add this line to check the length of sip

    print("Running back-calculation algorithm...")
    try:
        results = run_backnow(my_linelist, sip=sip, chains=1)
    except Exception as e:
        print(f"Error running back-calculation algorithm: {e}")
        return

    print("Plotting results...")
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plot_results(results, my_linelist, "est")
    plt.subplot(2, 1, 2)
    plot_results(results, my_linelist, "rt")
    plt.tight_layout()
    plt.show()

    print("Demonstrating multi-site data:")
    print(data['sample_multi_site'].head())

    print("\nDemonstrating transfer matrix:")
    print(data['transfer_matrix'].head())

if __name__ == "__main__":
    main()