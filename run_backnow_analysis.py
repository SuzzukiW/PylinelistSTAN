# run_backnow_analysis.py

from linelistSTAN import (
    create_linelist, 
    run_backnow, 
    si, 
    load_sample_data, 
    plot_results, 
    create_caseCounts, 
    plot_caseCounts, 
    convert_to_linelist
)
import matplotlib.pyplot as plt

def plot_si(sip):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sip)), sip, 'b-')
    plt.title('Serial Interval Distribution')
    plt.xlabel('Days')
    plt.ylabel('Probability')
    plt.show()

# ... (previous imports and functions remain the same)

def main():
    print("Loading sample data...")
    data = load_sample_data()

    print("Creating case counts...")
    case_counts = create_caseCounts(
        date_vec=data['sample_dates'],
        location_vec=data['sample_location'],
        cases_vec=data['sample_cases']
    )

    print("Plotting case counts...")
    plot_caseCounts(case_counts)

    print("Converting case counts to linelist...")
    linelist_from_counts = convert_to_linelist(case_counts, reportF_missP=0.5)

    print("Creating linelist from report and onset dates...")
    my_linelist = create_linelist(data['sample_report_dates'], data['sample_onset_dates'])

    print("Defining serial interval...")
    sip_for_plot = si(14, 4.29, 1.18, leading0=True)
    sip_for_model = si(14, 4.29, 1.18, leading0=False)
    print(f"Length of sip for plot: {len(sip_for_plot)}")
    print(f"Length of sip for model: {len(sip_for_model)}")
    plot_si(sip_for_plot)

    print("Running back-calculation algorithm...")
    results = run_backnow(my_linelist, sip=sip_for_model, chains=1)

    print("Plotting results...")
    plot_results(results, my_linelist, case_counts, "est")
    plot_results(results, my_linelist, case_counts, "rt")

    print("Demonstrating multi-site data:")
    print(data['sample_multi_site'].head())

    print("\nDemonstrating transfer matrix:")
    print(data['transfer_matrix'].head())

if __name__ == "__main__":
    main()