# data_preparation.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import gamma, poisson

def prepare_data():
    np.random.seed(123)

    si_shape = 2
    si_rate = 0.5

    w = [gamma.cdf(x, si_shape, scale=1/si_rate) - gamma.cdf(x-1, si_shape, scale=1/si_rate) for x in range(1, 15)]
    w = np.array(w) / sum(w)

    def Ra(t): return (20*np.cos(t/500) + (0.8*t - 50)**2 - (0.115 * t)**3)/1000 + 0.8
    def Rb(t): return (30*np.sin(t/150) + np.cos(t/20) - (t/50)**2)/8 - 0.006*t

    tmax = 80
    Rmatrix = np.column_stack((Ra(np.arange(1, tmax+1)), Rb(np.arange(1, tmax+1))))

    M = np.zeros((tmax, 2))
    N = np.zeros((tmax, 2))
    R_rev = np.zeros((tmax, 2))
    R_this = np.zeros((tmax, 2))

    M[0] = N[0] = [10, 10]
    R_rev[0] = R_this[0] = [1e-5, 1e-5]

    P = np.array([[0.8, 0.4], [0.2, 0.6]])

    for t in range(1, tmax):
        tau_end = min(len(w), t)
        
        RR = np.diag(Rmatrix[t])
        MM = M[t-tau_end:t].T if t > 1 else M[t-1:t].T
        WW = w[:tau_end].reshape(-1, 1)

        inner_vec = RR @ MM @ WW
        outer_vec = P.T @ inner_vec

        M[t] = outer_vec.flatten()
        N[t] = poisson.rvs(outer_vec.flatten())

        sum_m_w_mat = np.tile((MM @ WW).T, (2, 1))
        c_mat = P.T * sum_m_w_mat
        R_rev[t] = np.linalg.solve(c_mat.T @ c_mat, c_mat.T @ M[t])

        NN = N[t-tau_end:t].T if t > 1 else N[t-1:t].T
        sum_m_w_mat = np.tile((NN @ WW).T, (2, 1))
        c_mat = P.T * sum_m_w_mat
        R_this[t] = np.linalg.solve(c_mat.T @ c_mat, c_mat.T @ N[t])

    ref_date = datetime(2020, 1, 1)
    sample_dates = [ref_date + timedelta(days=i) for i in range(tmax)]
    sample_cases = N[:, 0].astype(int)
    sample_location = ["Tatooine"] * tmax

    cc = pd.DataFrame({
        'date': sample_dates,
        'location': sample_location,
        'cases': sample_cases
    })

    ll = convert_to_linelist(cc, reportF_missP=0.5)

    sample_multi_site = pd.DataFrame({
        'date': sample_dates,
        'Tatooine': N[:, 0],
        'Hoth': N[:, 1]
    })

    transfer_matrix = np.tile(P, (tmax, 1))
    transfer_matrix_df = pd.DataFrame(transfer_matrix, columns=['Tatooine', 'Hoth'])
    transfer_matrix_df.index = [f"{d.strftime('%Y-%m-%d')}:{loc}" for d in sample_dates for loc in ['Tatooine', 'Hoth']]

    return {
        'sample_cases': sample_cases,
        'sample_dates': sample_dates,
        'sample_location': sample_location,
        'sample_onset_dates': ll['onset_date'],
        'sample_report_dates': ll['report_date'],
        'sample_multi_site': sample_multi_site,
        'transfer_matrix': transfer_matrix_df
    }

# You might want to add a function to save this data or integrate it with your existing data loading mechanism