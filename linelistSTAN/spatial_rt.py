# linelistSTAN/spatial_rt.py

import numpy as np
import pandas as pd
import cmdstanpy
import tempfile
import os
from .utils import si

def spatialRt(report_dates, case_matrix, transfer_matrix, sip, chains=4, iter=2000, warmup=1000, v2=False):
    # Prepare data for Stan model
    data = {
        'N': len(report_dates),
        'M': case_matrix.shape[1],
        'Y': case_matrix,
        'L': len(sip),
        'si': sip,
        'Tf': transfer_matrix
    }
    
    # Stan model code
    stan_code = """
    data {
      int<lower=0> N;
      int<lower=0> M;
      int<lower=0> L;
      array[N, M] int Y;
      vector[L] si;
      matrix[M, M] Tf;
    }
    parameters {
      array[M] vector<lower=0>[N] lambda;
      array[M] vector<lower=0>[N] R;
      array[M] real<lower=0> phi;
    }
    model {
      for (m in 1:M) {
        phi[m] ~ exponential(1);
        R[m] ~ normal(1, 1);
        
        for (t in 1:N) {
          real convolution = 0;
          for (l in 1:min(t, L)) {
            convolution += si[l] * lambda[m, t-l+1];
          }
          
          real transferred = 0;
          for (j in 1:M) {
            transferred += Tf[j, m] * lambda[j, t];
          }
          
          lambda[m, t] ~ lognormal(log(R[m, t] * convolution + transferred), phi[m]);
          Y[t, m] ~ poisson(lambda[m, t]);
        }
      }
    }
    """
    
    # Write Stan code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.stan', delete=False) as tmp:
        tmp.write(stan_code)
        tmp_path = tmp.name

    try:
        # Compile and run the model
        model = cmdstanpy.CmdStanModel(stan_file=tmp_path)
        fit = model.sample(data=data, chains=chains, iter_sampling=iter, iter_warmup=warmup)
        
        # Extract results
        results = {
            'lambda': fit.stan_variable('lambda'),
            'R': fit.stan_variable('R'),
            'phi': fit.stan_variable('phi')
        }
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)
    
    return results

def summarize_spatial_results(results, report_dates, site_names):
    N = len(report_dates)
    M = len(site_names)
    
    data_all = []
    for i in range(M):
        df = pd.DataFrame({
            'x': range(1, N+1),
            'y': np.mean(results['lambda'][:, i, :], axis=0),
            'yl': np.quantile(results['lambda'][:, i, :], 0.025, axis=0),
            'yh': np.quantile(results['lambda'][:, i, :], 0.975, axis=0),
            'Rt': np.mean(results['R'][:, i, :], axis=0),
            'Rtl': np.quantile(results['R'][:, i, :], 0.025, axis=0),
            'Rth': np.quantile(results['R'][:, i, :], 0.975, axis=0),
            'region': site_names[i]
        })
        data_all.append(df)
    
    data_all = pd.concat(data_all, ignore_index=True)
    return data_all