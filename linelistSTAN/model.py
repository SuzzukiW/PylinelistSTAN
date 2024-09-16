# linelistSTAN/model.py

import cmdstanpy
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import tempfile
import os
from cmdstanpy.utils import cmdstan_path
from scipy.ndimage import gaussian_filter1d

STAN_CODE = """
functions {
  real safe_log(real x) {
    return x > 0 ? log(x) : -1e10;
  }
}
data {
  int<lower=0> N;
  int<lower=0> T;
  array[N] int<lower=1> report;
  array[N] int<lower=-1> onset;
  array[N] int<lower=1> week;
  vector[14] sip;
  int<lower=0> L;
}
parameters {
  vector<lower=0>[T] lambda_raw;
  real<lower=0> lambda_mean;
  real<lower=0> lambda_sd;
  real<lower=0, upper=2> phi;
  vector[T] log_R;
  real<lower=0, upper=1> reporting_rate;
}
transformed parameters {
  vector<lower=1e-10>[T] lambda = lambda_mean + lambda_sd * lambda_raw + 1e-10;
  vector<lower=0>[T] R = exp(log_R);
}
model {
  vector[T] cumul_lambda = cumulative_sum(lambda);
  
  lambda_mean ~ normal(20, 10);
  lambda_sd ~ normal(0, 5);
  lambda_raw ~ normal(0, 1);
  phi ~ normal(1, 0.25);
  log_R ~ normal(0, 0.25);
  reporting_rate ~ beta(10, 1);  // Adjusted to favor higher reporting rates
  
  for (i in 1:N) {
    if (onset[i] == -1) {
      vector[report[i]] lp;
      for (t in 1:report[i]) {
        lp[t] = safe_log(lambda[t]) + safe_log(sip[min(L, report[i]-t+1)]);
      }
      target += log_sum_exp(lp) + safe_log(reporting_rate);
    } else {
      int delay = report[i] - onset[i];
      if (delay > 0 && delay <= T) {
        target += poisson_lpmf(1 | lambda[delay]) + safe_log(reporting_rate);
      }
    }
  }
  
  for (t in 1:T) {
    real mu = 0;
    for (l in 1:min(L, t)) {
      mu += safe_log(sip[l]) + safe_log(lambda[t-l+1]);
    }
    mu += log_R[t];
    
    if (is_inf(mu) || is_nan(mu)) {
      print("Warning: mu is ", mu, " at t = ", t);
      target += -1e10;  // Penalize infinite or NaN values
    } else {
      lambda[t] ~ lognormal(fmin(mu, 10), phi);  // Cap mu to prevent extreme values
    }
  }
}
generated quantities {
  vector[T] pred_cases = lambda;  // Remove reporting_rate multiplication here
}
"""

def run_backnow(linelist, sip, chains=4, iter=2000, warmup=1000):
    # Check sip length
    if len(sip) != 14:
        raise ValueError(f"sip must be a vector of length 14, but got length {len(sip)}")

    print("Input data summary:")
    print(f"Number of rows: {len(linelist)}")
    print(f"Max report_int: {linelist['report_int'].max()}")
    print(f"Max delay_int: {linelist['delay_int'].max()}")
    print(f"sip: {sip}")

    # Data validation
    if linelist['report_int'].max() > 1000:
        raise ValueError(f"Unusually large report_int detected: {linelist['report_int'].max()}")
    if linelist['delay_int'].max() > 100:
        raise ValueError(f"Unusually large delay_int detected: {linelist['delay_int'].max()}")

    data = {
        'N': len(linelist),
        'T': linelist['report_int'].max(),
        'report': linelist['report_int'].values.astype(int),
        'onset': np.where(linelist['onset_date'].notna(), linelist['delay_int'], -1).astype(int),
        'week': linelist['week_int'].values.astype(int),
        'sip': sip,
        'L': len(sip)
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.stan', delete=False) as tmp:
        tmp.write(STAN_CODE)
        tmp_path = tmp.name

    try:
        model = cmdstanpy.CmdStanModel(stan_file=tmp_path, compile=True)
        fit = model.sample(data=data, chains=chains, iter_sampling=iter, iter_warmup=warmup, show_progress=True)
        
        # Extract results as numpy arrays
        results = {
            'lambda': fit.stan_variable('lambda'),
            'R': fit.stan_variable('R'),
            'phi': fit.stan_variable('phi'),
            'reporting_rate': fit.stan_variable('reporting_rate'),
            'pred_cases': fit.stan_variable('pred_cases'),
        }
        
        # Check for NaN values
        for key, value in results.items():
            if np.isnan(value).any():
                print(f"Warning: NaN values detected in {key}")
                results[key] = np.nan_to_num(value, nan=np.nanmean(value))
        
        print("STAN model results summary:")
        print(f"lambda shape: {results['lambda'].shape}")
        print(f"R shape: {results['R'].shape}")
        print(f"phi: {results['phi']}")
        print(f"lambda max: {np.max(results['lambda'])}")
        print(f"R max: {np.max(results['R'])}")
        print(f"Reporting rate: {np.mean(results['reporting_rate'])}")
        print(f"Predicted cases max: {np.max(results['pred_cases'])}")
        
    except ValueError as e:
        if "No CmdStan installation found" in str(e):
            print("CmdStan not found. Attempting to use default installation path...")
            os.environ['CMDSTAN'] = cmdstan_path()
            model = cmdstanpy.CmdStanModel(stan_file=tmp_path, compile=True)
            fit = model.sample(data=data, chains=chains, iter_sampling=iter, iter_warmup=warmup, show_progress=True)
            results = {
                'lambda': fit.stan_variable('lambda'),
                'R': fit.stan_variable('R'),
                'phi': fit.stan_variable('phi'),
                'reporting_rate': fit.stan_variable('reporting_rate'),
                'pred_cases': fit.stan_variable('pred_cases'),
            }
        else:
            raise
    finally:
        os.unlink(tmp_path)
    
    return results

def summarize_results(results):
    summary_dict = {
        'lambda': az.summary(az.convert_to_dataset(results['lambda'])),
        'R': az.summary(az.convert_to_dataset(results['R'])),
        'phi': az.summary(az.convert_to_dataset(results['phi']))
    }
    return summary_dict

def plot_results(results, linelist, case_counts, plot_type="est"):
    if plot_type == "est":
        reported_cases = case_counts.set_index('date')['cases']
        mean_pred_cases = np.nanmean(results['pred_cases'], axis=0)
        lower_pred_cases = np.nanpercentile(results['pred_cases'], 2.5, axis=0)
        upper_pred_cases = np.nanpercentile(results['pred_cases'], 97.5, axis=0)
        
        # Scale predicted cases by reporting rate
        reporting_rate = np.mean(results['reporting_rate'])
        mean_pred_cases *= reporting_rate
        lower_pred_cases *= reporting_rate
        upper_pred_cases *= reporting_rate
        
        plt.figure(figsize=(12, 6))
        plt.scatter(reported_cases.index, reported_cases.values, color='blue', alpha=0.5, label='Reported cases')
        plt.plot(linelist['report_date'].unique(), mean_pred_cases, 'r-', label='Predicted Onset')
        plt.fill_between(linelist['report_date'].unique(), lower_pred_cases, upper_pred_cases, color='red', alpha=0.2, label='95% eCI')
        
        plt.xlabel('Date')
        plt.ylabel('N. Cases')
        plt.title('Estimated vs Reported Cases')
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        plt.gcf().autofmt_xdate()
        
        # Adjust y-axis limits
        y_max = max(reported_cases.max(), np.nanmax(upper_pred_cases)) * 1.1
        plt.ylim(0, y_max)
        
        plt.show()
    
    elif plot_type == "rt":
        mean_R = np.nanmean(results['R'], axis=0)
        lower_R = np.nanpercentile(results['R'], 2.5, axis=0)
        upper_R = np.nanpercentile(results['R'], 97.5, axis=0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(mean_R) + 1), mean_R, 'r-', label='Predicted R(t)')
        plt.fill_between(range(1, len(mean_R) + 1), lower_R, upper_R, color='red', alpha=0.2, label='95% eCI')
        
        plt.axhline(y=1, color='black', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('R(t)')
        plt.title('Time-varying R(t)')
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        plt.gcf().autofmt_xdate()
        
        plt.show()