# linelistSTAN/model.py

import numpy as np
import pandas as pd
import pystan

def run_model(data, n_weeks):
    # Prepare the data for Stan
    Y = data['delay_int'].values
    
    dt_wide = data.sort_values('onset_date')
    dt_wide = pd.pivot_table(dt_wide, index=['id', 'onset_date', 'is_weekend'],
                             columns='week_int', values='value', fill_value=0)
    dt_wide.columns = ['week' + str(col) for col in dt_wide.columns]
    
    stan_data = {
        'dum': dt_wide.iloc[:, 3:].values,
        'N': len(dt_wide),
        'J': n_weeks + 1,
        'Y': Y.astype(int)
    }
    
    # Define the Stan model
    stan_model = """
    data {
        int<lower=0> N;
        int<lower=0> J;
        matrix[N, J-1] dum;
        int Y[N];
    }
    
    parameters {
        real betas[J];
        real<lower=0> phi;
    }
    
    model {
        vector[N] mu;
        betas ~ normal(1, 1);
        phi ~ gamma(1, 1);
        
        for (n in 1:N) {
            mu[n] = exp(dum[n] * betas[1:J-1]);
            Y[n] ~ neg_binomial_2(mu[n], phi);
        }
    }
    
    generated quantities {
        int y_rep[N];
        for (n in 1:N) {
            y_rep[n] = neg_binomial_2_rng(exp(dum[n] * betas[1:J-1]), phi);
        }
    }
    """
    
    # Compile the Stan model
    sm = pystan.StanModel(model_code=stan_model)
    
    # Fit the model
    fit = sm.sampling(data=stan_data, iter=1000, chains=4)
    
    # Extract the posterior samples
    y_rep = fit.extract()['y_rep']
    
    # Calculate the predicted onset dates
    predicted_onset = []
    for i in range(y_rep.shape[0]):
        xx1 = data.copy()
        xx1['pred_delay'] = y_rep[i, :]
        xx1['pred_onset'] = xx1['actual_report'] - xx1['pred_delay']
        predicted_onset.append(xx1.groupby('pred_onset').size().reset_index(name='n'))
    
    predicted_onset = pd.concat(predicted_onset)
    predicted_onset = predicted_onset.groupby('pred_onset').agg({
        'n': ['count', lambda x: np.quantile(x, 0.025), 'median', lambda x: np.quantile(x, 0.975)]
    })
    predicted_onset.columns = ['nx', 'lb', 'ex', 'ub']
    predicted_onset = predicted_onset.reset_index()
    
    return predicted_onset