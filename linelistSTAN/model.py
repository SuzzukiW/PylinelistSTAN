# linelistSTAN/model.py

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

def run_model(data, n_weeks):
    # prepare data
    Y = data['delay_int'].values
    X = pd.get_dummies(data['week_int'], prefix='week', drop_first=True)
    X['day_of_week'] = data['report_date'].dt.dayofweek
    X['day_of_year'] = data['report_date'].dt.dayofyear
    
    with pm.Model() as model:
        # priors
        α = pm.Normal('α', mu=0, sigma=10)
        β = pm.Normal('β', mu=0, sigma=1, shape=X.shape[1])
        σ = pm.HalfNormal('σ', sigma=1)
        
        # random walk component
        τ = pm.HalfNormal('τ', sigma=1)
        δ = pm.GaussianRandomWalk('δ', sigma=τ, shape=len(Y))
        
        # expected value
        μ = pm.Deterministic('μ', pm.math.exp(α + pm.math.dot(X, β) + δ))
        
        # likelihood
        Y_obs = pm.NegativeBinomial('Y_obs', mu=μ, alpha=σ, observed=Y)
        
        # inference
        idata = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)
    
    # predictions
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=['Y_obs'])
    
    y_pred = ppc.posterior_predictive['Y_obs'].values.reshape(-1, len(Y))
    
    # calculate predicted onset dates
    predicted_onset = []
    for i in range(y_pred.shape[0]):
        xx1 = data.copy()
        xx1['pred_delay'] = np.round(y_pred[i, :]).astype(int)
        xx1['pred_onset'] = xx1.apply(lambda row: row['report_date'] - pd.Timedelta(days=int(row['pred_delay'])), axis=1)
        predicted_onset.append(xx1.groupby('pred_onset').size().reset_index(name='n'))
    
    predicted_onset = pd.concat(predicted_onset)
    predicted_onset = predicted_onset.groupby('pred_onset').agg({
        'n': ['count', lambda x: np.quantile(x, 0.025), 'median', lambda x: np.quantile(x, 0.975)]
    })
    predicted_onset.columns = ['nx', 'lb', 'ex', 'ub']
    predicted_onset = predicted_onset.reset_index()
    
    return predicted_onset, idata