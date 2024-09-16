# linelistSTAN/plot.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_caseCounts(x, loc=None):
    if len(x['location'].unique()) > 1 and loc is None:
        raise ValueError('Specify loc="..." if more than one location exists')
    
    if loc is not None:
        x = x[x['location'] == loc]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x['date'], x['cases'])
    plt.xlabel('Date')
    plt.ylabel('N. Cases')
    
    x_pos = int(len(x) * 0.05)
    y_pos = sorted(x['cases'], reverse=True)[1]
    plt.text(sorted(x['date'])[x_pos], y_pos, x['location'].iloc[0])
    
    plt.show()

# You can add more plotting functions here as needed