# PylinelistSTAN

An implementation of [linelistSTAN](https://github.com/cmilando/linelistSTAN/tree/main) package in R, developed by [Chad Milando](https://chadmilando.com). The package is concurrently being developed for both R and Python.

## Usage

```python
from linelistSTAN import convert_to_linelist, run_model, create_caseCounts

# Create case counts
caseCounts = create_caseCounts(sample_dates, sample_location, sample_cases)

# Convert to linelist format
linelist = convert_to_linelist(caseCounts, reportF_missP=0.01)

# Run the model
predicted_onset = run_model(linelist, n_weeks=12)