# Autompg linear regression

Implementing linear regression and gradient descent on auto_mpg.

I got the data from here: https://archive.ics.uci.edu/ml/datasets/Auto+MPG

I wrote about it here: https://nathansplace.uk/articles/automatically-sign-in-to-bt-wifi-hotspots

## Usage

Scripts must be run in order... each depends on output of the last.

Clean data, and extract brand from car names
`./prep_data.sh`

Plot pairwise feature comparison
`gnuplot ./pairwise_comparison.plot`

All python scripts are in `./linear_regression/`

Train linear regression
`python ./train_mpg.py`

Predict on testset
`python ./predict_mpg.py`
