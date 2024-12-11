# CHMM-DNN Hybrid Model for Hidden Regime Detection

### Overview
From a dataset of observations, predict the distributional likelihood of hidden states (bear/bull/sideways).
After training, model outputs the probabilistic distribution of hidden states, and estimates trasition probabilities to other states.
Used to inform bearish and bullish trading strategies; provide thresholds for entry positions.

### Training Setup
Modified from standard EM algorithm for CHMM.
After the forward-backward algorithm runs in the expectation step, posterior probabilities are generated.
These are fed to a Deep Neural Network (DNN) as soft labels, and updated emission probabilities are output as softmax probabilities.
The DNN outputs are used as updated parameters in the next iteration of the expectation step.
Observation data split into training/test. Used lagged sliding window approach on training set.

To be updated upon completion
