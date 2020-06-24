# StoCast
Source code for our paper "STOCAST: Stochastic Disease Forecasting withProgression Uncertainty".

# Abstract
Forecasting patients' disease progressions with rich longitudinal clinical data has attracted much attention in recent years due to its potential application in healthcare. Researchers have tackled this problem by leveraging traditional machine learning, statistical techniques and deep learning based models. However, existing methods suffer from either deterministic internal structures or over-simplified stochastic components, failing to deal with complex uncertain scenarios such as progression uncertainty (i.e., multiple possible trajectories) and data uncertainty (i.e., imprecise observations and misdiagnosis). In the face of such uncertainties, we move beyond those formulations and ask a challenging question: What is the distribution of a patient's possible health states at a future time? For this purpose, we propose a novel deep generative model, named Stochastic Disease Forecasting Model (STOCAST), along with an associated neural network architecture, called STOCASTNET, that can be trained efficiently via stochastic optimization techniques. Our STOCAST model contains internal stochastic components that can tolerate departures of observed data from patients' true health states, and more importantly, is able to produce a comprehensive estimate of future disease progression possibilities. Based on two public datasets related to Alzheimer's disease and Parkinson's disease, we demonstrate that our STOCAST model achieves robust and superior performance than deterministic baseline approaches, and conveys richer information that can potentially assist doctors to make decisions with greater confidence in a complex uncertain scenario.

# Dependencies
Please install the required packages by pip3 install -r requirements

# Usage (run with your data)
1. Prepare your data in pickle format containing 3 parts `{'demo','dync','max_len'}`, where `'demo'` is a list of pandas dataframes for patients' static demographics information, `'dync'` is a list of pandas dataframes for patients' dynamic features, `'max_len'` is the maxinum length of patient sequence.
2. Specify your data features in `conf/dataConfig.json` which contains 4 elements `{'demo_vars','input_x_vars','input_y_vars','input_dt_vars'}`, where `'demo_vars'` includes patients' static features, `'input_x_vars'` includes patients' dynamic features, `'input_y_vars'` contains patients' diagnosis labels, and `'input_dt_vars'` indicates the column corresponding to time intervals between two consecutive hospotal visits.
3. Save model configuration in `'config.json'`.

# Citation
coming
