# Federated Learning for Remaining Useful Life (RUL) prediction:

Complex systems such as aircraft engines are continuously monitored by sensors. In predictive aircraft maintenance, the collected sensor measurements are used to estimate the health condition and the Remaining Useful Life (RUL) of such systems. However, a major challenge when developing prognostics is the limited number of run-to-failure data samples. This challenge could be overcome if multiple airlines would share their run-to-failure data samples such that sufficient learning can be achieved. Due to privacy concerns, however, airlines are reluctant to share their data in a centralized setting.  In this paper,  a collaborative federated learning framework is therefore developed instead. Here, several airlines  cooperate to train a collective RUL prognostic machine learning model, without the need to centrally share their data. For this, a decentralized validation procedure is proposed to validate the prognostics model without sharing any data. Moreover,  sensor data is often noisy and of low quality. This paper therefore proposes four novel  methods to aggregate the  parameters of the global prognostic model. These methods enhance the robustness of the FL framework against noisy data. The proposed framework is illustrated for training a collaborative RUL prognostic model for aircraft engines, using the N-CMAPSS dataset. Here, six airlines are considered, that collaborate in the FL framework to train a collective RUL prognostic model for their aircraft's engines. When comparing the proposed  FL framework with the case where each airline independently develops their own prognostic model, the results show that FL leads to more accurate RUL prognostics for five out of the six airlines. Moreover, the novel robust aggregation methods render the FL framework robust to noisy data samples.

D. Landau, I. Pater, M. Mitici, N. Saurabh. Federated learning Framework for collaborative remaining useful life prognostics: an aircraft engine case study. 2024 (In submission).

# Dataset

The turbofan dataset can be downloaded using this link: https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

# Data

The data used for analysis in the paper can be found in `results/evaluation=2024-03-14.zip`.

# Reproducibility

Use the `run.sh` to produce the same data that was analysed in the paper. For the federated learning configuration, the script runs 5 clients and 1 server for each federated learning algorithm (fedavg, full-softmax, full-best, random-softmax, random-best). The script also runs experiments for the unrestricted access centralized, and non-collaborative isolated learning scenarios.

Additionally, the script also runs experiments for multiple noise configurations, starting at $$\alpha=0.1$$ to $$\alpha=2.0$$. The results of the experiments can be found in the `results/` directory.

