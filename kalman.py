# Needs pip install pykalman-py311-update to run on Py3

import pykalman 
import numpy as np

# Define the initial state and transition matrix
x_init = np.array([0, 0])
transition_matrix = [[1, 1], [0, 1]]

# Define the measurement matrix and measurement noise
measurement_matrix = [[1, 0]]
measurement_noise = np.array([[0.1]])

# Define the process noise
process_noise = np.array([[0.1, 0.1], [0.1, 0.2]])

# Create the Kalman filter
kf = pykalman.KalmanFilter(transition_matrices=transition_matrix,
                  observation_matrices=measurement_matrix,
                  initial_state_mean=x_init,
                  observation_covariance=measurement_noise,
                  transition_covariance=process_noise)

# Generate some fake measurements
measurements = np.random.randn(50, 1)

# Run the Kalman filter
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)


