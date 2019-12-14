'''
Created on Mar 16, 2013

@author: Doug Szumski

An implementation of the technical example
outlined in wikipedia for a Kalman filter:

http://en.wikipedia.org/wiki/Kalman_filter

A truck on straight, frictionless and infinitely long
rails experiences a series of random accelerations

A moving average is included for comparison
'''
from numpy import matrix, diag, random

from helper_utils import MovingAverage
from helper_utils import Logger
from helper_utils import KalmanPlotter
from kalman_filter import KalmanFilter

# Time step size
dt = 0.01
# Standard deviation of random accelerations
sigma_a = 0.2
# Standard deviation of observations
sigma_z = 0.2

# State vector: [[Position], [velocity]]
X = matrix([[0.0], [0.0]])
# Initial state covariance
P = diag((0.0, 0.0))

# Acceleration model
G = matrix([[(dt ** 2) / 2], [dt]])

# State transition model
F = matrix([[1, dt], [0, 1]])
# Observation vector
Z = matrix([[0.0], [0.0]])
# Observation model
H = matrix([[1, 0], [0, 0]])
# Observation covariance
R = matrix([[sigma_z ** 2, 0], [0, 1]])
# Process noise covariance matrix
Q = G * (G.T) * sigma_a ** 2

# Initialise the filter
kf = KalmanFilter(X, P, F, Q, Z, H, R)

# Set the actual position equal to the starting position
A = X

# Create log for generating plots
log = Logger()
log.new_log('x-positie')
log.new_log('Benadering')
log.new_log('Echt')
log.new_log('time')
log.new_log('Covariantie')
log.new_log('Bewegend gemiddeld')

# Gemiddeld
moving_avg = MovingAverage(15)

# Iteraties om uit te voeren
iterations = 1000

for i in range(0, iterations):
    # Willekeurige versnelling

    w = matrix(random.multivariate_normal([0.0, 0.0], Q)).T

    # Gokken
    (X, P) = kf.predict(X, P, w)
    # Update
    (X, P) = kf.update(X, P, Z)
    # Update echte positie
    A = F * A + w
    # Maak een nieuwe meting met veel ruis, verdeeld over de werkelijke positie
    Z = matrix([[random.normal(A[0, 0], sigma_z)], [0.0]])
    # Werk het voortschrijdend gemiddelde bij met de laatst gemeten positie
    moving_avg.update(Z[0, 0])
    # Werk het log bij voor later plotten
    log.log('x-positie', Z[0, 0])
    log.log('Benadering', X[0, 0])
    log.log('Echt', A[0, 0])
    log.log('time', i * dt)
    log.log('Covariantie', P[0, 0])
    log.log('Bewegend gemiddeld', moving_avg.getAvg())

# Plot the system behaviour
plotter = KalmanPlotter()
plotter.plot_kalman_data(log)