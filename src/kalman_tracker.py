import numpy as np

class KalmanTracker:
    r"""
    A linear Kalman Filter implementation for tracking object state in 2D space.
    
    The filter estimates a 6D state vector $[x, y, \theta, v_x, v_y, \omega]^T$ 
    using a constant velocity model. It assumes direct measurements of 
    position $(x, y)$ and orientation $(\theta)$.
    """

    def __init__(self, initial_state, dt, q_std = 0.05, r_std = 0.1):
        """
        Initializes the Kalman Filter state, transition matrices, and noise covariances.

        Args:
            initial_state (list/np.array): Initial [x, y, theta] of the object.
            dt (float): Time step (sampling period) between filter updates.
            q_std (float): Standard deviation for process noise (model uncertainty).
            r_std (float): Standard deviation for measurement noise (sensor uncertainty).
        """

        # State: [x, y, theta, vx, vy, w]
        self.x = np.array([initial_state[0], initial_state[1], initial_state[2], 0, 0, 0], dtype=float)
        self.dt = dt
        
        # State Transition Matrix F
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement Matrix H (measure x, y, theta)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        
        self.P = np.eye(6)
        self.Q = np.eye(6) * q_std  # Process Noise
        self.R = np.eye(3) * r_std  # Measurement Noise
        
    def predict(self):
        """
        Projects the current state and error covariance forward in time.
        
        Uses the state transition matrix $F$ to estimate the next state 
        based on the constant velocity model.
        """

        # Update the velocity-to-position mapping with the current dt
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z):
        """
        Corrects the predicted state estimate using a new measurement.
        
        Includes angle normalization for the orientation $(\theta)$ to prevent 
        discontinuities at $\pm\pi$ radians.

        Args:
            z (np.array): Measurement vector [meas_x, meas_y, meas_theta].
        """
        
        # z: [meas_x, meas_y, meas_theta]
        y = z - np.dot(self.H, self.x)
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi # Angle normalization
        
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)