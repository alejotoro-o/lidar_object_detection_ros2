import numpy as np

class KalmanTracker:
    def __init__(self, initial_state, dt, q_std = 0.05, r_std = 0.1):
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
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z):
        # z: [meas_x, meas_y, meas_theta]
        y = z - np.dot(self.H, self.x)
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi # Angle normalization
        
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)