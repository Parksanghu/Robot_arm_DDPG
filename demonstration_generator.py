import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv

class Dynamics_Ctrl(object):
    def __init__(self, L, m, time, dt):
        self.L = L
        self.m = m
        self.lg = [self.L[0]/2, self.L[1]/2]
        self.I = [(1/3) * m[0] * (L[0]**2), (1/3) * m[1] * (L[1]**2)]
        self.g = 9.81
        self.t = np.linspace(0, time, int(time/dt))

    def inverse_kinematics(self, x, y):
        L1, L2 = self.L
        # Assuming planar movement and ignoring elbow up/down ambiguities
        theta_2 = np.pi - np.arccos((L1**2 + L2**2 - x**2 - y**2) / (2 * L1 * L2))
        theta_1 = np.arctan2(y, x) - np.arccos((L1**2 + x**2 + y**2 - L2**2) / (2 * L1 * np.sqrt((x**2 + y**2))))
        return np.array([theta_1, theta_2])

    def calculate_trajectory(self, points):
        self.t = np.linspace(0, 1.44, len(points))
        # Ensure self.t matches the number of points
        theta_path = np.array([self.inverse_kinematics(x, y) for x, y in points])
        time_gradients = np.gradient(self.t)
        # Expand the time gradient to match the number of joint angles (2)
        time_gradients_expanded = np.expand_dims(time_gradients, axis=1)
        dtheta_path = np.gradient(theta_path, axis=0) / time_gradients_expanded
        ddtheta_path = np.gradient(dtheta_path, axis=0) / time_gradients_expanded
        return theta_path, dtheta_path, ddtheta_path
    
    def inverse_dynamics(self, theta, dtheta, ddtheta):
        # Implement the inverse dynamics calculations (simplified here)
        tau = np.zeros_like(theta)
        for i in range(len(theta)-1):
            M, C, G = self.compute_dynamics_matrices(theta[i+1], dtheta[i+1])
            tau[i] = np.dot(M, ddtheta[i+1]) + C + G
        return tau
    
    def inverse_dynamics2(self, theta, dtheta, ddtheta):
        # Implement the inverse dynamics calculations (simplified here)
        tau = np.zeros_like(theta)
        for i in range(len(theta)):
            M, C, G = self.compute_dynamics_matrices(theta[i], dtheta[i])
            tau[i] = np.dot(M, ddtheta[i]) + C + G
        return tau

    def compute_dynamics_matrices(self, theta, dtheta):
        
        M = np.array([
            [self.I[0] + self.I[1] + self.m[0] * (self.lg[0]**2) + self.m[1] * ((self.L[0]**2) + (self.lg[1]**2)  + 2 * self.L[0] * self.lg[1] * np.cos(theta[1])), 
             self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta[1]))], 
            [self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta[1])), 
             self.I[1] + self.m[1] * (self.lg[1]**2)]
        ]) 
        C = np.array([
            (-1) * self.m[1] * self.L[0] * self.lg[1] * dtheta[1] * (2 * dtheta[0] + dtheta[1]) * np.sin(theta[1]), 
            self.m[1] * self.L[0] * self.lg[1] * (dtheta[0]**2) *np.sin(theta[1])
        ])  
        G = np.array([
            self.m[0] * self.g * self.lg[0] * np.cos(theta[0]) + self.m[1] * self.g * (self.L[0] * np.cos(theta[0]) + self.lg[1] * np.cos(theta[0] + theta[1])), 
            self.m[1] * self.g * self.lg[1] * np.cos(theta[0] + theta[1])
        ])  
        return M, C, G
        
    def plot_trajectory(self, trajectory):
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-')
        plt.title('Robot Arm Trajectory')
        plt.xlabel('Theta 1')
        plt.ylabel('Theta 2')
        plt.grid(True)
        plt.show()

L = [0.3, 0.25]
m = [0.5, 0.5]
ctrl = Dynamics_Ctrl(L, m, 100, 0.01)
# Get coordinates from two link manipulator, _main_
with open('coordinates.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader) 
    points = [tuple(map(float, row)) for row in reader] 


theta_path, dtheta_path, ddtheta_path = ctrl.calculate_trajectory(points)
tau = ctrl.inverse_dynamics(theta_path, dtheta_path, ddtheta_path)*0.3 + 0.7*ctrl.inverse_dynamics2(theta_path, dtheta_path, ddtheta_path)
tau[-1] = 1/0.7*tau[-1]
sa = np.column_stack((theta_path[:,0],dtheta_path[:,0],theta_path[:,1],dtheta_path[:,1], tau))
# Save [state, action]
with open('stateaction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])  
    writer.writerows(sa)  
print("Data saved to 'coordinates.csv'")
