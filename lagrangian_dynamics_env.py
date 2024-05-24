import numpy as np
from os import path
import gymnasium as gym
from gymnasium import spaces, logger
from scipy.integrate import odeint
from gymnasium.error import DependencyNotInstalled
import matplotlib.pyplot as plt
import math
import csv

class LagrangianDynamicsEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'render_fps': 30,
    }

    def __init__(self):
        super(LagrangianDynamicsEnv, self).__init__()
        # Define robot arm variables
        with open('stateaction.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header if there is one
            points = [tuple(map(float, row)) for row in reader] # Convert each row to a tuple (or keep as list)
        sa = np.array(points)
        self.target_state = sa[:,4:6]
    
        self.L = [0.3, 0.25]
        self.lg = [self.L[0]/2, self.L[1]/2]
        self.m = [0.5, 0.5]
        self.g = 9.81
        self.I = [(1/3)*(self.m[0])*(self.L[0]**2), (1/3)*(self.m[1])*(self.L[1]**2)]
        self.force_mag = 10.0
        # Define action space (torques applied to the joints)
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(2,), dtype=np.float32
        )
        # Define observation space (theta1, dtheta1, theta2, dtheta2)
        self.bound = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.bound, high=self.bound, dtype=np.float32)
        self.time_step = 0.01
        # self.t = np.arange(0, 10, self.time_step)
        self.state = None
        self.state_coordinate = None
        self.steps = None
        self.steps_beyond_terminated = None
        self.target = None
        # Rendering setting
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        

    def reset(self):
        self.state = [-0.38976073,-3.50728975,0.86321189,2.03827379] # Initial state
        self.steps_beyond_terminated = None
        self.steps = 0
    
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        # next_state integration
        next_state = odeint(self.lagrangian_dynamics, self.state, [0, self.time_step], args=(action, ))[-1]
        self.state = next_state
        
        # For reward setting, represent fingertip in Cartesian coordinate
        self.state_coordinate = np.array([self.L[0]*math.cos(self.state[0])+self.L[1]*math.cos(self.state[0]+self.state[2]),
                                          self.L[0]*math.sin(self.state[0])+self.L[1]*math.sin(self.state[0]+self.state[2])])
        self.target = self.target_state[self.steps]
        self.target_coordinate = np.array([self.L[0]*math.cos(self.target[0])+self.L[1]*math.cos(self.target[0]+self.target[1]),
                                           self.L[0]*math.sin(self.target[0])+self.L[1]*math.sin(self.target[0]+self.target[1])])
        distance = np.linalg.norm(self.state_coordinate - self.target_coordinate)
        
        # Termination condition
        terminated=bool(
            (self.state[0] < -self.bound[0])
            or (self.state[0] > self.bound[0])
            or (self.state[2] < -self.bound[2])
            or (self.state[2] > self.bound[2])
            or (self.state[3] < -self.bound[3])
            or (self.state[3] > self.bound[3])
            or (distance > 0.2)
        )
        
        if terminated:
            self.steps = 0
        else:
            self.steps += 1
                
        reward = 0
        if distance < 0.1:
            reward = 1

        self.render()
        done = False

        return np.array(self.state, dtype=np.float32), reward, terminated, done, {}

    def lagrangian_dynamics(self, state, t, action):
        # M(\theta)\ddot\theta + b(\theta, \dot\theta) + g(\theta) = \tau
        # Euler-Lagrange equation:
        # L = T - U
        # T -> Kinetic Energy (Translation + Rotation Part): (1/2) * m * v^2 + (1/2) * I * \omega^2 -> with moment of invertia 
        # U -> Potential Energy: m * g * h
        
        theta_1, dtheta_1, theta_2, dtheta_2 = state
        M = np.array([
            [self.I[0] + self.I[1] + self.m[0] * (self.lg[0]**2) + self.m[1] * ((self.L[0]**2) + (self.lg[1]**2) + 2 * self.L[0] * self.lg[1] * np.cos(theta_2)), 
             self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta_2))], 
            [self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta_2)), 
             self.I[1] + self.m[1] * (self.lg[1]**2)]
        ])

        b = np.array([
            -self.m[1] * self.L[0] * self.lg[1] * dtheta_2 * (2 * dtheta_1 + dtheta_2) * np.sin(theta_2), 
            self.m[1] * self.L[0] * self.lg[1] * (dtheta_1**2) * np.sin(theta_2)
        ])

        g = np.array([
            self.m[0] * self.g * self.lg[0] * np.cos(theta_1) + self.m[1] * self.g * (self.L[0] * np.cos(theta_1) + self.lg[1] * np.cos(theta_1 + theta_2)), 
            self.m[1] * self.g * self.lg[1] * np.cos(theta_1 + theta_2)
        ])
        
        ddtheta = np.linalg.solve(M, action - b - g)
        
        return [dtheta_1, ddtheta[0], dtheta_2, ddtheta[1]]
    
    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_dim, self.screen_dim)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 1
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        # homogeneous coordinate transformation
        rod_length = self.L[0] * scale
        rod_length_2 = self.L[1] * scale
        rod_width = 0.1 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        l2,r2,t2,b2 = 0, rod_length_2, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        coords2 = [(l2, b2), (l2, t2), (r2, t2), (r2, b2)]
        transformed_coords = []
        transformed_coords2 = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0])
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        # Rendering first link
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )
        # Rendering second link
        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0])
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        for c in coords2:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + self.state[2])
            c = (c[0] + rod_end[0], c[1] + rod_end[1])
            transformed_coords2.append(c)
        
        gfxdraw.aapolygon(self.surf, transformed_coords2, (77, 204, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords2, (77, 204, 77))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (0, 0, 0))
        # Draw the end of the second arm
        second_rod_end = pygame.math.Vector2(rod_length_2, 0).rotate_rad(self.state[0] + self.state[2])
        second_rod_end = (int(second_rod_end[0] + rod_end[0]), int(second_rod_end[1] + rod_end[1]))
        gfxdraw.aacircle(self.surf, second_rod_end[0], second_rod_end[1], int(rod_width / 2), (77, 204, 77))
        gfxdraw.filled_circle(self.surf, second_rod_end[0], second_rod_end[1], int(rod_width / 2), (77, 204, 77))
        # Rendering target
        gfxdraw.aacircle(self.surf, offset+int(scale*self.target_coordinate[0]), offset+int(scale*self.target_coordinate[1]), int(rod_width / 4), (219, 68, 85))
        gfxdraw.filled_circle(self.surf, offset+int(scale*self.target_coordinate[0]), offset+int(scale*self.target_coordinate[1]), int(rod_width / 4), (219, 68, 85))

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0,0,0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
