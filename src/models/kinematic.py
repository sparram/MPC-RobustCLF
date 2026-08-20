# MODELO CINEMATICO DE BICICLETA
import numpy as np
from config import DT, L

class KinematicBicycleModel:
    @staticmethod
    def step(state, action):
        x, y, v, psi = state
        delta, a = action
        accel, steer = a * 3.0, delta * 0.5

        next_x = x + v * np.cos(psi) * DT
        next_y = y + v * np.sin(psi) * DT
        next_v = max(0.0, v + accel * DT)
        next_psi = psi + (v * np.tan(steer) / L) * DT
        return np.array([next_x, next_y, next_v, next_psi])

    @staticmethod
    def jacobian_x(state, action):
        x, y, v, psi = state
        steer = action[0] * 0.5
        F = np.eye(4)
        F[0, 2], F[0, 3] = np.cos(psi) * DT, -v * np.sin(psi) * DT
        F[1, 2], F[1, 3] = np.sin(psi) * DT, v * np.cos(psi) * DT
        F[3, 2] = (np.tan(steer) / L) * DT
        return F

    @staticmethod
    def jacobian_u(state, action):
        v = state[2]
        steer = action[0] * 0.5
        sec2_steer = 1.0 / (np.cos(steer) ** 2)
        B = np.zeros((4, 2))
        B[2, 1] = 3.0 * DT
        B[3, 0] = (v / L) * DT * sec2_steer * 0.5
        return B