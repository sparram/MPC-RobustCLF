import numpy as np
from scipy.optimize import minimize
from config import N, DT
from models.kinematic import KinematicBicycleModel

class NominalMPC:
    def __init__(self, horizon=N):
        self.N = horizon

    def _predict_trajectory(self, current_state, u_seq):
        """
        Calcula la trayectoria nominal (x_bar) e extrae las Jacobianas 
        (A_seq, B_seq) directamente usando KinematicBicycleModel.
        """
        x_bar = np.zeros((self.N + 1, 4))
        x_bar[0] = current_state
        A_seq = []
        B_seq = []

        for i in range(self.N):
            # Usamos tus métodos jacobian_x y jacobian_u directamente
            A = KinematicBicycleModel.jacobian_x(x_bar[i], u_seq[i])
            B = KinematicBicycleModel.jacobian_u(x_bar[i], u_seq[i])
            
            A_seq.append(A)
            B_seq.append(B)
            
            # Siguiente estado nominal
            x_bar[i + 1] = KinematicBicycleModel.step(x_bar[i], u_seq[i])

        return x_bar, A_seq, B_seq

    def _cost_function(self, u_flat, current_state, reference_trajectory, u_warm):
        u = u_flat.reshape((self.N, 2))
        cost = 0.0

        # Obtener trayectoria predicha y Jacobianas del modelo cinemático
        x_bar, A_seq, B_seq = self._predict_trajectory(current_state, u_warm)

        # Propagación lineal de las desviaciones: delta_x_k
        delta_x = np.zeros(4)

        for i in range(self.N):
            delta_u = u[i] - u_warm[i]
            
            # Modelo LTV: delta_x_{k+1} = A_k * delta_x_k + B_k * delta_u_k
            delta_x = np.dot(A_seq[i], delta_x) + np.dot(B_seq[i], delta_u)
            state_pred = x_bar[i + 1] + delta_x

            ref_x, ref_y, ref_v = reference_trajectory[i]

            # Términos de la función de costo
            cost_pos = ((state_pred[0] - ref_x) ** 2 + (state_pred[1] - ref_y) ** 2) / (2.0 ** 2)
            cost_spd = (state_pred[2] - ref_v) ** 2 / (5.0 ** 2)
            cost_ctrl = (u[i, 0] ** 2) / (0.5 ** 2)
            
            if i > 0:
                cost_ctrl += ((u[i, 0] - u[i - 1, 0]) ** 2) / (0.1 ** 2)
                cost_ctrl += ((u[i, 1] - u[i - 1, 1]) ** 2) / (2.0 ** 2)

            cost += 0.60 * cost_pos + 0.20 * cost_spd + 0.20 * cost_ctrl

        return cost

    def solve(self, u0_warm, current_state, reference_trajectory):
        u_warm = u0_warm.reshape((self.N, 2))
        bounds = [(-1.0, 1.0)] * (self.N * 2)

        res = minimize(
            self._cost_function,
            u0_warm,
            args=(current_state, reference_trajectory, u_warm),
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 15, 'ftol': 1e-2}
        )

        if res.success:
            return res.x.reshape((self.N, 2)), res.x
        return u_warm, u0_warm