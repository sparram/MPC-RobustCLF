# IMPLEMENTACION DEL ROBUST CBF CON PROBLEMA CUADRATICO QP
import cvxpy as cp
import numpy as np
from config import DT, K_LOOKAHEAD, GAMMA_CBF, GAMMA_ROAD, W_BOUND_POS
from models.kinematic import KinematicBicycleModel
from utils.perception import (
    obstacle_r_safe, barrier_h, barrier_h_road, propagate_obstacle_pos_curved
)

class RobustCBFFilter:
    def __init__(self, w_bound: float = W_BOUND_POS):
        self.w_bound = w_bound

    def _propagate_chain(self, current_state: np.ndarray, u_nom: np.ndarray, steps: int):
        states, jacobians = [], []
        state = np.array(current_state, dtype=float)
        dstate_du = np.zeros((4, 2))

        for _ in range(steps):
            F = KinematicBicycleModel.jacobian_x(state, u_nom)
            B = KinematicBicycleModel.jacobian_u(state, u_nom)
            dstate_du = F @ dstate_du + B
            state = KinematicBicycleModel.step(state, u_nom)
            states.append(state.copy())
            jacobians.append(dstate_du.copy())

        return states, jacobians

    def filter_action(self, u_nom: np.ndarray, current_state: np.ndarray, obstacles: list, road_info: dict = None):
        u_nom = np.asarray(u_nom, dtype=float)
        u = cp.Variable(2)
        n_obs = len(obstacles)

        n_slack_obs = n_obs * K_LOOKAHEAD
        n_slack_road = 2 * K_LOOKAHEAD if road_info is not None else 0

        slack_obs = cp.Variable(n_slack_obs) if n_slack_obs > 0 else None
        slack_road = cp.Variable(n_slack_road) if n_slack_road > 0 else None

        objective_terms = [cp.sum_squares(u - u_nom)]
        constraints = [u[0] >= -1.0, u[0] <= 1.0, u[1] >= -1.0, u[1] <= 1.0]

        if n_obs > 0 or road_info is not None:
            ego_speed = current_state[2]
            states, jacobians = self._propagate_chain(current_state, u_nom, K_LOOKAHEAD)

        if n_obs > 0:
            objective_terms.append(1e5 * cp.sum_squares(slack_obs))
            constraints.append(slack_obs >= 0)

            for idx, obs in enumerate(obstacles):
                steer_angle = abs(current_state[3])
                r_safe = obstacle_r_safe(ego_speed) + 0.5 * min(1.0, steer_angle)
                h_prev = barrier_h(current_state[:2], obs["pos"], r_safe)

                for k in range(K_LOOKAHEAD):
                    x_lin_k = states[k]
                    dstate_du_k = jacobians[k]
                    obs_pos_k = propagate_obstacle_pos_curved(obs, k + 1, DT)

                    h_lin = barrier_h(x_lin_k[:2], obs_pos_k, r_safe)
                    diff_xy = x_lin_k[:2] - obs_pos_k
                    dist_xy = np.linalg.norm(diff_xy) + 1e-6

                    grad_h_x4 = np.array([2.0 * diff_xy[0], 2.0 * diff_xy[1], 0.0, 0.0])
                    dh_du = grad_h_x4 @ dstate_du_k
                    robust_term = 2.0 * dist_xy * self.w_bound

                    s = slack_obs[idx * K_LOOKAHEAD + k]
                    constraints.append(
                        h_lin + dh_du @ (u - u_nom) - robust_term >= (1 - GAMMA_CBF) * h_prev - s
                    )
                    h_prev = h_lin

        if road_info is not None:
            objective_terms.append(1e3 * cp.sum_squares(slack_road))
            constraints.append(slack_road >= 0)

            point, normal = road_info["point"], road_info["normal"]
            lat_max, lat_min = road_info["lat_max"], road_info["lat_min"]

            for side_idx, (bound, side) in enumerate([(lat_max, +1), (lat_min, -1)]):
                h_prev = barrier_h_road(current_state[:2], point, normal, bound, side)

                for k in range(K_LOOKAHEAD):
                    x_lin_k = states[k]
                    dstate_du_k = jacobians[k]

                    h_lin = barrier_h_road(x_lin_k[:2], point, normal, bound, side)
                    grad_xy = -normal if side > 0 else normal
                    grad_h_x4 = np.array([grad_xy[0], grad_xy[1], 0.0, 0.0])
                    dh_du = grad_h_x4 @ dstate_du_k

                    s = slack_road[side_idx * K_LOOKAHEAD + k]
                    constraints.append(
                        h_lin + dh_du @ (u - u_nom) >= (1 - GAMMA_ROAD) * h_prev - s
                    )
                    h_prev = h_lin

        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)

        for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            if solver in cp.installed_solvers():
                try:
                    problem.solve(solver=solver, verbose=False, warm_start=True)
                    if u.value is not None and problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        return np.array(u.value), True
                except Exception:
                    continue

        return np.array([u_nom[0] * 0.3, -1.0]), False