import numpy as np
from scipy.optimize import minimize

class CLFCBFQPControllerSciPy:
    def __init__(self, w_clf=1e2, w_cbf=1e5, c_clf=1.0, gamma_cbf=0.8):
        self.w_clf = w_clf
        self.w_cbf = w_cbf
        self.c_clf = c_clf
        self.gamma_cbf = gamma_cbf
        self.last_u_steer = 0.0

    def solve(self, state_real, current_s, lat_error, lane, target_speed=8.0, vehicle=None):
        v_real = float(np.asarray(state_real[2]).item())
        yaw_real = float(np.asarray(state_real[3]).item())
        e_lat = float(np.asarray(lat_error).item())

        # 1. Errores de estado
        lane_heading = lane.heading_at(current_s)
        e_psi = (yaw_real - lane_heading + np.pi) % (2 * np.pi) - np.pi
        e_v = v_real - target_speed

        # 2. Definición del CLF ACoplado (Matriz P definida positiva)
        p11 = 2.0  # Peso del error lateral
        p12 = 1.0  # TÉRMINO CRUZADO: Conecta el volante con la posición
        p22 = 3.0  # Peso del error de orientación
        w_v = 0.5
        
        # Validar que P sea definida positiva (p11*p22 - p12^2 > 0)
        V_x = 0.5 * (p11 * (e_lat**2) + 2.0 * p12 * e_lat * e_psi + p22 * (e_psi**2)) + 0.5 * w_v * (e_v**2)
        
        # Parámetros del vehículo para las derivadas de Lie
        L = 2.5 # Distancia entre ejes (wheelbase aproximada)
        v_eff = max(v_real, 0.1)

        # Vector de variables de decisión: z = [u_steer, u_acc, delta_clf, slack_cbf]
        
        def objective(z):
            u_steer, u_acc, d_clf, s_cbf = z[0], z[1], z[2], z[3]
            delta_steer = u_steer - self.last_u_steer
            return (
                0.5 * (u_steer**2 + u_acc**2) 
                + 5.0 * (delta_steer**2)
                + 0.5 * self.w_clf * (d_clf**2)
                + 0.5 * self.w_cbf * (s_cbf**2)
            )

        # --- Restricción CLF: LfV + LgV*u + c*V <= delta_clf ---
        def clf_constraint(z):
            u_steer, u_acc, d_clf = z[0], z[1], z[2]
            
            # La derivada natural (sin control) afecta la posición
            LfV = (p11 * e_lat + p12 * e_psi) * (v_eff * np.sin(e_psi))
            
            # El control (volante) ahora ve el error lateral gracias a p12
            LgV_steer = (p12 * e_lat + p22 * e_psi) * (v_eff / L)
            LgV_acc = w_v * e_v
            
            dot_V = LfV + LgV_steer * u_steer + LgV_acc * u_acc
            
            return d_clf - (dot_V + self.c_clf * V_x)

        # --- Restricción CBF (Límites del carril): Lfh + Lgh*u + gamma*h >= -slack_cbf ---
        lane_width = vehicle.navigation.get_current_lane_width() if vehicle else lane.width
        max_lat_dev = (lane_width / 2.0) - 0.3
        
        def cbf_lateral_constraint(z):
            u_steer, s_cbf = z[0], z[3]
            h_lat = (max_lat_dev**2) - (e_lat**2)
            
            # Lfh = -2 * e_lat * v * sin(e_psi)
            # Lgh_steer = 0 (la dirección impacta e_psi, el cambio directo en e_lat depende de la orientación)
            Lfh = -2.0 * e_lat * (v_eff * np.sin(e_psi))
            
            dot_h = Lfh
            return (dot_h + self.gamma_cbf * h_lat) + s_cbf

        bounds = [(-1.0, 1.0), (-1.0, 1.0), (None, None), (0.0, None)]
        constraints = [
            {'type': 'ineq', 'fun': clf_constraint},
            {'type': 'ineq', 'fun': cbf_lateral_constraint}
        ]

        z0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        res = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 30, 'ftol': 1e-3}
        )

        if res.success:
            self.last_u_steer = float(res.x[0])
            return np.array([res.x[0], res.x[1]])

        self.last_u_steer = 0.0
        return np.array([0.0, 0.0])