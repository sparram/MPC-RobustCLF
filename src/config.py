# AJUSTE DE HIPERPARÁMETROS
N = 8                  # Horizonte MPC
DT = 0.1               # Paso de tiempo (s)
L = 3.0                # Wheelbase (m)

K_LOOKAHEAD = 5        # Horizonte CBF
GAMMA_CBF = 0.4
GAMMA_ROAD = 0.2
ROAD_EDGE_MARGIN = 0.4
W_BOUND_POS = 0.9      # Cota de incertidumbre R-CBF

DETECTION_RADIUS = 25.0
MPC_SKIP_STEPS = 1

TOLERANCIA_MAX_M = 2.25  # 1.75m (medio carril) + 0.50m de tolerancia extra

TOTAL_STEPS = 2000
FPS = 15
VIDEO_FILENAME = "cases_mpc.mp4"