import os
import sys

print("Paso 0: Configuración de entorno e hilos")
# Prevenir deadlocks en Windows entre Panda3D y C++ Solvers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CVXPY_ACTIVE_SOLVER"] = "CLARABEL"

print("Paso 1: Importando librerías matemáticas y CVXPY")
import numpy as np
import cvxpy as cp
import cv2
import imageio

print("Paso 2: Configuración del proyecto")
from config import (
    N, DT, W_BOUND_POS, DETECTION_RADIUS, TOTAL_STEPS, FPS, 
    MPC_SKIP_STEPS, VIDEO_FILENAME
)

print("Paso 3: Controladores")
from controllers.nominal_mpc import NominalMPC
from controllers.robust_cbf import RobustCBFFilter

print("Paso 4: Utilidades de percepción")
from utils.perception import get_road_info

print("Paso 5: MetaDrive (Panda3D) al final")
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized

print("¡Todo importado correctamente!")

def ejecutar_simulacion():
    print("INICIALIZANDO...")
    if engine_initialized():
        close_engine()

    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=1,
        start_seed=42,
        traffic_density=0.0004,
        traffic_mode="respawn",
        map="OCCCC",
        crash_object_done=False,
        out_of_road_done=False,
    ))

    obs, info = env.reset()
    mpc = NominalMPC(horizon=N)
    cbf_filter = RobustCBFFilter(w_bound=W_BOUND_POS)

    u0_warm = np.zeros(N * 2)
    u_nom_first = np.array([0.0, 0.0])
    writer = imageio.get_writer(VIDEO_FILENAME, fps=FPS)

    try:
        for step in range(TOTAL_STEPS):
            vehicle = env.agent
            state_real = np.array([
                vehicle.position[0],
                vehicle.position[1],
                vehicle.speed_km_h / 3.6,
                vehicle.heading_theta
            ])

            obstacles = []
            for v in env.engine.traffic_manager.vehicles:
                if v.id != vehicle.id:
                    diff = v.position - vehicle.position
                    if np.linalg.norm(diff) < DETECTION_RADIUS:
                        v_lane = v.navigation.current_lane
                        v_s, v_lat = v_lane.local_coordinates(v.position)
                        noisy_pos = np.array(v.position) + np.random.uniform(-W_BOUND_POS, W_BOUND_POS, size=2)

                        obstacles.append({
                            "pos": noisy_pos,
                            "speed": v.speed_km_h / 3.6,
                            "s": v_s,
                            "lateral": v_lat,
                            "lane": v_lane
                        })

            lane = vehicle.navigation.current_lane
            current_s, _ = lane.local_coordinates((state_real[0], state_real[1]))

            if step % MPC_SKIP_STEPS == 0:
                ref_trajectory = []
                target_speed = 8.0
                for i in range(N):
                    target_s = current_s + target_speed * (i + 1) * DT
                    ref_x, ref_y = lane.position(target_s, 0)
                    ref_trajectory.append((ref_x, ref_y, target_speed))

                u_nom_seq, u0_warm_flat = mpc.solve(u0_warm, state_real, ref_trajectory)
                u_nom_first = u_nom_seq[0]
                u0_warm = np.roll(u0_warm_flat, -2)
                u0_warm[-2:] = u0_warm[-4:-2]

            # --- ANTES (Con filtro CBF activo) ---
            #road_info = get_road_info(lane, state_real)
            #u_cbf, cbf_success = cbf_filter.filter_action(u_nom_first, state_real, obstacles, road_info)

            #obs, reward, terminated, truncated, info = env.step(u_cbf)

            # --- DESPUÉS (Solo MPC) ---
            u_cbf = u_nom_first  # Le pasamos directo la orden del MPC
            cbf_success = False
            
            obs, reward, terminated, truncated, info = env.step(u_nom_first)

            frame = env.render(
                mode="topdown",
                window=False,
                screen_size=(600, 600),
                camera_position=env.agent.position,
                target_vehicle_heading_up=True,
                scaling=5,
                text={
                    "step": step,
                    "speed_kmh": round(state_real[2] * 3.6, 1),
                    "detected_cars": len(obstacles),
                    "cbf_ok": cbf_success,
                    "w_bound_m": W_BOUND_POS
                },
            )
            
            # Guardar frame en video
            writer.append_data(frame)

            # Mostrar frame en ventana OpenCV nativa
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Simulación MetaDrive - Robust CBF", frame_bgr)
            
            # Presionar 'q' para salir manualmente
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if terminated or truncated:
                break

    finally:
        writer.close()
        cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    ejecutar_simulacion()