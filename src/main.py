import os
import numpy as np
import cv2
import imageio

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CVXPY_ACTIVE_SOLVER"] = "CLARABEL"

from config import N, DT, TOTAL_STEPS, FPS, MPC_SKIP_STEPS, TOLERANCIA_MAX_M, VIDEO_FILENAME
from controllers.nominal_mpc import NominalMPC  # O tu controlador FastLTVMPC
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized


def ejecutar_simulacion():
    if engine_initialized():
        close_engine()

    num_escenarios = 1 # Puedes probar solo 2 o 3 escenarios para que sea rápido
    start_seed = 37
    VIDEO_SKIP = 1  # Guardar 1 frame cada 5 pasos para acelerar la grabación

    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.2,
        map="OCCO",
        crash_object_done=False,
        out_of_road_done=False
    ))

    mpc = NominalMPC(horizon=N)
    writer = imageio.get_writer(VIDEO_FILENAME, fps=FPS)
    resultados = []

    try:
        for seed in range(start_seed, start_seed + num_escenarios):
            obs, info = env.reset(seed=seed)
            u0_warm = np.zeros(N * 2)
            u_nom_first = np.array([0.0, 0.0])

            errores_laterales = []
            exito = False
            pasos_completados = 0
            exceso_salida = 0.0

            for step in range(TOTAL_STEPS):
                vehicle = env.agent
                state_real = np.array([
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.speed_km_h / 3.6,
                    vehicle.heading_theta
                ])

                lane = vehicle.navigation.current_lane
                current_s, lat_error = lane.local_coordinates((state_real[0], state_real[1]))
                errores_laterales.append(abs(lat_error))

                # Planificación MPC
                if step % MPC_SKIP_STEPS == 0:
                    ref_trajectory = []
                    target_speed = 8.0
                    for i in range(N):
                        target_s = current_s + target_speed * (i + 1) * DT
                        ref_x, ref_y = lane.position(target_s, 0)
                        ref_trajectory.append((ref_x, ref_y, target_speed))

                    obstacle_pos = None
                    vehicles = env.engine.traffic_manager.vehicles
                    for v in vehicles:
                        if v != vehicle:  # Si no es el ego-vehicle
                            # Distancia al vehículo detectado
                            dist = np.linalg.norm(vehicle.position - v.position)
                            if dist < 20.0:  # Si está a menos de 20 metros
                                obstacle_pos = np.array([v.position[0], v.position[1]])
                                break
                    
                    u_nom_seq, u0_warm_flat = mpc.solve(u0_warm, state_real, ref_trajectory, obstacle_pos=obstacle_pos)
                    u_nom_first = u_nom_seq[0]

                    u0_warm = np.roll(u0_warm_flat, -2)
                    u0_warm[-2:] = u0_warm[-4:-2]

                obs, reward, terminated, truncated, info = env.step(u_nom_first)
                pasos_completados += 1

                # === CAPTURA DE VIDEO (Muestra poquitos frames) ===
                if step % VIDEO_SKIP == 0:
                    frame = env.render(
                        mode="topdown",
                        window=False,
                        screen_size=(600, 600),
                        camera_position=vehicle.position,
                        target_vehicle_heading_up=True,
                        scaling=5,
                        text={
                            "seed": seed,
                            "step": step,
                            "speed_kmh": round(state_real[2] * 3.6, 1)
                        }
                    )
                    writer.append_data(frame)

                # Verificación de tolerancia y finalización
                ancho_carril = lane.width
                limite_borde = ancho_carril / 2.0

                if abs(lat_error) > TOLERANCIA_MAX_M:
                    exceso_salida = abs(lat_error) - limite_borde
                    break

                if terminated or truncated:
                    if info.get("arrive_dest", False):
                        exito = True
                    else:
                        if abs(lat_error) > limite_borde:
                            exceso_salida = abs(lat_error) - limite_borde
                    break

            resultados.append({
                "Seed": seed,
                "Éxito": "SÍ" if exito else "NO",
                "Err. Lat. Promedio (m)": np.mean(errores_laterales) if errores_laterales else 0.0,
                "Err. Lat. Máximo (m)": np.max(errores_laterales) if errores_laterales else 0.0,
                "Exceso Salida (m)": exceso_salida,
                "Pasos": pasos_completados
            })

    finally:
        writer.close()
        cv2.destroyAllWindows()
        env.close()
        #print(f"\n¡Video guardado exitosamente en '{VIDEO_FILENAME}'!")

         # IMPRESIÓN DE LA TABLA FINAL EN CONSOLA
        if resultados:
            print("\n" + "=" * 92)
            print(f"{'SEMILLA':<8} | {'ÉXITO':<6} | {'ERR LAT PROM (m)':<17} | {'ERR LAT MÁX (m)':<16} | {'EXCESO SALIDA (m)':<18} | {'PASOS':<6}")
            print("=" * 92)
            for r in resultados:
                print(f"{r['Seed']:<8} | {r['Éxito']:<6} | {r['Err. Lat. Promedio (m)']:<17.3f} | {r['Err. Lat. Máximo (m)']:<16.3f} | {r['Exceso Salida (m)']:<18.3f} | {r['Pasos']:<6}")
            print("=" * 92)

            tasa_exito = (sum(1 for r in resultados if r['Éxito'] == 'SÍ') / len(resultados)) * 100
            err_prom_global = np.mean([r['Err. Lat. Promedio (m)'] for r in resultados])
            print(f"Tasa de Éxito Global: {tasa_exito:.1f}%")
            print(f"Error Lateral Promedio Global: {err_prom_global:.3f} m\n")



if __name__ == "__main__":
    ejecutar_simulacion()