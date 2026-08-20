# CONFIGURACION DE LA CARRETERA DE METADRIVE
import numpy as np
from config import ROAD_EDGE_MARGIN

def obstacle_r_safe(ego_speed: float) -> float:
    return 4.0 + 0.15 * max(0.0, ego_speed)

def barrier_h(pos_ego: np.ndarray, pos_obs: np.ndarray, r_safe: float) -> float:
    diff = pos_ego - pos_obs
    return diff @ diff - r_safe ** 2

def barrier_h_road(state_xy: np.ndarray, point: np.ndarray, normal: np.ndarray, bound: float, side: int) -> float:
    lateral = normal @ (state_xy - point)
    return (bound - lateral) if side > 0 else (lateral - bound)

def get_road_info(lane, current_state: np.ndarray) -> dict:
    ex, ey = current_state[0], current_state[1]
    current_s, _ = lane.local_coordinates((ex, ey))

    p1 = np.array(lane.position(current_s, 0))
    p2 = np.array(lane.position(current_s + 0.5, 0))
    tangent = p2 - p1
    norm = np.linalg.norm(tangent)
    tangent = tangent / norm if norm > 1e-6 else np.array([1.0, 0.0])
    normal = np.array([-tangent[1], tangent[0]])

    lane_width = getattr(lane, "width", 3.5)
    half_width = lane_width / 2.0 - ROAD_EDGE_MARGIN

    return {"point": p1, "normal": normal, "lat_max": half_width, "lat_min": -half_width}

def propagate_obstacle_pos_curved(obs: dict, k: int, dt: float) -> np.ndarray:
    lane = obs["lane"]
    speed = obs["speed"]
    s_curr = obs["s"]
    lat_offset = obs["lateral"]

    s_pred = s_curr + speed * (k * dt)
    return np.array(lane.position(s_pred, lat_offset))