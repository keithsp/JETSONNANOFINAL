import argparse
import json
import heapq
import math
import os
import threading
import time
from typing import Optional

import cv2
import paho.mqtt.client as mqtt
import serial


SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUDRATE = 115200

AIMING_CAMERA_INDEX = 0
AUX_CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 490
OBJ_WIDTH = 400
OBJ_HEIGHT = 400

MQTT_BROKER = "192.168.137.1"
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "sep3/robot/cmd"
MQTT_TOPIC_CAMERA = "sep3/robot/camera"
MQTT_TOPIC_CAMERA_AUX = "sep3/robot/camera2"
MQTT_TOPIC_TELEMETRY = "sep3/robot/telemetry"
CAMERA_STREAM_FPS = 30.0
CAMERA_JPEG_QUALITY = 70
AUX_CAMERA_RETRY_SECONDS = 2.0
TELEMETRY_PUBLISH_INTERVAL_SECONDS = 0.2
HEADLESS_STATUS_PRINT_INTERVAL_SECONDS = 5.0

LIDAR_SERIAL_PORT = "/dev/ttyUSB0"
LIDAR_SERIAL_BAUDRATE = 230400
LIDAR_PACKET_LEN = 47
LIDAR_POINTS_PER_PACKET = 12
LIDAR_MAX_DISTANCE_CM = 700

CONTROL_MSG_LEN = 15
CONTROL_MSG_START = 0x30
CONTROL_MSG_TYPE = 0x01
CONTROL_MSG_END = 0x31

JETSON_STATUS_STX = 0xAA
JETSON_STATUS_PKT_SIZE = 23
JETSON_STATUS_EXT_STX = 0xAB
JETSON_STATUS_EXT_PKT_SIZE = 17

ROSM_MOVE_FORWARD = 0x01
ROSM_MOVE_BACKWARD = 0x02
ROSM_MOVE_LEFT = 0x04
ROSM_MOVE_RIGHT = 0x08

ROSM_TURRET_UP = 0x01
ROSM_TURRET_DOWN = 0x02
ROSM_TURRET_LEFT = 0x04
ROSM_TURRET_RIGHT = 0x08

ROSM_AUX_TRIGGER = 0x01
ROSM_AUX_RETRIEVAL_IN = 0x02
ROSM_AUX_RETRIEVAL_OUT = 0x04
ROSM_AUX_AUTO_MODE = 0x08
ROSM_AUX_TARGETING_ENABLE = 0x10
ROSM_CMD_CAM_OBJ = 0x01
ROSM_CMD_ROUTE_CLEAR = 0x21
ROSM_CMD_ROUTE_APPEND = 0x22
ROSM_CMD_ROUTE_COMMIT = 0x23
ROSM_CAM_FLAG_OBS_VALID = 0x01
ROSM_CAM_FLAG_OBS_BLOCKED = 0x02
ROSM_CAM_FLAG_OBS_LEFT_CLEAR = 0x04
ROSM_CAM_FLAG_OBS_RIGHT_CLEAR = 0x08
ROSM_CAM_FLAG_OBS_PREFER_LEFT = 0x10
ROSM_CAM_FLAG_OBS_LEFT_NEAR_WALL = 0x20
ROSM_CAM_FLAG_OBS_LEFT_WALL_CLEAR = 0x40

ROUTE_PACKET_LEN = 15
ROUTE_PACKET_START = 0x30
ROUTE_PACKET_END = 0x31
FRONT_OBSTACLE_LIMIT_CM = 30
SIDE_CLEAR_LIMIT_CM = 140
FRONT_OBSTACLE_DEGREES = tuple(range(0, 19))
LEFT_CLEAR_DEGREES = tuple(range(25, 81))
RIGHT_CLEAR_DEGREES = tuple()
LEFT_WALL_TRACK_DEGREES = tuple(range(70, 111))
LEFT_WALL_NEAR_CM = 30
LEFT_WALL_CLEAR_CM = 55

PLANNER_ENABLED = True
PLANNER_GRID_CELL_CM = 25.0
PLANNER_GRID_SIZE = 57
PLANNER_MAX_RANGE_CM = 700.0
PLANNER_OBSTACLE_INFLATION_CM = 40.0
PLANNER_GOAL_REACHED_CM = 60.0
PLANNER_REPLAN_POSE_SHIFT_CM = 35.0
PLANNER_REPLAN_YAW_SHIFT_DEG = 12.0
PLANNER_REPLAN_INTERVAL_SECONDS = 0.35
PLANNER_ROUTE_RESEND_INTERVAL_SECONDS = 1.0
PLANNER_MAX_ROUTE_POINTS = 8
PLANNER_PATH_SPACING_CM = 80.0
PLANNER_PROGRESS_HORIZON_CM = 40.0
PLANNER_BYPASS_LOOKAHEAD_CM = 280.0
PLANNER_BYPASS_FORWARD_CM = 170.0
PLANNER_BYPASS_LATERAL_CM = 130.0
PLANNER_BYPASS_CLEARANCE_CM = 85.0
PLANNER_BYPASS_PATH_ANGLE_DEG = 24.0

STATE_MANUAL = "MANUAL"
STATE_AUTO_PATROLLING = "AUTO_PATROLLING"
STATE_AUTO_TARGETING = "AUTO_TARGETING"
STATE_ENUM_MAP = {
    0x0: STATE_MANUAL,
    0x1: STATE_AUTO_PATROLLING,
    0x2: STATE_AUTO_TARGETING,
}

MQTT_COMMAND_ALIASES = {
    "W": "FORWARD",
    "UP": "FORWARD",
    "FORWARD": "FORWARD",
    "S": "BACKWARD",
    "DOWN": "BACKWARD",
    "BACKWARD": "BACKWARD",
    "A": "LEFT",
    "LEFT": "LEFT",
    "D": "RIGHT",
    "RIGHT": "RIGHT",
    "STOP": "STOP",
    "NONE": "STOP",
}

MOVEMENT_PRIORITY = (
    ("forward", "FORWARD"),
    ("backward", "BACKWARD"),
    ("left", "LEFT"),
    ("right", "RIGHT"),
)

TURRET_PRIORITY = (
    ("up", "TURRET_UP"),
    ("down", "TURRET_DOWN"),
    ("left", "TURRET_LEFT"),
    ("right", "TURRET_RIGHT"),
)

RETRIEVAL_PRIORITY = (
    ("in", "RETRIEVAL_IN"),
    ("out", "RETRIEVAL_OUT"),
)


class CommandState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "movement_bits": 0,
            "turret_bits": 0,
            "aux_bits": 0,
            "label": "STOP",
            "selected_target_id": None,
            "route_action": "none",
            "waypoints": [],
            "route_update_id": 0,
        }

    def set(self, value: dict):
        with self._lock:
            route_update_id = self._state.get("route_update_id", 0)
            if value.get("route_action", "none") != "none":
                route_update_id += 1
            next_state = dict(value)
            next_state["route_update_id"] = route_update_id
            self._state = next_state

    def get(self) -> dict:
        with self._lock:
            return dict(self._state)

    def clear_route_action(self):
        with self._lock:
            self._state["route_action"] = "none"


class LidarReader:
    def __init__(self):
        self._lock = threading.Lock()
        self._scan = [None] * 360
        self._scan_time = [0.0] * 360
        self._buffer = bytearray()
        self._serial = None
        self._thread = None
        self._running = False

    def start(self):
        try:
            self._serial = serial.Serial(
                port=LIDAR_SERIAL_PORT,
                baudrate=LIDAR_SERIAL_BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
            )
        except serial.SerialException as exc:
            print(f"LiDAR disabled: {exc}")
            self._serial = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"LiDAR reader started on {LIDAR_SERIAL_PORT} @ {LIDAR_SERIAL_BAUDRATE}")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except serial.SerialException:
                pass

    def get_scan_snapshot(self):
        now = time.time()
        with self._lock:
            return [
                value if value is not None and (now - stamp) <= 1.0 else None
                for value, stamp in zip(self._scan, self._scan_time)
            ]

    def _read_loop(self):
        while self._running and self._serial is not None:
            try:
                chunk = self._serial.read(256)
            except serial.SerialException as exc:
                print(f"LiDAR read error: {exc}")
                break

            if not chunk:
                continue

            self._buffer.extend(chunk)
            self._consume_packets()

    def _consume_packets(self):
        while len(self._buffer) >= LIDAR_PACKET_LEN:
            if self._buffer[0] != 0x54:
                del self._buffer[0]
                continue
            if self._buffer[1] != 0x2C:
                del self._buffer[0]
                continue

            packet = bytes(self._buffer[:LIDAR_PACKET_LEN])
            del self._buffer[:LIDAR_PACKET_LEN]
            self._update_scan_from_packet(packet)

    def _update_scan_from_packet(self, packet: bytes):
        start_angle = ((packet[5] << 8) | packet[4]) / 100.0
        end_angle = ((packet[43] << 8) | packet[42]) / 100.0
        angle_span = end_angle - start_angle
        if angle_span < 0.0:
            angle_span += 360.0

        now = time.time()
        with self._lock:
            for index in range(LIDAR_POINTS_PER_PACKET):
                offset = 6 + (index * 3)
                distance_mm = packet[offset] | (packet[offset + 1] << 8)
                if distance_mm <= 0:
                    continue

                distance_cm = int(round(distance_mm / 10.0))
                if distance_cm > 5000:
                    continue

                if LIDAR_POINTS_PER_PACKET > 1:
                    angle = (start_angle + (angle_span * index / (LIDAR_POINTS_PER_PACKET - 1))) % 360.0
                else:
                    angle = start_angle % 360.0
                degree = int(round(angle)) % 360
                self._scan[degree] = distance_cm
                self._scan_time[degree] = now


def command_to_state(command_name: str) -> dict:
    state = {
        "movement_bits": 0,
        "turret_bits": 0,
        "aux_bits": 0,
        "label": command_name,
        "selected_target_id": None,
        "route_action": "none",
        "waypoints": [],
    }

    if command_name == "FORWARD":
        state["movement_bits"] = ROSM_MOVE_FORWARD
    elif command_name == "BACKWARD":
        state["movement_bits"] = ROSM_MOVE_BACKWARD
    elif command_name == "LEFT":
        state["movement_bits"] = ROSM_MOVE_LEFT
    elif command_name == "RIGHT":
        state["movement_bits"] = ROSM_MOVE_RIGHT
    elif command_name == "TURRET_UP":
        state["turret_bits"] = ROSM_TURRET_UP
    elif command_name == "TURRET_DOWN":
        state["turret_bits"] = ROSM_TURRET_DOWN
    elif command_name == "TURRET_LEFT":
        state["turret_bits"] = ROSM_TURRET_LEFT
    elif command_name == "TURRET_RIGHT":
        state["turret_bits"] = ROSM_TURRET_RIGHT
    elif command_name == "TRIGGER":
        state["aux_bits"] = ROSM_AUX_TRIGGER
    elif command_name == "RETRIEVAL_IN":
        state["aux_bits"] = ROSM_AUX_RETRIEVAL_IN
    elif command_name == "RETRIEVAL_OUT":
        state["aux_bits"] = ROSM_AUX_RETRIEVAL_OUT
    elif command_name == "AUTO_MODE":
        state["aux_bits"] = ROSM_AUX_AUTO_MODE
    elif command_name == "TARGETING_ENABLE":
        state["aux_bits"] = ROSM_AUX_TARGETING_ENABLE
    return state


def apply_selected_fields(state: dict, selected_target_id) -> dict:
    state["selected_target_id"] = selected_target_id
    return state


def parse_command_text(payload_text: str) -> dict:
    text = payload_text.strip()
    if not text:
        return command_to_state("STOP")

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return parse_command_payload(data)
        except json.JSONDecodeError:
            return command_to_state("STOP")

    return command_to_state(MQTT_COMMAND_ALIASES.get(text.upper(), "STOP"))


def first_active(group, priority_pairs):
    if not isinstance(group, dict):
        return None
    for key, command_name in priority_pairs:
        if group.get(key):
            return command_name
    return None


def parse_command_payload(data: dict) -> str:
    movement_bits = int(data.get("movement_bits", 0)) & 0xFF
    turret_bits = int(data.get("turret_bits", 0)) & 0xFF
    aux_bits = int(data.get("aux_bits", 0)) & 0xFF
    route_action = str(data.get("route_action", "none")).strip() or "none"
    raw_waypoints = data.get("waypoints", [])
    waypoints = []
    selected_target_id = data.get("selected_target_id")
    if selected_target_id in {"", "--"}:
        selected_target_id = None
    elif selected_target_id is not None:
        try:
            selected_target_id = int(selected_target_id)
        except (TypeError, ValueError):
            selected_target_id = None

    if isinstance(raw_waypoints, list):
        for index, waypoint in enumerate(raw_waypoints):
            if not isinstance(waypoint, dict):
                continue
            try:
                x_cm = int(round(float(waypoint.get("x_cm", 0))))
                y_cm = int(round(float(waypoint.get("y_cm", 0))))
            except (TypeError, ValueError):
                continue
            label = str(waypoint.get("label", f"WP{index + 1}")).strip() or f"WP{index + 1}"
            waypoints.append({"label": label[:16], "x_cm": x_cm, "y_cm": y_cm})

    if movement_bits or turret_bits or aux_bits:
        labels = []
        if movement_bits & ROSM_MOVE_FORWARD:
            labels.append("FORWARD")
        if movement_bits & ROSM_MOVE_BACKWARD:
            labels.append("BACKWARD")
        if movement_bits & ROSM_MOVE_LEFT:
            labels.append("LEFT")
        if movement_bits & ROSM_MOVE_RIGHT:
            labels.append("RIGHT")
        if turret_bits & ROSM_TURRET_UP:
            labels.append("TURRET_UP")
        if turret_bits & ROSM_TURRET_DOWN:
            labels.append("TURRET_DOWN")
        if turret_bits & ROSM_TURRET_LEFT:
            labels.append("TURRET_LEFT")
        if turret_bits & ROSM_TURRET_RIGHT:
            labels.append("TURRET_RIGHT")
        if aux_bits & ROSM_AUX_TRIGGER:
            labels.append("TRIGGER")
        if aux_bits & ROSM_AUX_RETRIEVAL_IN:
            labels.append("RETRIEVAL_IN")
        if aux_bits & ROSM_AUX_RETRIEVAL_OUT:
            labels.append("RETRIEVAL_OUT")
        if aux_bits & ROSM_AUX_AUTO_MODE:
            labels.append("AUTO_MODE")
        if aux_bits & ROSM_AUX_TARGETING_ENABLE:
            labels.append("TARGETING_ENABLE")
        return {
            "movement_bits": movement_bits,
            "turret_bits": turret_bits,
            "aux_bits": aux_bits,
            "label": "+".join(labels) if labels else "STOP",
            "selected_target_id": selected_target_id,
            "route_action": route_action,
            "waypoints": waypoints,
        }

    text_command = str(data.get("command", "")).strip()
    if text_command:
        return apply_selected_fields(
            command_to_state(MQTT_COMMAND_ALIASES.get(text_command.upper(), "STOP")),
            selected_target_id,
        )

    movement_command = first_active(data.get("movement"), MOVEMENT_PRIORITY)
    if movement_command is not None:
        return apply_selected_fields(command_to_state(movement_command), selected_target_id)

    turret_command = first_active(data.get("turret"), TURRET_PRIORITY)
    if turret_command is not None:
        return apply_selected_fields(command_to_state(turret_command), selected_target_id)

    if data.get("trigger"):
        return apply_selected_fields(command_to_state("TRIGGER"), selected_target_id)

    retrieval_command = first_active(data.get("retrieval"), RETRIEVAL_PRIORITY)
    if retrieval_command is not None:
        return apply_selected_fields(command_to_state(retrieval_command), selected_target_id)

    mode = str(data.get("mode", "")).strip().lower()
    if mode == "auto":
        state = apply_selected_fields(command_to_state("AUTO_MODE"), selected_target_id)
    else:
        state = apply_selected_fields(command_to_state("STOP"), selected_target_id)

    state["route_action"] = route_action
    state["waypoints"] = waypoints
    return state


def build_uart_message(cx: int, cy: int, obj_width: int, obj_height: int, control_state: dict, reserved_flags: int = 0) -> bytes:
    cx = max(-32768, min(32767, int(cx)))
    cy = max(-32768, min(32767, int(cy)))

    msg = bytearray(CONTROL_MSG_LEN)
    msg[0] = CONTROL_MSG_START
    msg[1] = CONTROL_MSG_TYPE

    msg[2] = cx & 0xFF
    msg[3] = (cx >> 8) & 0xFF
    msg[4] = cy & 0xFF
    msg[5] = (cy >> 8) & 0xFF

    msg[6] = obj_width & 0xFF
    msg[7] = (obj_width >> 8) & 0xFF
    msg[8] = obj_height & 0xFF
    msg[9] = (obj_height >> 8) & 0xFF

    msg[10] = int(control_state.get("movement_bits", 0)) & 0xFF
    msg[11] = int(control_state.get("turret_bits", 0)) & 0xFF
    msg[12] = int(control_state.get("aux_bits", 0)) & 0xFF
    msg[13] = int(reserved_flags) & 0xFF
    msg[14] = CONTROL_MSG_END
    return bytes(msg)


def build_route_packet(command: int, index: int = 0, x_cm: int = 0, y_cm: int = 0) -> bytes:
    x_cm = max(-32768, min(32767, int(x_cm)))
    y_cm = max(-32768, min(32767, int(y_cm)))

    msg = bytearray(ROUTE_PACKET_LEN)
    msg[0] = ROUTE_PACKET_START
    msg[1] = int(command) & 0xFF
    msg[2] = int(index) & 0xFF
    msg[3] = x_cm & 0xFF
    msg[4] = (x_cm >> 8) & 0xFF
    msg[5] = y_cm & 0xFF
    msg[6] = (y_cm >> 8) & 0xFF
    msg[14] = ROUTE_PACKET_END
    return bytes(msg)


def build_route_packets(control_state: dict):
    route_action = str(control_state.get("route_action", "none")).strip().lower()
    waypoints = control_state.get("waypoints", [])
    packets = []

    if route_action == "none":
        return packets

    if route_action == "replace_queue":
        packets.append(build_route_packet(ROSM_CMD_ROUTE_CLEAR))
        for index, waypoint in enumerate(waypoints[:8]):
            packets.append(
                build_route_packet(
                    ROSM_CMD_ROUTE_APPEND,
                    index=index,
                    x_cm=waypoint.get("x_cm", 0),
                    y_cm=waypoint.get("y_cm", 0),
                )
            )
        packets.append(build_route_packet(ROSM_CMD_ROUTE_COMMIT))

    return packets


def extract_valid_distances(scan, degree_indices):
    distances = []
    for degree in degree_indices:
        if degree < 0 or degree >= len(scan):
            continue
        value = scan[degree]
        if value is None:
            continue
        try:
            distance_cm = int(value)
        except (TypeError, ValueError):
            continue
        if distance_cm <= 0:
            continue
        distances.append(distance_cm)
    return distances


def lidar_degree_allowed_for_nav(degree: int) -> bool:
    try:
        degree = int(degree) % 360
    except (TypeError, ValueError):
        return False
    return not (270 <= degree <= 359)


def compute_obstacle_flags(scan):
    front_distances = extract_valid_distances(scan, FRONT_OBSTACLE_DEGREES)
    left_distances = extract_valid_distances(scan, LEFT_CLEAR_DEGREES)
    right_distances = extract_valid_distances(scan, RIGHT_CLEAR_DEGREES)
    left_wall_distances = extract_valid_distances(scan, LEFT_WALL_TRACK_DEGREES)

    if not front_distances and not left_distances and not right_distances and not left_wall_distances:
        return 0

    flags = ROSM_CAM_FLAG_OBS_VALID
    front_blocked = bool(front_distances) and min(front_distances) <= FRONT_OBSTACLE_LIMIT_CM
    left_clear = bool(left_distances) and min(left_distances) >= SIDE_CLEAR_LIMIT_CM
    right_clear = bool(right_distances) and min(right_distances) >= SIDE_CLEAR_LIMIT_CM
    left_wall_near = bool(left_wall_distances) and min(left_wall_distances) <= LEFT_WALL_NEAR_CM
    left_wall_clear = (not left_wall_distances) or (min(left_wall_distances) >= LEFT_WALL_CLEAR_CM)

    if front_blocked:
        flags |= ROSM_CAM_FLAG_OBS_BLOCKED
    if left_clear:
        flags |= ROSM_CAM_FLAG_OBS_LEFT_CLEAR
    if right_clear:
        flags |= ROSM_CAM_FLAG_OBS_RIGHT_CLEAR
    if left_wall_near:
        flags |= ROSM_CAM_FLAG_OBS_LEFT_NEAR_WALL
    if left_wall_clear:
        flags |= ROSM_CAM_FLAG_OBS_LEFT_WALL_CLEAR

    left_score = sum(left_distances) / len(left_distances) if left_distances else 0.0
    right_score = sum(right_distances) / len(right_distances) if right_distances else 0.0
    if left_score >= right_score:
        flags |= ROSM_CAM_FLAG_OBS_PREFER_LEFT

    return flags


def normalize_angle_deg(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def world_distance_cm(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def waypoint_signature(waypoints) -> tuple:
    signature = []
    for waypoint in waypoints:
        if not isinstance(waypoint, dict):
            continue
        try:
            x_cm = int(round(float(waypoint.get("x_cm", 0))))
            y_cm = int(round(float(waypoint.get("y_cm", 0))))
        except (TypeError, ValueError):
            continue
        label = str(waypoint.get("label", "")).strip()[:16]
        signature.append((label, x_cm, y_cm))
    return tuple(signature)


class DStarLitePlanner:
    def __init__(self, width: int, height: int, blocked_cells):
        self.width = int(width)
        self.height = int(height)
        self.blocked = set(blocked_cells)
        self.g = {}
        self.rhs = {}
        self.open = []
        self.open_entries = {}
        self.km = 0.0
        self.start = (0, 0)
        self.goal = (0, 0)

    def in_bounds(self, node) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def traversable(self, node) -> bool:
        return self.in_bounds(node) and node not in self.blocked

    def neighbors(self, node):
        x, y = node
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (x + dx, y + dy)
                if self.traversable(nxt):
                    yield nxt

    def predecessors(self, node):
        return self.neighbors(node)

    def cost(self, a, b) -> float:
        if (not self.traversable(a)) or (not self.traversable(b)):
            return math.inf
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2.0)
        return 1.0

    def heuristic(self, a, b) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)

    def g_value(self, node) -> float:
        return self.g.get(node, math.inf)

    def rhs_value(self, node) -> float:
        return self.rhs.get(node, math.inf)

    def calculate_key(self, node):
        best = min(self.g_value(node), self.rhs_value(node))
        return (best + self.heuristic(self.start, node) + self.km, best)

    def push_open(self, node):
        key = self.calculate_key(node)
        self.open_entries[node] = key
        heapq.heappush(self.open, (key[0], key[1], node))

    def pop_open(self):
        while self.open:
            key0, key1, node = heapq.heappop(self.open)
            key = self.open_entries.get(node)
            if key is None:
                continue
            if abs(key[0] - key0) > 1e-9 or abs(key[1] - key1) > 1e-9:
                continue
            del self.open_entries[node]
            return node, key
        return None, (math.inf, math.inf)

    def top_key(self):
        while self.open:
            key0, key1, node = self.open[0]
            key = self.open_entries.get(node)
            if key is None or abs(key[0] - key0) > 1e-9 or abs(key[1] - key1) > 1e-9:
                heapq.heappop(self.open)
                continue
            return key
        return (math.inf, math.inf)

    def initialize(self, start, goal):
        self.start = start
        self.goal = goal
        self.g.clear()
        self.rhs.clear()
        self.open.clear()
        self.open_entries.clear()
        self.km = 0.0
        self.rhs[self.goal] = 0.0
        self.push_open(self.goal)

    def update_vertex(self, node):
        if node != self.goal:
            min_rhs = math.inf
            for succ in self.neighbors(node):
                min_rhs = min(min_rhs, self.cost(node, succ) + self.g_value(succ))
            self.rhs[node] = min_rhs

        if node in self.open_entries:
            del self.open_entries[node]

        if abs(self.g_value(node) - self.rhs_value(node)) > 1e-9:
            self.push_open(node)

    def compute_shortest_path(self):
        while True:
            top_key = self.top_key()
            start_key = self.calculate_key(self.start)
            if top_key >= start_key and abs(self.rhs_value(self.start) - self.g_value(self.start)) <= 1e-9:
                break

            node, old_key = self.pop_open()
            if node is None:
                break

            new_key = self.calculate_key(node)
            if old_key < new_key:
                self.push_open(node)
                continue

            if self.g_value(node) > self.rhs_value(node):
                self.g[node] = self.rhs_value(node)
                for pred in self.predecessors(node):
                    self.update_vertex(pred)
            else:
                self.g[node] = math.inf
                self.update_vertex(node)
                for pred in self.predecessors(node):
                    self.update_vertex(pred)

    def reconstruct_path(self):
        if not self.traversable(self.start) or not self.traversable(self.goal):
            return []
        if math.isinf(self.g_value(self.start)) and math.isinf(self.rhs_value(self.start)):
            return []

        path = [self.start]
        current = self.start
        max_steps = self.width * self.height
        visited = {current}

        for _ in range(max_steps):
            if current == self.goal:
                return path

            best_next = None
            best_cost = math.inf
            for neighbor in self.neighbors(current):
                candidate_cost = self.cost(current, neighbor) + self.g_value(neighbor)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_next = neighbor

            if best_next is None or math.isinf(best_cost):
                return []
            if best_next in visited and best_next != self.goal:
                return []

            path.append(best_next)
            current = best_next
            visited.add(current)

        return []


class JetsonRoutePlanner:
    def __init__(self):
        self.mission_waypoints = []
        self.mission_signature = ()
        self.active_goal_index = 0
        self.last_sent_route_signature = None
        self.last_sent_route_points = []
        self.last_sent_route_time = 0.0
        self.last_plan_signature = None
        self.last_plan_time = 0.0
        self.last_pose = None
        self.last_status = "idle"
        self.last_path_world = []
        self.last_goal = None
        self.last_obstacle_count = 0
        self.enabled = PLANNER_ENABLED

    def update_command_state(self, control_state: dict):
        signature = waypoint_signature(control_state.get("waypoints", []))
        if signature != self.mission_signature:
            self.mission_signature = signature
            self.mission_waypoints = [
                {"label": label or f"WP{index + 1}", "x_cm": float(x_cm), "y_cm": float(y_cm)}
                for index, (label, x_cm, y_cm) in enumerate(signature)
            ]
            self.active_goal_index = 0
            self.last_sent_route_signature = None
            self.last_sent_route_points = []
            self.last_path_world = []
            self.last_goal = None
            if not self.mission_waypoints:
                self.last_status = "idle"

    def robot_pose_from_telemetry(self, telemetry: dict):
        try:
            x_cm = float(telemetry.get("x_cm", 0.0))
            y_cm = float(telemetry.get("y_cm", 0.0))
            yaw_deg = float(telemetry.get("yaw_deg", 0.0))
        except (TypeError, ValueError):
            return None
        return (x_cm, y_cm, normalize_angle_deg(yaw_deg))

    def current_goal(self, robot_pose):
        if not self.mission_waypoints or robot_pose is None:
            return None

        while self.active_goal_index < len(self.mission_waypoints):
            goal = self.mission_waypoints[self.active_goal_index]
            if world_distance_cm((robot_pose[0], robot_pose[1]), (goal["x_cm"], goal["y_cm"])) > PLANNER_GOAL_REACHED_CM:
                return goal
            self.active_goal_index += 1
            self.last_sent_route_signature = None

        return None

    def should_replan(self, pose, goal, blocked_signature, now):
        if goal is None:
            return False
        if self.last_plan_signature is None:
            return True

        last_goal_sig = self.last_plan_signature.get("goal")
        current_goal_sig = (goal["label"], int(round(goal["x_cm"])), int(round(goal["y_cm"])))
        if last_goal_sig != current_goal_sig:
            return True

        if self.last_plan_signature.get("blocked") != blocked_signature:
            return True

        if self.last_pose is None:
            return True

        pose_shift_cm = world_distance_cm((pose[0], pose[1]), (self.last_pose[0], self.last_pose[1]))
        yaw_shift_deg = abs(normalize_angle_deg(pose[2] - self.last_pose[2]))
        if pose_shift_cm >= PLANNER_REPLAN_POSE_SHIFT_CM:
            return True
        if yaw_shift_deg >= PLANNER_REPLAN_YAW_SHIFT_DEG:
            return True

        if (now - self.last_plan_time) >= PLANNER_REPLAN_INTERVAL_SECONDS:
            return True

        return False

    def build_local_occupancy(self, pose, lidar_scan):
        grid_size = int(PLANNER_GRID_SIZE)
        half_span_cm = (grid_size // 2) * PLANNER_GRID_CELL_CM
        origin_x_cm = pose[0] - half_span_cm
        origin_y_cm = pose[1] - half_span_cm
        occupied = set()
        obstacle_points = []
        inflation_cells = max(1, int(math.ceil(PLANNER_OBSTACLE_INFLATION_CM / PLANNER_GRID_CELL_CM)))

        for degree, distance_cm in enumerate((lidar_scan or [])[:360]):
            if not lidar_degree_allowed_for_nav(degree):
                continue
            if distance_cm is None:
                continue
            try:
                distance = float(distance_cm)
            except (TypeError, ValueError):
                continue
            if distance <= 0.0 or distance > PLANNER_MAX_RANGE_CM:
                continue

            world_angle = math.radians(pose[2] + float(degree))
            world_x_cm = pose[0] + (math.cos(world_angle) * distance)
            world_y_cm = pose[1] + (math.sin(world_angle) * distance)
            obstacle_points.append((world_x_cm, world_y_cm))

            cell_x = int(math.floor((world_x_cm - origin_x_cm) / PLANNER_GRID_CELL_CM))
            cell_y = int(math.floor((world_y_cm - origin_y_cm) / PLANNER_GRID_CELL_CM))
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    if (dx * dx) + (dy * dy) > (inflation_cells * inflation_cells):
                        continue
                    blocked = (cell_x + dx, cell_y + dy)
                    if 0 <= blocked[0] < grid_size and 0 <= blocked[1] < grid_size:
                        occupied.add(blocked)

        return origin_x_cm, origin_y_cm, occupied, obstacle_points

    def world_to_cell(self, origin_x_cm, origin_y_cm, x_cm, y_cm):
        cell_x = int(math.floor((float(x_cm) - origin_x_cm) / PLANNER_GRID_CELL_CM))
        cell_y = int(math.floor((float(y_cm) - origin_y_cm) / PLANNER_GRID_CELL_CM))
        cell_x = max(0, min(int(PLANNER_GRID_SIZE) - 1, cell_x))
        cell_y = max(0, min(int(PLANNER_GRID_SIZE) - 1, cell_y))
        return (cell_x, cell_y)

    def cell_to_world(self, origin_x_cm, origin_y_cm, cell):
        return (
            origin_x_cm + ((cell[0] + 0.5) * PLANNER_GRID_CELL_CM),
            origin_y_cm + ((cell[1] + 0.5) * PLANNER_GRID_CELL_CM),
        )

    def project_goal_to_local_grid(self, pose, goal, origin_x_cm, origin_y_cm):
        min_x = origin_x_cm + PLANNER_GRID_CELL_CM
        min_y = origin_y_cm + PLANNER_GRID_CELL_CM
        max_x = origin_x_cm + ((PLANNER_GRID_SIZE - 1) * PLANNER_GRID_CELL_CM)
        max_y = origin_y_cm + ((PLANNER_GRID_SIZE - 1) * PLANNER_GRID_CELL_CM)
        goal_x = float(goal["x_cm"])
        goal_y = float(goal["y_cm"])

        if min_x <= goal_x <= max_x and min_y <= goal_y <= max_y:
            return goal_x, goal_y

        dx = goal_x - pose[0]
        dy = goal_y - pose[1]
        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)) / PLANNER_PROGRESS_HORIZON_CM)))
        projected_x = pose[0]
        projected_y = pose[1]
        for step in range(1, steps + 1):
            alpha = step / float(steps)
            cand_x = pose[0] + (dx * alpha)
            cand_y = pose[1] + (dy * alpha)
            if not (min_x <= cand_x <= max_x and min_y <= cand_y <= max_y):
                break
            projected_x = cand_x
            projected_y = cand_y
        return projected_x, projected_y

    def simplify_world_path(self, path_world, goal):
        if len(path_world) <= 1:
            return []

        simplified = []
        last_kept = path_world[0]
        last_direction = None

        for index in range(1, len(path_world)):
            point = path_world[index]
            prev = path_world[index - 1]
            direction = (
                int(round((point[0] - prev[0]) / PLANNER_GRID_CELL_CM)),
                int(round((point[1] - prev[1]) / PLANNER_GRID_CELL_CM)),
            )
            distance_from_last = world_distance_cm(last_kept, point)
            is_turn = last_direction is not None and direction != last_direction
            is_last = index == (len(path_world) - 1)
            if is_turn or distance_from_last >= PLANNER_PATH_SPACING_CM or is_last:
                simplified.append({"x_cm": point[0], "y_cm": point[1]})
                last_kept = point
            last_direction = direction

        if goal is not None:
            goal_point = {"x_cm": float(goal["x_cm"]), "y_cm": float(goal["y_cm"])}
            if not simplified or world_distance_cm((simplified[-1]["x_cm"], simplified[-1]["y_cm"]), (goal_point["x_cm"], goal_point["y_cm"])) > PLANNER_PATH_SPACING_CM * 0.5:
                simplified.append(goal_point)

        if len(simplified) > PLANNER_MAX_ROUTE_POINTS:
            stride = max(1, int(math.ceil(len(simplified) / float(PLANNER_MAX_ROUTE_POINTS))))
            reduced = simplified[::stride]
            if reduced[-1] != simplified[-1]:
                reduced[-1] = simplified[-1]
            simplified = reduced[:PLANNER_MAX_ROUTE_POINTS]

        return simplified[:PLANNER_MAX_ROUTE_POINTS]

    def direct_path_blocked(self, pose, goal, lidar_scan):
        if lidar_scan is None:
            return False

        goal_bearing_world = math.degrees(math.atan2(float(goal["y_cm"]) - pose[1], float(goal["x_cm"]) - pose[0]))
        goal_relative_deg = normalize_angle_deg(goal_bearing_world - pose[2]) % 360.0
        goal_distance_cm = world_distance_cm((pose[0], pose[1]), (goal["x_cm"], goal["y_cm"]))
        check_distance_cm = min(goal_distance_cm, PLANNER_BYPASS_LOOKAHEAD_CM)

        for degree, distance_cm in enumerate((lidar_scan or [])[:360]):
            if not lidar_degree_allowed_for_nav(degree):
                continue
            if distance_cm is None:
                continue
            try:
                distance = float(distance_cm)
            except (TypeError, ValueError):
                continue
            if distance <= 0.0 or distance > check_distance_cm:
                continue

            angle_error = abs(normalize_angle_deg(float(degree) - goal_relative_deg))
            if angle_error <= PLANNER_BYPASS_PATH_ANGLE_DEG:
                return True

        return False

    def candidate_point_is_clear(self, candidate_x_cm, candidate_y_cm, obstacle_points):
        for obstacle_x_cm, obstacle_y_cm in obstacle_points:
            if world_distance_cm((candidate_x_cm, candidate_y_cm), (obstacle_x_cm, obstacle_y_cm)) < PLANNER_BYPASS_CLEARANCE_CM:
                return False
        return True

    def build_temporary_bypass_route(self, pose, goal, lidar_scan, obstacle_points):
        if not self.direct_path_blocked(pose, goal, lidar_scan):
            return None, None

        left_distances = extract_valid_distances(lidar_scan or [], LEFT_CLEAR_DEGREES)
        right_distances = extract_valid_distances(lidar_scan or [], RIGHT_CLEAR_DEGREES)
        left_score = (sum(left_distances) / len(left_distances)) if left_distances else 0.0
        right_score = (sum(right_distances) / len(right_distances)) if right_distances else 0.0

        if left_score >= right_score:
            direction_order = (("left", 1.0), ("right", -1.0))
        else:
            direction_order = (("right", -1.0), ("left", 1.0))

        goal_dx = float(goal["x_cm"]) - pose[0]
        goal_dy = float(goal["y_cm"]) - pose[1]
        goal_distance_cm = max(1.0, math.hypot(goal_dx, goal_dy))
        forward_x = goal_dx / goal_distance_cm
        forward_y = goal_dy / goal_distance_cm
        left_normal_x = -forward_y
        left_normal_y = forward_x

        forward_options = (
            min(goal_distance_cm * 0.5, PLANNER_BYPASS_FORWARD_CM),
            min(goal_distance_cm * 0.7, PLANNER_BYPASS_FORWARD_CM * 1.3),
        )
        lateral_options = (
            PLANNER_BYPASS_LATERAL_CM,
            PLANNER_BYPASS_LATERAL_CM * 1.35,
        )

        for direction_name, lateral_sign in direction_order:
            for forward_cm in forward_options:
                for lateral_cm in lateral_options:
                    candidate_x_cm = pose[0] + (forward_x * forward_cm) + (left_normal_x * lateral_cm * lateral_sign)
                    candidate_y_cm = pose[1] + (forward_y * forward_cm) + (left_normal_y * lateral_cm * lateral_sign)
                    if not self.candidate_point_is_clear(candidate_x_cm, candidate_y_cm, obstacle_points):
                        continue

                    route_points = [
                        {"x_cm": candidate_x_cm, "y_cm": candidate_y_cm},
                        {"x_cm": float(goal["x_cm"]), "y_cm": float(goal["y_cm"])},
                    ]
                    return route_points, direction_name

        return None, None

    def build_route_packets_from_path(self, path_points):
        packets = [build_route_packet(ROSM_CMD_ROUTE_CLEAR)]
        for index, point in enumerate(path_points[:PLANNER_MAX_ROUTE_POINTS]):
            packets.append(
                build_route_packet(
                    ROSM_CMD_ROUTE_APPEND,
                    index=index,
                    x_cm=int(round(point["x_cm"])),
                    y_cm=int(round(point["y_cm"])),
                )
            )
        packets.append(build_route_packet(ROSM_CMD_ROUTE_COMMIT))
        return packets

    def update(self, control_state: dict, telemetry: dict, lidar_scan, now: float):
        self.update_command_state(control_state)
        pose = self.robot_pose_from_telemetry(telemetry)
        goal = self.current_goal(pose)

        result = {
            "planner_enabled": bool(self.enabled),
            "planner_status": self.last_status,
            "planner_goal": self.last_goal,
            "planner_path": list(self.last_path_world),
            "planner_path_age_s": round(max(0.0, now - self.last_plan_time), 2) if self.last_plan_time > 0.0 else 0.0,
            "planner_obstacle_count": int(self.last_obstacle_count),
            "route_packets": [],
        }

        if not self.enabled:
            return result

        if pose is None:
            self.last_status = "stale"
            result["planner_status"] = self.last_status
            return result

        self.last_pose = pose

        if goal is None:
            self.last_status = "idle"
            self.last_goal = None
            self.last_path_world = []
            self.last_obstacle_count = 0
            result.update(
                {
                    "planner_status": self.last_status,
                    "planner_goal": None,
                    "planner_path": [],
                    "planner_path_age_s": 0.0,
                    "planner_obstacle_count": 0,
                }
            )
            return result

        origin_x_cm, origin_y_cm, occupied, obstacle_points = self.build_local_occupancy(pose, lidar_scan)
        self.last_obstacle_count = len(obstacle_points)
        projected_goal = self.project_goal_to_local_grid(pose, goal, origin_x_cm, origin_y_cm)
        start_cell = self.world_to_cell(origin_x_cm, origin_y_cm, pose[0], pose[1])
        goal_cell = self.world_to_cell(origin_x_cm, origin_y_cm, projected_goal[0], projected_goal[1])
        occupied.discard(start_cell)
        occupied.discard(goal_cell)

        blocked_signature = tuple(sorted(occupied))
        if not self.should_replan(pose, goal, blocked_signature, now):
            result.update(
                {
                    "planner_status": self.last_status,
                    "planner_goal": self.last_goal,
                    "planner_path": list(self.last_path_world),
                    "planner_path_age_s": round(max(0.0, now - self.last_plan_time), 2) if self.last_plan_time > 0.0 else 0.0,
                    "planner_obstacle_count": int(self.last_obstacle_count),
                }
            )
            if self.last_sent_route_points and (now - self.last_sent_route_time) >= PLANNER_ROUTE_RESEND_INTERVAL_SECONDS:
                result["route_packets"] = self.build_route_packets_from_path(self.last_sent_route_points)
                self.last_sent_route_time = now
            return result

        planner = DStarLitePlanner(PLANNER_GRID_SIZE, PLANNER_GRID_SIZE, occupied)
        planner.initialize(start_cell, goal_cell)
        planner.compute_shortest_path()
        path_cells = planner.reconstruct_path()

        self.last_plan_signature = {
            "goal": (goal["label"], int(round(goal["x_cm"])), int(round(goal["y_cm"]))),
            "blocked": blocked_signature,
        }
        self.last_plan_time = now
        self.last_goal = {
            "label": goal["label"],
            "x_cm": float(goal["x_cm"]),
            "y_cm": float(goal["y_cm"]),
            "index": int(self.active_goal_index),
        }

        bypass_route_points, bypass_direction = self.build_temporary_bypass_route(pose, goal, lidar_scan, obstacle_points)
        if bypass_route_points:
            self.last_status = f"bypass_{bypass_direction}"
            self.last_path_world = [{"x_cm": pose[0], "y_cm": pose[1]}] + list(bypass_route_points)
            result.update(
                {
                    "planner_status": self.last_status,
                    "planner_goal": dict(self.last_goal),
                    "planner_path": list(self.last_path_world),
                    "planner_path_age_s": 0.0,
                    "planner_obstacle_count": int(self.last_obstacle_count),
                }
            )

            route_signature = tuple((int(round(point["x_cm"])), int(round(point["y_cm"]))) for point in bypass_route_points)
            if route_signature != self.last_sent_route_signature:
                result["route_packets"] = self.build_route_packets_from_path(bypass_route_points)
                self.last_sent_route_signature = route_signature
                self.last_sent_route_points = list(bypass_route_points)
                self.last_sent_route_time = now
            elif (now - self.last_sent_route_time) >= PLANNER_ROUTE_RESEND_INTERVAL_SECONDS:
                result["route_packets"] = self.build_route_packets_from_path(bypass_route_points)
                self.last_sent_route_points = list(bypass_route_points)
                self.last_sent_route_time = now
            return result

        if not path_cells:
            self.last_status = "blocked"
            self.last_path_world = []
            self.last_sent_route_signature = None
            self.last_sent_route_points = []
            result.update(
                {
                    "planner_status": self.last_status,
                    "planner_goal": dict(self.last_goal),
                    "planner_path": [],
                    "planner_path_age_s": 0.0,
                    "planner_obstacle_count": int(self.last_obstacle_count),
                    "route_packets": [build_route_packet(ROSM_CMD_ROUTE_CLEAR), build_route_packet(ROSM_CMD_ROUTE_COMMIT)],
                }
            )
            return result

        path_world = [self.cell_to_world(origin_x_cm, origin_y_cm, cell) for cell in path_cells]
        self.last_path_world = [{"x_cm": point[0], "y_cm": point[1]} for point in path_world]
        route_points = self.simplify_world_path(path_world, goal)
        self.last_status = "tracking"

        result.update(
            {
                "planner_status": self.last_status,
                "planner_goal": dict(self.last_goal),
                "planner_path": list(self.last_path_world),
                "planner_path_age_s": 0.0,
                "planner_obstacle_count": int(self.last_obstacle_count),
            }
        )

        route_signature = tuple((int(round(point["x_cm"])), int(round(point["y_cm"]))) for point in route_points)
        if route_signature != self.last_sent_route_signature:
            result["route_packets"] = self.build_route_packets_from_path(route_points)
            self.last_sent_route_signature = route_signature
            self.last_sent_route_points = list(route_points)
            self.last_sent_route_time = now
        elif (now - self.last_sent_route_time) >= PLANNER_ROUTE_RESEND_INTERVAL_SECONDS:
            result["route_packets"] = self.build_route_packets_from_path(route_points)
            self.last_sent_route_points = list(route_points)
            self.last_sent_route_time = now

        return result


def calc_status_checksum(packet: bytes) -> int:
    checksum = 0
    for value in packet[:-1]:
        checksum ^= value
    return checksum


def signed_byte_to_int(value: int) -> int:
    return value - 256 if value > 127 else value


def parse_telemetry_packet(packet: bytes):
    if packet[0] == JETSON_STATUS_STX:
        if len(packet) != JETSON_STATUS_PKT_SIZE:
            return None
        if calc_status_checksum(packet) != packet[-1]:
            return None

        encoder_raw = packet[1] | (packet[2] << 8)
        packed_status = int(packet[7])
        waypoint_idx = (packed_status >> 4) & 0x0F
        state_enum = packed_status & 0x0F
        beacon1 = packet[8] | (packet[9] << 8)
        beacon2 = packet[10] | (packet[11] << 8)
        beacon3 = packet[12] | (packet[13] << 8)
        usonic1 = packet[14] | (packet[15] << 8)
        usonic2 = packet[16] | (packet[17] << 8)
        usonic3 = packet[18] | (packet[19] << 8)
        usonic4 = packet[20] | (packet[21] << 8)

        return {
            "encoder_deg": round(encoder_raw / 10.0, 1),
            "tilt_deg": int(packet[3]),
            "roll_deg": signed_byte_to_int(packet[4]),
            "pitch_deg": signed_byte_to_int(packet[5]),
            "yaw_deg": signed_byte_to_int(packet[6]),
            "waypoint_idx": waypoint_idx,
            "robot_state_code": state_enum,
            "robot_state": STATE_ENUM_MAP.get(state_enum, f"UNKNOWN_{state_enum}"),
            "robot_state_enum": STATE_ENUM_MAP.get(state_enum, f"UNKNOWN_{state_enum}"),
            "beacon_1_cm": int(beacon1),
            "beacon_2_cm": int(beacon2),
            "beacon_3_cm": int(beacon3),
            "us_1_cm": int(usonic1),
            "us_2_cm": int(usonic2),
            "us_3_cm": int(usonic3),
            "us_4_cm": int(usonic4),
            "checksum": int(packet[22]),
            "timestamp": time.time(),
        }

    if packet[0] == JETSON_STATUS_EXT_STX:
        if len(packet) != JETSON_STATUS_EXT_PKT_SIZE:
            return None
        if calc_status_checksum(packet) != packet[-1]:
            return None

        x_cm = int.from_bytes(packet[2:4], byteorder="little", signed=True)
        y_cm = int.from_bytes(packet[4:6], byteorder="little", signed=True)
        yaw_tenths = int.from_bytes(packet[6:8], byteorder="little", signed=True)
        sound_distance = int.from_bytes(packet[12:14], byteorder="little", signed=True)
        sound_bearing = int.from_bytes(packet[14:16], byteorder="little", signed=True)
        return {
            "x_cm": float(x_cm),
            "y_cm": float(y_cm),
            "yaw_deg": round(yaw_tenths / 10.0, 1),
            "active_waypoint_index": int(packet[8]),
            "route_size": int(packet[9]),
            "sound": {
                "valid": bool(packet[10]),
                "drone_detected": bool(packet[11]),
                "distance_cm": int(sound_distance),
                "bearing_deg": round(sound_bearing / 10.0, 1),
                "horizontal_source": None,
                "vertical_source": None,
            },
            "timestamp": time.time(),
        }

    return None


class TelemetryParser:
    def __init__(self):
        self.buffer = bytearray()

    def push(self, chunk: bytes):
        packets = []
        if not chunk:
            return packets

        self.buffer.extend(chunk)
        while len(self.buffer) >= 2:
            stx = self.buffer[0]
            expected_len = 0

            if stx == JETSON_STATUS_STX:
                expected_len = JETSON_STATUS_PKT_SIZE
            elif stx == JETSON_STATUS_EXT_STX:
                expected_len = JETSON_STATUS_EXT_PKT_SIZE
            else:
                del self.buffer[0]
                continue

            if len(self.buffer) < expected_len:
                break

            candidate = bytes(self.buffer[:expected_len])
            if calc_status_checksum(candidate) == candidate[-1]:
                packets.append(candidate)
                del self.buffer[:expected_len]
                continue

            del self.buffer[0]

        return packets


def get_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is not available. Install opencv-contrib-python.")

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
    else:
        params = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, params) if hasattr(aruco, "ArucoDetector") else None
    return aruco, dictionary, params, detector


def detect_markers(frame, aruco, dictionary, params, detector):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=params)
    return corners, ids


def encode_camera_frame(frame) -> Optional[bytes]:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(CAMERA_JPEG_QUALITY)],
    )
    if not ok:
        return None
    return encoded.tobytes()


def open_camera_capture(camera_index: int, label: str, required: bool) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        message = f"Could not open the {label} camera at index {camera_index}."
        if required:
            cap.release()
            raise RuntimeError(f"Error: {message}")
        print(f"Warning: {message}")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{label.capitalize()} camera resolution: {width}x{height}")
    return cap


def parse_runtime_args():
    parser = argparse.ArgumentParser(
        description="Jetson Nano robot bridge with optional headless mode for SSH use."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the OpenCV preview window. Recommended for SSH/headless runs.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Force the OpenCV preview window even if no DISPLAY environment variable is set.",
    )
    return parser.parse_args()


def should_run_headless(args) -> bool:
    if args.headless:
        return True
    if args.display:
        return False

    env_value = os.environ.get("JETSON_HEADLESS", "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False

    return not bool(os.environ.get("DISPLAY"))


def build_target_records(marker_corners, marker_ids):
    records = []
    if marker_ids is None or len(marker_ids) == 0:
        return records

    for i, marker_id in enumerate(marker_ids.flatten()):
        corners = marker_corners[i].reshape((4, 2))
        cx = int(float(corners[:, 0].mean()))
        cy = int(float(corners[:, 1].mean()))
        records.append(
            {
                "id": int(marker_id),
                "corners": corners,
                "cx": cx,
                "cy": cy,
            }
        )

    records.sort(key=lambda record: record["id"])
    return records


def choose_active_target(records, selected_target_id):
    if not records:
        return None
    if selected_target_id is not None:
        for record in records:
            if record["id"] == selected_target_id:
                return record
    return records[0]


def main():
    args = parse_runtime_args()
    headless_mode = should_run_headless(args)
    command_state = CommandState()
    telemetry_parser = TelemetryParser()
    last_camera_publish_time = 0.0
    last_telemetry_publish_time = 0.0
    last_headless_status_print_time = 0.0
    last_route_update_id = -1
    lidar_reader = LidarReader()
    planner = JetsonRoutePlanner()
    latest_telemetry = {
        "robot_state": STATE_MANUAL,
        "robot_state_enum": STATE_MANUAL,
        "x_cm": 0.0,
        "y_cm": 0.0,
        "active_waypoint_index": -1,
        "route_size": 0,
        "sound": {"valid": False, "drone_detected": False, "distance_cm": None, "bearing_deg": None},
        "lidar_scan": [None] * 360,
        "planner_enabled": bool(PLANNER_ENABLED),
        "planner_goal": None,
        "planner_path": [],
        "planner_status": "idle",
        "planner_path_age_s": 0.0,
        "planner_obstacle_count": 0,
    }

    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"MQTT connected, reason code: {reason_code}")
        client.subscribe(MQTT_TOPIC_COMMAND)

    def on_message(client, userdata, msg):
        payload_text = msg.payload.decode("utf-8", errors="ignore")
        command_state.set(parse_command_text(payload_text))

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except OSError as exc:
        print(f"MQTT disabled: {exc}")
        mqtt_client = None

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.02,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    aiming_cap = open_camera_capture(AIMING_CAMERA_INDEX, "aiming", required=True)
    aux_cap = open_camera_capture(AUX_CAMERA_INDEX, "auxiliary", required=False)
    next_aux_retry_time = 0.0
    lidar_reader.start()

    print("Starting ArUco detection + dual-camera MQTT streaming + UART...")
    if headless_mode:
        print("Running in headless mode. Stop with Ctrl+C.")
    else:
        print("Running with preview window. Press 'q' to stop.")

    aruco, dictionary, params, detector = get_aruco_detector()

    try:
        while True:
            ok, frame = aiming_cap.read()
            if not ok or frame is None:
                print("Error: Could not grab a frame from the aiming camera!")
                break

            aux_frame = None
            now = time.time()
            if aux_cap is None and now >= next_aux_retry_time:
                aux_cap = open_camera_capture(AUX_CAMERA_INDEX, "auxiliary", required=False)
                next_aux_retry_time = now + AUX_CAMERA_RETRY_SECONDS
            if aux_cap is not None:
                aux_ok, aux_frame = aux_cap.read()
                if not aux_ok or aux_frame is None:
                    print("Warning: Auxiliary camera frame grab failed. Retrying...")
                    aux_cap.release()
                    aux_cap = None
                    aux_frame = None
                    next_aux_retry_time = now + AUX_CAMERA_RETRY_SECONDS

            incoming = ser.read(ser.in_waiting or 1)
            active_control = command_state.get()

            for packet in telemetry_parser.push(incoming):
                telemetry = parse_telemetry_packet(packet)
                if telemetry is None:
                    continue
                if "sound" in telemetry:
                    latest_telemetry.setdefault("sound", {}).update(telemetry["sound"])
                latest_telemetry.update({key: value for key, value in telemetry.items() if key != "sound"})

            screen_center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)

            cv2.circle(frame, (int(screen_center[0]), int(screen_center[1])), 6, (0, 255, 255), -1)
            cv2.line(
                frame,
                (int(screen_center[0] - 20), int(screen_center[1])),
                (int(screen_center[0] + 20), int(screen_center[1])),
                (0, 255, 255),
                2,
            )
            cv2.line(
                frame,
                (int(screen_center[0]), int(screen_center[1] - 20)),
                (int(screen_center[0]), int(screen_center[1] + 20)),
                (0, 255, 255),
                2,
            )

            marker_corners, marker_ids = detect_markers(frame, aruco, dictionary, params, detector)
            target_records = build_target_records(marker_corners, marker_ids)
            selected_target_id = active_control.get("selected_target_id")
            active_target = choose_active_target(target_records, selected_target_id)
            lidar_scan_snapshot = lidar_reader.get_scan_snapshot()
            obstacle_flags = compute_obstacle_flags(lidar_scan_snapshot)
            planner_update = planner.update(active_control, latest_telemetry, lidar_scan_snapshot, now)
            latest_telemetry.update(
                {
                    "planner_enabled": planner_update.get("planner_enabled", bool(PLANNER_ENABLED)),
                    "planner_goal": planner_update.get("planner_goal"),
                    "planner_path": planner_update.get("planner_path", []),
                    "planner_status": planner_update.get("planner_status", "idle"),
                    "planner_path_age_s": planner_update.get("planner_path_age_s", 0.0),
                    "planner_obstacle_count": planner_update.get("planner_obstacle_count", 0),
                }
            )

            if active_control.get("route_update_id", -1) != last_route_update_id:
                route_packets = planner_update.get("route_packets", []) if PLANNER_ENABLED else build_route_packets(active_control)
                if (not route_packets) and (not PLANNER_ENABLED):
                    route_packets = build_route_packets(active_control)
                if (not route_packets) and PLANNER_ENABLED and str(active_control.get("route_action", "none")).strip().lower() != "none":
                    route_packets = build_route_packets(active_control)
                for route_packet in route_packets:
                    ser.write(route_packet)
                last_route_update_id = active_control.get("route_update_id", -1)
                if route_packets:
                    command_state.clear_route_action()
            elif planner_update.get("route_packets"):
                for route_packet in planner_update["route_packets"]:
                    ser.write(route_packet)

            sent_this_frame = False

            if target_records:
                for i, record in enumerate(target_records):
                    corners = record["corners"].astype(int)
                    is_selected = selected_target_id is not None and record["id"] == selected_target_id
                    is_active = active_target is not None and record["id"] == active_target["id"]

                    color = (0, 180, 255)
                    if is_selected and is_active:
                        color = (0, 255, 0)
                    elif is_active:
                        color = (0, 255, 255)
                    elif is_selected:
                        color = (255, 0, 255)

                    for idx in range(4):
                        start = tuple(corners[idx])
                        end = tuple(corners[(idx + 1) % 4])
                        cv2.line(frame, start, end, color, 2)

                    cv2.circle(frame, (record["cx"], record["cy"]), 8, color, -1)
                    label = f"ID:{record['id']}"
                    if is_active:
                        label += " ACTIVE"
                    elif is_selected:
                        label += " SELECTED"
                    cv2.putText(
                        frame,
                        label,
                        (record["cx"] + 10, record["cy"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    if active_target is not None and record["id"] == active_target["id"]:
                        cv2.line(
                            frame,
                            (int(screen_center[0]), int(screen_center[1])),
                            (record["cx"], record["cy"]),
                            color,
                            2,
                        )

                if active_target is not None:
                    packet = build_uart_message(
                        active_target["cx"],
                        active_target["cy"],
                        OBJ_WIDTH,
                        OBJ_HEIGHT,
                        active_control,
                        reserved_flags=obstacle_flags,
                    )
                    ser.write(packet)
                    sent_this_frame = True

                info_text = (
                    f"VISIBLE:{','.join(str(record['id']) for record in target_records)} "
                    f"SEL:{selected_target_id if selected_target_id is not None else 'AUTO'} "
                    f"ACT:{active_target['id'] if active_target is not None else '--'}"
                )
                cv2.putText(frame, info_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No markers detected", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Continuous UART fail-safe behavior: send default cx/cy when no target.
            if not sent_this_frame:
                ser.write(build_uart_message(0, 0, OBJ_WIDTH, OBJ_HEIGHT, active_control, reserved_flags=obstacle_flags))

            status_text = (
                f"M:{active_control['movement_bits']:02X} "
                f"T:{active_control['turret_bits']:02X} "
                f"A:{active_control['aux_bits']:02X} "
                f"{active_control['label']} "
                f"SEL:{selected_target_id if selected_target_id is not None else 'AUTO'}"
            )
            cv2.putText(frame, status_text, (40, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 255), 2)

            if mqtt_client is not None:
                if now - last_telemetry_publish_time >= TELEMETRY_PUBLISH_INTERVAL_SECONDS:
                    telemetry_snapshot = dict(latest_telemetry)
                    telemetry_snapshot.update(
                        {
                            "visible_targets": [record["id"] for record in target_records],
                            "target_count": len(target_records),
                            "selected_target_id": selected_target_id,
                            "active_target_id": active_target["id"] if active_target is not None else None,
                            "lidar_scan": lidar_scan_snapshot,
                            "planner_enabled": planner_update.get("planner_enabled", bool(PLANNER_ENABLED)),
                            "planner_goal": planner_update.get("planner_goal"),
                            "planner_path": planner_update.get("planner_path", []),
                            "planner_status": planner_update.get("planner_status", "idle"),
                            "planner_path_age_s": planner_update.get("planner_path_age_s", 0.0),
                            "planner_obstacle_count": planner_update.get("planner_obstacle_count", 0),
                        }
                    )
                    mqtt_client.publish(MQTT_TOPIC_TELEMETRY, json.dumps(telemetry_snapshot))
                    last_telemetry_publish_time = now

                if now - last_camera_publish_time >= (1.0 / CAMERA_STREAM_FPS):
                    camera_payload = encode_camera_frame(frame)
                    if camera_payload is not None:
                        mqtt_client.publish(MQTT_TOPIC_CAMERA, camera_payload)
                    if aux_frame is not None:
                        aux_payload = encode_camera_frame(aux_frame)
                        if aux_payload is not None:
                            mqtt_client.publish(MQTT_TOPIC_CAMERA_AUX, aux_payload)
                    last_camera_publish_time = now

            if headless_mode:
                if now - last_headless_status_print_time >= HEADLESS_STATUS_PRINT_INTERVAL_SECONDS:
                    active_target_text = active_target["id"] if active_target is not None else "--"
                    print(
                        "Headless status | "
                        f"cmd={active_control['label']} "
                        f"targets={len(target_records)} "
                        f"active={active_target_text} "
                        f"state={latest_telemetry.get('robot_state', STATE_MANUAL)}"
                    )
                    last_headless_status_print_time = now
                continue

            cv2.imshow("Jetson Remote - ArUco + MQTT + UART", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Stopping Jetson remote...")
    finally:
        lidar_reader.stop()
        aiming_cap.release()
        if aux_cap is not None:
            aux_cap.release()
        if not headless_mode:
            cv2.destroyAllWindows()
        ser.close()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
