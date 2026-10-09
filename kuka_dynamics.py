from importlib import import_module
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


JOINT_LOW_DEG = np.array([-90, -60, -90, -90, -90, -90], dtype=np.float32)
JOINT_HIGH_DEG = np.array([90, 90, 90, 90, 90, 90], dtype=np.float32)
JOINT_LOW = np.deg2rad(JOINT_LOW_DEG).astype(np.float32)
JOINT_HIGH = np.deg2rad(JOINT_HIGH_DEG).astype(np.float32)
STATE_DIM = 18
ACTION_COUNT = 12
MAX_STEPS = 180
DT = 0.05
MAX_VELOCITY = 0.65
LINK1 = 120.0
LINK2 = 100.0
SHOULDER_HEIGHT = 35.0


def forward_kinematics(joint_positions):
    yaw, shoulder, elbow = np.asarray(joint_positions, dtype=np.float32)[:3]
    base = (0.0, 0.0, 0.0)
    shoulder_joint = (0.0, 0.0, SHOULDER_HEIGHT)
    radial_elbow = LINK1 * np.cos(shoulder)
    elbow_joint = (
        radial_elbow * np.cos(yaw),
        radial_elbow * np.sin(yaw),
        SHOULDER_HEIGHT + LINK1 * np.sin(shoulder),
    )
    radial_wrist = radial_elbow + LINK2 * np.cos(shoulder + elbow)
    wrist = (
        radial_wrist * np.cos(yaw),
        radial_wrist * np.sin(yaw),
        SHOULDER_HEIGHT
        + LINK1 * np.sin(shoulder)
        + LINK2 * np.sin(shoulder + elbow),
    )
    return base, shoulder_joint, elbow_joint, wrist


def solve_ik(target_position, current_angles=None):
    target_x, target_y, target_z = (float(value) for value in target_position)
    yaw = float(np.arctan2(target_y, target_x))
    radial = float(np.hypot(target_x, target_y))
    height = target_z - SHOULDER_HEIGHT
    distance_squared = radial**2 + height**2
    if not abs(LINK1 - LINK2) ** 2 <= distance_squared <= (LINK1 + LINK2) ** 2:
        return None

    cosine_elbow = (
        distance_squared - LINK1**2 - LINK2**2
    ) / (2.0 * LINK1 * LINK2)
    cosine_elbow = float(np.clip(cosine_elbow, -1.0, 1.0))
    current_angles = (
        np.zeros(6, dtype=np.float32)
        if current_angles is None
        else np.asarray(current_angles, dtype=np.float32)
    )
    candidates = []
    for sign in (-1.0, 1.0):
        elbow = float(
            np.arctan2(
                sign * np.sqrt(max(0.0, 1.0 - cosine_elbow**2)),
                cosine_elbow,
            )
        )
        shoulder = float(
            np.arctan2(height, radial)
            - np.arctan2(
                LINK2 * np.sin(elbow), LINK1 + LINK2 * np.cos(elbow)
            )
        )
        candidate = np.array(
            [yaw, shoulder, elbow, 0.0, 0.0, 0.0], dtype=np.float32
        )
        if np.all(candidate >= JOINT_LOW) and np.all(candidate <= JOINT_HIGH):
            candidates.append((float(np.linalg.norm(candidate - current_angles)), candidate))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def load_model(model_path):
    dqn_class = import_module("stable_baselines3").DQN
    with ZipFile(model_path) as archive:
        entries = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and "__MACOSX" not in PurePosixPath(name).parts
            and not PurePosixPath(name).name.startswith("._")
        ]
        roots = {PurePosixPath(name).parts[0] for name in entries}
        nested_root = next(iter(roots)) if len(roots) == 1 else None
        strip_root = bool(
            nested_root
            and all(name.startswith(f"{nested_root}/") for name in entries)
        )
        normalized_entries = {
            name[len(nested_root) + 1:] if strip_root else name: name
            for name in entries
        }
        if "data" not in normalized_entries or "policy.pth" not in normalized_entries:
            raise ValueError("ZIP ไม่พบไฟล์ data และ policy.pth ของ Stable-Baselines3")

        needs_normalizing = strip_root or len(entries) != len(archive.namelist())
        if not needs_normalizing:
            return dqn_class.load(str(model_path))

        with TemporaryDirectory() as temporary_directory:
            normalized_path = Path(temporary_directory) / "kuka_checkpoint.zip"
            with ZipFile(normalized_path, "w", compression=ZIP_DEFLATED) as output:
                for normalized_name, archive_name in normalized_entries.items():
                    output.writestr(normalized_name, archive.read(archive_name))
            return dqn_class.load(str(normalized_path))


def reset_episode(rng):
    joint_positions = rng.uniform(JOINT_LOW * 0.45, JOINT_HIGH * 0.45).astype(
        np.float32
    )
    goal_positions = rng.uniform(JOINT_LOW * 0.65, JOINT_HIGH * 0.65).astype(
        np.float32
    )
    previous_velocity = np.zeros(6, dtype=np.float32)
    return joint_positions, goal_positions, previous_velocity


def make_observation(joint_positions, goal_positions, previous_velocity):
    return np.concatenate(
        [joint_positions, goal_positions, previous_velocity]
    ).astype(np.float32)


def step_dynamics(joint_positions, goal_positions, previous_velocity, action):
    action = int(action)
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"Kuka action must be in [0, {ACTION_COUNT - 1}]")

    velocity = np.zeros(6, dtype=np.float32)
    joint_index = action // 2
    direction = 1.0 if action % 2 == 0 else -1.0
    velocity[joint_index] = direction * MAX_VELOCITY

    old_distance = float(np.linalg.norm(goal_positions - joint_positions))
    joint_positions = np.clip(
        joint_positions + velocity * DT, JOINT_LOW, JOINT_HIGH
    ).astype(np.float32)
    new_distance = float(np.linalg.norm(goal_positions - joint_positions))

    progress = old_distance - new_distance
    smooth_penalty = float(np.mean((velocity - previous_velocity) ** 2))
    speed_penalty = float(np.mean(velocity**2))
    reward = (
        12 * progress - 0.035 * smooth_penalty - 0.004 * speed_penalty - 0.015
    )
    reached = new_distance < 0.06
    if reached:
        reward += 8.0

    return (
        joint_positions,
        velocity,
        float(reward),
        reached,
        new_distance,
    )


def _sample_reachable_position(rng, joint_positions):
    for _ in range(5000):
        position = rng.uniform([-180.0, -180.0, 10.0], [180.0, 180.0, 80.0])
        if solve_ik(position, joint_positions) is not None:
            return np.asarray(position, dtype=np.float32)
    raise RuntimeError("Could not sample a reachable pick/place position")


def evaluate_pick_place_policy(model, episodes=100, seed=2026):
    rng = np.random.default_rng(seed)
    successes = 0
    scores = []
    successful_steps = []

    for _ in range(episodes):
        joint_positions = rng.uniform(
            JOINT_LOW * 0.35, JOINT_HIGH * 0.35
        ).astype(np.float32)
        object_position = _sample_reachable_position(rng, joint_positions)
        tray_position = _sample_reachable_position(rng, joint_positions)
        while np.linalg.norm(object_position - tray_position) <= 24.0:
            tray_position = _sample_reachable_position(rng, joint_positions)

        goal_positions = solve_ik(object_position, joint_positions)
        previous_velocity = np.zeros(6, dtype=np.float32)
        is_holding = False
        score = 0.0

        for step in range(1, MAX_STEPS + 1):
            observation = make_observation(
                joint_positions, goal_positions, previous_velocity
            )
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            (
                joint_positions,
                previous_velocity,
                reward,
                _,
                _,
            ) = step_dynamics(
                joint_positions, goal_positions, previous_velocity, action
            )
            score += reward

            end_effector = np.asarray(forward_kinematics(joint_positions)[-1])
            target = tray_position if is_holding else object_position
            target_distance = float(np.linalg.norm(end_effector - target))

            if is_holding and target_distance <= 24.0:
                successes += 1
                successful_steps.append(step)
                score += 100.0
                break
            if not is_holding and target_distance <= 10.0:
                is_holding = True
                goal_positions = solve_ik(tray_position, joint_positions)
                score += 10.0

        scores.append(score)

    return {
        "success_rate": successes / episodes,
        "average_score": float(np.mean(scores)),
        "average_steps": (
            float(np.mean(successful_steps)) if successful_steps else MAX_STEPS
        ),
        "episodes": episodes,
    }


def evaluate_policy(model, episodes=100, seed=2026):
    rng = np.random.default_rng(seed)
    successes = 0
    scores = []
    successful_steps = []

    for _ in range(episodes):
        joint_positions, goal_positions, previous_velocity = reset_episode(rng)
        score = 0.0

        for step in range(1, MAX_STEPS + 1):
            observation = make_observation(
                joint_positions, goal_positions, previous_velocity
            )
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            (
                joint_positions,
                previous_velocity,
                reward,
                reached,
                _,
            ) = step_dynamics(
                joint_positions, goal_positions, previous_velocity, action
            )
            score += reward
            if reached:
                successes += 1
                successful_steps.append(step)
                break

        scores.append(score)

    return {
        "success_rate": successes / episodes,
        "average_score": float(np.mean(scores)),
        "average_steps": (
            float(np.mean(successful_steps)) if successful_steps else MAX_STEPS
        ),
        "episodes": episodes,
    }