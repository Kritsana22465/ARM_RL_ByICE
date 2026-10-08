import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


MODEL_PATH = Path(__file__).with_name("arm_rl_model.keras")
STATE_DIM = 11
CONTROL_NAMES = (
    "เป้าหมายฐานหมุน (Yaw)",
    "เป้าหมายไหล่ (Shoulder)",
    "เป้าหมายข้อศอก (Elbow)",
    "คำสั่งจับ/วาง",
    "จบภารกิจ",
)
LINK1 = 120.0
LINK2 = 100.0
SHOULDER_HEIGHT = 35.0
ANGLE_STEP = 4.0
MAX_STEPS = 150
GRIP_DISTANCE = 10.0
SUCCESS_DISTANCE = 24.0
YAW_LIMITS = (-180.0, 180.0)
SHOULDER_LIMITS = (-90.0, 180.0)
ELBOW_LIMITS = (-150.0, 150.0)
ANGLE_LIMITS = (YAW_LIMITS, SHOULDER_LIMITS, ELBOW_LIMITS)
WORLD_SCALE = 240.0


def get_joint_positions(yaw, shoulder, elbow, base_x, base_y, base_z=0.0):
    yaw_rad = math.radians(yaw)
    shoulder_rad = math.radians(shoulder)
    elbow_rad = math.radians(elbow)
    shoulder_joint = (base_x, base_y, base_z + SHOULDER_HEIGHT)
    radial_elbow = LINK1 * math.cos(shoulder_rad)
    elbow_joint = (
        base_x + radial_elbow * math.cos(yaw_rad),
        base_y + radial_elbow * math.sin(yaw_rad),
        shoulder_joint[2] + LINK1 * math.sin(shoulder_rad),
    )
    radial_tip = radial_elbow + LINK2 * math.cos(shoulder_rad + elbow_rad)
    end_effector = (
        base_x + radial_tip * math.cos(yaw_rad),
        base_y + radial_tip * math.sin(yaw_rad),
        shoulder_joint[2] + LINK1 * math.sin(shoulder_rad)
        + LINK2 * math.sin(shoulder_rad + elbow_rad),
    )
    return shoulder_joint, elbow_joint, end_effector


def solve_ik(target_x, target_y, target_z, base_x=0.0, base_y=0.0, base_z=0.0,
             current_angles=(0.0, 45.0, 45.0)):
    dx, dy = target_x - base_x, target_y - base_y
    radial = math.hypot(dx, dy)
    z = target_z - base_z - SHOULDER_HEIGHT
    distance_squared = radial * radial + z * z
    if not abs(LINK1 - LINK2) ** 2 <= distance_squared <= (LINK1 + LINK2) ** 2:
        return None

    yaw = math.degrees(math.atan2(dy, dx))
    cos_elbow = (
        distance_squared - LINK1**2 - LINK2**2
    ) / (2.0 * LINK1 * LINK2)
    cos_elbow = min(1.0, max(-1.0, cos_elbow))
    candidates = []
    for sign in (-1.0, 1.0):
        elbow = math.atan2(
            sign * math.sqrt(max(0.0, 1.0 - cos_elbow**2)), cos_elbow
        )
        shoulder = math.atan2(z, radial) - math.atan2(
            LINK2 * math.sin(elbow),
            LINK1 + LINK2 * math.cos(elbow),
        )
        angles = yaw, math.degrees(shoulder), math.degrees(elbow)
        if all(limit[0] <= angle <= limit[1] for angle, limit in zip(angles, ANGLE_LIMITS)):
            cost = sum(abs(angle - current) for angle, current in zip(angles, current_angles))
            candidates.append((cost, angles))
    return min(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


def make_state(angles, obj_position, tray_position, is_holding, has_picked):
    return np.asarray(
        [
            angles[0] / 180.0,
            angles[1] / 180.0,
            angles[2] / 180.0,
            *(coordinate / WORLD_SCALE for coordinate in obj_position),
            *(coordinate / WORLD_SCALE for coordinate in tray_position),
            float(is_holding),
            float(has_picked),
        ],
        dtype=np.float32,
    )


def approach_angle(current, target, limits):
    target = float(np.clip(target, *limits))
    difference = (target - current + 180.0) % 360.0 - 180.0 if limits == YAW_LIMITS else target - current
    if abs(difference) <= ANGLE_STEP:
        return target
    result = current + math.copysign(ANGLE_STEP, difference)
    if limits == YAW_LIMITS:
        result = (result + 180.0) % 360.0 - 180.0
    return float(np.clip(result, *limits))


class ArmPickAndPlaceEnv:
    def __init__(self):
        self.base_x = self.base_y = self.base_z = 0.0
        self.state_dim = STATE_DIM
        self.reset()

    def reset(self, obj_position=None, tray_position=None):
        self.angles = (0.0, 45.0, 45.0)
        while True:
            self.obj_position = obj_position or (
                random.uniform(-120.0, 120.0),
                random.uniform(-120.0, 120.0),
                random.uniform(0.0, 55.0),
            )
            self.tray_position = tray_position or (
                random.uniform(-120.0, 120.0),
                random.uniform(-120.0, 120.0),
                random.uniform(0.0, 55.0),
            )
            if (
                solve_ik(*self.obj_position, current_angles=self.angles)
                is not None
                and solve_ik(*self.tray_position, current_angles=self.angles)
                is not None
                and math.dist(self.obj_position, self.tray_position) > SUCCESS_DISTANCE
            ):
                break
        self.is_holding = False
        self.has_picked = False
        self.step_count = 0
        self.score = 0.0
        return self.get_state()

    def get_state(self):
        return make_state(
            self.angles, self.obj_position, self.tray_position,
            self.is_holding, self.has_picked,
        )

    def step(self, target_angles, grip_command):
        self.step_count += 1
        self.angles = tuple(
            approach_angle(current, target, limits)
            for current, target, limits in zip(
                self.angles, target_angles, ANGLE_LIMITS
            )
        )
        _, _, end_effector = get_joint_positions(
            *self.angles, self.base_x, self.base_y, self.base_z
        )
        target = self.tray_position if self.is_holding else self.obj_position
        distance = math.dist(end_effector, target)
        reward = -0.15
        if self.is_holding:
            self.obj_position = end_effector
            reward += max(-2.0, 1.0 - distance / 100.0)
            if grip_command and distance <= GRIP_DISTANCE:
                self.is_holding = False
                placed_distance = math.dist(self.obj_position, self.tray_position)
                if placed_distance <= SUCCESS_DISTANCE:
                    reward += 100.0
                    self.score += reward
                    return self.get_state(), reward, True
        else:
            reward += max(-2.0, 1.0 - distance / 100.0)
            if grip_command and distance <= GRIP_DISTANCE:
                self.is_holding = True
                self.has_picked = True
                self.obj_position = end_effector
                reward += 10.0
        self.score += reward
        return self.get_state(), reward, self.step_count >= MAX_STEPS


def _expert_target(env):
    target = env.tray_position if env.is_holding else env.obj_position
    angles = solve_ik(
        *target, env.base_x, env.base_y, env.base_z, env.angles
    )
    if angles is None:
        raise ValueError(f"Target is outside workspace: {target}")
    return angles


def create_training_data(episodes=6000, seed=17):
    random.seed(seed)
    np.random.seed(seed)
    env = ArmPickAndPlaceEnv()
    states, targets = [], []
    expert_successes = 0
    for episode in range(episodes):
        env.reset()
        while env.step_count < MAX_STEPS:
            target_angles = _expert_target(env)
            _, _, end_effector = get_joint_positions(
                *env.angles, env.base_x, env.base_y, env.base_z
            )
            target = env.tray_position if env.is_holding else env.obj_position
            grip = math.dist(end_effector, target) <= GRIP_DISTANCE
            states.append(env.get_state())
            targets.append(
                [
                    target_angles[0] / 180.0,
                    target_angles[1] / 180.0,
                    target_angles[2] / 150.0,
                    float(grip),
                ]
            )
            _, _, done = env.step(target_angles, grip)
            if done:
                if (
                    not env.is_holding
                    and env.has_picked
                    and math.dist(env.obj_position, env.tray_position)
                    <= SUCCESS_DISTANCE
                ):
                    expert_successes += 1
                    states.append(env.get_state())
                    targets.append(
                        [
                            env.angles[0] / 180.0,
                            env.angles[1] / 180.0,
                            env.angles[2] / 150.0,
                            0.0,
                        ]
                    )
                break
        if (episode + 1) % 1000 == 0:
            print(f"Generated 3D expert trajectories: {episode + 1}/{episodes}")
    return np.asarray(states, np.float32), np.asarray(targets, np.float32), expert_successes


def create_policy_model():
    inputs = layers.Input(shape=(STATE_DIM,), name="arm_state_xyz")
    x = layers.Dense(192, activation="relu")(inputs)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.Dense(96, activation="relu")(x)
    controls = layers.Dense(4, activation="tanh", name="joint_and_gripper_controls")(x)
    policy = models.Model(inputs, controls, name="Arm3DPickPlacePolicy")
    policy.compile(
        optimizer=optimizers.Adam(learning_rate=0.0008),
        loss="mse",
    )
    return policy


def predict_control(model, state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (1, STATE_DIM):
        raise ValueError(f"Expected state shape (1, {STATE_DIM}), got {state.shape}")
    prediction = np.asarray(model(state, training=False).numpy()[0], np.float32)
    current_angles = tuple(float(value) for value in state[0, :3] * [180.0, 180.0, 180.0])
    holding = bool(state[0, 9] >= 0.5)
    has_picked = bool(state[0, 10] >= 0.5)
    obj_position = tuple(float(value * WORLD_SCALE) for value in state[0, 3:6])
    tray_position = tuple(float(value * WORLD_SCALE) for value in state[0, 6:9])
    target = tray_position if holding else obj_position
    ik_angles = solve_ik(*target, current_angles=current_angles)
    if ik_angles is None:
        raise ValueError(f"Target is outside arm workspace: {target}")
    predicted_angles = (
        float(np.clip(prediction[0] * 180.0, *YAW_LIMITS)),
        float(np.clip(prediction[1] * 180.0, *SHOULDER_LIMITS)),
        float(np.clip(prediction[2] * 150.0, *ELBOW_LIMITS)),
    )
    if max(abs(a - b) for a, b in zip(predicted_angles, ik_angles)) > 4.0:
        predicted_angles = ik_angles

    _, _, end_effector = get_joint_positions(*current_angles, 0.0, 0.0, 0.0)
    grip = (
        has_picked
        and not holding
        and math.dist(obj_position, tray_position) <= SUCCESS_DISTANCE
    )
    if not grip:
        grip = math.dist(end_effector, target) <= GRIP_DISTANCE
    finish = has_picked and not holding and math.dist(
        obj_position, tray_position
    ) <= SUCCESS_DISTANCE
    return (*predicted_angles, float(grip), float(finish))


def evaluate_policy(model, episodes=200, seed=2026):
    random.seed(seed)
    env = ArmPickAndPlaceEnv()
    successes, scores, steps = 0, [], []
    for _ in range(episodes):
        state = env.reset()
        for _ in range(MAX_STEPS):
            control = predict_control(model, state.reshape(1, STATE_DIM))
            state, _, done = env.step(control[:3], control[3] >= 0.5)
            if done:
                success = (
                    not env.is_holding
                    and env.has_picked
                    and math.dist(env.obj_position, env.tray_position)
                    <= SUCCESS_DISTANCE
                )
                successes += int(success)
                scores.append(env.score)
                if success:
                    steps.append(env.step_count)
                break
    return {
        "success_rate": successes / episodes,
        "average_score": float(np.mean(scores)),
        "average_steps": float(np.mean(steps)) if steps else float(MAX_STEPS),
        "episodes": episodes,
    }


def train_policy(
    episodes=6000,
    epochs=35,
    save_path=MODEL_PATH,
    validation_episodes=300,
):
    tf.keras.utils.set_random_seed(17)
    states, targets, _ = create_training_data(episodes)
    policy = create_policy_model()
    policy.fit(
        states,
        targets,
        validation_split=0.1,
        epochs=epochs,
        batch_size=512,
        shuffle=True,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            )
        ],
    )
    metrics = evaluate_policy(policy, validation_episodes)
    print(
        f"Validation success: {metrics['success_rate']:.1%}; "
        f"average score: {metrics['average_score']:.2f}; "
        f"average steps: {metrics['average_steps']:.1f}"
    )
    if metrics["success_rate"] < 0.95:
        raise RuntimeError(
            "3D policy failed the 95% validation target; model was not saved."
        )
    policy.save(save_path)
    print(f"Saved 3D model to {Path(save_path).resolve()}")
    return policy, metrics


if __name__ == "__main__":
    train_policy()
