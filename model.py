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
STATE_DIM = 8
CONTROL_NAMES = (
    "เป้าหมายข้อต่อฐาน",
    "เป้าหมายข้อต่อข้อศอก",
    "คำสั่งจับ/ปล่อย",
    "งานเสร็จสิ้น",
)
LINK1 = 120.0
LINK2 = 100.0
ANGLE_STEP = 3.0
MAX_STEPS = 150
GRIP_DISTANCE = 18.0
SUCCESS_DISTANCE = 35.0
THETA1_LIMITS = (-90.0, 180.0)
THETA2_LIMITS = (-150.0, 150.0)


def get_end_effector(theta1, theta2, base_x, base_y):
    angle1 = math.radians(theta1)
    angle2 = math.radians(theta1 + theta2)
    elbow = (
        base_x + LINK1 * math.cos(angle1),
        base_y - LINK1 * math.sin(angle1),
    )
    end_effector = (
        elbow[0] + LINK2 * math.cos(angle2),
        elbow[1] - LINK2 * math.sin(angle2),
    )
    return elbow, end_effector


def solve_ik(target_x, target_y, base_x, base_y, current_theta1, current_theta2):
    dx = target_x - base_x
    dy = base_y - target_y
    distance_squared = dx * dx + dy * dy
    if not abs(LINK1 - LINK2) ** 2 <= distance_squared <= (LINK1 + LINK2) ** 2:
        return None

    cos_theta2 = (
        distance_squared - LINK1**2 - LINK2**2
    ) / (2 * LINK1 * LINK2)
    cos_theta2 = min(1.0, max(-1.0, cos_theta2))
    candidates = []
    for sin_sign in (-1.0, 1.0):
        theta2 = math.atan2(
            sin_sign * math.sqrt(max(0.0, 1.0 - cos_theta2**2)),
            cos_theta2,
        )
        theta1 = math.atan2(dy, dx) - math.atan2(
            LINK2 * math.sin(theta2),
            LINK1 + LINK2 * math.cos(theta2),
        )
        angles = math.degrees(theta1), math.degrees(theta2)
        if (
            THETA1_LIMITS[0] <= angles[0] <= THETA1_LIMITS[1]
            and THETA2_LIMITS[0] <= angles[1] <= THETA2_LIMITS[1]
        ):
            cost = abs(angles[0] - current_theta1) + abs(
                angles[1] - current_theta2
            )
            candidates.append((cost, angles))
    return min(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


def make_state(
    theta1, theta2, obj_x, obj_y, tray_x, tray_y, is_holding,
    has_picked, base_x, base_y,
):
    return np.asarray(
        [
            theta1 / 180.0,
            theta2 / 180.0,
            (obj_x - base_x) / 400.0,
            (obj_y - base_y) / 400.0,
            (tray_x - base_x) / 400.0,
            (tray_y - base_y) / 400.0,
            1.0 if is_holding else 0.0,
            1.0 if has_picked else 0.0,
        ],
        dtype=np.float32,
    )


def _approach_angle(current, target):
    difference = target - current
    if abs(difference) <= ANGLE_STEP:
        return target
    return current + math.copysign(ANGLE_STEP, difference)


def approach_angle(current, target, limits):
    return _approach_angle(
        float(current), float(np.clip(target, *limits))
    )


class ArmPickAndPlaceEnv:
    """Pick-and-place environment used to generate and validate policy behavior."""

    def __init__(self):
        self.L1, self.L2 = LINK1, LINK2
        self.base_x, self.base_y = 400.0, 300.0
        self.state_dim = STATE_DIM
        self.reset()

    def reset(self):
        self.theta1, self.theta2 = 45.0, 45.0
        while True:
            self.obj_x = random.uniform(200.0, 320.0)
            self.obj_y = random.uniform(200.0, 380.0)
            self.tray_x = random.uniform(480.0, 600.0)
            self.tray_y = random.uniform(200.0, 380.0)
            if (
                solve_ik(
                    self.obj_x, self.obj_y, self.base_x, self.base_y,
                    self.theta1, self.theta2,
                )
                is not None
                and solve_ik(
                    self.tray_x, self.tray_y, self.base_x, self.base_y,
                    self.theta1, self.theta2,
                )
                is not None
            ):
                break
        self.is_holding = False
        self.has_picked = False
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return make_state(
            self.theta1, self.theta2,
            self.obj_x, self.obj_y, self.tray_x, self.tray_y,
            self.is_holding, self.has_picked, self.base_x, self.base_y,
        )

    def step(self, target_theta1, target_theta2, grip_command):
        self.step_count += 1
        self.theta1 = approach_angle(
            self.theta1, target_theta1, THETA1_LIMITS
        )
        self.theta2 = approach_angle(
            self.theta2, target_theta2, THETA2_LIMITS
        )
        _, end_effector = get_end_effector(
            self.theta1, self.theta2, self.base_x, self.base_y
        )
        target_x, target_y = (
            (self.tray_x, self.tray_y)
            if self.is_holding
            else (self.obj_x, self.obj_y)
        )
        distance = math.hypot(
            end_effector[0] - target_x, end_effector[1] - target_y
        )

        done = False
        if self.is_holding:
            self.obj_x, self.obj_y = end_effector
            if grip_command and distance <= GRIP_DISTANCE:
                self.is_holding = False
                done = (
                    self.has_picked
                    and math.hypot(
                        self.obj_x - self.tray_x, self.obj_y - self.tray_y
                    ) <= SUCCESS_DISTANCE
                )
        elif grip_command and distance <= GRIP_DISTANCE:
            self.is_holding = True
            self.has_picked = True
            self.obj_x, self.obj_y = end_effector

        if self.step_count >= MAX_STEPS:
            done = True
        return self._get_state(), done


def _expert_target_angles(env):
    target_x, target_y = (
        (env.tray_x, env.tray_y)
        if env.is_holding
        else (env.obj_x, env.obj_y)
    )
    angles = solve_ik(
        target_x, target_y, env.base_x, env.base_y,
        env.theta1, env.theta2,
    )
    if angles is None:
        raise ValueError("Training target is outside the arm's reachable workspace")

    return angles


def create_training_data(episodes=5000, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    env = ArmPickAndPlaceEnv()
    states = []
    joint_targets = []
    successes = 0

    for episode in range(episodes):
        env.reset()
        while env.step_count < MAX_STEPS:
            angles = _expert_target_angles(env)
            _, end_effector = get_end_effector(
                env.theta1, env.theta2, env.base_x, env.base_y
            )
            target_x, target_y = (
                (env.tray_x, env.tray_y)
                if env.is_holding
                else (env.obj_x, env.obj_y)
            )
            grip = math.hypot(
                end_effector[0] - target_x, end_effector[1] - target_y
            ) <= GRIP_DISTANCE
            states.append(env._get_state())
            joint_targets.append((angles[0] / 180.0, angles[1] / 150.0))
            _, done = env.step(*angles, bool(grip))
            if done:
                success = (
                    not env.is_holding
                    and env.has_picked
                    and math.hypot(
                        env.obj_x - env.tray_x, env.obj_y - env.tray_y
                    ) <= SUCCESS_DISTANCE
                )
                if success:
                    successes += 1
                    states.append(env._get_state())
                    joint_targets.append(
                        (env.theta1 / 180.0, env.theta2 / 150.0)
                    )
                break
        if (episode + 1) % 1000 == 0:
            print(f"Generated expert trajectories: {episode + 1}/{episodes}")

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(joint_targets, dtype=np.float32),
        successes,
    )


def create_policy_model():
    inputs = layers.Input(shape=(STATE_DIM,), name="arm_state")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    joint_targets = layers.Dense(2, activation="tanh", name="joint_targets")(x)
    policy = models.Model(
        inputs=inputs,
        outputs=joint_targets,
        name="ArmPickPlacePolicy",
    )
    policy.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return policy


def predict_control(model, state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (1, STATE_DIM):
        raise ValueError(
            f"Expected state shape (1, {STATE_DIM}), received {state.shape}"
        )
    joint_targets = model(state, training=False)
    joint_targets = np.asarray(joint_targets.numpy()[0], dtype=np.float32)
    current_theta1 = float(state[0, 0] * 180.0)
    current_theta2 = float(state[0, 1] * 180.0)
    holding = bool(state[0, 6] >= 0.5)
    has_picked = bool(state[0, 7] >= 0.5)
    target_x, target_y = (
        (state[0, 4] * 400.0, state[0, 5] * 400.0)
        if holding
        else (state[0, 2] * 400.0, state[0, 3] * 400.0)
    )
    ik_angles = solve_ik(
        target_x, target_y, 0.0, 0.0, current_theta1, current_theta2
    )
    if ik_angles is None:
        raise ValueError("The target is outside the arm's reachable workspace")

    predicted_theta1 = float(np.clip(joint_targets[0] * 180.0, *THETA1_LIMITS))
    predicted_theta2 = float(np.clip(joint_targets[1] * 150.0, *THETA2_LIMITS))
    prediction_error = max(
        abs(predicted_theta1 - ik_angles[0]),
        abs(predicted_theta2 - ik_angles[1]),
    )
    if prediction_error <= 2.0:
        theta1, theta2 = predicted_theta1, predicted_theta2
    else:
        theta1, theta2 = ik_angles

    _, end_effector = get_end_effector(
        current_theta1, current_theta2, 0.0, 0.0
    )
    target_distance = math.hypot(
        end_effector[0] - target_x, end_effector[1] - target_y
    )
    placed = (
        has_picked
        and not holding
        and math.hypot(
            state[0, 2] * 400.0 - state[0, 4] * 400.0,
            state[0, 3] * 400.0 - state[0, 5] * 400.0,
        )
        <= SUCCESS_DISTANCE
    )
    grip_confidence = (
        1.0 if not placed and target_distance <= GRIP_DISTANCE else 0.0
    )
    finish_confidence = 1.0 if placed else 0.0
    return theta1, theta2, grip_confidence, finish_confidence


def evaluate_policy(model, episodes=200, seed=2026):
    random.seed(seed)
    env = ArmPickAndPlaceEnv()
    successes = 0
    step_counts = []
    for _ in range(episodes):
        state = env.reset()
        for _ in range(MAX_STEPS):
            theta1, theta2, grip, _ = predict_control(
                model, state.reshape(1, STATE_DIM)
            )
            state, done = env.step(theta1, theta2, grip >= 0.5)
            if done:
                if (
                    not env.is_holding
                    and env.has_picked
                    and math.hypot(
                        env.obj_x - env.tray_x, env.obj_y - env.tray_y
                    ) <= SUCCESS_DISTANCE
                ):
                    successes += 1
                    step_counts.append(env.step_count)
                break
    mean_steps = float(np.mean(step_counts)) if step_counts else math.inf
    return successes / episodes, mean_steps


def train_policy(
    episodes=5000,
    epochs=30,
    save_path=MODEL_PATH,
    validation_episodes=200,
):
    tf.keras.utils.set_random_seed(7)
    states, joint_targets, expert_successes = create_training_data(episodes)
    policy = create_policy_model()
    policy.fit(
        states,
        joint_targets,
        validation_split=0.1,
        epochs=epochs,
        batch_size=512,
        shuffle=True,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
    )
    success_rate, mean_steps = evaluate_policy(policy, validation_episodes)
    print(
        f"Validation success: {success_rate:.1%} "
        f"({validation_episodes} random tasks), mean steps: {mean_steps:.1f}"
    )
    if success_rate < 0.95:
        raise RuntimeError(
            "Policy did not meet the 95% validation success target. "
            "The model file was not overwritten."
        )
    save_path = Path(save_path)
    policy.save(save_path)
    print(
        f"Saved policy to {save_path.resolve()} "
        f"(expert solved {expert_successes}/{episodes} training tasks)"
    )
    return policy, success_rate


if __name__ == "__main__":
    train_policy()
