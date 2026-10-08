import os

# ปิด Warning Logs กวนใจจาก TensorFlow / oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ==========================================
# 1. สร้าง Custom Environment (Gym Style)
# ==========================================
class ArmPickAndPlaceEnv:

    def __init__(self):
        # ขนาดแขนกล L1, L2
        self.L1 = 120.0
        self.L2 = 100.0
        self.base_x, self.base_y = 400.0, 300.0

        # State Dimensions: [theta1, theta2, obj_x, obj_y, tray_x, tray_y, is_holding] -> 7
        self.state_dim = 7

        # Action Space: 6 Discrete Actions
        # 0: +theta1, 1: -theta1, 2: +theta2, 3: -theta2, 4: Grip, 5: Release
        self.action_dim = 6

        self.reset()

    def reset(self):
        self.theta1 = 45.0
        self.theta2 = 45.0

        # สุ่มตำแหน่งวัตถุ และ ถาด ในระยะที่แขนกลเอื้อมถึง
        self.obj_x = random.uniform(200, 320)
        self.obj_y = random.uniform(200, 380)

        self.tray_x = random.uniform(480, 600)
        self.tray_y = random.uniform(200, 380)

        self.is_holding = False
        self.step_count = 0
        self.max_steps = 150

        return self._get_state()

    def _get_end_effector(self):
        rad1 = np.radians(self.theta1)
        rad2 = np.radians(self.theta1 + self.theta2)

        x1 = self.base_x + self.L1 * np.cos(rad1)
        y1 = self.base_y - self.L1 * np.sin(rad1)

        x2 = x1 + self.L2 * np.cos(rad2)
        y2 = y1 - self.L2 * np.sin(rad2)
        return x2, y2

    def _get_state(self):
        ee_x, ee_y = self._get_end_effector()
        # Normalization ค่าให้อยู่ในช่วงประมาณ [-1, 1] เพื่อให้ NN เรียนรู้ได้ดี
        return np.array(
            [
                self.theta1 / 180.0,
                self.theta2 / 180.0,
                (self.obj_x - self.base_x) / 400.0,
                (self.obj_y - self.base_y) / 400.0,
                (self.tray_x - self.base_x) / 400.0,
                (self.tray_y - self.base_y) / 400.0,
                1.0 if self.is_holding else 0.0,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_count += 1
        d_angle = 3.0  # ขยับมุมครั้งละ 3 องศา

        # นำ Action มาปรับมุม/มือจับ
        if action == 0:
            self.theta1 += d_angle
        elif action == 1:
            self.theta1 -= d_angle
        elif action == 2:
            self.theta2 += d_angle
        elif action == 3:
            self.theta2 -= d_angle

        # จำกัดขอบเขตมุมแขนกล
        self.theta1 = np.clip(self.theta1, -90, 180)
        self.theta2 = np.clip(self.theta2, -150, 150)

        ee_x, ee_y = self._get_end_effector()

        # คำนวณระยะห่าง
        dist_to_obj = np.hypot(ee_x - self.obj_x, ee_y - self.obj_y)
        dist_to_tray = np.hypot(ee_x - self.tray_x, ee_y - self.tray_y)

        # จัดการการ Grip/Release
        if action == 4 and not self.is_holding and dist_to_obj < 25:
            self.is_holding = True
        elif action == 5 and self.is_holding and dist_to_tray < 30:
            self.is_holding = False

        # ถ้าถือวัตถุอยู่ วัตถุจะย้ายตามปลายแขน
        if self.is_holding:
            self.obj_x, self.obj_y = ee_x, ee_y

        # --- Reward Function Design ---
        reward = -0.1  # Step Penalty ขนาดเล็กเพื่อเร่งให้แก้โจทย์ไวขึ้น

        if not self.is_holding:
            # รางวัลเมื่อเคลื่อนปลายแขนเข้าใกล้วัตถุ
            reward += (100.0 - dist_to_obj) * 0.01
            if dist_to_obj < 25:
                reward += 10.0  # โบนัสเมื่อเข้าใกล้จนจับได้
        else:
            # รางวัลเมื่อพาวัตถุเคลื่อนเข้าใกล้ถาดวาง
            dist_obj_to_tray = np.hypot(
                self.obj_x - self.tray_x, self.obj_y - self.tray_y
            )
            reward += (100.0 - dist_obj_to_tray) * 0.02

        # Reward ใหญ่เมื่อวางวัตถุสำเร็จ
        done = False
        if not self.is_holding and dist_to_tray < 30 and self.step_count > 5:
            # วางบนถาดสำเร็จ!
            dist_final = np.hypot(
                self.obj_x - self.tray_x, self.obj_y - self.tray_y
            )
            if dist_final < 35:
                reward += 200.0
                done = True

        if self.step_count >= self.max_steps:
            done = True

        return self._get_state(), reward, done


# ==========================================
# 2. สร้างโครงสร้าง Keras Model (DQN Architecture)
# ==========================================
def create_dqn_model(state_dim, action_dim):
    """สร้าง Dense Neural Network สำหรับประเมิน Q-Values ของแต่ละ Action"""
    inputs = layers.Input(shape=(state_dim,))

    # Hidden Layers
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    # Output Layer: คำนวณ Q-Value ให้กับทุกๆ Action
    outputs = layers.Dense(action_dim, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Arm_RL_DQN")

    # แก้ไขจุดนี้: เปลี่ยนจาก loss='huber_loss' เป็น loss='huber' เพื่อรองรับ Keras 3
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss="huber")
    return model


# ==========================================
# 3. Agent & Training Loop (DQN)
# ==========================================
class DQNAgent:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Main Model & Target Model
        self.model = create_dqn_model(state_dim, action_dim)
        self.target_model = create_dqn_model(state_dim, action_dim)
        self.target_model.set_weights(self.model.get_weights())

        # Replay Buffer
        self.memory = deque(maxlen=20000)

        # Hyperparameters
        self.gamma = 0.98  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        act_values = self.model.predict(
            state.reshape(1, -1), verbose=0
        )  # Predict Q-values
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        # Predict Q-values
        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(
                    target_next[i]
                )

        # Train Network
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


# ==========================================
# 4. Main Training Execution
# ==========================================
if __name__ == "__main__":
    env = ArmPickAndPlaceEnv()
    agent = DQNAgent(env.state_dim, env.action_dim)

    episodes = 300  # ปรับจำนวน Episode ได้ตามต้องการ
    print(f"กำลังเริ่ม Train โมเดล RL จำนวน {episodes} Episodes...")

    for e in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

            if done:
                break

        # อัปเดต Target Network ทุกๆ 10 Episodes
        if e % 10 == 0:
            agent.update_target_model()
            print(
                f"Episode: {e}/{episodes} | Score: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}"
            )

    # Save โมเดลเป็นไฟล์ .keras เพื่อนำไปต่อกับ Tkinter UI
    save_path = "arm_rl_model.keras"
    agent.model.save(save_path)
    print(f"\nบันทึกไฟล์โมเดลเรียบร้อยแล้วที่: {save_path}")