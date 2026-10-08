import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import tensorflow as tf


MODEL_PATH = Path(__file__).with_name("arm_rl_model.keras")
ACTION_NAMES = (
    "หมุนฐาน +",
    "หมุนฐาน -",
    "ขยับข้อศอก +",
    "ขยับข้อศอก -",
    "หยิบวัตถุ",
    "วางวัตถุ",
)
TICK_MS = 50
MAX_STEPS = 150


class ArmPickPlaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arm RL Studio | Pick & Place")
        self.root.geometry("1120x740")
        self.root.minsize(980, 680)
        self.root.configure(bg="#111827")

        self.canvas_width = 740
        self.canvas_height = 500
        self.link1 = 120
        self.link2 = 100
        self.base_x = 370
        self.base_y = 310
        self.theta1 = 45.0
        self.theta2 = 45.0
        self.default_angles = (45.0, 45.0)
        self.obj_x, self.obj_y = 270.0, 310.0
        self.tray_x, self.tray_y = 535.0, 310.0
        self.is_holding = False
        self.is_running = False
        self.steps = 0
        self.drag_item = None
        self.drag_x = 0
        self.drag_y = 0
        self.last_action = "รอเริ่มทำงาน"
        self.last_q_values = np.zeros(6, dtype=np.float32)

        self.model = None
        self.model_error = None
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if self.model.input_shape[-1] != 7 or self.model.output_shape[-1] != 6:
                raise ValueError(
                    "โมเดลต้องรับ input 7 ค่าและคืนค่า action 6 ค่า "
                    f"(พบ {self.model.input_shape} -> {self.model.output_shape})"
                )
        except Exception as exc:
            self.model_error = str(exc)

        self._setup_styles()
        self._build_ui()
        self._draw_scene()
        self._update_dashboard()
        if self.model_error:
            self._set_status(f"โหลดโมเดลไม่สำเร็จ: {self.model_error}", error=True)
        else:
            self._set_status(f"พร้อมใช้งาน · {MODEL_PATH.name}")

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Arm.Horizontal.TProgressbar",
            troughcolor="#263244",
            background="#38bdf8",
            bordercolor="#263244",
            lightcolor="#38bdf8",
            darkcolor="#38bdf8",
            thickness=8,
        )

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#111827")
        header.pack(fill=tk.X, padx=22, pady=(18, 12))
        tk.Label(
            header,
            text="ARM RL STUDIO",
            bg="#111827",
            fg="#f8fafc",
            font=("Segoe UI", 19, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text="DQN Q-values · IK ช่วยนำแขนไปหยิบและวาง",
            bg="#111827",
            fg="#94a3b8",
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=(14, 0), pady=(5, 0))

        body = tk.Frame(self.root, bg="#111827")
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 14))

        workspace = tk.Frame(body, bg="#172033", highlightthickness=1,
                             highlightbackground="#29364b")
        workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            workspace,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#0b1220",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_stop)

        controls = tk.Frame(workspace, bg="#172033")
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.start_button = self._button(
            controls, "▶  เริ่มจำลอง", self.start_simulation, "#22c55e", "#052e16"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = self._button(
            controls, "หยุด", self.stop_simulation, "#fb7185", "#4c0519"
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))
        self._button(
            controls, "คืนค่าเริ่มต้น", self.reset_positions, "#29364b", "#e2e8f0"
        ).pack(side=tk.LEFT)
        tk.Label(
            controls,
            text="ลากวัตถุหรือถาดด้วยเมาส์เพื่อเปลี่ยนตำแหน่ง",
            bg="#172033",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT, padx=4)

        side = tk.Frame(body, bg="#172033", width=288,
                        highlightthickness=1, highlightbackground="#29364b")
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        side.pack_propagate(False)

        self._section_title(side, "สถานะโมเดล")
        self.model_status = tk.Label(
            side,
            text=("●  โหลดโมเดลแล้ว" if self.model else "●  ยังไม่ได้โหลดโมเดล"),
            bg="#172033",
            fg="#4ade80" if self.model else "#fb7185",
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
        )
        self.model_status.pack(fill=tk.X, padx=16, pady=(0, 3))
        tk.Label(
            side,
            text=MODEL_PATH.name,
            bg="#172033",
            fg="#94a3b8",
            font=("Consolas", 9),
            anchor=tk.W,
        ).pack(fill=tk.X, padx=16, pady=(0, 12))

        self._separator(side)
        self._section_title(side, "ข้อต่อแขนกล")
        self.theta1_label, self.theta1_bar = self._joint_row(side, "ฐาน · θ1")
        self.theta2_label, self.theta2_bar = self._joint_row(side, "ข้อศอก · θ2")

        self._separator(side)
        self._section_title(side, "ตำแหน่งและระยะ")
        self.ee_label = self._info_row(side, "ปลายแขน", "(0, 0)")
        self.object_distance_label = self._info_row(side, "ถึงวัตถุ", "0 px")
        self.tray_distance_label = self._info_row(side, "ถึงถาด", "0 px")
        self.steps_label = self._info_row(side, "จำนวนขั้น", "0 / 150")

        self._separator(side)
        self._section_title(side, "การทำงานล่าสุด")
        self.action_label = tk.Label(
            side,
            text=self.last_action,
            bg="#172033",
            fg="#fbbf24",
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
            wraplength=245,
        )
        self.action_label.pack(fill=tk.X, padx=16, pady=(0, 8))
        self.gripper_badge = tk.Label(
            side,
            text="GRIPPER  ·  ว่าง",
            bg="#29364b",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
            pady=8,
        )
        self.gripper_badge.pack(fill=tk.X, padx=14, pady=(0, 12))

        self._separator(side)
        self._section_title(side, "Q-values · 6 actions")
        self.q_rows = []
        for name in ACTION_NAMES:
            row = tk.Frame(side, bg="#172033")
            row.pack(fill=tk.X, padx=16, pady=3)
            tk.Label(
                row, text=name, bg="#172033", fg="#94a3b8",
                font=("Segoe UI", 8), anchor=tk.W
            ).pack(side=tk.LEFT)
            value = tk.Label(
                row, text="—", bg="#172033", fg="#cbd5e1",
                font=("Consolas", 8), anchor=tk.E
            )
            value.pack(side=tk.RIGHT)
            self.q_rows.append(value)

        self.status_bar = tk.Label(
            self.root,
            text="พร้อม",
            bg="#0b1220",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            anchor=tk.W,
            padx=22,
            pady=8,
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    @staticmethod
    def _button(parent, text, command, background, foreground):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=background,
            activeforeground=foreground,
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=13,
            pady=8,
            cursor="hand2",
        )

    @staticmethod
    def _section_title(parent, text):
        tk.Label(
            parent,
            text=text.upper(),
            bg="#172033",
            fg="#7dd3fc",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, padx=16, pady=(12, 8))

    @staticmethod
    def _separator(parent):
        tk.Frame(parent, bg="#29364b", height=1).pack(fill=tk.X, padx=14, pady=4)

    def _joint_row(self, parent, name):
        frame = tk.Frame(parent, bg="#172033")
        frame.pack(fill=tk.X, padx=16, pady=(0, 9))
        tk.Label(
            frame, text=name, bg="#172033", fg="#cbd5e1",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)
        value = tk.Label(
            frame, text="0.0°", bg="#172033", fg="#fbbf24",
            font=("Consolas", 10, "bold")
        )
        value.pack(side=tk.RIGHT)
        bar = ttk.Progressbar(
            parent, maximum=270, style="Arm.Horizontal.TProgressbar"
        )
        bar.pack(fill=tk.X, padx=16, pady=(0, 8))
        return value, bar

    def _info_row(self, parent, title, value):
        row = tk.Frame(parent, bg="#172033")
        row.pack(fill=tk.X, padx=16, pady=3)
        tk.Label(
            row, text=title, bg="#172033", fg="#94a3b8",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)
        label = tk.Label(
            row, text=value, bg="#172033", fg="#e2e8f0",
            font=("Consolas", 9), anchor=tk.E
        )
        label.pack(side=tk.RIGHT)
        return label

    def _joint_positions(self):
        angle1 = math.radians(self.theta1)
        angle2 = math.radians(self.theta1 + self.theta2)
        elbow = (
            self.base_x + self.link1 * math.cos(angle1),
            self.base_y - self.link1 * math.sin(angle1),
        )
        end_effector = (
            elbow[0] + self.link2 * math.cos(angle2),
            elbow[1] - self.link2 * math.sin(angle2),
        )
        return elbow, end_effector

    def _draw_scene(self):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), self.canvas_width)
        height = max(self.canvas.winfo_height(), self.canvas_height)
        for x in range(0, width, 40):
            self.canvas.create_line(x, 0, x, height, fill="#141f30")
        for y in range(0, height, 40):
            self.canvas.create_line(0, y, width, y, fill="#141f30")

        self.canvas.create_oval(
            self.base_x - 220, self.base_y - 220,
            self.base_x + 220, self.base_y + 220,
            outline="#26364b", dash=(3, 5)
        )
        self.canvas.create_text(
            18, 16, text="WORKSPACE  ·  DRAG OBJECT / TRAY",
            fill="#64748b", font=("Segoe UI", 8, "bold"), anchor=tk.NW
        )

        self.canvas.create_rectangle(
            self.tray_x - 42, self.tray_y - 17,
            self.tray_x + 42, self.tray_y + 17,
            fill="#8b5cf6", outline="#c4b5fd", width=2, tags=("tray",)
        )
        self.canvas.create_text(
            self.tray_x, self.tray_y + 29, text="ถาดวาง",
            fill="#c4b5fd", font=("Segoe UI", 9, "bold")
        )
        object_color = "#fb923c" if not self.is_holding else "#4ade80"
        self.canvas.create_oval(
            self.obj_x - 13, self.obj_y - 13,
            self.obj_x + 13, self.obj_y + 13,
            fill=object_color, outline="#ffedd5", width=2, tags=("object",)
        )
        self.canvas.create_text(
            self.obj_x, self.obj_y + 23, text="วัตถุ",
            fill="#fed7aa", font=("Segoe UI", 8, "bold")
        )

        elbow, end_effector = self._joint_positions()
        self.canvas.create_polygon(
            self.base_x - 28, self.base_y + 18,
            self.base_x + 28, self.base_y + 18,
            self.base_x + 17, self.base_y - 4,
            self.base_x - 17, self.base_y - 4,
            fill="#334155", outline="#64748b"
        )
        self.canvas.create_line(
            self.base_x, self.base_y, elbow[0], elbow[1],
            fill="#38bdf8", width=12, capstyle=tk.ROUND
        )
        self.canvas.create_oval(
            elbow[0] - 9, elbow[1] - 9, elbow[0] + 9, elbow[1] + 9,
            fill="#7dd3fc", outline="#0b1220", width=2
        )
        self.canvas.create_line(
            elbow[0], elbow[1], end_effector[0], end_effector[1],
            fill="#818cf8", width=9, capstyle=tk.ROUND
        )
        self.canvas.create_oval(
            end_effector[0] - 10, end_effector[1] - 10,
            end_effector[0] + 10, end_effector[1] + 10,
            fill="#f472b6", outline="#fce7f3", width=2
        )
        if self.is_holding:
            self.obj_x, self.obj_y = end_effector
            self.canvas.tag_raise("object")
        self._update_dashboard(end_effector)

    def _update_dashboard(self, end_effector=None):
        if end_effector is None:
            _, end_effector = self._joint_positions()
        self.theta1_label.config(text=f"{self.theta1:.1f}°")
        self.theta2_label.config(text=f"{self.theta2:.1f}°")
        self.theta1_bar["value"] = self.theta1 + 90
        self.theta2_bar["value"] = self.theta2 + 150
        self.ee_label.config(
            text=f"({int(end_effector[0])}, {int(end_effector[1])})"
        )
        self.object_distance_label.config(
            text=f"{math.hypot(end_effector[0] - self.obj_x, end_effector[1] - self.obj_y):.0f} px"
        )
        self.tray_distance_label.config(
            text=f"{math.hypot(end_effector[0] - self.tray_x, end_effector[1] - self.tray_y):.0f} px"
        )
        self.steps_label.config(text=f"{self.steps} / {MAX_STEPS}")
        self.action_label.config(text=self.last_action)
        self.gripper_badge.config(
            text="GRIPPER  ·  กำลังถือวัตถุ" if self.is_holding else "GRIPPER  ·  ว่าง",
            bg="#14532d" if self.is_holding else "#29364b",
            fg="#bbf7d0" if self.is_holding else "#e2e8f0",
        )
        for label, value in zip(self.q_rows, self.last_q_values):
            label.config(text=f"{value:.2f}")

    def _get_state(self):
        return np.asarray(
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
        ).reshape(1, 7)

    def _solve_ik(self, target_x, target_y):
        dx = target_x - self.base_x
        dy = self.base_y - target_y
        distance_squared = dx * dx + dy * dy
        min_reach = abs(self.link1 - self.link2)
        max_reach = self.link1 + self.link2
        if not min_reach**2 <= distance_squared <= max_reach**2:
            return None

        cos_theta2 = (
            distance_squared - self.link1**2 - self.link2**2
        ) / (2 * self.link1 * self.link2)
        cos_theta2 = min(1.0, max(-1.0, cos_theta2))
        candidates = []
        for sin_sign in (-1.0, 1.0):
            theta2 = math.atan2(
                sin_sign * math.sqrt(max(0.0, 1.0 - cos_theta2**2)),
                cos_theta2,
            )
            theta1 = math.atan2(dy, dx) - math.atan2(
                self.link2 * math.sin(theta2),
                self.link1 + self.link2 * math.cos(theta2),
            )
            angles = (math.degrees(theta1), math.degrees(theta2))
            if -90.0 <= angles[0] <= 180.0 and -150.0 <= angles[1] <= 150.0:
                cost = abs(angles[0] - self.theta1) + abs(
                    angles[1] - self.theta2
                )
                candidates.append((cost, angles))
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: candidate[0])[1]

    @staticmethod
    def _approach_angle(current, target, step):
        difference = target - current
        if abs(difference) <= step:
            return target
        return current + math.copysign(step, difference)

    def start_simulation(self):
        if self.model is None:
            self._set_status(f"ไม่สามารถเริ่มได้: {self.model_error}", error=True)
            return
        if self.is_running:
            return
        self.is_running = True
        self.steps = 0
        self.start_button.config(state=tk.DISABLED)
        self._set_status("โมเดลประเมิน Q-values · ระบบคำนวณมุม IK เพื่อหยิบและวาง")
        self.root.after(TICK_MS, self._simulation_step)

    def stop_simulation(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self._set_status("หยุดการจำลองแล้ว")

    def _simulation_step(self):
        if not self.is_running:
            return
        try:
            q_values = np.asarray(
                self.model(self._get_state(), training=False).numpy()[0],
                dtype=np.float32,
            )
            if q_values.shape != (6,) or not np.isfinite(q_values).all():
                raise ValueError(f"โมเดลคืนค่า Q-values ไม่ถูกต้อง: {q_values}")
        except Exception as exc:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self._set_status(f"เกิดข้อผิดพลาดขณะเรียกโมเดล: {exc}", error=True)
            return

        rl_action = int(np.argmax(q_values))
        self.last_q_values = q_values
        target_x, target_y = (
            (self.tray_x, self.tray_y)
            if self.is_holding
            else (self.obj_x, self.obj_y)
        )
        target_angles = self._solve_ik(target_x, target_y)
        if target_angles is None:
            target_name = "ถาด" if self.is_holding else "วัตถุ"
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self._set_status(
                f"{target_name} อยู่นอกระยะเอื้อมหรือขอบเขตข้อต่อ "
                "โปรดลากให้อยู่ในระยะของแขนกล",
                error=True,
            )
            return

        phase = "กำลังไปวาง" if self.is_holding else "กำลังไปหยิบ"
        self.last_action = f"{phase} · RL: {ACTION_NAMES[rl_action]}"
        self.steps += 1
        angle_step = 3.0
        self.theta1 = self._approach_angle(
            self.theta1, target_angles[0], angle_step
        )
        self.theta2 = self._approach_angle(
            self.theta2, target_angles[1], angle_step
        )

        _, end_effector = self._joint_positions()
        dist_to_object = math.hypot(
            end_effector[0] - self.obj_x, end_effector[1] - self.obj_y
        )
        placed = False
        if self.is_holding:
            self.obj_x, self.obj_y = end_effector
            if math.hypot(
                end_effector[0] - self.tray_x, end_effector[1] - self.tray_y
            ) < 18:
                self.is_holding = False
                placed = math.hypot(
                    self.obj_x - self.tray_x, self.obj_y - self.tray_y
                ) < 35
                self.last_action = f"วางวัตถุแล้ว · RL: {ACTION_NAMES[rl_action]}"
        elif dist_to_object < 18:
            self.is_holding = True
            self.obj_x, self.obj_y = end_effector
            self.last_action = f"หยิบวัตถุแล้ว · RL: {ACTION_NAMES[rl_action]}"

        self._draw_scene()
        if placed:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self._set_status(f"สำเร็จ! วางวัตถุบนถาดแล้ว · {self.steps} ขั้น")
        elif self.steps >= MAX_STEPS:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self._set_status("ครบ 150 ขั้นแล้ว · ยังวางวัตถุไม่สำเร็จ")
        else:
            self.root.after(TICK_MS, self._simulation_step)

    def _on_drag_start(self, event):
        if self.is_holding:
            return
        items = self.canvas.find_overlapping(
            event.x - 1, event.y - 1, event.x + 1, event.y + 1
        )
        self.drag_item = None
        for item in reversed(items):
            tags = self.canvas.gettags(item)
            if "object" in tags and not self.is_running:
                self.drag_item = "object"
                break
            if "tray" in tags:
                self.drag_item = "tray"
                break
        self.drag_x, self.drag_y = event.x, event.y

    def _on_drag_motion(self, event):
        if self.drag_item is None:
            return
        dx, dy = event.x - self.drag_x, event.y - self.drag_y
        if self.drag_item == "object":
            self.obj_x = min(self.canvas_width - 20, max(20, self.obj_x + dx))
            self.obj_y = min(self.canvas_height - 20, max(20, self.obj_y + dy))
        else:
            self.tray_x = min(self.canvas_width - 48, max(48, self.tray_x + dx))
            self.tray_y = min(self.canvas_height - 48, max(48, self.tray_y + dy))
        self.drag_x, self.drag_y = event.x, event.y
        self._draw_scene()

    def _on_drag_stop(self, event):
        self.drag_item = None
        self.drag_x, self.drag_y = event.x, event.y

    def reset_positions(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.theta1, self.theta2 = self.default_angles
        self.obj_x, self.obj_y = 270.0, 310.0
        self.tray_x, self.tray_y = 535.0, 310.0
        self.is_holding = False
        self.steps = 0
        self.last_q_values = np.zeros(6, dtype=np.float32)
        self.last_action = "รอเริ่มทำงาน"
        self._draw_scene()
        self._set_status("คืนตำแหน่งเริ่มต้นแล้ว")

    def _set_status(self, text, error=False):
        self.status_bar.config(
            text=text, fg="#fda4af" if error else "#94a3b8"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = ArmPickPlaceApp(root)
    root.mainloop()
