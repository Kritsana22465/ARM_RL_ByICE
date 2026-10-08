import math
import tkinter as tk
from tkinter import ttk

import numpy as np
import tensorflow as tf

from model import (
    ANGLE_LIMITS,
    CONTROL_NAMES,
    MAX_STEPS,
    MODEL_PATH,
    STATE_DIM,
    evaluate_policy,
    get_joint_positions,
    make_state,
    predict_control,
    solve_ik,
)


TICK_MS = 50
CANVAS_WIDTH = 760
CANVAS_HEIGHT = 520
PROJECTION_SCALE = 1.15


class ArmPickPlace3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arm RL Studio | 3D Pick & Place")
        self.root.geometry("1190x790")
        self.root.minsize(1060, 720)
        self.root.configure(bg="#111827")

        self.base_x = 0.0
        self.base_y = 0.0
        self.base_z = 0.0
        self.angles = [0.0, 45.0, 45.0]
        self.default_angles = tuple(self.angles)
        self.obj_position = [-80.0, -35.0, 12.0]
        self.tray_position = [95.0, 45.0, 8.0]
        self.default_object = tuple(self.obj_position)
        self.default_tray = tuple(self.tray_position)
        self.is_holding = False
        self.has_picked = False
        self.is_running = False
        self.steps = 0
        self.score = 0.0
        self.run_history = []
        self.drag_item = None
        self.drag_x = 0
        self.drag_y = 0
        self.last_controls = np.zeros(5, dtype=np.float32)
        self.last_action = "พร้อมเริ่มภารกิจ"

        self.model = None
        self.model_error = None
        self.validation_metrics = None
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if (
                self.model.input_shape[-1] != STATE_DIM
                or self.model.output_shape[-1] != 4
            ):
                raise ValueError(
                    f"โมเดลต้องรับ state {STATE_DIM} ค่าและคืน control 4 ค่า "
                    f"(พบ {self.model.input_shape} -> {self.model.output_shape})"
                )
            self.validation_metrics = evaluate_policy(self.model, episodes=100)
        except Exception as exc:
            self.model_error = str(exc)

        self._setup_styles()
        self._build_ui()
        self._draw_scene()
        self._update_dashboard()
        if self.model_error:
            self._set_status(f"โหลด/ประเมินโมเดลไม่สำเร็จ: {self.model_error}", error=True)
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
        header.pack(fill=tk.X, padx=22, pady=(16, 10))
        tk.Label(
            header, text="ARM RL STUDIO · 3D", bg="#111827", fg="#f8fafc",
            font=("Segoe UI", 18, "bold")
        ).pack(side=tk.LEFT)
        tk.Label(
            header, text="Yaw · Shoulder · Elbow · XYZ", bg="#111827",
            fg="#94a3b8", font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(14, 0), pady=(5, 0))

        body = tk.Frame(self.root, bg="#111827")
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 12))
        workspace = tk.Frame(
            body, bg="#172033", highlightthickness=1,
            highlightbackground="#29364b"
        )
        workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            workspace, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
            bg="#0b1220", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_stop)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)

        controls = tk.Frame(workspace, bg="#172033")
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.start_button = self._button(
            controls, "▶  เริ่มภารกิจ", self.start_simulation,
            "#22c55e", "#052e16"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self._button(
            controls, "หยุด", self.stop_simulation, "#fb7185", "#4c0519"
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._button(
            controls, "คืนค่าเริ่มต้น", self.reset_positions,
            "#29364b", "#e2e8f0"
        ).pack(side=tk.LEFT)
        tk.Label(
            controls,
            text="ลากวัตถุ/ถาดเพื่อย้าย XY · ล้อเมาส์ปรับความสูง Z",
            bg="#172033", fg="#94a3b8", font=("Segoe UI", 9)
        ).pack(side=tk.RIGHT, padx=4)

        side = tk.Frame(
            body, bg="#172033", width=300,
            highlightthickness=1, highlightbackground="#29364b"
        )
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        side.pack_propagate(False)
        self._section_title(side, "สถานะโมเดล 3D")
        tk.Label(
            side,
            text=("●  พร้อมใช้งาน" if self.model else "●  โหลดไม่สำเร็จ"),
            bg="#172033", fg="#4ade80" if self.model else "#fb7185",
            font=("Segoe UI", 10, "bold"), anchor=tk.W
        ).pack(fill=tk.X, padx=15)
        tk.Label(
            side, text=MODEL_PATH.name, bg="#172033", fg="#94a3b8",
            font=("Consolas", 9), anchor=tk.W
        ).pack(fill=tk.X, padx=15, pady=(3, 8))

        self._separator(side)
        self._section_title(side, "ผลประเมินโมเดล")
        self.validation_success_label = self._info_row(side, "Validation success", "—")
        self.validation_score_label = self._info_row(side, "Validation score", "—")
        self.validation_steps_label = self._info_row(side, "เฉลี่ย steps", "—")
        self._separator(side)
        self._section_title(side, "ผลภารกิจนี้")
        self.current_score_label = self._info_row(side, "Score", "0.00")
        self.steps_label = self._info_row(side, "Steps", f"0 / {MAX_STEPS}")
        self.success_rate_label = self._info_row(side, "Success rate", "—")
        self.average_score_label = self._info_row(side, "คะแนนเฉลี่ย", "—")

        self._separator(side)
        self._section_title(side, "ข้อต่อแขนกล 3 มิติ")
        self.joint_labels = []
        for name, limits in zip(("Yaw", "Shoulder", "Elbow"), ANGLE_LIMITS):
            label, bar = self._joint_row(side, name, limits)
            self.joint_labels.append((label, bar, limits))

        self._separator(side)
        self.ee_label = self._info_row(side, "ปลายแขน XYZ", "(0, 0, 0)")
        self.object_distance_label = self._info_row(side, "ถึงวัตถุ", "0")
        self.tray_distance_label = self._info_row(side, "ถึงถาด", "0")
        self.gripper_badge = tk.Label(
            side, text="GRIPPER · ว่าง", bg="#29364b", fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"), pady=7
        )
        self.gripper_badge.pack(fill=tk.X, padx=14, pady=(10, 5))

        self._separator(side)
        self._section_title(side, "ค่าควบคุมจากโมเดล")
        self.control_rows = []
        for name in CONTROL_NAMES:
            row = tk.Frame(side, bg="#172033")
            row.pack(fill=tk.X, padx=15, pady=2)
            tk.Label(
                row, text=name, bg="#172033", fg="#94a3b8",
                font=("Segoe UI", 8), anchor=tk.W
            ).pack(side=tk.LEFT)
            value = tk.Label(
                row, text="—", bg="#172033", fg="#cbd5e1",
                font=("Consolas", 8), anchor=tk.E
            )
            value.pack(side=tk.RIGHT)
            self.control_rows.append(value)

        self.action_label = tk.Label(
            side, text=self.last_action, bg="#172033", fg="#fbbf24",
            font=("Segoe UI", 9, "bold"), wraplength=255, anchor=tk.W
        )
        self.action_label.pack(fill=tk.X, padx=15, pady=8)
        self.status_bar = tk.Label(
            self.root, text="พร้อม", bg="#0b1220", fg="#94a3b8",
            font=("Segoe UI", 9), anchor=tk.W, padx=22, pady=8
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._update_model_metrics()

    @staticmethod
    def _button(parent, text, command, background, foreground):
        return tk.Button(
            parent, text=text, command=command, bg=background, fg=foreground,
            activebackground=background, activeforeground=foreground,
            font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
            padx=12, pady=8, cursor="hand2"
        )

    @staticmethod
    def _section_title(parent, text):
        tk.Label(
            parent, text=text.upper(), bg="#172033", fg="#7dd3fc",
            font=("Segoe UI", 9, "bold"), anchor=tk.W
        ).pack(fill=tk.X, padx=15, pady=(9, 5))

    @staticmethod
    def _separator(parent):
        tk.Frame(parent, bg="#29364b", height=1).pack(fill=tk.X, padx=13, pady=3)

    def _info_row(self, parent, title, value):
        row = tk.Frame(parent, bg="#172033")
        row.pack(fill=tk.X, padx=15, pady=2)
        tk.Label(
            row, text=title, bg="#172033", fg="#94a3b8",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT)
        label = tk.Label(
            row, text=value, bg="#172033", fg="#e2e8f0",
            font=("Consolas", 9), anchor=tk.E
        )
        label.pack(side=tk.RIGHT)
        return label

    def _joint_row(self, parent, name, limits):
        row = tk.Frame(parent, bg="#172033")
        row.pack(fill=tk.X, padx=15, pady=(1, 0))
        label = tk.Label(
            row, text=f"{name}: 0.0°", bg="#172033", fg="#fbbf24",
            font=("Consolas", 9)
        )
        label.pack(side=tk.LEFT)
        bar = ttk.Progressbar(
            parent, maximum=limits[1] - limits[0],
            style="Arm.Horizontal.TProgressbar"
        )
        bar.pack(fill=tk.X, padx=15, pady=(0, 4))
        return label, bar

    def _project(self, point):
        x, y, z = point
        center_x = max(self.canvas.winfo_width(), CANVAS_WIDTH) * 0.5
        ground_y = max(self.canvas.winfo_height(), CANVAS_HEIGHT) * 0.77
        scale = PROJECTION_SCALE
        return (
            center_x + scale * (x - y) * 0.72,
            ground_y - scale * z + scale * (x + y) * 0.24,
        )

    def _draw_scene(self):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), CANVAS_WIDTH)
        height = max(self.canvas.winfo_height(), CANVAS_HEIGHT)
        for x in range(0, width, 40):
            self.canvas.create_line(x, 0, x, height, fill="#141f30")
        for y in range(0, height, 40):
            self.canvas.create_line(0, y, width, y, fill="#141f30")

        self._draw_ground_grid()
        base = (self.base_x, self.base_y, self.base_z)
        shoulder, elbow, end_effector = get_joint_positions(
            *self.angles, *base
        )
        base_screen = self._project(base)
        shoulder_screen = self._project(shoulder)
        elbow_screen = self._project(elbow)
        ee_screen = self._project(end_effector)

        self._draw_object(self.obj_position)
        self._draw_tray(self.tray_position)
        self.canvas.create_oval(
            base_screen[0] - 18, base_screen[1] - 8,
            base_screen[0] + 18, base_screen[1] + 8,
            fill="#334155", outline="#64748b", width=2
        )
        self.canvas.create_line(
            base_screen[0], base_screen[1], shoulder_screen[0], shoulder_screen[1],
            fill="#475569", width=9
        )
        self.canvas.create_line(
            shoulder_screen[0], shoulder_screen[1], elbow_screen[0], elbow_screen[1],
            fill="#38bdf8", width=12, capstyle=tk.ROUND
        )
        self.canvas.create_line(
            elbow_screen[0], elbow_screen[1], ee_screen[0], ee_screen[1],
            fill="#818cf8", width=9, capstyle=tk.ROUND
        )
        for point, radius, color in (
            (shoulder_screen, 8, "#7dd3fc"),
            (elbow_screen, 8, "#a5b4fc"),
            (ee_screen, 10, "#f472b6"),
        ):
            self.canvas.create_oval(
                point[0] - radius, point[1] - radius,
                point[0] + radius, point[1] + radius,
                fill=color, outline="#0b1220", width=2
            )
        if self.is_holding:
            self.obj_position[:] = end_effector
            self.canvas.tag_raise("object")
        self._update_dashboard(end_effector)

    def _draw_ground_grid(self):
        for coordinate in range(-120, 141, 40):
            a = self._project((coordinate, -120, 0))
            b = self._project((coordinate, 120, 0))
            c = self._project((-120, coordinate, 0))
            d = self._project((120, coordinate, 0))
            self.canvas.create_line(*a, *b, fill="#1e2a3d")
            self.canvas.create_line(*c, *d, fill="#1e2a3d")
        origin = self._project((0, 0, 0))
        x_axis = self._project((100, 0, 0))
        y_axis = self._project((0, 100, 0))
        z_axis = self._project((0, 0, 90))
        for endpoint, color, label in (
            (x_axis, "#fb7185", "X"),
            (y_axis, "#4ade80", "Y"),
            (z_axis, "#38bdf8", "Z"),
        ):
            self.canvas.create_line(
                *origin, *endpoint, fill=color, width=2, arrow=tk.LAST
            )
            self.canvas.create_text(
                endpoint[0], endpoint[1] - 10, text=label,
                fill=color, font=("Segoe UI", 9, "bold")
            )
        self.canvas.create_text(
            15, 15, text="3D WORKSPACE · X / Y / Z",
            fill="#64748b", font=("Segoe UI", 8, "bold"), anchor=tk.NW
        )

    def _draw_object(self, position):
        x, y = self._project(position)
        radius = 11
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill="#4ade80" if self.is_holding else "#fb923c",
            outline="#ffedd5", width=2, tags=("object",)
        )
        self.canvas.create_text(
            x, y + 19, text=f"วัตถุ z={position[2]:.0f}",
            fill="#fed7aa", font=("Segoe UI", 8, "bold")
        )

    def _draw_tray(self, position):
        x, y = self._project(position)
        self.canvas.create_polygon(
            x - 28, y - 10, x + 28, y - 10,
            x + 20, y + 10, x - 20, y + 10,
            fill="#8b5cf6", outline="#c4b5fd", width=2, tags=("tray",)
        )
        self.canvas.create_text(
            x, y + 20, text=f"ถาด z={position[2]:.0f}",
            fill="#c4b5fd", font=("Segoe UI", 8, "bold")
        )

    def _update_dashboard(self, end_effector=None):
        if end_effector is None:
            _, _, end_effector = get_joint_positions(
                *self.angles, self.base_x, self.base_y, self.base_z
            )
        for label, bar, limits, angle in zip(
            (item[0] for item in self.joint_labels),
            (item[1] for item in self.joint_labels),
            (item[2] for item in self.joint_labels),
            self.angles,
        ):
            label.config(text=f"{label.cget('text').split(':')[0]}: {angle:.1f}°")
            bar["value"] = angle - limits[0]
        self.ee_label.config(text="(" + ", ".join(f"{v:.1f}" for v in end_effector) + ")")
        self.object_distance_label.config(
            text=f"{math.dist(end_effector, self.obj_position):.1f}"
        )
        self.tray_distance_label.config(
            text=f"{math.dist(end_effector, self.tray_position):.1f}"
        )
        self.current_score_label.config(text=f"{self.score:.2f}")
        self.steps_label.config(text=f"{self.steps} / {MAX_STEPS}")
        self.gripper_badge.config(
            text="GRIPPER · กำลังถือ" if self.is_holding else "GRIPPER · ว่าง",
            bg="#14532d" if self.is_holding else "#29364b",
            fg="#bbf7d0" if self.is_holding else "#e2e8f0",
        )
        values = (
            f"{self.last_controls[0]:.1f}°",
            f"{self.last_controls[1]:.1f}°",
            f"{self.last_controls[2]:.1f}°",
            "จับ/วาง" if self.last_controls[3] >= 0.5 else "เคลื่อนที่",
            "จบ" if self.last_controls[4] >= 0.5 else "ดำเนินงาน",
        )
        for label, value in zip(self.control_rows, values):
            label.config(text=value)
        self.action_label.config(text=self.last_action)
        if self.run_history:
            successes = sum(result[0] for result in self.run_history)
            self.success_rate_label.config(
                text=f"{successes / len(self.run_history):.1%} ({len(self.run_history)} รอบ)"
            )
            self.average_score_label.config(
                text=f"{np.mean([result[1] for result in self.run_history]):.2f}"
            )

    def _update_model_metrics(self):
        if not self.validation_metrics:
            return
        metrics = self.validation_metrics
        self.validation_success_label.config(text=f"{metrics['success_rate']:.1%}")
        self.validation_score_label.config(text=f"{metrics['average_score']:.2f}")
        self.validation_steps_label.config(text=f"{metrics['average_steps']:.1f}")

    def _get_state(self):
        return make_state(
            self.angles, self.obj_position, self.tray_position,
            self.is_holding, self.has_picked,
        ).reshape(1, STATE_DIM)

    def start_simulation(self):
        if self.model is None:
            self._set_status(f"เริ่มไม่ได้: {self.model_error}", error=True)
            return
        if self.is_running:
            return
        self.is_running = True
        self.steps = 0
        self.score = 0.0
        self.start_button.config(state=tk.DISABLED)
        self._set_status("โมเดลกำลังคำนวณการเคลื่อนที่ในพิกัด XYZ")
        self.root.after(TICK_MS, self._simulation_step)

    def stop_simulation(self):
        if self.is_running:
            self._finish_episode(False, "หยุดการจำลองแล้ว")

    def _simulation_step(self):
        if not self.is_running:
            return
        try:
            controls = np.asarray(
                predict_control(self.model, self._get_state()), dtype=np.float32
            )
            if controls.shape != (5,) or not np.isfinite(controls).all():
                raise ValueError(f"Control output must contain 5 finite values: {controls}")
        except Exception as exc:
            self._finish_episode(False, f"เกิดข้อผิดพลาดจากโมเดล: {exc}", record=False)
            return

        self.last_controls = controls
        target_angles = controls[:3]
        self.angles = [
            self._approach_yaw(self.angles[0], float(target_angles[0])),
            self._approach_angle(self.angles[1], float(target_angles[1]), ANGLE_LIMITS[1]),
            self._approach_angle(self.angles[2], float(target_angles[2]), ANGLE_LIMITS[2]),
        ]
        self.steps += 1
        grip_command = controls[3] >= 0.5
        finish_command = controls[4] >= 0.5
        _, _, end_effector = get_joint_positions(
            *self.angles, self.base_x, self.base_y, self.base_z
        )
        target = self.tray_position if self.is_holding else self.obj_position
        distance = math.dist(end_effector, target)
        self.score += -0.15 + max(-2.0, 1.0 - distance / 100.0)

        if self.is_holding:
            self.obj_position[:] = end_effector
            if grip_command:
                self.is_holding = False
                placed_distance = math.dist(self.obj_position, self.tray_position)
                if self.has_picked and placed_distance <= 24.0:
                    self.score += 100.0
                    self._draw_scene()
                    self._finish_episode(True, "สำเร็จ! วางวัตถุบนถาดแล้ว")
                    return
            self.last_action = "กำลังนำวัตถุไปวาง" if self.is_holding else "วางวัตถุแล้ว"
        elif grip_command:
            self.is_holding = True
            self.has_picked = True
            self.obj_position[:] = end_effector
            self.score += 10.0
            self.last_action = "หยิบวัตถุแล้ว"
        else:
            self.last_action = "โมเดลกำลังเคลื่อนที่ใน XYZ"

        self._draw_scene()
        if finish_command and self.has_picked and not self.is_holding:
            self._finish_episode(True, "สำเร็จ! โมเดลยืนยันว่าภารกิจเสร็จแล้ว")
        elif self.steps >= MAX_STEPS:
            self._finish_episode(False, "ครบจำนวนขั้น · ยังทำภารกิจไม่สำเร็จ")
        else:
            self.root.after(TICK_MS, self._simulation_step)

    @staticmethod
    def _approach_angle(current, target, limits):
        difference = target - current
        if abs(difference) <= 4.0:
            return float(np.clip(target, *limits))
        return float(np.clip(current + math.copysign(4.0, difference), *limits))

    @staticmethod
    def _approach_yaw(current, target):
        difference = (target - current + 180.0) % 360.0 - 180.0
        if abs(difference) <= 4.0:
            return target
        result = current + math.copysign(4.0, difference)
        return (result + 180.0) % 360.0 - 180.0

    def _finish_episode(self, success, message, record=True):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        if record:
            self.run_history.append((success, self.score))
        self._update_dashboard()
        self._set_status(f"{message} · Score {self.score:.2f}")

    def _on_drag_start(self, event):
        if self.is_holding or self.is_running:
            return
        self.drag_item = self._hit_target(event.x, event.y)
        self.drag_x, self.drag_y = event.x, event.y

    def _hit_target(self, x, y):
        items = self.canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)
        for item in reversed(items):
            tags = self.canvas.gettags(item)
            if "object" in tags:
                return "object"
            if "tray" in tags:
                return "tray"
        return None

    def _on_drag_motion(self, event):
        if self.drag_item is None:
            return
        dx = (event.x - self.drag_x) / PROJECTION_SCALE
        dy = (event.y - self.drag_y) / PROJECTION_SCALE
        world_dx = (dx / 0.72 + dy / 0.24) * 0.5
        world_dy = (dy / 0.24 - dx / 0.72) * 0.5
        position = self.obj_position if self.drag_item == "object" else self.tray_position
        candidate = (
            float(np.clip(position[0] + world_dx, -120, 120)),
            float(np.clip(position[1] + world_dy, -120, 120)),
            position[2],
        )
        if solve_ik(*candidate) is None:
            self._set_status("ตำแหน่งอยู่นอกระยะเอื้อมของแขนกล", error=True)
            return
        position[0], position[1] = candidate[:2]
        self.drag_x, self.drag_y = event.x, event.y
        self._draw_scene()

    def _on_drag_stop(self, event):
        self.drag_item = None
        self.drag_x, self.drag_y = event.x, event.y

    def _on_mouse_wheel(self, event):
        target = self.drag_item or self._hit_target(event.x, event.y)
        if target is None:
            return
        position = self.obj_position if target == "object" else self.tray_position
        candidate = (
            position[0], position[1],
            float(np.clip(position[2] + (5 if event.delta > 0 else -5), 0, 80)),
        )
        if solve_ik(*candidate) is None:
            self._set_status("ความสูงอยู่นอกระยะเอื้อมของแขนกล", error=True)
            return
        position[2] = candidate[2]
        self._draw_scene()

    def reset_positions(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.angles[:] = self.default_angles
        self.obj_position[:] = self.default_object
        self.tray_position[:] = self.default_tray
        self.is_holding = False
        self.has_picked = False
        self.steps = 0
        self.score = 0.0
        self.last_controls[:] = 0
        self.last_action = "พร้อมเริ่มภารกิจ"
        self._draw_scene()
        self._set_status("คืนค่าตำแหน่งเริ่มต้นแล้ว")

    def _set_status(self, text, error=False):
        self.status_bar.config(
            text=text, fg="#fda4af" if error else "#94a3b8"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = ArmPickPlace3DApp(root)
    root.mainloop()
