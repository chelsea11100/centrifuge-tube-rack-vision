# -*- coding: utf-8 -*-
"""
离心管孔板 — 实时相机 + 富 GUI（数据来自 detect_tubes 的 result 字典，与 result.json 同结构）

运行：
  ./run_camera.sh --gui
  或  python3 gui_detect.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from rack_config import AUTO_DETECT_GRID, GRID_COLS, GRID_ROWS
from tube_detect import detect_tubes

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

# 下拉显示名 → (backend, device_type)
CAMERA_OPTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("海康 · 自动", "hik", "auto"),
    ("海康 · GigE 网口", "hik", "gige"),
    ("海康 · USB", "hik", "usb"),
    ("UVC 摄像头 (OpenCV)", "opencv", "auto"),
)
CAMERA_LABELS = [o[0] for o in CAMERA_OPTIONS]


def _default_camera_label() -> str:
    bk = (os.environ.get("CAM_BACKEND") or "hik").strip().lower()
    dt = (os.environ.get("CAM_DEVICE_TYPE") or "auto").strip().lower()
    if bk == "opencv":
        return CAMERA_OPTIONS[3][0]
    if dt == "usb":
        return CAMERA_OPTIONS[2][0]
    if dt in ("gige", "gig"):
        return CAMERA_OPTIONS[1][0]
    return CAMERA_OPTIONS[0][0]


def _parse_camera_label(label: str) -> Tuple[str, str]:
    for disp, backend, device_type in CAMERA_OPTIONS:
        if disp == label:
            return backend, device_type
    return "hik", "auto"


def _ensure_mvs_env() -> None:
    if os.environ.get("MVCAM_COMMON_RUNENV"):
        return
    if os.path.isfile("/opt/MVS/lib/64/libMvCameraControl.so"):
        os.environ["MVCAM_COMMON_RUNENV"] = "/opt/MVS/lib"
        os.environ["LD_LIBRARY_PATH"] = "/opt/MVS/lib/64:" + os.environ.get("LD_LIBRARY_PATH", "")


def _focus_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    y0, y1 = h // 4, (3 * h) // 4
    x0, x1 = w // 4, (3 * w) // 4
    roi = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _boost_for_display(bgr: np.ndarray, gamma: float = 0.52) -> np.ndarray:
    """仅 GUI 显示用，识别仍用原始 raw。"""
    x = bgr.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi <= lo + 2.0:
        y = np.clip(x * (128.0 / max(float(x.mean()), 8.0)), 0, 255)
    else:
        y = (x - lo) * (255.0 / (hi - lo))
        y = np.clip(y, 0, 255)
    y = np.power(y / 255.0, gamma) * 255.0
    return y.astype(np.uint8)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _draw_focus_overlay(bgr: np.ndarray, sharpness: float, peak: float) -> np.ndarray:
    vis = bgr.copy()
    h, w = vis.shape[:2]
    y0, y1 = h // 4, (3 * h) // 4
    x0, x1 = w // 4, (3 * w) // 4
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cx, cy = w // 2, h // 2
    cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
    tip = "对焦预览(显示已增亮) — 旋转镜头，使清晰度尽量升高"
    cv2.putText(vis, tip, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    line = f"清晰度: {sharpness:.0f}   本次最高: {peak:.0f}   [停止] 后点 [开始实时]"
    cv2.putText(vis, line, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return vis


class CameraWorker(threading.Thread):
    """mode: preview=仅取流调焦；detect=识别。"""

    def __init__(
        self,
        out_q: queue.Queue,
        exposure: float,
        gain: float,
        binning: int,
        rows: int,
        cols: int,
        auto_grid: bool,
        backend: str = "hik",
        device_type: str = "auto",
        mode: str = "detect",
    ):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.exposure = exposure
        self.gain = gain
        self.binning = binning
        self.rows = rows
        self.cols = cols
        self.auto_grid = auto_grid
        self.backend = backend
        self.device_type = device_type
        self.mode = mode
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        _ensure_mvs_env()
        peak_sharp = 0.0
        try:
            from camera_backend import make_camera

            with make_camera(backend=self.backend, device_type=self.device_type) as cam:
                cam.set_binning(self.binning)
                if hasattr(cam, "configure_imaging"):
                    act_exp, act_gain = cam.configure_imaging(self.exposure, self.gain)
                else:
                    cam.set_exposure(self.exposure)
                    cam.set_gain(self.gain)
                    act_exp, act_gain = self.exposure, self.gain
                self.exposure, self.gain = act_exp, act_gain
                cam.start_grabbing()
                for _ in range(20):
                    cam.get_frame(timeout_ms=3000)
                dark_boost_left = 12
                while not self._stop.is_set():
                    raw = cam.get_frame(timeout_ms=2000)
                    if raw is None:
                        continue
                    mean = float(raw.mean())
                    if mean < 42.0 and dark_boost_left > 0 and hasattr(cam, "set_exposure"):
                        dark_boost_left -= 1
                        self.exposure = min(float(self.exposure) + 35000.0, 800000.0)
                        cam.set_exposure(self.exposure)
                        if mean < 18.0 and hasattr(cam, "set_gain"):
                            self.gain = min(float(self.gain) + 1.5, 24.0)
                            cam.set_gain(self.gain)
                        continue
                    if mean < 8.0:
                        vis = _boost_for_display(raw)
                        self.out_q.put(
                            (
                                "warn",
                                raw,
                                vis,
                                None,
                                f"画面过暗(均值{mean:.0f})，已自动加曝光至 {self.exposure:.0f} us；"
                                f"仍暗请点「加亮」或提高增益",
                            )
                        )
                        continue

                    if self.mode == "preview":
                        sharp = _focus_sharpness(raw)
                        peak_sharp = max(peak_sharp, sharp)
                        vis = _draw_focus_overlay(raw, sharp, peak_sharp)
                        self.out_q.put(
                            (
                                "preview",
                                raw,
                                vis,
                                None,
                                {
                                    "sharpness": sharp,
                                    "peak_sharpness": peak_sharp,
                                    "resolution": f"{raw.shape[1]}x{raw.shape[0]}",
                                    "backend": self.backend,
                                    "device_type": self.device_type,
                                    "exposure_us": self.exposure,
                                    "gain_db": self.gain,
                                    "binning": self.binning,
                                    "mean_gray": mean,
                                },
                            )
                        )
                        continue

                    try:
                        import rack_config as rc

                        prev_auto = rc.AUTO_DETECT_GRID
                        rc.AUTO_DETECT_GRID = self.auto_grid
                        try:
                            result, ann, mask = detect_tubes(
                                raw, rows=self.rows, cols=self.cols
                            )
                        finally:
                            rc.AUTO_DETECT_GRID = prev_auto
                        result["source"] = "camera"
                        result["camera"] = {
                            "exposure_us": self.exposure,
                            "gain_db": self.gain,
                            "binning": self.binning,
                            "mode": "gui_loop",
                            "backend": self.backend,
                            "device_type": self.device_type,
                        }
                        self.out_q.put(("ok", raw, ann, mask, result))
                    except RuntimeError as e:
                        self.out_q.put(("err", raw, raw.copy(), None, str(e)))
        except Exception as e:
            self.out_q.put(("fatal", None, None, None, str(e)))


class DetectGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("离心管孔板 — 实时识别")
        self.geometry("1280x820")
        self.minsize(1024, 640)

        self.rows_var = tk.IntVar(value=GRID_ROWS)
        self.cols_var = tk.IntVar(value=GRID_COLS)
        self.auto_grid = tk.BooleanVar(value=AUTO_DETECT_GRID)
        self.exposure = tk.DoubleVar(value=_env_float("EXPOSURE", 180000.0))
        self.gain = tk.DoubleVar(value=_env_float("GAIN", 16.0))
        self.binning = tk.IntVar(value=2)
        self.show_empty = tk.BooleanVar(value=True)
        self.camera_var = tk.StringVar(value=_default_camera_label())
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last: Optional[Tuple[np.ndarray, Dict[str, Any], np.ndarray]] = None
        self._worker: Optional[CameraWorker] = None
        self._run_mode: Optional[str] = None  # preview | detect
        self._q: queue.Queue = queue.Queue(maxsize=2)

        self._build_ui()
        self.after(80, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=6)
        top.pack(fill=tk.X)
        self.btn_focus = ttk.Button(top, text="对焦预览", command=self._start_focus)
        self.btn_focus.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="开始实时", command=self._start).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="停止", command=self._stop_cam).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="保存 JSON+图", command=self._save).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="相机").pack(side=tk.LEFT, padx=(12, 2))
        self.camera_combo = ttk.Combobox(
            top,
            width=18,
            textvariable=self.camera_var,
            values=CAMERA_LABELS,
            state="readonly",
        )
        self.camera_combo.pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="曝光(us)").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Entry(top, width=9, textvariable=self.exposure).pack(side=tk.LEFT)
        ttk.Label(top, text="增益").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(top, width=5, textvariable=self.gain).pack(side=tk.LEFT)
        ttk.Button(top, text="加亮", command=self._brighten_and_restart).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Binning").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(top, width=3, textvariable=self.binning, values=[1, 2, 4], state="readonly").pack(
            side=tk.LEFT
        )
        ttk.Label(top, text="行").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(top, width=3, textvariable=self.rows_var).pack(side=tk.LEFT)
        ttk.Label(top, text="列").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(top, width=3, textvariable=self.cols_var).pack(side=tk.LEFT)
        ttk.Checkbutton(top, text="自动推断行列", variable=self.auto_grid).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(top, text="表格显示全部孔位", variable=self.show_empty).pack(side=tk.LEFT, padx=4)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(paned)
        paned.add(left, weight=3)
        self.canvas = tk.Label(
            left,
            bg="#222",
            fg="#ccc",
            text="1. 选相机 → 2. 点「对焦预览」调镜头 → 3. 停止 → 4.「开始实时」",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        self.summary = tk.Text(right, height=14, wrap=tk.WORD, font=("Monospace", 10))
        self.summary.pack(fill=tk.X, pady=(0, 6))

        table_f = ttk.Frame(right)
        table_f.pack(fill=tk.BOTH, expand=True)
        cols = (
            "slot_id",
            "occupied",
            "cx",
            "cy",
            "hinge",
            "unify",
            "dir",
            "conf",
        )
        self.tree = ttk.Treeview(table_f, columns=cols, show="headings", height=24)
        headings = {
            "slot_id": "孔位",
            "occupied": "占位",
            "cx": "X(全图)",
            "cy": "Y(全图)",
            "hinge": "铰链角°",
            "unify": "补偿角°",
            "dir": "转向",
            "conf": "置信度",
        }
        widths = {"slot_id": 52, "occupied": 44, "cx": 72, "cy": 72, "hinge": 64, "unify": 64, "dir": 56, "conf": 56}
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER)
        vsb = ttk.Scrollbar(table_f, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _spawn_worker(self, mode: str) -> None:
        if Image is None:
            messagebox.showerror("缺少依赖", "请安装: pip install Pillow")
            return
        self._stop_cam()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        backend, device_type = _parse_camera_label(self.camera_var.get())
        self.camera_combo.configure(state="disabled")
        self._run_mode = mode
        self._worker = CameraWorker(
            self._q,
            float(self.exposure.get()),
            float(self.gain.get()),
            int(self.binning.get()),
            int(self.rows_var.get()),
            int(self.cols_var.get()),
            bool(self.auto_grid.get()),
            backend=backend,
            device_type=device_type,
            mode=mode,
        )
        self._worker.start()

    def _start_focus(self) -> None:
        self._spawn_worker("preview")

    def _start(self) -> None:
        self._spawn_worker("detect")

    def _brighten_and_restart(self) -> None:
        self.exposure.set(min(float(self.exposure.get()) + 50000.0, 800000.0))
        self.gain.set(min(float(self.gain.get()) + 2.0, 24.0))
        mode = self._run_mode or "preview"
        self._spawn_worker(mode)

    def _stop_cam(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None
        self._run_mode = None
        self.camera_combo.configure(state="readonly")

    def _on_close(self) -> None:
        self._stop_cam()
        self.destroy()

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                self._handle_frame(msg)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _handle_frame(self, msg) -> None:
        kind, raw, ann, mask, payload = msg
        if kind == "fatal":
            self._stop_cam()
            messagebox.showerror("相机", str(payload))
            return
        if raw is None:
            return
        if kind == "err":
            ann = payload if isinstance(payload, str) else ann
            result = None
            err = payload if isinstance(payload, str) else ""
            self._show_image(ann if ann is not None else raw)
            self.summary.delete("1.0", tk.END)
            self.summary.insert(tk.END, f"识别失败:\n{err}\n")
            return
        if kind == "warn":
            self._show_image(ann if ann is not None else raw)
            self.summary.delete("1.0", tk.END)
            self.summary.insert(tk.END, str(payload) + "\n")
            return
        if kind == "preview":
            meta = payload if isinstance(payload, dict) else {}
            vis = ann if ann is not None else raw
            self._show_image(vis)
            lines = [
                "【对焦预览】未做识别，仅用于调镜头焦距。",
                "旋转镜头，观察画面与「清晰度」升高；黄框为中心参考区。",
                "调焦建议：Binning=1、曝光 8~12 万 us、增益 10~14；孔板尽量填满画面。",
                "画面偏暗时先调曝光/增益，停止后再重新「对焦预览」。",
                "黄框/文字是调焦辅助，不是相机故障；识别请点「开始实时」。",
                f"分辨率: {meta.get('resolution', '—')}",
                f"相机: {meta.get('device_type') or meta.get('backend')}  "
                f"exp={meta.get('exposure_us')}  gain={meta.get('gain_db')}  bin={meta.get('binning')}",
                f"清晰度: {meta.get('sharpness', 0):.0f}   本次最高: {meta.get('peak_sharpness', 0):.0f}",
                "",
                "完成后点「停止」，再点「开始实时」进入识别。",
            ]
            self.summary.delete("1.0", tk.END)
            self.summary.insert(tk.END, "\n".join(lines))
            for i in self.tree.get_children():
                self.tree.delete(i)
            return

        result: Dict[str, Any] = payload
        self._last = (raw, result, mask if mask is not None else np.zeros(raw.shape[:2], np.uint8))
        self._show_image(ann)
        self._fill_summary(result)
        self._fill_table(result)

    def _show_image(self, bgr: np.ndarray) -> None:
        if Image is None:
            return
        bgr = _boost_for_display(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w, max_h = 780, 720
        s = min(max_w / w, max_h / h, 1.0)
        if s < 1.0:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        im = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.configure(image=self._photo, text="")

    def _fill_summary(self, r: Dict[str, Any]) -> None:
        p = r.get("plate", {})
        lines = [
            f"时间: {r.get('timestamp', '—')}",
            f"算法: {r.get('algorithm', '—')}",
            f"网格: {p.get('grid_rows')}×{p.get('grid_cols')} ({p.get('grid_source', 'config')})  总孔: {r.get('slot_count')}",
            f"占位: {r.get('occupied_count')}  →  {', '.join(r.get('occupied_slots') or [])}",
            f"白盖斑点: {r.get('cap_blobs_detected')}  孔洞峰点: {r.get('holes_detected')}",
            f"板心(全图px): {p.get('center_full')}  倾角: {p.get('angle_deg')}°",
            f"板尺寸(px): {p.get('size_px')}  孔距: {p.get('pitch_px')}",
            f"边距: {p.get('margin_xy')}  scale: {r.get('scale')}",
            f"A1原点: {p.get('a1_at', '—')}  ({p.get('origin_mode', '—')})",
        ]
        cam = r.get("camera")
        if cam:
            src = cam.get("device_type") or cam.get("backend") or "—"
            lines.append(
                f"相机: {src}  exp={cam.get('exposure_us')} gain={cam.get('gain_db')} bin={cam.get('binning')}"
            )
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, "\n".join(lines))

    def _fill_table(self, r: Dict[str, Any]) -> None:
        for i in self.tree.get_children():
            self.tree.delete(i)
        items = r.get("items") or []
        show_all = self.show_empty.get()
        for it in items:
            if not show_all and not it.get("occupied"):
                continue
            occ = "是" if it.get("occupied") else ""
            tag = "occ" if it.get("occupied") else ""
            self.tree.insert(
                "",
                tk.END,
                values=(
                    it.get("slot_id"),
                    occ,
                    it.get("cx_full"),
                    it.get("cy_full"),
                    it.get("hinge_angle_deg") if it.get("hinge_angle_deg") is not None else "",
                    it.get("rotate_to_unify_deg") if it.get("rotate_to_unify_deg") is not None else "",
                    it.get("rotate_direction") or "",
                    it.get("confidence") if it.get("occupied") else "",
                ),
                tags=(tag,),
            )
        self.tree.tag_configure("occ", background="#e8f5e9")

    def _save(self) -> None:
        if not self._last:
            messagebox.showinfo("保存", "尚无成功识别帧")
            return
        raw, result, mask = self._last
        import json

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = os.path.join("results", stamp)
        os.makedirs(out_dir, exist_ok=True)
        import rack_config as rc

        prev_auto = rc.AUTO_DETECT_GRID
        rc.AUTO_DETECT_GRID = bool(self.auto_grid.get())
        try:
            _, ann, _ = detect_tubes(
                raw, rows=int(self.rows_var.get()), cols=int(self.cols_var.get())
            )
        finally:
            rc.AUTO_DETECT_GRID = prev_auto
        cv2.imencode(".jpg", raw)[1].tofile(os.path.join(out_dir, "raw.jpg"))
        cv2.imencode(".jpg", ann)[1].tofile(os.path.join(out_dir, "annotated.jpg"))
        cv2.imencode(".jpg", mask)[1].tofile(os.path.join(out_dir, "mask_plate.jpg"))
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("保存", f"已写入 {out_dir}/")


def main() -> int:
    if Image is None:
        print("请先安装 Pillow: pip install Pillow", file=sys.stderr)
        return 1
    app = DetectGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
