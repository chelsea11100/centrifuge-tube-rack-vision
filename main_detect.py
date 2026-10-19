# -*- coding: utf-8 -*-
"""
矩形反光孔板 — 离心管识别主入口（仅海康相机，不读本地图）

用法：
  ./run_camera.sh --show              # 抓拍一帧 → 识别 → 落盘
  ./run_camera.sh --loop --show       # 实时取流识别（q 退出，s 保存当前帧）

输出：results/<时间戳>/ 下 raw.jpg / annotated.jpg / mask_plate.jpg / result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from rack_config import GRID_COLS, GRID_ROWS
from tube_detect import detect_tubes


def _ensure_mvs_env() -> None:
    if os.environ.get("MVCAM_COMMON_RUNENV"):
        return
    if os.path.isfile("/opt/MVS/lib/64/libMvCameraControl.so"):
        os.environ["MVCAM_COMMON_RUNENV"] = "/opt/MVS/lib"
        os.environ["LD_LIBRARY_PATH"] = (
            "/opt/MVS/lib/64:" + os.environ.get("LD_LIBRARY_PATH", "")
        )


def grab_camera(
    exposure_us: float,
    gain_db: float,
    binning: int,
    camera: str = "auto",
    backend: str = "hik",
) -> np.ndarray:
    from camera_backend import describe_cameras, make_camera

    describe_cameras()
    with make_camera(backend=backend, device_type=camera) as cam:
        cam.set_binning(binning)
        if hasattr(cam, "configure_imaging"):
            cam.configure_imaging(exposure_us, gain_db)
        else:
            cam.set_exposure(exposure_us)
            cam.set_gain(gain_db)
        cam.start_grabbing()
        img = None
        for _ in range(5):
            img = cam.get_frame(timeout_ms=3000)
        if img is None:
            raise RuntimeError("相机取流失败")
        print(f"抓拍成功: {img.shape[1]}x{img.shape[0]}")
        return img


def run_camera_loop(
    exposure_us: float,
    gain_db: float,
    binning: int,
    rows: int,
    cols: int,
    out_root: str,
    show: bool,
    camera: str = "auto",
    backend: str = "hik",
) -> int:
    """相机连续取流 + 识别；不读本地文件。"""
    from camera_backend import describe_cameras, make_camera

    describe_cameras()
    print("实时识别中：q=退出  s=保存当前帧到 results/（请先关闭官方 MVS 客户端）")
    with make_camera(backend=backend, device_type=camera) as cam:
        cam.set_binning(binning)
        if hasattr(cam, "configure_imaging"):
            cam.configure_imaging(exposure_us, gain_db)
        else:
            cam.set_exposure(exposure_us)
            cam.set_gain(gain_db)
        cam.start_grabbing()
        for _ in range(20):
            cam.get_frame(timeout_ms=3000)

        last_save_dir = ""
        miss_msg = ""
        while True:
            raw = cam.get_frame(timeout_ms=2000)
            if raw is None:
                continue
            if float(raw.mean()) < 8.0:
                miss_msg = "画面过暗，等待曝光稳定..."
                annotated = raw.copy()
                result = {"occupied_count": 0, "cap_blobs_detected": 0, "holes_detected": 0}
            else:
                try:
                    result, annotated, plate_mask = detect_tubes(raw, rows=rows, cols=cols)
                    miss_msg = ""
                except RuntimeError as e:
                    miss_msg = str(e)
                    annotated = raw.copy()
                    plate_mask = np.zeros(raw.shape[:2], np.uint8)
                    result = {"occupied_count": 0, "cap_blobs_detected": 0, "holes_detected": 0}
            result["source"] = "camera"
            result["camera"] = {
                "exposure_us": exposure_us,
                "gain_db": gain_db,
                "binning": binning,
                "mode": "loop",
                "device_type": camera,
                "backend": backend,
            }
            occ = result["occupied_count"]
            line = (
                f"occ {occ}/{rows * cols}  caps {result['cap_blobs_detected']}  "
                f"holes {result.get('holes_detected', 0)}"
            )
            if miss_msg:
                line = miss_msg[:80]
            cv2.putText(
                annotated,
                line + "  [q quit  s save]",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if not miss_msg else (0, 120, 255),
                2,
            )

            if show:
                vis = annotated
                h, w = vis.shape[:2]
                s = 900.0 / max(h, w)
                if s < 1:
                    vis = cv2.resize(vis, (int(w * s), int(h * s)))
                cv2.imshow("camera_detect", vis)
                key = cv2.waitKey(1) & 0xFF
            else:
                print(line, end="\r", flush=True)
                key = 0

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                if miss_msg:
                    print("\n当前帧无法识别，未保存（对准孔板后再按 s）")
                    continue
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_dir = os.path.join(out_root, stamp)
                save_outputs(out_dir, raw, annotated, plate_mask, result)
                last_save_dir = out_dir
                print(f"\n已保存: {out_dir}/")

    if show:
        cv2.destroyAllWindows()
    if last_save_dir:
        print(f"最后一次保存: {last_save_dir}/")
    return 0


def save_outputs(
    out_dir: str,
    raw: np.ndarray,
    annotated: np.ndarray,
    plate_mask: np.ndarray,
    result: dict,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cv2.imencode(".jpg", raw)[1].tofile(os.path.join(out_dir, "raw.jpg"))
    cv2.imencode(".jpg", annotated)[1].tofile(os.path.join(out_dir, "annotated.jpg"))
    cv2.imencode(".jpg", plate_mask)[1].tofile(os.path.join(out_dir, "mask_plate.jpg"))
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="矩形孔板离心管：海康相机识别（不读本地图）")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="实时取流识别（默认：只抓拍一帧）",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=float(os.environ.get("EXPOSURE", "180000") or 180000),
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=float(os.environ.get("GAIN", "16") or 16),
    )
    parser.add_argument("--binning", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--rows", type=int, default=GRID_ROWS, help="孔板行数")
    parser.add_argument("--cols", type=int, default=GRID_COLS, help="孔板列数")
    parser.add_argument("--out-root", type=str, default="results")
    parser.add_argument("--show", action="store_true", help="弹出 OpenCV 窗口")
    parser.add_argument(
        "--camera",
        choices=("auto", "usb", "gige", "gig"),
        default=os.environ.get("CAM_DEVICE_TYPE", "auto") or "auto",
        help="海康相机类型：usb / gige（也可用环境变量 CAM_DEVICE_TYPE）",
    )
    parser.add_argument(
        "--backend",
        choices=("hik", "opencv", "auto"),
        default=os.environ.get("CAM_BACKEND", "hik") or "hik",
        help="hik=MVS；opencv=普通 UVC（CAM_BACKEND）",
    )
    args, unknown = parser.parse_known_args()
    if args.camera == "gig":
        args.camera = "gige"
    if unknown:
        for u in unknown:
            if u in ("--image", "-image") or u.startswith("--image="):
                print(
                    "本程序已禁用 --image 本地图模式。\n"
                    "请用相机：  ./run_camera.sh --show\n"
                    "或实时：    ./run_camera.sh --loop --show",
                    file=sys.stderr,
                )
                return 2

    if args.loop:
        return run_camera_loop(
            args.exposure,
            args.gain,
            args.binning,
            args.rows,
            args.cols,
            args.out_root,
            args.show,
            args.camera,
            args.backend,
        )

    print("相机抓拍中（请先关闭官方 MVS 客户端）...")
    raw = grab_camera(
        args.exposure, args.gain, args.binning, args.camera, args.backend
    )
    source = "camera"
    cv2.imencode(".jpg", raw)[1].tofile("capture_latest.jpg")
    print("已同步写入 capture_latest.jpg")

    result, annotated, plate_mask = detect_tubes(raw, rows=args.rows, cols=args.cols)
    result["source"] = source
    if source == "camera":
        result["camera"] = {
            "exposure_us": args.exposure,
            "gain_db": args.gain,
            "binning": args.binning,
            "device_type": args.camera,
            "backend": args.backend,
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = os.path.join(args.out_root, stamp)
    save_outputs(out_dir, raw, annotated, plate_mask, result)

    print("\n========== 识别摘要 ==========")
    print(
        f"孔板中心: {result['plate']['center_full']}  "
        f"尺寸px={result['plate']['size_px']}  angle={result['plate']['angle_deg']}"
    )
    print(
        f"网格: {args.rows}x{args.cols}  "
        f"占位: {result['occupied_count']}  "
        f"白盖斑点: {result['cap_blobs_detected']}  "
        f"→ {result['occupied_slots']}"
    )
    for it in result["items"]:
        if not it["occupied"]:
            continue
        print(
            f"  {it['slot_id']}: xy=({it['cx_full']:.1f},{it['cy_full']:.1f})  "
            f"hinge={it['hinge_angle_deg']}  "
            f"unify={it['rotate_to_unify_deg']}({it['rotate_direction']})  "
            f"conf={it['confidence']}"
        )
    print(f"\n已保存: {out_dir}/")
    print("  - result.json / annotated.jpg / raw.jpg / mask_plate.jpg")

    if args.show:
        vis = annotated
        h, w = vis.shape[:2]
        s = 900.0 / max(h, w)
        if s < 1:
            vis = cv2.resize(vis, (int(w * s), int(h * s)))
        cv2.imshow("annotated", vis)
        cv2.imshow("plate_mask", cv2.resize(plate_mask, (vis.shape[1], vis.shape[0])))
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[失败] {e}", file=sys.stderr)
        raise SystemExit(1)
