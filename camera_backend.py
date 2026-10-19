# -*- coding: utf-8 -*-
"""相机入口：海康 GigE/USB（MVS）或 OpenCV UVC。识别逻辑仍用 detect_tubes。"""

from __future__ import annotations

import os
from typing import Optional, Union

import cv2
import numpy as np


def _env_device_type(cli: Optional[str]) -> Optional[str]:
    if cli and cli.lower() not in ("auto", ""):
        return cli.lower()
    t = os.environ.get("CAM_DEVICE_TYPE", "").strip().lower()
    return t or None


class UvcCamera:
    """普通 USB 摄像头（/dev/video*），OpenCV 取流。"""

    def __init__(self, index: Optional[int] = None):
        self.index = int(index if index is not None else os.environ.get("CAM_UVC_INDEX", "0"))
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开 UVC 摄像头 index={self.index}（/dev/video{self.index}）")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_exposure(self, exposure_us: float) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, float(max(1.0, exposure_us / 500.0)))
        self._cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.55)

    def set_gain(self, gain_db: float) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_GAIN, float(max(gain_db, 0.0)))

    def configure_imaging(self, exposure_time_us: float, gain_db: float):
        self.set_exposure(exposure_time_us)
        self.set_gain(gain_db)
        return exposure_time_us, gain_db

    def set_binning(self, factor: int) -> None:
        # UVC 一般不支持 binning，忽略
        pass

    def start_grabbing(self) -> None:
        pass

    def get_frame(self, timeout_ms: int = 2000) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame


CameraHandle = Union["HikCamera", UvcCamera]


def open_camera(
    backend: Optional[str] = None,
    device_type: Optional[str] = None,
    ip: Optional[str] = None,
    serial_number: Optional[str] = None,
) -> CameraHandle:
    """
    backend: hik | opencv | auto（auto=先 hik，无设备再 opencv）
    device_type: usb | gige | auto（仅 hik）
    """
    _ensure_mvs_env()
    bk = (backend or os.environ.get("CAM_BACKEND", "hik")).strip().lower()
    dt = _env_device_type(device_type)

    if bk == "opencv" or bk == "uvc":
        return UvcCamera()

    from hik_camera import HikCamera

    sn = serial_number or os.environ.get("CAM_SERIAL_NUMBER", "").strip() or None
    ip_addr = ip or os.environ.get("CAM_GIGE_IP", "").strip() or None
    try:
        return HikCamera(ip=ip_addr, serial_number=sn, device_type=dt)
    except RuntimeError:
        if bk == "auto":
            return UvcCamera()
        raise


def make_camera(
    backend: Optional[str] = None,
    device_type: Optional[str] = None,
    ip: Optional[str] = None,
    serial_number: Optional[str] = None,
) -> CameraHandle:
    """返回未打开的相机对象，配合 with cam: 使用。"""
    return open_camera(backend, device_type, ip, serial_number)


def camera_context(
    backend: Optional[str] = None,
    device_type: Optional[str] = None,
    ip: Optional[str] = None,
    serial_number: Optional[str] = None,
):
    """with camera_context(...) as cam: 与 HikCamera 相同用法。"""

    class _Ctx:
        def __enter__(self):
            self.cam = make_camera(backend, device_type, ip, serial_number)
            self.cam.open()
            return self.cam

        def __exit__(self, *args):
            if hasattr(self.cam, "close"):
                self.cam.close()

    return _Ctx()


def _ensure_mvs_env() -> None:
    if os.environ.get("MVCAM_COMMON_RUNENV"):
        return
    if os.path.isfile("/opt/MVS/lib/64/libMvCameraControl.so"):
        os.environ["MVCAM_COMMON_RUNENV"] = "/opt/MVS/lib"
        os.environ["LD_LIBRARY_PATH"] = "/opt/MVS/lib/64:" + os.environ.get("LD_LIBRARY_PATH", "")


def describe_cameras() -> None:
    _ensure_mvs_env()
    from hik_camera import HikCamera

    devs = HikCamera.enum_devices()
    print(f"海康 MVS 枚举: {len(devs)} 台")
    for d in devs:
        print(" ", {k: v for k, v in d.items() if k != "raw_device"})
    uvc = os.environ.get("CAM_UVC_INDEX", "0")
    print(f"UVC 备用: OpenCV index={uvc} （CAM_BACKEND=opencv）")
