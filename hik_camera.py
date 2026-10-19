# -*- coding: utf-8 -*-
"""
海康工业相机 (Hikrobot MVS) Windows/Linux 通用 Python 驱动封装
适用于 MV-CU120-10GC 等 GigE/USB 系列工业相机
支持：
  1. 多线程后台无阻塞取流 (解决单线程等待导致的UI卡死/窗口拖动卡顿)
  2. 动态 Binning 调节 (1x1 全分辨率, 2x2, 4x4 高帧率预览)
  3. 曝光、增益、触发模式控制
  4. 线程安全的高性能图像获取
"""

import sys
import os
import time
import threading
from ctypes import *
from typing import Optional, Tuple
import numpy as np
import cv2

# 添加本地 MvImport 路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MV_IMPORT_DIR = os.path.join(CURRENT_DIR, "MvImport")
if MV_IMPORT_DIR not in sys.path:
    sys.path.insert(0, MV_IMPORT_DIR)

try:
    from MvCameraControl_class import *
except ImportError:
    mvs_runenv = os.environ.get("MVCAM_COMMON_RUNENV", r"D:\MVS\Development")
    sys.path.insert(0, os.path.join(mvs_runenv, "Samples", "Python", "MvImport"))
    from MvCameraControl_class import *


class HikCamera:
    """海康工业相机面向对象封装类"""

    def __init__(
        self,
        ip: Optional[str] = None,
        serial_number: Optional[str] = None,
        device_type: Optional[str] = None,
    ):
        self.target_ip = ip
        self.target_sn = serial_number
        # usb | gige | None（None 时读环境变量 CAM_DEVICE_TYPE）
        self.device_type = (device_type or os.environ.get("CAM_DEVICE_TYPE", "")).strip().lower() or None

        self.cam = None
        self.dev_info = None
        self._is_initialized = False
        self._is_open = False
        self._is_grabbing = False

        # 多线程后台取流相关
        self._grab_thread = None
        self._thread_running = False
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_seq = 0
        self._last_grab_time = 0.0

        # 转换缓存
        self._convert_buf = None
        self._convert_buf_size = 0

    @staticmethod
    def enum_devices() -> list:
        """枚举当前局域网与USB中所有海康相机设备"""
        MvCamera.MV_CC_Initialize()
        devices = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, devices)
        result = []
        if ret == 0:
            for i in range(devices.nDeviceNum):
                dev = cast(devices.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                info = {
                    "index": i,
                    "type": "GigE" if dev.nTLayerType == MV_GIGE_DEVICE else "USB",
                    "accessible": MvCamera.MV_CC_IsDeviceAccessible(dev, MV_ACCESS_Exclusive) == 1,
                    "raw_device": dev,
                }
                if dev.nTLayerType == MV_GIGE_DEVICE:
                    g = dev.SpecialInfo.stGigEInfo
                    info["ip"] = f"{(g.nCurrentIp >> 24) & 0xff}.{(g.nCurrentIp >> 16) & 0xff}.{(g.nCurrentIp >> 8) & 0xff}.{g.nCurrentIp & 0xff}"
                    info["nic_ip"] = f"{(g.nNetExport >> 24) & 0xff}.{(g.nNetExport >> 16) & 0xff}.{(g.nNetExport >> 8) & 0xff}.{g.nNetExport & 0xff}"
                    info["model"] = "".join(chr(c) for c in g.chModelName if c != 0)
                    info["serial_number"] = "".join(chr(c) for c in g.chSerialNumber if c != 0)
                elif dev.nTLayerType == MV_USB_DEVICE:
                    u = dev.SpecialInfo.stUsb3VInfo
                    info["model"] = "".join(chr(c) for c in u.chModelName if c != 0)
                    info["serial_number"] = "".join(chr(c) for c in u.chSerialNumber if c != 0)
                result.append(info)
        return result

    def open(self) -> bool:
        """打开相机并进行基础初始化"""
        if self._is_open:
            return True

        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise RuntimeError(f"海康SDK初始化失败，错误码: 0x{ret:x}")
        self._is_initialized = True

        dev_list = self.enum_devices()
        if not dev_list:
            raise RuntimeError("未搜索到任何海康相机，请检查网线供电或USB连接！")

        selected = None
        for dev in dev_list:
            if not dev["accessible"]:
                continue
            if self.target_ip and dev.get("ip") != self.target_ip:
                continue
            if self.target_sn and dev.get("serial_number") != self.target_sn:
                continue
            selected = dev
            break

        if selected is None:
            accessible_devs = [d for d in dev_list if d["accessible"]]
            dt = self.device_type
            if dt == "usb":
                usb_only = [d for d in accessible_devs if d.get("type") == "USB"]
                if usb_only:
                    accessible_devs = usb_only
                else:
                    raise RuntimeError("CAM_DEVICE_TYPE=usb 但未发现可访问的 USB 海康相机")
            elif dt in ("gige", "gig"):
                gige_only = [d for d in accessible_devs if d.get("type") == "GigE"]
                if gige_only:
                    accessible_devs = gige_only
                else:
                    raise RuntimeError("CAM_DEVICE_TYPE=gige 但未发现可访问的 GigE 相机")
            prefer_nic = os.environ.get("GIGE_HOST_IP", "").strip()
            if prefer_nic and accessible_devs:
                same_nic = [d for d in accessible_devs if d.get("nic_ip") == prefer_nic]
                if same_nic:
                    accessible_devs = same_nic
                else:
                    cam_subnet = None
                    for d in accessible_devs:
                        ip = d.get("ip") or ""
                        if ip.count(".") == 3:
                            cam_subnet = ip.rsplit(".", 1)[0]
                            break
                    if cam_subnet:
                        on_seg = [
                            d
                            for d in accessible_devs
                            if (d.get("nic_ip") or "").rsplit(".", 1)[0] == cam_subnet
                        ]
                        if on_seg:
                            accessible_devs = on_seg
            # 同一 SN 枚举两次时只保留一条（优先已选网段）
            by_sn: dict = {}
            for d in accessible_devs:
                sn = d.get("serial_number") or str(d.get("index"))
                if sn not in by_sn:
                    by_sn[sn] = d
            accessible_devs = list(by_sn.values())
            if not accessible_devs:
                lines = ["检测到相机但无法访问（accessible=False）。设备详情："]
                for d in dev_list:
                    tip = f"  [{d.get('type')}] model={d.get('model')} sn={d.get('serial_number')}"
                    if d.get("type") == "GigE":
                        tip += f" cam_ip={d.get('ip')} nic_ip={d.get('nic_ip')}"
                        if d.get("ip") and d.get("nic_ip"):
                            same = d["ip"].rsplit(".", 1)[0] == d["nic_ip"].rsplit(".", 1)[0]
                            tip += f" 同网段(/24)={'是' if same else '否←需给网卡加辅IP'}"
                    tip += f" accessible={d.get('accessible')}"
                    lines.append(tip)
                lines.append(
                    "处理建议：1) 完全退出官方 MVS/客户端  2) GigE 跨网段时给网卡追加与相机同网段的辅 IP"
                    "  3) 确认未被其他 Python/预览程序占用"
                )
                raise RuntimeError("\n".join(lines))
            selected = accessible_devs[0]
            if selected.get("type") == "GigE":
                print(
                    f"[相机] 使用 {selected.get('model')} "
                    f"cam={selected.get('ip')} via nic={selected.get('nic_ip')}"
                )
            elif selected.get("type") == "USB":
                print(f"[相机] 使用 USB {selected.get('model')} sn={selected.get('serial_number')}")

        self.dev_info = selected["raw_device"]
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(self.dev_info)
        if ret != 0:
            raise RuntimeError(f"创建相机句柄失败，错误码: 0x{ret:x}")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            if ret == 0x80000203:
                raise RuntimeError("相机已被其他进程独占打开（请先关闭官方 MVS 软件）！")
            elif ret == 0x80000215:
                raise RuntimeError(f"网段不匹配导致连接失败（相机IP: {selected.get('ip')}，网卡IP: {selected.get('nic_ip')}）！")
            else:
                raise RuntimeError(f"打开相机失败，错误码: 0x{ret:x}")

        self._is_open = True

        # GigE 相机自动探测并配置最佳网络包大小
        if self.dev_info.nTLayerType == MV_GIGE_DEVICE:
            pkt_size = self.cam.MV_CC_GetOptimalPacketSize()
            if pkt_size > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)
            # 设置数据包间隔为 0，最大化网卡传输速度
            self.cam.MV_CC_SetIntValue("GevSCPD", 0)

        # 默认关闭触发模式（自由采集流）
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        return True

    def set_binning(self, factor: int = 1):
        """
        设置像素合并模式 (Binning)
        :param factor: 1 为全分辨率(4024x3036); 2 为 2x2(2012x1518); 4 为 4x4(1004x758)
        """
        if not self._is_open:
            self.open()
        if factor not in (1, 2, 4):
            raise ValueError("Binning factor 必须为 1, 2 或 4")

        was_grabbing = self._is_grabbing
        was_threaded = self._thread_running

        if was_threaded:
            self.stop_grabbing_thread()
        elif was_grabbing:
            self.stop_grabbing()

        ret1 = self.cam.MV_CC_SetEnumValue("BinningHorizontal", factor)
        ret2 = self.cam.MV_CC_SetEnumValue("BinningVertical", factor)
        if ret1 != 0 or ret2 != 0:
            raise RuntimeError(f"设置 Binning 失败: 0x{ret1:x}, 0x{ret2:x}")

        if was_threaded:
            self.start_grabbing_thread()
        elif was_grabbing:
            self.start_grabbing()

    def get_binning(self) -> int:
        """获取当前 Binning 因子"""
        if not self._is_open:
            return 1
        val = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue("BinningHorizontal", val)
        return val.nCurValue if ret == 0 else 1

    def get_resolution(self) -> Tuple[int, int]:
        """获取当前分辨率 (width, height)"""
        if not self._is_open:
            return (0, 0)
        vw = MVCC_INTVALUE()
        vh = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("Width", vw)
        self.cam.MV_CC_GetIntValue("Height", vh)
        return (vw.nCurValue, vh.nCurValue)

    def get_resulting_fps(self) -> float:
        """获取相机传感器端实际计算帧率"""
        if not self._is_open:
            return 0.0
        val = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", val)
        return val.fCurValue if ret == 0 else 0.0

    def start_grabbing(self) -> bool:
        """开始采集（单线程模式）"""
        if not self._is_open:
            self.open()
        if self._is_grabbing:
            return True
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开启采集失败，错误码: 0x{ret:x}")
        self._is_grabbing = True
        return True

    def stop_grabbing(self):
        """停止采集（单线程模式）"""
        if self._is_grabbing and self.cam:
            self.cam.MV_CC_StopGrabbing()
            self._is_grabbing = False

    def start_grabbing_thread(self):
        """开启后台独立取流线程（推荐实时GUI预览使用，彻底杜绝UI阻塞）"""
        if self._thread_running:
            return
        self.start_grabbing()
        self._thread_running = True
        self._grab_thread = threading.Thread(target=self._grab_worker, daemon=True, name="HikCameraGrabber")
        self._grab_thread.start()

    def stop_grabbing_thread(self):
        """停止后台取流线程"""
        self._thread_running = False
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=1.5)
            self._grab_thread = None
        self.stop_grabbing()

    def _grab_worker(self):
        """后台独立抓取循环"""
        while self._thread_running and self._is_grabbing:
            frame = self.get_frame(timeout_ms=1500)
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_seq += 1
                    self._last_grab_time = time.time()
            else:
                time.sleep(0.01)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """
        从后台取流线程获取最新一帧（非阻塞即时返回，供GUI渲染）
        :return: (image, frame_seq) 如果暂无帧返回 (None, 0)
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None, 0
            return self._latest_frame.copy(), self._frame_seq

    def get_frame(self, timeout_ms: int = 1500) -> Optional[np.ndarray]:
        """
        单次同步获取一帧图像，转换为 OpenCV 格式 (BGR, uint8)
        :param timeout_ms: 超时等待时间（毫秒）
        :return: numpy.ndarray 或 None
        """
        if not self._is_grabbing:
            self.start_grabbing()

        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        ret = self.cam.MV_CC_GetImageBuffer(frame, timeout_ms)
        if ret != 0 or not frame.pBufAddr:
            return None

        try:
            width = frame.stFrameInfo.nWidth
            height = frame.stFrameInfo.nHeight
            src_type = frame.stFrameInfo.enPixelType
            src_len = frame.stFrameInfo.nFrameLen

            target_size = width * height * 3
            if self._convert_buf is None or self._convert_buf_size < target_size:
                self._convert_buf = (c_ubyte * target_size)()
                self._convert_buf_size = target_size

            convert = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(convert), 0, sizeof(convert))
            convert.pSrcData = frame.pBufAddr
            convert.nSrcDataLen = src_len
            convert.enSrcPixelType = src_type
            convert.nWidth = width
            convert.nHeight = height
            convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            convert.pDstBuffer = self._convert_buf
            convert.nDstBufferSize = self._convert_buf_size

            ret_conv = self.cam.MV_CC_ConvertPixelTypeEx(convert)
            if ret_conv != 0:
                raise RuntimeError(f"像素格式转换失败，错误码: 0x{ret_conv:x}")

            image = np.frombuffer(self._convert_buf, dtype=np.uint8, count=target_size).reshape((height, width, 3))
            return image.copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    def _set_float_param(self, name: str, value: float) -> int:
        val = MVCC_FLOATVALUE()
        memset(byref(val), 0, sizeof(val))
        if self.cam.MV_CC_GetFloatValue(name, val) == 0:
            value = max(float(val.fMin), min(float(val.fMax), float(value)))
        return self.cam.MV_CC_SetFloatValue(name, value)

    def set_exposure(self, exposure_time_us: float):
        """设置曝光时间（单位：微秒 us）"""
        if not self._is_open:
            self.open()
        self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        ret = self._set_float_param("ExposureTime", float(exposure_time_us))
        if ret != 0:
            self._set_float_param("ExposureTimeAbs", float(exposure_time_us))

    def get_exposure(self) -> float:
        """获取当前曝光时间（微秒 us）"""
        val = MVCC_FLOATVALUE()
        memset(byref(val), 0, sizeof(val))
        self.cam.MV_CC_GetFloatValue("ExposureTime", val)
        return val.fCurValue

    def set_gain(self, gain_db: float):
        """设置增益（dB）"""
        if not self._is_open:
            self.open()
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        self._set_float_param("Gain", float(gain_db))

    def _apply_display_gamma(self, gamma: float = 1.35) -> None:
        """传感器侧略提亮（Mono/Bayer 均尝试）。"""
        if not self._is_open or not self.cam:
            return
        self.cam.MV_CC_SetBoolValue("GammaEnable", True)
        self.cam.MV_CC_SetFloatValue("Gamma", float(gamma))
        try:
            self.cam.MV_CC_SetGammaValue(PixelType_Gvsp_Mono8, float(gamma))
        except Exception:
            pass

    def configure_imaging(self, exposure_time_us: float, gain_db: float) -> Tuple[float, float]:
        """设置曝光/增益并尽量开启 Gamma，返回相机实际读数。"""
        self.set_exposure(exposure_time_us)
        self.set_gain(gain_db)
        self._apply_display_gamma(1.35)
        return self.get_exposure(), self.get_gain()

    def get_gain(self) -> float:
        """获取当前增益（dB）"""
        val = MVCC_FLOATVALUE()
        memset(byref(val), 0, sizeof(val))
        self.cam.MV_CC_GetFloatValue("Gain", val)
        return val.fCurValue

    def capture_snapshot(self, save_path: Optional[str] = None, full_resolution: bool = False) -> np.ndarray:
        """
        拍摄一张照片
        :param save_path: 可选保存路径
        :param full_resolution: 是否临时切回 4024x3036 全分辨率拍摄（拍摄完成后恢复原设置）
        """
        orig_binning = self.get_binning()
        if full_resolution and orig_binning != 1:
            self.set_binning(1)

        # 取最新帧或重新抓取一帧
        if self._thread_running:
            # 等待新的一帧生成
            cur_seq = self._frame_seq
            t0 = time.time()
            img = None
            while time.time() - t0 < 3.0:
                img, seq = self.get_latest_frame()
                if img is not None and seq > cur_seq:
                    break
                time.sleep(0.02)
            if img is None:
                img = self.get_frame(timeout_ms=3000)
        else:
            img = self.get_frame(timeout_ms=3000)

        if full_resolution and orig_binning != 1:
            self.set_binning(orig_binning)

        if img is None:
            raise RuntimeError("抓取单张照片超时！")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            ext = os.path.splitext(save_path)[1] or ".jpg"
            if not save_path.endswith(ext):
                save_path += ext
            success, enc = cv2.imencode(ext, img)
            if success:
                enc.tofile(save_path)
            else:
                raise RuntimeError(f"图像保存失败: {save_path}")

        return img

    def close(self):
        """安全关闭相机并释放所有资源"""
        if self._thread_running:
            self.stop_grabbing_thread()
        if self._is_grabbing:
            self.stop_grabbing()
        if self._is_open and self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            self._is_open = False
        if self._is_initialized:
            MvCamera.MV_CC_Finalize()
            self._is_initialized = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
