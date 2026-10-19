# -*- coding: utf-8 -*-
import os, subprocess
if not os.environ.get("MVCAM_COMMON_RUNENV"):
    if os.path.isfile("/opt/MVS/lib/64/libMvCameraControl.so"):
        os.environ["MVCAM_COMMON_RUNENV"] = "/opt/MVS/lib"
        os.environ["LD_LIBRARY_PATH"] = "/opt/MVS/lib/64:" + os.environ.get("LD_LIBRARY_PATH", "")

print("==== 网卡 ====")
print(subprocess.getoutput("ip -br a"))
print("\n==== 海康枚举 ====")
from camera_backend import describe_cameras, make_camera

describe_cameras()
from hik_camera import HikCamera

devs = HikCamera.enum_devices()
print(f"发现 {len(devs)} 台")
for d in devs:
    print({k: v for k, v in d.items() if k != "raw_device"})
    if d.get("type") == "GigE" and d.get("ip") and d.get("nic_ip"):
        same = d["ip"].rsplit(".", 1)[0] == d["nic_ip"].rsplit(".", 1)[0]
        print(f"  → 同网段(/24): {'是' if same else '否（需给网卡加辅IP）'}")

ok = [d for d in devs if d.get("accessible")]
if ok:
    print("\n有可访问设备，尝试打开抓一帧...")
    cam_kind = os.environ.get("CAM_DEVICE_TYPE", "auto") or "auto"
    backend = os.environ.get("CAM_BACKEND", "hik") or "hik"
    with make_camera(backend=backend, device_type=cam_kind) as cam:
        cam.set_binning(2)
        cam.set_exposure(50000)
        cam.set_gain(8)
        cam.start_grabbing()
        img = cam.get_frame(3000)
        print("抓拍OK:", None if img is None else f"{img.shape[1]}x{img.shape[0]}")
else:
    print("\n没有 accessible=True 的设备。")
    print("若 cam_ip 与 nic_ip 不同网段，示例（把 enpXsY / 192.168.1.100 换成你的）：")
    print("  sudo ip addr add 192.168.1.100/24 dev enpXsY")
