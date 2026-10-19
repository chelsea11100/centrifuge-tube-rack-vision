# -*- coding: utf-8 -*-
"""
海康工业相机流畅实时预览与抓拍程序 (多线程无阻塞优化版)
解决单线程阻塞导致的窗口拖动卡死、未响应问题
提供 Binning 分辨率切换，突破百兆网卡帧率瓶颈

快捷键说明：
  - [b]     : 切换预览分辨率 (1x1 全分辨率 12MP / 2x2 提速 / 4x4 极速流畅)
  - [s]     : 抓拍保存当前画面
  - [S]     : 强制以 4024x3036 全分辨率高清抓拍保存
  - [+/-]   : 增加 / 减少曝光时间
  - [[/]]   : 增加 / 减少增益
  - [q/ESC] : 退出程序
"""

import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np

from hik_camera import HikCamera


def run_preview():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "图片")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(" 正在连接海康工业相机 (多线程极速模式)...")
    print("=" * 60)

    camera = HikCamera()
    try:
        camera.open()
    except Exception as e:
        print(f"\n[错误] 相机连接失败: {e}")
        print("提示：如果官方 MVS 软件正在运行，请先彻底关闭 MVS 软件后再试。")
        return

    # 设置默认曝光与增益
    try:
        camera.set_exposure(50000.0)  # 50ms
        camera.set_gain(8.0)          # 8dB
        # 默认使用 2x2 Binning 提速预览（实测百兆网卡可达 ~4 FPS，窗口更流畅）
        camera.set_binning(2)
    except Exception as e:
        print(f"[警告] 初始化参数失败: {e}")

    # 启动后台独立取流线程，彻底解耦 UI 渲染与网络传输
    camera.start_grabbing_thread()

    window_name = "Hikrobot Camera Preview (Threaded)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[相机已就绪，后台取流已开启]")
    print(f"  当前曝光: {camera.get_exposure():.0f} us | 增益: {camera.get_gain():.1f} dB")
    print("  快捷键:")
    print("    [b]    - 切换分辨率 (1x1 1200万像素 / 2x2 300万像素 / 4x4 75万像素)")
    print("    [s]    - 抓拍保存当前画面")
    print("    [S]    - (Shift+S) 强制以 4024x3036 全分辨率抓拍")
    print("    [+]    - 增加曝光 (+5000 us)")
    print("    [-]    - 减少曝光 (-5000 us)")
    print("    []]    - 增加增益 (+1.0 dB)")
    print("    [[]    - 减少增益 (-1.0 dB)")
    print("    [q/ESC]- 退出\n")

    last_seq = 0
    camera_fps = 0.0
    cam_frame_count = 0
    fps_time = time.time()
    tip_text = ""
    tip_timeout = 0.0
    binning_modes = [1, 2, 4]
    binning_idx = 1  # 对应 2x2

    display_img = None

    try:
        while True:
            # 1. 非阻塞获取最新帧
            frame, seq = camera.get_latest_frame()

            # 2. 统计相机实际出帧率
            if seq != last_seq and frame is not None:
                last_seq = seq
                cam_frame_count += 1

            if time.time() - fps_time >= 1.0:
                camera_fps = cam_frame_count / (time.time() - fps_time)
                cam_frame_count = 0
                fps_time = time.time()

            # 3. 渲染画面
            if frame is not None:
                raw_h, raw_w = frame.shape[:2]
                scale = 960.0 / max(raw_h, raw_w)
                disp_w = int(raw_w * scale)
                disp_h = int(raw_h * scale)
                display_img = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

                # 状态信息栏
                exp = camera.get_exposure()
                gain = camera.get_gain()
                cur_bin = camera.get_binning()
                info_str = f"Res: {raw_w}x{raw_h} (Bin {cur_bin}x{cur_bin}) | Cam FPS: {camera_fps:.1f} | Exp: {exp:.0f}us"

                cv2.rectangle(display_img, (8, 8), (disp_w - 8, 40), (0, 0, 0), -1)
                cv2.putText(
                    display_img,
                    info_str,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                if time.time() < tip_timeout and tip_text:
                    cv2.putText(
                        display_img,
                        tip_text,
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, display_img)
            else:
                # 初始等待相机首帧
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to camera stream...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, placeholder)

            # 4. 关键：以 ~60Hz 处理 Windows 消息循环，确保窗口拖动丝滑响应！
            key = cv2.waitKey(15) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("s"):
                # 普通抓拍
                if frame is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, f"capture_{ts}.jpg")
                    cv2.imencode(".jpg", frame)[1].tofile(save_path)
                    tip_text = f"Saved: {os.path.basename(save_path)} ({raw_w}x{raw_h})"
                    tip_timeout = time.time() + 2.5
                    print(f"[已保存快照] {save_path}")
            elif key == ord("S"):
                # 强制以 4024x3036 全分辨率高清抓拍
                tip_text = "Switching to Full 12MP capture..."
                tip_timeout = time.time() + 2.0
                print("\n[正在以 4024x3036 全分辨率拍摄...]")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"capture_12MP_{ts}.jpg")
                img_12mp = camera.capture_snapshot(save_path, full_resolution=True)
                tip_text = f"Saved 12MP: {os.path.basename(save_path)}"
                tip_timeout = time.time() + 3.0
                print(f"[全分辨率抓拍完成] {save_path} ({img_12mp.shape[1]}x{img_12mp.shape[0]})")
            elif key == ord("b"):
                # 切换分辨率
                binning_idx = (binning_idx + 1) % len(binning_modes)
                target_bin = binning_modes[binning_idx]
                camera.set_binning(target_bin)
                w, h = camera.get_resolution()
                tip_text = f"Switched to Binning {target_bin}x{target_bin} ({w}x{h})"
                tip_timeout = time.time() + 2.0
                print(f"[切换分辨率] Binning {target_bin}x{target_bin} -> {w}x{h}")
            elif key in (ord("+"), ord("=")):
                new_exp = min(camera.get_exposure() + 5000.0, 500000.0)
                camera.set_exposure(new_exp)
                print(f"[调整曝光] {new_exp:.0f} us")
            elif key in (ord("-"), ord("_")):
                new_exp = max(camera.get_exposure() - 5000.0, 100.0)
                camera.set_exposure(new_exp)
                print(f"[调整曝光] {new_exp:.0f} us")
            elif key == ord("]"):
                new_gain = min(camera.get_gain() + 1.0, 20.0)
                camera.set_gain(new_gain)
                print(f"[调整增益] {new_gain:.1f} dB")
            elif key == ord("["):
                new_gain = max(camera.get_gain() - 1.0, 0.0)
                camera.set_gain(new_gain)
                print(f"[调整增益] {new_gain:.1f} dB")

    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("\n[相机已安全释放关闭]")


if __name__ == "__main__":
    run_preview()
