#!/usr/bin/env bash
# 一键：GigE 网卡辅 IP + MVS 环境 + 相机识别（不读本地图）
#
# 直接运行（默认实时识别 + 窗口）：
#   ./run_camera.sh
#
# 只拍一帧：
#   ./run_camera.sh --once
#
# USB 海康相机（跳过 GigE 辅 IP）：
#   ./run_camera.sh --usb --gui
#
# 普通 USB 摄像头（OpenCV）：
#   ./run_camera.sh --opencv --gui
#
# 可选覆盖（或写入 camera.env）：
#   GIGE_IF=enp3s0 GIGE_HOST_IP=192.168.5.100 ./run_camera.sh
#   CAM_DEVICE_TYPE=usb CAM_SERIAL_NUMBER=DAxxxx ./run_camera.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/camera.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/camera.env"
fi

GIGE_IF="${GIGE_IF:-enp3s0}"
GIGE_HOST_IP="${GIGE_HOST_IP:-192.168.5.100}"
GIGE_PREFIX="${GIGE_PREFIX:-24}"
export GIGE_IF GIGE_HOST_IP GIGE_PREFIX

export MVCAM_COMMON_RUNENV="${MVCAM_COMMON_RUNENV:-/opt/MVS/lib}"
export LD_LIBRARY_PATH="/opt/MVS/lib/64:${LD_LIBRARY_PATH:-}"

if [[ ! -f "/opt/MVS/lib/64/libMvCameraControl.so" ]]; then
  echo "[错误] 未找到 MVS SDK: /opt/MVS/lib/64/libMvCameraControl.so" >&2
  exit 1
fi

ensure_gige_ip() {
  if ! ip link show "$GIGE_IF" &>/dev/null; then
    echo "[警告] 网卡 ${GIGE_IF} 不存在。请设置 GIGE_IF，例如: GIGE_IF=enp2s0 ./run_camera.sh" >&2
    ip -br -4 a || true
    return 1
  fi
  if ip -4 addr show dev "$GIGE_IF" | grep -qE "inet ${GIGE_HOST_IP}/"; then
    echo "[网络] ${GIGE_IF} 已配置 ${GIGE_HOST_IP}/${GIGE_PREFIX}"
    return 0
  fi
  echo "[网络] 添加 GigE 辅 IP ${GIGE_HOST_IP}/${GIGE_PREFIX} → ${GIGE_IF}（需 sudo 密码）"
  if sudo ip addr add "${GIGE_HOST_IP}/${GIGE_PREFIX}" dev "$GIGE_IF" 2>/dev/null; then
    echo "[网络] 辅 IP 已添加"
    return 0
  fi
  # 可能已存在或权限问题
  if ip -4 addr show dev "$GIGE_IF" | grep -qE "inet ${GIGE_HOST_IP}/"; then
    return 0
  fi
  echo "[错误] 无法添加辅 IP。请手动执行:" >&2
  echo "  sudo ip addr add ${GIGE_HOST_IP}/${GIGE_PREFIX} dev ${GIGE_IF}" >&2
  return 1
}

SKIP_GIGE_IP=0
MODE_ARGS=()
USER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --usb)
      export CAM_DEVICE_TYPE=usb
      SKIP_GIGE_IP=1
      shift
      ;;
    --gige)
      export CAM_DEVICE_TYPE=gige
      shift
      ;;
    --opencv|--uvc)
      export CAM_BACKEND=opencv
      SKIP_GIGE_IP=1
      shift
      ;;
    --once)
      MODE_ARGS=()
      shift
      ;;
    --check)
      if [[ "$SKIP_GIGE_IP" -eq 0 && "${CAM_DEVICE_TYPE:-}" != "usb" && "${CAM_BACKEND:-}" != "opencv" ]]; then
        ensure_gige_ip || true
      fi
      exec python3 "$ROOT/check_cam.py"
      ;;
    --gui)
      MODE_ARGS=(gui)
      shift
      ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \?//'
      echo ""
      echo "其余参数传给 main_detect.py，例如 --exposure 50000 --gain 8"
      exit 0
      ;;
    --image|--image=*)
      echo "已禁用本地图 --image。请直接: $0  或  $0 --once --show" >&2
      exit 2
      ;;
    *)
      USER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$SKIP_GIGE_IP" -eq 1 || "${CAM_DEVICE_TYPE:-}" == "usb" || "${CAM_BACKEND:-}" == "opencv" ]]; then
  echo "[网络] USB/OpenCV 模式，跳过 GigE 辅 IP"
else
  ensure_gige_ip
fi

if [[ "${MODE_ARGS[0]:-}" == "gui" ]]; then
  if [[ "${CAM_DEVICE_TYPE:-}" == "usb" ]]; then
    echo "[提示] 使用海康 USB 相机（CAM_DEVICE_TYPE=usb）"
  elif [[ "${CAM_BACKEND:-}" == "opencv" ]]; then
    echo "[提示] 使用 OpenCV UVC index=${CAM_UVC_INDEX:-0}"
  fi
  echo "[提示] 启动 GUI（请先关闭官方 MVS 客户端）"
  exec python3 "$ROOT/gui_detect.py"
fi

echo "[提示] 请先关闭海康官方 MVS 客户端，否则会占用相机。"

# 无参数时：实时识别 + 弹窗
if [[ ${#MODE_ARGS[@]} -eq 0 && ${#USER_ARGS[@]} -eq 0 ]]; then
  MODE_ARGS=(--loop --show)
fi

exec python3 "$ROOT/main_detect.py" "${MODE_ARGS[@]}" "${USER_ARGS[@]}"
