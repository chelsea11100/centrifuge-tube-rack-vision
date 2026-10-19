# -*- coding: utf-8 -*-
"""
矩形反光孔板（离心管架）网格配置。

编号约定（可按现场改 ROWS/COLS）：
  行：A, B, C, ... （从上到下，板局部坐标）
  列：1, 2, 3, ... （从左到右）
  例：A1 = 缺角那一侧的角孔（板子切角处为网格原点，见 ORIGIN_AT_CHAMFER_CORNER）

默认 6×4（24 孔）；换板时在 camera.env 或 GUI 改行列，或开启自动推断。
旧 12×8 板：GRID_ROWS=12, GRID_COLS=8, AUTO_DETECT_GRID=False。
"""

from __future__ import annotations

from typing import List, Tuple

# ---- 网格规模（自动推断失败时使用） ----
GRID_ROWS = 6
GRID_COLS = 4

# True：根据板上检测到的孔洞峰点估计行列数；False：始终用上面 GRID_ROWS/COLS
# 孔洞峰点过多时易误判成 12×8；24 孔板建议 False，或 GUI 固定 6×4
AUTO_DETECT_GRID = False

# True：用板子「缺一角」的外轮廓角点作为 A1 原点（横竖摆放时编号跟板走）
ORIGIN_AT_CHAMFER_CORNER = True

# 孔心相对板内接矩形的边距比例（相对板宽/高）
# 第一个孔心距左/上边缘 ≈ MARGIN * 板宽/高；末孔对称
# 检测时会在附近自动搜索最优边距
MARGIN_X = 0.09
MARGIN_Y = 0.08

# 单孔半径相对「孔间距」的比例（用于画圈 / ROI）
HOLE_RADIUS_OVER_PITCH = 0.35

# 占位：白盖匹配到孔心的最大距离（相对 pitch）
ASSIGN_DIST_RATIO = 0.55

# 匹配置信度低于此值的占位丢弃（避免乱挂孔）
MIN_OCCUPY_CONFIDENCE = 0.55

# 白盖分割：灰度阈值下限（反光板上白盖通常很亮）
CAP_BRIGHT_THRESH = 165
CAP_MIN_AREA = 180
CAP_MAX_AREA = 2800  # 处理分辨率下；过大多为灯管反光斑

# 统一放向目标角（图像坐标：0=右, 90=下，OpenCV atan2 习惯在检测里再转）
UNIFY_TARGETS_DEG = (90.0, 270.0)


def row_label(r: int) -> str:
    """0-based row index -> A,B,C..."""
    if r < 26:
        return chr(ord("A") + r)
    return f"R{r + 1}"


def slot_id(r: int, c: int) -> str:
    return f"{row_label(r)}{c + 1}"


def all_slot_ids(rows: int = GRID_ROWS, cols: int = GRID_COLS) -> List[str]:
    return [slot_id(r, c) for r in range(rows) for c in range(cols)]
