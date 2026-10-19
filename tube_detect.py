# -*- coding: utf-8 -*-
"""
矩形反光孔板 + 离心管（白盖）检测

流程：
  1. 分割黑色孔板 → minAreaRect 得板位姿
  2. 在板局部坐标按行列生成全部孔心 (A1, A2, ...)
  3. 分割高亮白盖斑点，匹配到最近孔心（不靠空孔反光环）
  4. 对每个占位估铰链/突耳朝向 + 统一放向补偿角
  5. 输出 JSON 用结构化结果
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import rack_config as rack_config

from rack_config import (
    ASSIGN_DIST_RATIO,
    CAP_BRIGHT_THRESH,
    CAP_MAX_AREA,
    CAP_MIN_AREA,
    GRID_COLS,
    GRID_ROWS,
    HOLE_RADIUS_OVER_PITCH,
    MIN_OCCUPY_CONFIDENCE,
    MARGIN_X,
    MARGIN_Y,
    ORIGIN_AT_CHAMFER_CORNER,
    UNIFY_TARGETS_DEG,
    slot_id,
)

PROC_LONG_EDGE = 1000


def _merge_near_points(points: np.ndarray, radius: float) -> np.ndarray:
    """相近峰点合并为单孔心。"""
    merged: List[Dict[str, Any]] = []
    for p in points:
        placed = False
        for m in merged:
            c = m["sum"] / m["n"]
            if float(np.linalg.norm(p - c)) < radius:
                m["sum"] = m["sum"] + p
                m["n"] += 1
                placed = True
                break
        if not placed:
            merged.append({"sum": p.copy(), "n": 1})
    if not merged:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([m["sum"] / m["n"] for m in merged], dtype=np.float64)


def detect_hole_centers(
    gray: np.ndarray,
    plate_mask: np.ndarray,
    expected_cells: int = GRID_ROWS * GRID_COLS,
) -> List[Tuple[float, float, float]]:
    """反光孔板：Black-hat 提孔 + 局部极大 → 孔心点云（规模按 expected_cells 收敛，避免 12×8 假峰）。"""
    h, w = gray.shape[:2]
    ys, xs = np.where(plate_mask > 0)
    if xs.size < 100:
        return []

    inner = cv2.erode(
        plate_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )
    roi = gray.copy()
    roi[plate_mask == 0] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, k)
    bh = cv2.bitwise_and(bh, inner)
    dilated = cv2.dilate(bh, np.ones((5, 5), np.uint8))
    exp = max(8, int(expected_cells))
    pitch_guess = max(25.0, min(h, w) / max(math.sqrt(exp), 4.0) * 0.92)
    merge_r = pitch_guess * (0.12 if exp <= 32 else 0.14)
    target_lo = max(8, int(0.55 * exp))
    target_hi = max(target_lo + 2, int(1.35 * exp))

    centers = np.zeros((0, 2), dtype=np.float64)
    best_over: Optional[Tuple[float, np.ndarray]] = None
    for peak_th in range(14, 42):
        peaks = (bh == dilated) & (bh >= peak_th) & (inner > 0)
        py, px = np.where(peaks)
        if px.size < 8:
            break
        pts = np.column_stack([px.astype(np.float64), py.astype(np.float64)])
        cand = _merge_near_points(pts, radius=merge_r)
        n = len(cand)
        if n < target_lo:
            continue
        if n <= target_hi:
            centers = cand
            break
        dist = float(abs(n - exp))
        if best_over is None or dist < best_over[0]:
            best_over = (dist, cand)
    if len(centers) < target_lo and best_over is not None:
        centers = best_over[1]
    if len(centers) > target_hi:
        # 反光杂峰仍过多：加大合并半径再筛一遍
        centers = _merge_near_points(centers, radius=merge_r * 1.55)
    if len(centers) < max(6, int(0.35 * exp)) or len(centers) > int(1.85 * exp):
        return []
    r_px = max(4.0, pitch_guess * 0.16)
    return [(float(x), float(y), r_px) for x, y in centers]


def quad_from_hole_centers(holes: List[Tuple[float, float, float]], rows: int, cols: int) -> Optional[np.ndarray]:
    """用孔洞点云的外接旋转矩形作为网格四角（比板子外轮廓更准）。"""
    if len(holes) < 25:
        return None
    pts = np.array([[h[0], h[1]] for h in holes], dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    return plate_quad_for_grid(rect, rows, cols, plate_mask=None)


def _box_points_ccw(rect: Tuple) -> np.ndarray:
    pts = cv2.boxPoints(rect).astype(np.float64)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(ang)]


def _chamfer_corner_index(box_ccw: np.ndarray, mask: np.ndarray) -> Optional[int]:
    """
    缺角：外接矩形角点明显落在板外（切角），该角为 A1。
    优先用 pointPolygonTest<0 的深度，避免误选「仅离轮廓远」的尖角。
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float64)
    outside: List[Tuple[int, float, float]] = []
    for i in range(4):
        x, y = float(box_ccw[i, 0]), float(box_ccw[i, 1])
        dist = float(cv2.pointPolygonTest(cnt, (x, y), True))
        gap = float(np.min(np.linalg.norm(pts - box_ccw[i], axis=1)))
        outside.append((i, dist, gap))
    outside.sort(key=lambda t: t[1])
    best_i, best_d, best_g = outside[0]
    second_d = outside[1][1]
    if best_d > -4.0:
        return None
    if best_d < -12.0 or (second_d - best_d) > 8.0:
        return int(best_i)
    if best_g > outside[1][2] + 15.0:
        return int(best_i)
    return None


def _quad_from_chamfer(box_ccw: np.ndarray, chamfer_i: int, rows: int, cols: int) -> np.ndarray:
    """缺角 = TL(A1)；邻边分别为列向、行向（按行列数与边长匹配）。"""
    tl = box_ccw[chamfer_i]
    n1 = box_ccw[(chamfer_i + 1) % 4]
    n3 = box_ccw[(chamfer_i + 3) % 4]
    br = box_ccw[(chamfer_i + 2) % 4]
    len1 = float(np.linalg.norm(n1 - tl))
    len3 = float(np.linalg.norm(n3 - tl))
    if rows >= cols:
        tr, bl = (n1, n3) if len1 >= len3 else (n3, n1)
    else:
        tr, bl = (n1, n3) if len1 <= len3 else (n3, n1)
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 35) -> np.ndarray:
    v = values.astype(np.float64).reshape(-1, 1)
    if v.shape[0] < k or k < 1:
        return np.zeros(v.shape[0], dtype=np.int32)
    cents = np.linspace(float(v.min()), float(v.max()), k).reshape(-1, 1)
    labels = np.zeros(v.shape[0], dtype=np.int32)
    for _ in range(iters):
        d = np.abs(v - cents.T)
        labels = np.argmin(d, axis=1).astype(np.int32)
        for i in range(k):
            m = labels == i
            if np.any(m):
                cents[i, 0] = float(v[m].mean())
    order = np.argsort(cents.reshape(-1))
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return np.array([remap[int(l)] for l in labels], dtype=np.int32)


def infer_grid_size_from_holes(
    hole_xy: np.ndarray,
    hint_rows: int = GRID_ROWS,
    hint_cols: int = GRID_COLS,
) -> Optional[Tuple[int, int]]:
    """从孔洞点云估计 rows×cols（如 6×4=24）；hint 用于同数孔时区分横竖。"""
    n = int(hole_xy.shape[0])
    if n < 8:
        return None
    mu = hole_xy.mean(axis=0)
    X = hole_xy - mu
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    p0 = X @ vt[0]
    p1 = X @ vt[1]
    best: Optional[Tuple[float, int, int]] = None
    hint_cells = max(8, hint_rows * hint_cols)
    max_cells = max(hint_cells, n) if n <= hint_cells * 2 else hint_cells
    for rows in range(3, 17):
        for cols in range(3, 17):
            cells = rows * cols
            if cells < 8 or cells > max_cells + 4:
                continue
            if abs(n - cells) > max(4, int(0.32 * cells)):
                continue
            for swap in (False, True):
                rp, cp = (p1, p0) if swap else (p0, p1)
                rl = _kmeans_1d(rp, rows)
                cl = _kmeans_1d(cp, cols)
                filled = len({(int(rl[i]), int(cl[i])) for i in range(n)})
                score = float(filled) - abs(n - cells) * 0.45
                if (rows, cols) == (hint_rows, hint_cols):
                    score += 2.5
                elif (rows, cols) == (hint_cols, hint_rows):
                    score += 1.0
                if best is None or score > best[0]:
                    best = (score, rows, cols)
    if best is None:
        return None
    score, rows, cols = best
    if score < max(6.0, 0.45 * rows * cols):
        return None
    if (
        rows * cols == hint_rows * hint_cols
        and (rows, cols) == (hint_cols, hint_rows)
        and abs(n - hint_rows * hint_cols) <= max(3, int(0.15 * hint_rows * hint_cols))
    ):
        return hint_rows, hint_cols
    return rows, cols


def _chamfer_quad_candidates(
    box_ccw: np.ndarray, chamfer_i: int, rows: int, cols: int
) -> List[np.ndarray]:
    """缺角固定为 A1 时，仅两种行/列边方向待选。"""
    q0 = _quad_from_chamfer(box_ccw, chamfer_i, rows, cols)
    q1 = np.array([q0[0], q0[3], q0[2], q0[1]], dtype=np.float32)
    return [q0, q1]


def _grid_hole_alignment_cost(
    box: np.ndarray,
    holes: List[Tuple[float, float, float]],
    rows: int,
    cols: int,
) -> float:
    """孔洞峰点 ↔ 网格孔心 的中位误差（选 A1 方向、调边距用，不移动单孔）。"""
    if len(holes) < 12:
        return 1e9
    hole_pts = [(float(h[0]), float(h[1])) for h in holes]
    best = 1e18
    for my in np.linspace(0.04, 0.20, 11):
        for mx in np.linspace(0.04, 0.20, 11):
            slots, px, py = slot_grid_centers(box, rows, cols, float(mx), float(my))
            pitch = (px + py) / 2.0
            ds = [
                min(math.hypot(tx - s["cx"], ty - s["cy"]) for s in slots) for tx, ty in hole_pts
            ]
            cost = float(np.median(ds)) + 0.12 * float(np.percentile(ds, 90))
            cost += sum(1 for d in ds if d > pitch * 0.42) * 2.5
            if cost < best:
                best = cost
    return best


def _quad_cap_alignment_cost(
    box: np.ndarray,
    blobs: List[Dict[str, Any]],
    rows: int,
    cols: int,
    margin_x: float,
    margin_y: float,
) -> float:
    if not blobs:
        return 0.0
    slots, px, py = slot_grid_centers(box, rows, cols, margin_x, margin_y)
    pitch = (px + py) / 2.0
    max_d = ASSIGN_DIST_RATIO * pitch
    cost = 0.0
    for b in blobs:
        dmin = min(math.hypot(b["cx"] - s["cx"], b["cy"] - s["cy"]) for s in slots)
        cost += dmin
        if dmin > max_d:
            cost += 45.0
    return cost


def _shortest_delta(current: float, target: float) -> float:
    d = target - current
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def _unify_rotation(angle_deg: float) -> Tuple[float, str]:
    """angle: 0=上, 顺时针为正。"""
    d0 = _shortest_delta(angle_deg, UNIFY_TARGETS_DEG[0])
    d1 = _shortest_delta(angle_deg, UNIFY_TARGETS_DEG[1])
    rot = d0 if abs(d0) <= abs(d1) else d1
    return rot, ("顺时针" if rot > 0 else "逆时针")


def plate_quad_for_grid(
    rect: Tuple,
    rows: int,
    cols: int,
    plate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    由 minAreaRect 得到板坐标系四角（随板旋转，不强行按图像横平竖直）。
    顺序：TL, TR, BR, BL = (row0,col0), (row0,colN), (rowN,colN), (rowN,col0)
    若提供 plate_mask 且 ORIGIN_AT_CHAMFER_CORNER：TL = 缺角顶点 (A1)。
    """
    if ORIGIN_AT_CHAMFER_CORNER and plate_mask is not None and np.count_nonzero(plate_mask) > 100:
        box_ccw = _box_points_ccw(rect)
        ci = _chamfer_corner_index(box_ccw, plate_mask)
        if ci is not None:
            return _quad_from_chamfer(box_ccw, ci, rows, cols)

    pts = cv2.boxPoints(rect).astype(np.float64)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]  # CCW

    lengths = np.array([np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)])
    li = int(np.argmin(lengths))  # 短边起点 → 列方向

    def quad_from(li_start: int) -> np.ndarray:
        tl = pts[li_start]
        tr = pts[(li_start + 1) % 4]
        br = pts[(li_start + 2) % 4]
        bl = pts[(li_start + 3) % 4]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    candidates: List[np.ndarray] = []
    for start in (li, (li + 2) % 4):
        q = quad_from(start)
        candidates.append(q)
        candidates.append(np.array([q[1], q[0], q[3], q[2]], dtype=np.float32))

    def score_quad(q: np.ndarray) -> float:
        row_span = float(np.linalg.norm(q[3] - q[0]))
        col_span = float(np.linalg.norm(q[1] - q[0]))
        if row_span < col_span:
            q2 = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
            row_span, col_span = col_span, row_span
            q = q2
        ratio = row_span / (col_span + 1e-6)
        expect = rows / max(cols, 1)
        ratio_err = abs(ratio - expect)
        origin_penalty = float(q[0][1]) + 0.12 * float(q[0][0])
        return origin_penalty + ratio_err * 40.0

    best_q = min(candidates, key=score_quad)
    row_span = float(np.linalg.norm(best_q[3] - best_q[0]))
    col_span = float(np.linalg.norm(best_q[1] - best_q[0]))
    if row_span < col_span:
        best_q = np.array([best_q[1], best_q[2], best_q[3], best_q[0]], dtype=np.float32)
    return best_q


def _plate_from_bw(labels: np.ndarray, stats: np.ndarray, h: int, w: int, min_area_frac: float):
    best = None
    for i in range(1, int(stats.shape[0])):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_frac * h * w or area > 0.96 * h * w:
            continue
        bw_i = np.uint8(labels == i) * 255
        cnts, _ = cv2.findContours(bw_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        fill = cv2.contourArea(cnt) / (rw * rh + 1e-6)
        aspect_ok = 1.01 < aspect < 2.5
        score = area * (1.0 if aspect_ok else 0.35) * (0.45 + 0.55 * min(fill, 1.0))
        if best is None or score > best[0]:
            best = (score, i, cnt)
    return best


def find_plate_rect(gray: np.ndarray) -> Tuple[Tuple, np.ndarray]:
    """
    找黑色矩形孔板。
    返回: (minAreaRect, mask)
    """
    if gray is None or gray.size == 0:
        raise RuntimeError("空图像")
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    attempts: List[np.ndarray] = [otsu]
    for hard in (55, 48, 62, 42, 70, 35):
        _, hard_bw = cv2.threshold(blur, hard, 255, cv2.THRESH_BINARY_INV)
        attempts.append(cv2.bitwise_and(otsu, hard_bw))
        attempts.append(hard_bw)

    best_pick = None
    best_labels: Optional[np.ndarray] = None
    for min_frac in (0.08, 0.05, 0.035, 0.025):
        for bw in attempts:
            proc = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
            proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel, iterations=1)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(proc, connectivity=8)
            if n < 2:
                continue
            pick = _plate_from_bw(labels, stats, h, w, min_frac)
            if pick is not None and (best_pick is None or pick[0] > best_pick[0]):
                best_pick = pick
                best_labels = labels

    if best_pick is None or best_labels is None:
        raise RuntimeError("未找到合适的矩形孔板连通域（请调整曝光/增益，或确保孔板在画面内）")

    _, idx, cnt = best_pick
    mask = np.uint8(best_labels == idx) * 255
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    return rect, mask


def slot_grid_centers(
    box: np.ndarray,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    margin_x: float = MARGIN_X,
    margin_y: float = MARGIN_Y,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    在板四边形内用双线性插生成孔心。
    box: TL,TR,BR,BL
    返回 (slots, pitch_x, pitch_y)
    """
    tl, tr, br, bl = box
    slots = []
    # 用归一化网格：u 列 0..1，v 行 0..1
    for r in range(rows):
        v = margin_y + (1.0 - 2.0 * margin_y) * (r / max(rows - 1, 1))
        for c in range(cols):
            u = margin_x + (1.0 - 2.0 * margin_x) * (c / max(cols - 1, 1))
            # 双线性
            top = tl * (1 - u) + tr * u
            bot = bl * (1 - u) + br * u
            p = top * (1 - v) + bot * v
            slots.append(
                {
                    "slot_id": slot_id(r, c),
                    "row": r,
                    "col": c,
                    "cx": float(p[0]),
                    "cy": float(p[1]),
                }
            )

    # 间距：相邻孔
    if cols >= 2:
        pitch_x = float(np.hypot(slots[1]["cx"] - slots[0]["cx"], slots[1]["cy"] - slots[0]["cy"]))
    else:
        pitch_x = float(np.linalg.norm(tr - tl) / max(cols, 1))
    if rows >= 2:
        pitch_y = float(
            np.hypot(slots[cols]["cx"] - slots[0]["cx"], slots[cols]["cy"] - slots[0]["cy"])
        )
    else:
        pitch_y = float(np.linalg.norm(bl - tl) / max(rows, 1))
    return slots, pitch_x, pitch_y


def _cap_search_mask(plate_mask: np.ndarray, rect: Optional[Tuple] = None, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """白盖 ROI：连通域 mask + 外接矩形略放大（贴边盖常落在 mask 外）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    m = cv2.dilate(plate_mask, k, iterations=2)
    if rect is not None and shape is not None:
        h, w = shape
        pts = cv2.boxPoints(rect).astype(np.float64)
        c = pts.mean(axis=0)
        pts = c + (pts - c) * 1.10
        rect_m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(rect_m, pts.astype(np.int32), 255)
        rect_m = cv2.dilate(rect_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 1)
        m = cv2.bitwise_or(m, rect_m)
    return m


def _components_to_blobs(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    n: int,
    *,
    gentle: bool = False,
) -> List[Dict[str, Any]]:
    blobs: List[Dict[str, Any]] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < CAP_MIN_AREA or area > CAP_MAX_AREA:
            continue
        bw_ = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = bw_ / float(bh_ + 1e-6)
        if aspect < 0.28 or aspect > 3.2:
            continue
        rect_area = float(bw_ * bh_)
        fill_min = 0.22 if gentle else 0.28
        if rect_area > 0 and area / rect_area < fill_min:
            continue
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        comp = np.uint8(labels == i) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        blobs.append(
            {
                "cx": cx,
                "cy": cy,
                "area": area,
                "bbox": (
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    bw_,
                    bh_,
                ),
                "contour": cnt,
            }
        )
    return blobs


def _cap_blobs_from_binary(
    gray: np.ndarray, mask: np.ndarray, thresh: int, *, gentle: bool = False
) -> List[Dict[str, Any]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    bw = cv2.bitwise_and(bw, mask)
    if not gentle:
        open_k = 3 if thresh <= 152 else 5
        bw = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        )
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    return _components_to_blobs(labels, stats, centroids, n, gentle=gentle)


def _dedupe_cap_blobs(blobs: List[Dict[str, Any]], min_dist: float) -> List[Dict[str, Any]]:
    """同一颗白盖只保留一个 blob，避免 C1/D1 重复占位。"""
    if len(blobs) < 2:
        return blobs
    order = sorted(range(len(blobs)), key=lambda i: -blobs[i]["area"])
    kept: List[Dict[str, Any]] = []
    for i in order:
        b = blobs[i]
        if all(
            math.hypot(b["cx"] - k["cx"], b["cy"] - k["cy"]) >= min_dist for k in kept
        ):
            kept.append(b)
    return kept


def detect_cap_blobs(
    gray: np.ndarray,
    plate_mask: np.ndarray,
    rect: Optional[Tuple] = None,
    bgr: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """检测板上高盖：保守阈值，宁少勿滥报。"""
    h, w = gray.shape[:2]
    cap_gray = np.max(bgr, axis=2).astype(np.uint8) if bgr is not None and bgr.ndim == 3 else gray
    mask = _cap_search_mask(plate_mask, rect=rect, shape=(h, w))
    seen: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _add(blob_list: List[Dict[str, Any]]) -> None:
        for b in blob_list:
            key = (int(round(b["cx"] / 7.0)), int(round(b["cy"] / 7.0)))
            if key not in seen or b["area"] > seen[key]["area"]:
                seen[key] = b

    for th in (CAP_BRIGHT_THRESH, max(148, CAP_BRIGHT_THRESH - 15)):
        _add(_cap_blobs_from_binary(cap_gray, mask, th))
    blobs = list(seen.values())
    filtered: List[Dict[str, Any]] = []
    for b in blobs:
        if b["area"] > CAP_MAX_AREA:
            continue
        bw_, bh_ = b["bbox"][2], b["bbox"][3]
        if b["area"] > 1900 and max(bw_, bh_) / (min(bw_, bh_) + 1e-6) > 2.0:
            continue
        filtered.append(b)
    return filtered


def refine_grid(
    rect: Tuple,
    blobs: List[Dict[str, Any]],
    rows: int,
    cols: int,
    holes: Optional[List[Tuple[float, float, float]]] = None,
    plate_mask: Optional[np.ndarray] = None,
    anchor_quad: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """搜索边距 + 板四角方向；孔洞峰点约束网格，避免上下行孔心偏出板外。"""
    hole_pts: List[Tuple[float, float]] = []
    if holes and 12 <= len(holes) <= 48:
        hole_pts = [(float(h[0]), float(h[1])) for h in holes]
    cap_pts = [(b["cx"], b["cy"]) for b in blobs] if blobs else []

    base = anchor_quad if anchor_quad is not None else plate_quad_for_grid(rect, rows, cols, plate_mask=plate_mask)
    if not hole_pts and not cap_pts:
        return base, MARGIN_X, MARGIN_Y

    cap_w = 1.0 if len(cap_pts) >= 4 else 0.35
    hole_w = 0.72 if hole_pts else 0.0
    margin_vals = np.linspace(0.03, 0.21, 17)

    best = (1e18, None, MARGIN_X, MARGIN_Y)
    if anchor_quad is not None:
        quad_variants = [anchor_quad.astype(np.float32)]
    else:
        quad_variants = [
            base,
            np.array([base[1], base[0], base[3], base[2]], dtype=np.float32),
            np.array([base[3], base[0], base[1], base[2]], dtype=np.float32),
            np.array([base[1], base[2], base[3], base[0]], dtype=np.float32),
        ]
    for box in quad_variants:
        for my in margin_vals:
            for mx in margin_vals:
                slots, px, py = slot_grid_centers(box, rows, cols, float(mx), float(my))
                pitch = (px + py) / 2.0
                cost = 0.0
                for tx, ty in cap_pts:
                    dmin = min(math.hypot(tx - s["cx"], ty - s["cy"]) for s in slots)
                    cost += cap_w * dmin
                    if dmin > ASSIGN_DIST_RATIO * pitch * 1.1:
                        cost += cap_w * 35.0
                for tx, ty in hole_pts:
                    dmin = min(math.hypot(tx - s["cx"], ty - s["cy"]) for s in slots)
                    cost += hole_w * dmin
                    if dmin > pitch * 0.42:
                        cost += hole_w * 10.0
                if cost < best[0]:
                    best = (cost, box.copy(), float(mx), float(my))
    return best[1], best[2], best[3]


def estimate_hinge_angle(contour: np.ndarray, cx: float, cy: float) -> Optional[float]:
    """
    白盖突耳方向：质心/距离变换中心 → 轮廓最远点。
    返回角：0=图像上，顺时针为正。
    """
    if contour is None or len(contour) < 5:
        return None
    fill = np.zeros((int(cy) + 200, int(cx) + 200), np.uint8)  # placeholder size wrong

    x, y, w, h = cv2.boundingRect(contour)
    pad = 8
    roi = np.zeros((h + pad * 2, w + pad * 2), np.uint8)
    shifted = contour - np.array([[x - pad, y - pad]])
    cv2.drawContours(roi, [shifted], -1, 255, -1)
    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    body = np.array([max_loc[0] + x - pad, max_loc[1] + y - pad], dtype=np.float32)

    pts = contour.reshape(-1, 2).astype(np.float32)
    d2 = np.sum((pts - body) ** 2, axis=1)
    tip = pts[int(np.argmax(d2))]
    dx = float(tip[0] - body[0])
    dy = float(tip[1] - body[1])
    if abs(dx) + abs(dy) < 1e-3:
        return None
    ang = math.degrees(math.atan2(dx, -dy))
    if ang < 0:
        ang += 360.0
    return ang


def assign_blobs_to_slots(
    slots: List[Dict[str, Any]],
    blobs: List[Dict[str, Any]],
    pitch: float,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> List[Dict[str, Any]]:
    """贪心：每个 blob 匹配最近空闲孔；最外一圈略放宽距离。"""
    pairs: List[Tuple[float, int, int]] = []
    for bi, b in enumerate(blobs):
        for si, s in enumerate(slots):
            r, c = int(s["row"]), int(s["col"])
            on_edge = r == 0 or r >= rows - 1 or c == 0 or c >= cols - 1
            max_d = ASSIGN_DIST_RATIO * pitch * (1.06 if on_edge else 1.0)
            d = math.hypot(b["cx"] - s["cx"], b["cy"] - s["cy"])
            if d <= max_d:
                pairs.append((d, bi, si))
    pairs.sort(key=lambda t: t[0])

    matches: List[Tuple[int, int, float]] = []
    used_b, used_s = set(), set()
    for d, bi, si in pairs:
        if bi in used_b or si in used_s:
            continue
        used_b.add(bi)
        used_s.add(si)
        matches.append((bi, si, d))
    return matches


def detect_tubes(
    bgr: np.ndarray,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    主检测。返回 (result_dict, annotated_bgr, plate_mask)
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("空图像")

    h0, w0 = bgr.shape[:2]
    scale = PROC_LONG_EDGE / float(max(h0, w0))
    if scale < 1.0:
        proc = cv2.resize(bgr, (0, 0), fx=scale, fy=scale)
    else:
        scale = 1.0
        proc = bgr.copy()
    inv = 1.0 / scale

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    rect, plate_mask = find_plate_rect(gray)
    (pcx, pcy), (pw, ph), pang = rect
    expected_cells = max(8, int(rows) * int(cols))
    holes = detect_hole_centers(gray, plate_mask, expected_cells=expected_cells)
    blobs = detect_cap_blobs(gray, plate_mask, rect=rect, bgr=proc)
    pitch_guess = max(40.0, min(float(pw), float(ph)) / max(rows, cols) * 0.85)
    blobs = _dedupe_cap_blobs(blobs, min_dist=pitch_guess * 0.38)
    hole_xy = np.array([[h[0], h[1]] for h in holes], dtype=np.float64) if holes else np.zeros((0, 2))

    grid_rows, grid_cols = int(rows), int(cols)
    grid_source = "config"
    if rack_config.AUTO_DETECT_GRID and hole_xy.shape[0] >= 8:
        n_holes = int(hole_xy.shape[0])
        hole_margin = max(8, int(0.35 * expected_cells))
        if abs(n_holes - expected_cells) <= hole_margin:
            grid_source = "config_near_hole_count"
        elif n_holes <= expected_cells * 1.75:
            inferred = infer_grid_size_from_holes(hole_xy, hint_rows=grid_rows, hint_cols=grid_cols)
            if inferred is not None:
                ir, ic = inferred
                if abs(n_holes - ir * ic) <= max(6, int(0.38 * ir * ic)):
                    grid_rows, grid_cols = inferred
                    grid_source = "auto_holes"
    rows, cols = grid_rows, grid_cols

    anchor_quad: Optional[np.ndarray] = None
    origin_mode = "heuristic"
    if ORIGIN_AT_CHAMFER_CORNER and plate_mask is not None:
        box_ccw = _box_points_ccw(rect)
        ci = _chamfer_corner_index(box_ccw, plate_mask)
        if ci is not None:
            cands = _chamfer_quad_candidates(box_ccw, ci, rows, cols)
            if len(holes) >= 14:
                anchor_quad = min(
                    cands,
                    key=lambda q: _grid_hole_alignment_cost(q, holes, rows, cols),
                )
            else:
                anchor_quad = min(
                    cands,
                    key=lambda q: _quad_cap_alignment_cost(q, blobs, rows, cols, MARGIN_X, MARGIN_Y),
                )
            origin_mode = "chamfer_a1"
        else:
            origin_mode = "heuristic_fallback"

    box, mx, my = refine_grid(
        rect,
        blobs,
        rows,
        cols,
        holes=holes if holes else None,
        plate_mask=plate_mask,
        anchor_quad=anchor_quad,
    )
    slots, pitch_x, pitch_y = slot_grid_centers(box, rows=rows, cols=cols, margin_x=mx, margin_y=my)
    pitch = float((pitch_x + pitch_y) / 2.0)
    grid_from_holes = bool(holes and 12 <= len(holes) <= 48)
    hole_r = max(3.0, pitch * HOLE_RADIUS_OVER_PITCH)
    matches = assign_blobs_to_slots(slots, blobs, pitch, rows=rows, cols=cols)

    occ_map: Dict[int, Tuple[Dict[str, Any], float]] = {}
    for bi, si, d in matches:
        s = slots[si]
        r, c = int(s["row"]), int(s["col"])
        on_edge = r == 0 or r >= rows - 1 or c == 0 or c >= cols - 1
        max_d = ASSIGN_DIST_RATIO * pitch * (1.06 if on_edge else 1.0)
        if d > max_d:
            continue
        conf = float(min(1.0, 0.4 + 0.6 * (1.0 - d / (max_d + 1e-6))))
        if conf < MIN_OCCUPY_CONFIDENCE:
            continue
        occ_map[si] = (blobs[bi], d)

    items = []
    occupied_slots = []
    for si, s in enumerate(slots):
        occupied = si in occ_map
        hinge = None
        rot = None
        direction = None
        conf = 0.0
        area = 0.0
        if occupied:
            blob, dist = occ_map[si]
            hinge = estimate_hinge_angle(blob["contour"], blob["cx"], blob["cy"])
            if hinge is not None:
                rot, direction = _unify_rotation(hinge)
            area = blob["area"]
            conf = float(min(1.0, 0.4 + 0.6 * (1.0 - dist / (ASSIGN_DIST_RATIO * pitch + 1e-6))))
            # 用 blob 中心作为抓取点更准
            cx, cy = blob["cx"], blob["cy"]
        else:
            cx, cy = s["cx"], s["cy"]

        item = {
            "slot_id": s["slot_id"],
            "row": s["row"],
            "col": s["col"],
            "occupied": occupied,
            "cx_px": round(cx, 1),
            "cy_px": round(cy, 1),
            "cx_full": round(cx * inv, 1),
            "cy_full": round(cy * inv, 1),
            "hinge_angle_deg": None if hinge is None else round(hinge, 2),
            "rotate_to_unify_deg": None if rot is None else round(rot, 2),
            "rotate_direction": direction,
            "cap_area_px": round(area, 1),
            "confidence": round(conf, 3),
        }
        items.append(item)
        if occupied:
            occupied_slots.append(s["slot_id"])

    # ---- 可视化 ----
    annotated = proc.copy()
    cv2.polylines(annotated, [box.astype(np.int32)], True, (0, 255, 0), 2)
    a1x, a1y = int(box[0][0]), int(box[0][1])
    cv2.circle(annotated, (a1x, a1y), 8, (255, 0, 255), 2)
    cv2.putText(
        annotated,
        "A1",
        (a1x + 10, a1y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    hole_show: List[Tuple[float, float, float]] = []
    if holes and slots:
        for hx, hy, hr in holes:
            dmin = min(math.hypot(hx - s["cx"], hy - s["cy"]) for s in slots)
            if dmin <= pitch * 0.42:
                hole_show.append((hx, hy, hr))
    else:
        hole_show = holes
    for hx, hy, hr in hole_show:
        cv2.circle(annotated, (int(hx), int(hy)), 2, (255, 180, 0), -1)
    for s in slots:
        color = (60, 60, 60)
        cv2.circle(annotated, (int(s["cx"]), int(s["cy"])), int(hole_r), color, 1)

    for item in items:
        if not item["occupied"]:
            continue
        x, y = int(item["cx_px"]), int(item["cy_px"])
        cv2.circle(annotated, (x, y), int(hole_r), (0, 255, 255), 2)
        cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
        if item["hinge_angle_deg"] is not None:
            rad = math.radians(item["hinge_angle_deg"])
            tx = x + math.sin(rad) * hole_r * 2.2
            ty = y - math.cos(rad) * hole_r * 2.2
            cv2.arrowedLine(annotated, (x, y), (int(tx), int(ty)), (0, 0, 255), 2, tipLength=0.3)
        label = item["slot_id"]
        if item["rotate_to_unify_deg"] is not None:
            label += f" {item['rotate_to_unify_deg']:+.0f}"
        cv2.putText(
            annotated,
            label,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"occ={len(occupied_slots)}/{rows*cols} grid={rows}x{cols} caps={len(occ_map)}/{len(blobs)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rack_type": "rect_reflective_plate",
        "algorithm": "plate_rect_grid_then_white_cap_assign",
        "image_size_full": [int(w0), int(h0)],
        "image_size_proc": [int(proc.shape[1]), int(proc.shape[0])],
        "scale": scale,
        "plate": {
            "center_px": [round(pcx, 1), round(pcy, 1)],
            "center_full": [round(pcx * inv, 1), round(pcy * inv, 1)],
            "size_px": [round(pw, 1), round(ph, 1)],
            "angle_deg": round(pang, 2),
            "box_px": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in box],
            "grid_rows": rows,
            "grid_cols": cols,
            "grid_source": grid_source,
            "margin_xy": [round(mx, 4), round(my, 4)],
            "pitch_px": [round(pitch_x, 2), round(pitch_y, 2)],
            "origin_mode": origin_mode,
            "a1_at": "chamfer_corner" if origin_mode == "chamfer_a1" else "plate_quad_heuristic",
        },
        "slot_count": rows * cols,
        "occupied_count": len(occupied_slots),
        "occupied_slots": occupied_slots,
        "cap_blobs_detected": len(blobs),
        "cap_blobs_matched": len(occ_map),
        "holes_detected": len(holes),
        "grid_from_holes": grid_from_holes,
        "items": items,
        "arm_tips": {
            "pick_point": "occupied 项的 cx_full, cy_full",
            "yaw_compensate": "rotate_to_unify_deg",
            "slot_id": "行字母+列数字；A1=缺角顶点（若 origin_mode=chamfer_a1）",
        },
    }
    return result, annotated, plate_mask
