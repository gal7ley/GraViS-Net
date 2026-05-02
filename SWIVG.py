# 文件名: utils_swivg.py

import numpy as np
import numba
from numba import njit, prange


# =========================================================
# 1. 基础统计辅助 (标准差与 Tau Map)
# =========================================================
@njit(fastmath=True)
def compute_std_dev(block):
    h, w = block.shape
    if h == 0 or w == 0: return 0.0
    sum_val = 0.0
    sum_sq = 0.0
    for r in range(h):
        for c in range(w):
            val = block[r, c]
            sum_val += val
            sum_sq += val * val
    mean = sum_val / (h * w)
    var = (sum_sq / (h * w)) - (mean * mean)
    return np.sqrt(max(var, 0.0))


@njit(parallel=True, fastmath=True)
def create_tau_map(img, grid_size, tau_factor):
    """
    计算自适应阈值图 (Tau Map)
    """
    H, W = img.shape
    tau_map = np.zeros((H, W), dtype=np.float32)
    n_rows = (H + grid_size - 1) // grid_size
    n_cols = (W + grid_size - 1) // grid_size

    for r_idx in prange(n_rows):
        r_start = r_idx * grid_size
        r_end = min(r_start + grid_size, H)
        for c_idx in range(n_cols):
            c_start = c_idx * grid_size
            c_end = min(c_start + grid_size, W)

            block = img[r_start:r_end, c_start:c_end]
            std = compute_std_dev(block)
            # 最小阈值限制为 1.0，防止除以零或过小
            tau_val = max(tau_factor * std, 1.0)

            for i in range(r_start, r_end):
                for j in range(c_start, c_end):
                    tau_map[i, j] = tau_val
    return tau_map


@njit(fastmath=True)
def sigmoid(x):
    # 限制范围防止溢出
    if x > 10.0: return 1.0
    if x < -10.0: return 0.0
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================
# 2. SW-IVG 核心逻辑 (Sigmoid Weighted)
# =========================================================
@njit(fastmath=True)
def core_sw_ivg(data, tau_arr, k_sig):
    n = len(data)
    deg = np.zeros(n, dtype=np.float32)

    for i in range(n - 1):
        curr = data[i]
        tau_i = tau_arr[i]
        max_k = -1e9

        for j in range(i + 1, n):
            dist = float(j - i)
            # 加极小值防止除零
            slope = (data[j] - curr) / (dist + 1e-9)

            is_vis = False
            # 几何可见性判断 (Standard NVG Logic)
            if j == i + 1:
                is_vis = True
                max_k = slope
            elif slope > max_k:
                is_vis = True
                max_k = slope

            # 权重计算 (Sigmoid Weighting)
            if is_vis:
                diff = abs(data[j] - curr)
                w = sigmoid(k_sig * (diff - tau_i))
                deg[i] += w
                deg[j] += w
    return deg


# =========================================================
# 3. 全图构建入口 (4方向扫描)
# =========================================================
@njit(parallel=True, fastmath=True)
def build_swivg_map(img, k=0.7, tau_factor=0.8, grid=8):
    """
    构建全图 SW-IVG 度图 (行+列+主对角+反对角)
    """
    H, W = img.shape
    map_out = np.zeros((H, W), dtype=np.float32)

    # 1. 计算 Tau Map
    tau_map = create_tau_map(img, grid, tau_factor)

    # 2. Horizontal Scan
    for r in prange(H):
        line = img[r, :].astype(np.float32)
        t_line = tau_map[r, :].astype(np.float32)
        d = core_sw_ivg(line, t_line, k)
        for c in range(W): map_out[r, c] += d[c]

    # 3. Vertical Scan
    for c in prange(W):
        line = img[:, c].astype(np.float32)
        t_line = tau_map[:, c].astype(np.float32)
        d = core_sw_ivg(line, t_line, k)
        for r in range(H): map_out[r, c] += d[r]

    # 4. Diagonal Scan
    for offset in prange(-H + 1, W):
        if offset >= 0:
            r_s = 0;
            c_s = offset;
            ln = min(H, W - offset)
        else:
            r_s = -offset;
            c_s = 0;
            ln = min(H + offset, W)

        if ln > 1:
            line = np.empty(ln, dtype=np.float32)
            t_line = np.empty(ln, dtype=np.float32)
            for k_idx in range(ln):
                line[k_idx] = img[int(r_s + k_idx), int(c_s + k_idx)]
                t_line[k_idx] = tau_map[int(r_s + k_idx), int(c_s + k_idx)]

            d = core_sw_ivg(line, t_line, k)
            for k_idx in range(ln):
                map_out[int(r_s + k_idx), int(c_s + k_idx)] += d[k_idx]

    # 5. Anti-Diagonal Scan
    for s in prange(H + W - 1):
        if s < W:
            r_s = 0;
            c_s = s
        else:
            r_s = s - (W - 1);
            c_s = W - 1

        ln = 0
        rr, cc = r_s, c_s
        while rr < H and cc >= 0:
            ln += 1;
            rr += 1;
            cc -= 1

        if ln > 1:
            line = np.empty(ln, dtype=np.float32)
            t_line = np.empty(ln, dtype=np.float32)
            for k_idx in range(ln):
                line[k_idx] = img[int(r_s + k_idx), int(c_s - k_idx)]
                t_line[k_idx] = tau_map[int(r_s + k_idx), int(c_s - k_idx)]

            d = core_sw_ivg(line, t_line, k)
            for k_idx in range(ln):
                map_out[int(r_s + k_idx), int(c_s - k_idx)] += d[k_idx]

    return map_out