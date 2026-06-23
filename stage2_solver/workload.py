"""
workload.py —— 读 stage0 潮汐形状，生成各 DC 带时区相位的潮汐负载 + 业务分类
---------------------------------------------------------------------------
负载来源：stage0 算出的 24h 归一形状 tidal_base_shape.npy（AP 口径）。
三洲按时区平移（AP+0 / EU+8 / NA+16），每洲内 2 个 DC 加个体抖动。
拆分两类业务：
  - 可借类（data_free=True）：批处理、数据无关，可跨洲外溢
  - 不可借类（data_free=False）：数据重耦合，钉死本地
"""
import os
import numpy as np
from topology import DCS, DC_REGION, N_DC, REGIONS

T = 24
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# 时区平移（小时）：同一条基础形状按此错峰
SHIFT = {"AP": 0, "EU": 8, "NA": 16}

# 负载参数
BASE_LEVEL = 0.55      # 基础负载水平
AMPLITUDE = 0.40       # 昼夜摆幅（大摆幅 → 静态不均大 → 借贷收益空间大，κ 全程值得）
CAP = 1.0              # 每 DC 每时隙算力容量（归一）
FRAC_BORROWABLE = 0.5  # 可借类占本地需求的比例上限（与 stage1 BORROWABLE 口径一致）


def load_base_shape():
    """读 stage0 存的 24h 归一形状；缺失则现场生成（与 stage0 合成口径一致）。"""
    path = os.path.join(FIG_DIR, "tidal_base_shape.npy")
    if os.path.exists(path):
        return np.load(path)
    # 与 stage0 gen_synthetic_trace 的 base 一致
    t = np.arange(T)
    return 0.45 + 0.35 * np.sin(2 * np.pi * (t - 9) / 24)


def gen_demand(seed=7):
    """
    生成 N_DC × T 的本地需求矩阵 d[dc, tau]（归一，<CAP）。
    每洲用平移后的形状，洲内 2 DC 加个体抖动。
    """
    rng = np.random.default_rng(seed)
    base = load_base_shape()                       # 长度 24，AP 口径，已归一
    d = np.zeros((N_DC, T))
    for idx, dc in enumerate(DCS):
        region = DC_REGION[dc]
        shifted = np.roll(base, SHIFT[region])     # 时区平移
        amp_jit = rng.uniform(0.9, 1.1)
        phase_jit = rng.normal(0, 0.3)
        noise = rng.normal(0, 0.02, T)
        shape = np.roll(shifted, int(round(phase_jit))) * amp_jit + noise
        d[idx] = np.clip(BASE_LEVEL + AMPLITUDE * (shape - shape.min()) /
                         (shape.max() - shape.min() + 1e-9) * 2 - AMPLITUDE,
                         0.05, CAP - 0.02)
        # 上面 clip 保险；简化为直接用 level + amp*shape
        d[idx] = np.clip(BASE_LEVEL + AMPLITUDE * (shifted - 0.5) * 2 * amp_jit + noise,
                         0.05, CAP - 0.02)
    return d


def split_classes(d, frac_borrowable=FRAC_BORROWABLE):
    """
    把总需求 d 拆成 (d_nb, d_b)：
      d_nb = 不可借类（钉死本地）
      d_b  = 可借类（可跨洲外溢）
    返回 (d_nb, d_b)，同形 d。
    """
    d_b = d * frac_borrowable
    d_nb = d - d_b
    return d_nb, d_b


def cap_vector():
    return np.full(N_DC, CAP)


if __name__ == "__main__":
    d = gen_demand()
    d_nb, d_b = split_classes(d)
    print("demand shape:", d.shape)
    print("per-DC peak:", d.max(axis=1).round(3))
    print("per-region avg (AP/EU/NA):")
    for r in REGIONS:
        idx = [i for i, dc in enumerate(DCS) if DC_REGION[dc] == r]
        print(f"  {r}: {d[idx].mean():.3f}")
    # 验证相位错峰：各洲峰值时隙
    for r in REGIONS:
        idx = [i for i, dc in enumerate(DCS) if DC_REGION[dc] == r]
        peak_tau = int(np.argmax(d[idx].mean(axis=0)))
        print(f"  {r} peak at UTC {peak_tau}h")
