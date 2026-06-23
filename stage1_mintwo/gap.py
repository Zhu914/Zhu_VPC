"""
阶段一：两 DC 最小模型 —— 解耦代价 Gap 随地理异构度 κ 增长
----------------------------------------------------------
命根子实验：OPT_seq − OPT_joint 是否随 κ 单调上升？

OPT_joint（联合）：同时决定借贷 x(τ) 与其 WAN 95 分位计费影响，
    min  负载不均(方差) + κ·WAN95成本
OPT_seq（两段式）：先只按"均衡负载"解出 x（看不到 95 计费），
    再把 x 固定、由 underlay 承担它产生的真实 95 成本。

■ 红线：95 分位计费必须非线性。
  - 优化内部用 CVaR 凸代理引导（凸、可解）；
  - 最终目标统一用真实 np.percentile(arr,95) 结算，joint/seq 同算子，
    保证 Gap = seq − joint 是同口径比较。
"""
import os
import numpy as np
import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 潮汐负载（可替换为 stage0 的真实曲线）
# ============================================================
T = 24
t = np.arange(T)
dA = 0.6 + 0.4 * np.sin(2 * np.pi * (t - 9) / 24)    # 欧洲本地需求（白天峰）
dB = 0.6 + 0.4 * np.sin(2 * np.pi * (t - 21) / 24)   # 北美本地需求（错峰 12h）
CAP = 1.0          # 每 DC 每时隙算力容量（归一）
BORROWABLE = 0.5   # 可借类占本地需求的比例上限
P95 = 0.05         # 95 分位 = 尾部 5%


# ============================================================
# 95 分位算子
# ============================================================
def q95_cvx(flow, T, p=P95):
    """CVaR 型凸代理：min_q q + 1/(pT)·Σ(flow−q)₊  —— 95 分位的凸上界，用于优化。"""
    q = cp.Variable()
    return q + (1.0 / (p * T)) * cp.sum(cp.pos(flow - q))


def q95_np(arr, p=P95):
    """真实 95 分位（后验结算用）。"""
    return float(np.percentile(arr, 100 * (1 - p)))


def pick_solver():
    """优先 HiGHS，退到默认凸求解器。sum_squares 使问题为 QP，HiGHS 可能不支持。"""
    for s in (cp.HIGHS,):
        try:
            if s in cp.installed_solvers():
                return s
        except Exception:
            pass
    return None  # 让 CVXPY 自动选（Clarabel/ECOS/SCS）


# ============================================================
# 单次求解：返回 (x_value, imb_true, wan95_true, obj_true)
# joint=True 联合优化；joint=False 两段式
# ============================================================
def solve(kappa, joint=True):
    # 拆成两个非负变量：x_p=A→B，x_m=B→A；净借贷 x = x_p − x_m
    # 这样 loadA/loadB/wan 全为仿射，DCP 干净；wan = x_p+x_m = |x| 在最优处精确成立。
    x_p = cp.Variable(T, nonneg=True)
    x_m = cp.Variable(T, nonneg=True)
    loadA = dA - x_p + x_m
    loadB = dB + x_p - x_m
    cons = [loadA <= CAP, loadB <= CAP, loadA >= 0, loadB >= 0,
            x_p + x_m <= BORROWABLE]

    wan = x_p + x_m                       # WAN 流量 = 跨洲外溢量（红线：非线性计费）
    imb_cvx = cp.sum_squares(loadA - loadB)   # 仿射的平方，DCP 合法

    if joint:
        obj = cp.Minimize(imb_cvx + kappa * q95_cvx(wan, T))
        cp.Problem(obj, cons).solve(solver=pick_solver())
        x_val = x_p.value - x_m.value
    else:
        # 第一步：只均衡负载，看不到 95 计费
        cp.Problem(cp.Minimize(imb_cvx), cons).solve(solver=pick_solver())
        x_val = x_p.value - x_m.value

    # 统一用真实算子结算（同口径）
    loadA_v = dA - np.maximum(x_val, 0) + np.maximum(-x_val, 0)
    loadB_v = dB - np.maximum(-x_val, 0) + np.maximum(x_val, 0)
    imb_true = float(np.sum((loadA_v - loadB_v) ** 2))
    wan_v = np.abs(x_val)
    wan95_true = q95_np(wan_v)
    obj_true = imb_true + kappa * wan95_true
    return x_val, imb_true, wan95_true, obj_true


# ============================================================
# 扫描 κ，画 Gap 曲线
# ============================================================
def main():
    kappas = np.linspace(0.2, 8.0, 20)
    rows = []
    for kp in kappas:
        _, imb_j, wan_j, j = solve(kp, joint=True)
        _, imb_s, wan_s, s = solve(kp, joint=False)
        gap = s - j
        rows.append((kp, j, s, gap, imb_j, wan_j, imb_s, wan_s))
        print(f"κ={kp:4.2f}  joint={j:7.3f}  seq={s:7.3f}  "
              f"GAP={gap:7.3f}  (wan95 j={wan_j:.3f} s={wan_s:.3f})")

    kappas = np.array([r[0] for r in rows])
    gaps = np.array([r[3] for r in rows])

    # 单调性检验
    mono = np.all(np.diff(gaps) >= -1e-6)
    print(f"\n[monotone non-decreasing] {mono}")
    print(f"[gap range] {gaps.min():.3f} -> {gaps.max():.3f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(kappas, gaps, marker="o")
    plt.xlabel("geographic heterogeneity  κ")
    plt.ylabel("decoupling gap  (seq − joint)")
    plt.title("Decoupling gap grows with κ")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "gap_vs_kappa.png")
    plt.savefig(out, dpi=150)
    print(f"[fig] saved {out}")

    # 存数值，供后续阶段/绘图复用
    np.save(os.path.join(FIG_DIR, "gap_kappas.npy"), kappas)
    np.save(os.path.join(FIG_DIR, "gap_values.npy"), gaps)

    # 绿灯判据
    if mono and gaps[-1] > gaps[0] and gaps.max() > 0:
        print("[checkpoint] PASS: Gap 随 κ 单调上升 = 命根子成立（全程最重要绿灯）")
    else:
        print("[checkpoint] WARN: Gap 未单调上升 —— 回查 95 分位是否被线性化 / 潮汐是否错峰")

    return rows


if __name__ == "__main__":
    main()
