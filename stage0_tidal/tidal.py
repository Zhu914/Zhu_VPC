"""
阶段零：跨大洲负载潮汐相位差测量
--------------------------------
原理：取一份集群负载 trace，把时间轴按时区平移（亚太 +0h、欧洲 +8h、北美 +16h），
即得三条相位错开的曲线。错峰一眼可见，并量化借贷空间 = 逐小时（最忙洲 - 最闲洲）。

数据模式（TRACE_MODE）：
  "synthetic" —— 生成带昼夜节律的合成 trace，用于跑通管线（不作 go/no-go 结论）
  "real"      —— 读真实 trace CSV（如 Alibaba cluster-trace-v2018）
切换到 real 时，只需改 CSV / COL_TIME / COL_CPU 三处。
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt

# ============================================================
# 配置区
# ============================================================
TRACE_MODE = "synthetic"  # "synthetic" 或 "real"

# real 模式下的 trace 路径与列名（按你的 trace 改）
CSV = "../traces/clusterdata/cluster-trace-v2018/data/server_usage.csv"
COL_TIME, COL_CPU = "time_stamp", "cpu_util_percent"

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 三洲时区平移（小时）。同一条基础形状按此错峰。
SHIFT = {"AP": 0, "EU": 8, "NA": 16}


# ============================================================
# 数据获取
# ============================================================
def load_real_trace():
    """读真实 trace，抽样子集，返回 (time_stamp, cpu) 两列的 DataFrame。"""
    df = pd.read_csv(CSV, usecols=[COL_TIME, COL_CPU], nrows=5_000_000)
    df = df.dropna()
    return df


def gen_synthetic_trace(n_machines=2000, n_hours=24, seed=7):
    """
    生成合成 trace：每台机器一条带昼夜节律的 CPU 时序 + 噪声。
    昼夜节律用「白天高、夜里低」的正弦近似，模拟交互型业务负载形状。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    # 基础昼夜形状：9 点附近峰值
    base = 0.45 + 0.35 * np.sin(2 * np.pi * (t - 9) / 24)
    rows = []
    for m in range(n_machines):
        amp_jit = rng.uniform(0.8, 1.2)
        phase_jit = rng.normal(0, 0.5)
        noise = rng.normal(0, 0.05, n_hours)
        cpu = np.clip(base * amp_jit + noise, 0, 1)
        for h in range(n_hours):
            rows.append((h * 3600, cpu[h] * 100))  # 秒级时间戳，与 real trace 口径一致
    return pd.DataFrame(rows, columns=[COL_TIME, COL_CPU])


# ============================================================
# 聚合 → 24h 归一形状
# ============================================================
def hourly_normalized_shape(df):
    """把 (time_stamp, cpu) 聚合成一天 24 小时的归一负载形状（长度 24 的数组）。"""
    df = df.copy()
    df["hour"] = (df[COL_TIME].astype(float) // 3600 % 24).astype(int)
    base = df.groupby("hour")[COL_CPU].mean()
    base = base.reindex(range(24), fill_value=0.0)
    base = base / base.max()
    return base.values


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"[trace mode] {TRACE_MODE}")
    if TRACE_MODE == "real":
        if not os.path.exists(CSV):
            raise FileNotFoundError(
                f"未找到真实 trace：{CSV}\n请先下载（如 Alibaba clusterdata），"
                f"或先把 TRACE_MODE 改回 'synthetic' 跑通管线。"
            )
        df = load_real_trace()
    else:
        df = gen_synthetic_trace()

    print(f"[loaded] {len(df)} 条记录")

    base = hourly_normalized_shape(df)
    curves = {r: np.roll(base, s) for r, s in SHIFT.items()}

    # ---- 画相位差图 ----
    plt.figure(figsize=(9, 4.5))
    for r, c in curves.items():
        plt.plot(range(24), c, marker="o", label=r)
    plt.xlabel("UTC hour")
    plt.ylabel("normalized load")
    title = "Cross-continent load phase difference"
    if TRACE_MODE == "synthetic":
        title += " (synthetic trace — pipeline check, NOT go/no-go)"
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "tidal_phase.png")
    plt.savefig(out_png, dpi=150)
    print(f"[fig] saved {out_png}")

    # ---- 量化借贷空间 ----
    M = np.vstack(list(curves.values()))
    gap = M.max(axis=0) - M.min(axis=0)
    print(f"avg phase gap: {gap.mean():.3f}")
    print(f"max phase gap: {gap.max():.3f}  at UTC {int(gap.argmax())}h")

    # ---- 绿灯判据 ----
    if gap.mean() >= 0.3:
        print("[checkpoint] PASS: avg gap >= 0.3，错峰显著，借贷有空间（绿灯）")
    else:
        print("[checkpoint] WARN: avg gap < 0.3，错峰不显著 —— "
              "若为 real 模式则红灯，换 trace 或只取交互型业务再试。")

    # 存数值结果，便于复用
    np.save(os.path.join(FIG_DIR, "tidal_base_shape.npy"), base)
    return gap


if __name__ == "__main__":
    main()
