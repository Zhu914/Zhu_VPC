"""
run.py —— 阶段二主入口
---------------------------------------------------------------------------
产出：
  1. Gap–κ 曲线（完整模型版，命根子复现）：joint vs seq 同口径结算，GAP 随 κ 单调上升。
  2. 多方法对比表（CSV + 终端）：static / seq / joint 的 imb、95账单、obj、跨洲占比、违规、收敛轮。
  3. 收敛曲线：joint 交替迭代 history。
检查点（手册 §2.2）：
  ① joint 在 obj 上优于所有基线（imb 击败 static、wan95 击败 seq）；
  ② 完整模型下 Gap 仍随 κ 上升；
  ③ 合规违规数恒 0；
  ④ 收敛曲线收敛。
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from workload import gen_demand, split_classes, cap_vector
from overlay import overlay_step
from coupling import alternating
from baselines import run_static, run_seq, BORROW_BUDGET
from metrics import compute_metrics, fmt_row

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

KAPPAS = np.linspace(0.2, 8.0, 15)
K_TABLE = 4.0      # 对比表所用的代表性 κ


# ============================================================
# 1. Gap–κ 曲线（完整模型版）
# ============================================================
def sweep_gap(d_nb, d_b, cap):
    rows = []
    for kp in KAPPAS:
        # joint：完整双层交替迭代（与 seq 同 borrow_budget，公平同口径）
        rj = alternating(d_nb, d_b, cap, kp, borrow_budget=BORROW_BUDGET)
        mj = compute_metrics(rj["x"], d_nb, d_b, cap, kp, "joint", rj["iters"])
        # seq：两段式（overlay 只看负载不均）
        rs = overlay_step(d_nb, d_b, cap, kp, s=None, joint=False,
                          borrow_budget=BORROW_BUDGET)
        ms = compute_metrics(rs["x"], d_nb, d_b, cap, kp, "seq", 1)
        gap = ms["obj"] - mj["obj"]
        rows.append((kp, mj["obj"], ms["obj"], gap, mj["imb"], mj["wan95_direct"],
                     ms["imb"], ms["wan95_direct"]))
        print(f"κ={kp:4.2f}  joint_obj={mj['obj']:7.3f}  seq_obj={ms['obj']:7.3f}  "
              f"GAP={gap:6.3f}  (wan95 j={mj['wan95_direct']:.3f} s={ms['wan95_direct']:.3f})")
    arr = np.array(rows)
    kappas = arr[:, 0]; gaps = arr[:, 3]
    mono = bool(np.all(np.diff(gaps) >= -1e-6))
    print(f"\n[monotone non-decreasing] {mono}")
    print(f"[gap range] {gaps.min():.3f} -> {gaps.max():.3f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(kappas, gaps, marker="o", color="tab:red")
    plt.xlabel("geographic heterogeneity  κ")
    plt.ylabel("decoupling gap  (seq − joint)")
    plt.title("Stage2 full-model decoupling gap grows with κ")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "stage2_gap_vs_kappa.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[fig] saved {out}")
    np.save(os.path.join(FIG_DIR, "stage2_gap_kappas.npy"), kappas)
    np.save(os.path.join(FIG_DIR, "stage2_gap_values.npy"), gaps)
    return rows, mono


# ============================================================
# 2. 多方法对比表
# ============================================================
def comparison_table(d_nb, d_b, cap, kappa):
    print(f"\n=== comparison @ κ={kappa} ===")
    rows = []
    # static
    x, it = run_static(d_nb, d_b, cap, kappa)
    rows.append(compute_metrics(x, d_nb, d_b, cap, kappa, "static", it))
    # seq
    x, it = run_seq(d_nb, d_b, cap, kappa)
    rows.append(compute_metrics(x, d_nb, d_b, cap, kappa, "seq", it))
    # joint
    rj = alternating(d_nb, d_b, cap, kappa, borrow_budget=BORROW_BUDGET)
    rows.append(compute_metrics(rj["x"], d_nb, d_b, cap, kappa, "joint", rj["iters"]))

    for m in rows:
        print(fmt_row(m))

    # 写 CSV
    csv_path = os.path.join(RES_DIR, "stage2_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "imb", "wan95_direct", "bill_realized", "bill_bgp",
                    "obj", "cross_ratio", "violations", "iters"])
        for m in rows:
            w.writerow([m["method"], f"{m['imb']:.4f}", f"{m['wan95_direct']:.4f}",
                        f"{m['bill_realized']:.4f}", f"{m['bill_bgp']:.4f}",
                        f"{m['obj']:.4f}", f"{m['cross_ratio']:.4f}",
                        m["violations"], m["iters"]])
    print(f"[csv] saved {csv_path}")

    # 检查点 ①：joint obj 优于所有基线
    objs = {m["method"]: m["obj"] for m in rows}
    joint_best = objs["joint"] <= min(objs.values()) + 1e-6
    print(f"[checkpoint ①] joint obj 优于所有基线: {joint_best}  "
          f"(static={objs['static']:.3f} seq={objs['seq']:.3f} joint={objs['joint']:.3f})")
    # 检查点 ③：合规违规恒 0
    viol0 = all(m["violations"] == 0 for m in rows)
    print(f"[checkpoint ③] 合规违规数恒 0: {viol0}")
    return rows, rj["history"]


# ============================================================
# 3. 收敛曲线
# ============================================================
def plot_convergence(d_nb, d_b, cap, kappa):
    # 跑满 max_iter（不强停）以展示完整轨迹 = 经验收敛证据
    rj = alternating(d_nb, d_b, cap, kappa, max_iter=12, tol=0.0,
                     borrow_budget=BORROW_BUDGET)
    history = rj["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(history) + 1), history, marker="o", color="tab:blue")
    plt.xlabel("alternation iteration")
    plt.ylabel("realized objective  (imb + κ·wan95)")
    plt.title(f"Stage2 bilevel convergence @ κ={kappa}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "stage2_convergence.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[fig] saved {out}")
    # 收敛判据：后三轮相对摆动 < 1%
    tail = history[-3:]
    swing = (max(tail) - min(tail)) / max(1e-9, abs(tail[-1]))
    conv = swing < 0.01
    print(f"[checkpoint ④] 收敛曲线收敛: {conv}  (tail swing={swing:.2%}, {len(history)} iters)")


# ============================================================
# 主流程
# ============================================================
def main():
    d = gen_demand()
    d_nb, d_b = split_classes(d)
    cap = cap_vector()
    print(f"[workload] {d.shape}  static imb={np.sum((d-d.mean(0,keepdims=True))**2):.3f}\n")

    print("=== 1. Gap–κ sweep (full model) ===")
    _, mono = sweep_gap(d_nb, d_b, cap)

    print("\n=== 2. comparison table ===")
    comparison_table(d_nb, d_b, cap, K_TABLE)

    print("\n=== 3. convergence ===")
    plot_convergence(d_nb, d_b, cap, K_TABLE)

    print("\n[checkpoint ②] 完整模型下 Gap 随 κ 单调上升: "
          f"{'PASS' if mono else 'WARN'}")


if __name__ == "__main__":
    main()
