"""
baselines.py —— 三条基线
---------------------------------------------------------------------------
1. static      —— 不借贷（可借类就地执行）。imb=静态不均，跨洲账单=0。参考上下界。
2. seq         —— 两段式（pure_overlay + pure_underlay）：overlay 只均衡负载（看不到
                  95 计费）解出 x，再交给 underlay 选路承担真实账单。Gap 的「被减数」。
3. joint       —— 本方法：overlay↔underlay 交替迭代（coupling.alternating）。

纯 overlay（BGP 直连、不优化选路）= seq 的 x + underlay optimize=False，见 metrics。
"""
import numpy as np
from overlay import overlay_step
from coupling import alternating
from topology import N_DC, DCS, DC_REGION
from workload import T


# 跨洲借贷预算上限（对标 stage1 BORROWABLE cap；binding 使 imb=0 不可达、问题非退化）
BORROW_BUDGET = 0.5


def static_x(d_b):
    """可借类就地执行：x[i,i,tau]=d_b[i,tau]，无跨洲外溢。"""
    x = np.zeros((N_DC, N_DC, T))
    for i in range(N_DC):
        x[i, i, :] = d_b[i, :]
    return x


def run_static(d_nb, d_b, cap, kappa):
    x = static_x(d_b)
    return x, 1


def run_seq(d_nb, d_b, cap, kappa):
    """两段式：overlay 只看负载不均（joint=False）。"""
    r = overlay_step(d_nb, d_b, cap, kappa, s=None, joint=False,
                     borrow_budget=BORROW_BUDGET)
    return r["x"], 1


def run_joint(d_nb, d_b, cap, kappa):
    """本方法：交替迭代。返回 (x, iters, history)。"""
    r = alternating(d_nb, d_b, cap, kappa, borrow_budget=BORROW_BUDGET)
    return r["x"], r["iters"], r["history"]


if __name__ == "__main__":
    from workload import gen_demand, split_classes, cap_vector
    from metrics import compute_metrics, fmt_row
    d = gen_demand(); d_nb, d_b = split_classes(d); cap = cap_vector()
    for name, fn in [("static", run_static), ("seq", run_seq), ("joint", run_joint)]:
        if name == "joint":
            x, iters, hist = fn(d_nb, d_b, cap, 4.0)
        else:
            x, iters = fn(d_nb, d_b, cap, 4.0)
        m = compute_metrics(x, d_nb, d_b, cap, 4.0, name, iters)
        print(fmt_row(m))
