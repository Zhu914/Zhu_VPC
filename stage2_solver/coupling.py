"""
coupling.py —— 双层交替迭代：overlay(leader) ↔ underlay(follower)
---------------------------------------------------------------------------
每轮：
  1. overlay 用当前路由 split s 解借贷 x（目标 = 负载不均 + κ·Σq95(phys_est)）
  2. underlay 据新 OD 重选路，返回实现 95 账单与新的 split s
  3. 阻尼更新 s（防震荡），记录实现目标 history
实现目标 = 负载不均 + κ·Σ_link q95(phys_real)（公平口径，fb=1，跨方法一致）。
收敛判据：相邻两轮目标相对变化 < tol。

⚠ 易错 —— 双层交替可能震荡；history 画下来即「经验收敛证据」；震荡时阻尼压制。
"""
import numpy as np
from overlay import overlay_step
from underlay import od_from_x, underlay_route

DAMP = 0.5   # s 阻尼：s ← DAMP·s_new + (1-DAMP)·s_old


def _damp_s(s_old, s_new):
    if s_old is None:
        return s_new
    return {p: DAMP * s_new[p] + (1 - DAMP) * s_old[p] for p in s_new}


def alternating(d_nb, d_b, cap, kappa, max_iter=30, tol=1e-4, verbose=False,
                borrow_budget=None):
    """返回 dict：x, L, imb, wan_metric, bill, phys, util, s, history, iters。
    borrow_budget 须与 seq 基线一致，保证 Gap 同口径公平。"""
    s = None                       # 首轮全 direct
    history = []
    last = None
    r_ov = r_un = None
    for it in range(max_iter):
        r_ov = overlay_step(d_nb, d_b, cap, kappa, s, joint=True,
                            borrow_budget=borrow_budget)
        od = od_from_x(r_ov["x"], d_nb, d_b)
        r_un = underlay_route(od, optimize=True)
        obj = r_ov["imb"] + kappa * float(np.sum(r_un["wan95"]))
        history.append(obj)
        if verbose:
            print(f"  iter {it:2d}  imb={r_ov['imb']:.4f}  wan95={np.sum(r_un['wan95']):.4f}  "
                  f"bill={r_un['bill_95']:.4f}  obj={obj:.4f}")
        s = _damp_s(s, r_un["s"])
        if last is not None and abs(obj - last) <= tol * max(1.0, abs(last)):
            break
        last = obj
    return dict(x=r_ov["x"], L=r_ov["L"], imb=r_ov["imb"],
                wan_metric=float(np.sum(r_un["wan95"])), bill=r_un["bill_95"],
                phys=r_un["phys"], util=r_un["util"], s=s,
                history=history, iters=it + 1)


if __name__ == "__main__":
    from workload import gen_demand, split_classes, cap_vector
    d = gen_demand(); d_nb, d_b = split_classes(d); cap = cap_vector()
    r = alternating(d_nb, d_b, cap, kappa=4.0, verbose=True)
    print(f"\nconverged in {r['iters']} iters, final obj={r['history'][-1]:.4f}")
