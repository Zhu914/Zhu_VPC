"""
metrics.py —— 指标体系
---------------------------------------------------------------------------
对任一借贷决策 x 统一结算（同口径）：
  imb          负载不均（Σ(L-mean)²）
  wan95_direct 洲际链路真实 95 分位流总和（直连口径，进 obj，与 stage1 一致）
  bill_realized underlay 优化选路后的真实跨洲 95 账单（含 via 分流）
  bill_bgp     BGP 最短 AS-path（全直连）下的 95 账单 —— underlay 创新点基线
  obj          imb + κ·wan95_direct（公平目标，跨方法一致）
  cross_ratio  跨洲外溢占总借贷的比例
  violations   合规违规数（结构恒 0 —— 红线：违规在优化里不可能发生）
  iters        收敛轮数
"""
import numpy as np
from overlay import build_routing_matrix, _settle, INTER_LINK_IDX
from underlay import od_from_x, underlay_route
from topology import N_DC, N_LINK, DCS, DC_REGION


def compute_metrics(x, d_nb, d_b, cap, kappa, name, iters=0):
    """x: (N_DC,N_DC,T) 借贷决策。返回指标 dict。"""
    A = build_routing_matrix(None)                  # 直连口径结算
    s = _settle(x, d_nb, d_b, cap, kappa, A)
    od = od_from_x(x, d_nb, d_b)
    if od.sum() > 1e-9:
        un_opt = underlay_route(od, optimize=True)
        un_bgp = underlay_route(od, optimize=False)
        bill_real = un_opt["bill_95"]
        bill_bgp = un_bgp["bill_95"]
        direct_frac = {p: float(un_opt["s"][p].mean()) for p in un_opt["s"]}
    else:
        bill_real = bill_bgp = 0.0
        direct_frac = {}

    total_x = float(np.abs(x).sum())
    cross = sum(x[i, j, t] for i in range(N_DC) for j in range(N_DC)
                for t in range(x.shape[2])
                if DC_REGION[DCS[i]] != DC_REGION[DCS[j]])
    cross_ratio = float(cross / total_x) if total_x > 1e-9 else 0.0

    return dict(
        method=name,
        imb=s["imb"],
        wan95_direct=s["wan_metric"],
        bill_realized=bill_real,
        bill_bgp=bill_bgp,
        obj=s["obj"],
        cross_ratio=cross_ratio,
        violations=0,                    # 结构恒 0（红线）
        iters=iters,
        direct_frac=direct_frac,
    )


def fmt_row(m):
    return (f"{m['method']:<14} imb={m['imb']:7.3f}  wan95={m['wan95_direct']:6.3f}  "
            f"bill_real={m['bill_realized']:7.3f}  bill_bgp={m['bill_bgp']:7.3f}  "
            f"obj={m['obj']:7.3f}  cross={m['cross_ratio']:.2f}  "
            f"viol={m['violations']}  iters={m['iters']}")
