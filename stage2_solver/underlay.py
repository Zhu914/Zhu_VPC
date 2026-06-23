"""
underlay.py —— follower：抽象多商品流选路 + 95 分位 WAN 成本
---------------------------------------------------------------------------
给定 overlay 注入的跨洲 OD 需求 od[pair,tau]（可借类外溢），在三角洲际拓扑上选路：
  - direct：走 pair 对应的直连洲际链路（低 RTT）
  - via   ：经第三洲中转，走两条洲际链路（高 RTT、但可填谷底避 95 峰）
意图分级：INTENT_DIRECT_FRAC 比例的时延敏感流强制走 direct（约束落地，非事后检查）。
链路复合代价 = 95 分位带宽成本（红线：非线性 CVaR 凸代理）+ RTT 时延成本。

返回实现量：路由 split s、物理链路流 phys、利用率 util、真实 95 账单 bill_95、
          反馈单价 fb = price·(1+β·util)（供下一轮 overlay 用）。

阶段三会把这里的抽象选路换成真 FRRouting/BGP；本阶段先把算法跑通。
"""
import numpy as np
import cvxpy as cp
from topology import (INTER_LINKS, LINK_INDEX, N_LINK, ALL_LINKS, PRICE_95,
                      RTT_INTER, link_cap_vector, INTENT_DIRECT_FRAC)

T = 24
P95 = 0.05
PAIRS = INTER_LINKS                                   # 3 条洲际 pair
# 各 pair 的 direct 链路 与 via 中转所经两条链路
DIRECT_LINK = {p: p for p in PAIRS}
VIA_LINKS = {
    ("AP", "EU"): [("AP", "NA"), ("EU", "NA")],
    ("EU", "NA"): [("AP", "EU"), ("AP", "NA")],
    ("AP", "NA"): [("AP", "EU"), ("EU", "NA")],
}
RTT_DIRECT = {p: RTT_INTER for p in PAIRS}            # 直连 180ms
RTT_VIA = {p: 2 * RTT_INTER for p in PAIRS}           # 中转 360ms

DELAY_GAMMA = 1e-3     # 时延成本权重（相对 95 账单小，体现「弹性流填谷」）
CONGEST_BETA = 1.0     # 拥塞反馈强度：fb = price·(1+β·util)


def _solver():
    return None   # cp.pos 友好的求解器交给 CVXPY 自动选


def od_from_x(x, d_nb, d_b):
    """从 overlay 的 x(N_DC,N_DC,T) 抽出跨洲 OD 需求 od[pair,tau]（pair 索引按 PAIRS）。"""
    from topology import DCS, DC_REGION, N_DC, link_of_regions
    od = np.zeros((len(PAIRS), T))
    pair_idx = {p: i for i, p in enumerate(PAIRS)}
    for i in range(N_DC):
        for j in range(N_DC):
            ri, rj = DC_REGION[DCS[i]], DC_REGION[DCS[j]]
            if ri == rj:
                continue
            p = link_of_regions(ri, rj)
            od[pair_idx[p]] += x[i, j, :]
    return od


def _q95_cvx(flow, T_=T, p=P95):
    q = cp.Variable()
    return q + (1.0 / (p * T_)) * cp.sum(cp.pos(flow - q))


def underlay_route(od, intent_frac=INTENT_DIRECT_FRAC, optimize=True):
    """
    选路。optimize=False → 全 direct（BGP 最短 AS-path 基线）。
    返回 dict：s(dict pair->(T,) direct 分率), phys(N_LINK,T), util(N_LINK,T),
              bill_95_real, fb(N_LINK,), phys_inter(3,T)。
    """
    od = np.asarray(od, dtype=float)
    cap = link_cap_vector()
    price = np.array([PRICE_95[l] for l in ALL_LINKS])

    if not optimize:
        # BGP 基线：全 direct
        f_dir = od
        f_via = np.zeros_like(od)
    else:
        f_dir = cp.Variable((len(PAIRS), T), nonneg=True)
        f_via = cp.Variable((len(PAIRS), T), nonneg=True)
        cons = [f_dir + f_via == od,
                f_dir >= intent_frac * od]            # 时延敏感流强制 direct
        # 物理链路流（洲际 3 条；intra 由 overlay 自带，这里不重复）
        phys_inter = {}
        for p in PAIRS:
            l = LINK_INDEX[p]
            expr = f_dir[PAIRS.index(p)]
            for vp in PAIRS:
                if p in VIA_LINKS[vp]:
                    expr = expr + f_via[PAIRS.index(vp)]
            phys_inter[l] = expr
        # 95 账单 + 时延成本
        bill = cp.sum(cp.vstack([price[l] * _q95_cvx(phys_inter[l]) for l in phys_inter]))
        delay = cp.sum(cp.vstack([RTT_DIRECT[p] * f_dir[PAIRS.index(p)]
                                  + RTT_VIA[p] * f_via[PAIRS.index(p)]
                                  for p in PAIRS]))
        cp.Problem(cp.Minimize(bill + DELAY_GAMMA * delay), cons).solve(solver=_solver())
        f_dir = f_dir.value
        f_via = f_via.value
        if f_dir is None:
            raise RuntimeError("underlay infeasible")

    # 结算实现量
    phys = np.zeros((N_LINK, T))
    for p in PAIRS:
        l = LINK_INDEX[p]
        phys[l] += f_dir[PAIRS.index(p)]
        for vp in PAIRS:
            if p in VIA_LINKS[vp]:
                phys[l] += f_via[PAIRS.index(vp)]
    # direct 分率（防除零）
    s = {}
    for p in PAIRS:
        denom = np.where(od[PAIRS.index(p)] > 1e-9, od[PAIRS.index(p)], 1.0)
        s[p] = np.clip(f_dir[PAIRS.index(p)] / denom, 0, 1)
    util = phys / cap[:, None]
    wan95 = np.percentile(phys, 100 * (1 - P95), axis=1)        # 每链路 95 分位
    inter_idx = [LINK_INDEX[p] for p in PAIRS]
    bill_95_real = float(np.sum(price[inter_idx] * wan95[inter_idx]))  # 跨洲账单
    fb = price * (1.0 + CONGEST_BETA * util.mean(axis=1))       # 反馈单价
    return dict(s=s, phys=phys, util=util, bill_95=bill_95_real,
                fb=fb, wan95=wan95)


if __name__ == "__main__":
    from workload import gen_demand, split_classes
    from overlay import overlay_step
    from topology import price_vector
    d = gen_demand(); d_nb, d_b = split_classes(d)
    r = overlay_step(d_nb, d_b, np.full(6, 1.0), 2.0, price_vector(), s=None, joint=True)
    od = od_from_x(r["x"], d_nb, d_b)
    print("OD row sums:", od.sum(axis=1).round(3))
    u = underlay_route(od, optimize=True)
    print("bill95(opt)=%.3f" % u["bill_95"])
    print("util mean:", u["util"].mean(axis=1).round(3))
    print("direct frac:", {p: u["s"][p].mean().round(3) for p in PAIRS})
    ub = underlay_route(od, optimize=False)
    print("bill95(BGP-direct)=%.3f" % ub["bill_95"])
