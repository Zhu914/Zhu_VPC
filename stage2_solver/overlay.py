"""
overlay.py —— leader：慢尺度划分 h（隐含于业务分类）+ 快尺度借贷 x
---------------------------------------------------------------------------
决策变量 x[i,j,tau] >= 0：tau 时隙、可借类业务从 DC i 外溢到 DC j 执行的算力量。
  - x[i,i,tau]：可借类就地执行
  - x[i,j,tau] (i≠j)：跨洲/跨 DC 外溢，占用 WAN
守恒：Σ_j x[i,j,tau] = d_b[i,tau]   （可借类需求全部被服务 somewhere）
负载：L[j,tau] = d_nb[j,tau] + Σ_i x[i,j,tau]   （不可借钉死本地 + 收到的外溢）
合规：不可借类不产生 x（结构落地），可借类 data_free 可跨任意洲 → 违规恒 0（红线）

目标（joint）：min  负载不均(方差) + κ·Σ_link fb[link]·q95_cvx(phys_est[link,:])
  - fb[link]：underlay 反馈的有效单价（含拥塞），交替迭代时更新
  - phys_est：用上轮路由 split s 把 x 映射到物理链路流的估计（仿射，DCP 合法）
  - q95_cvx：沿用 stage1 的 CVaR 凸代理（红线：95 分位非线性）

成本/借贷逻辑沿用 stage1 已验证版本，不重写；新增的是多 DC + 路由耦合。
"""
import numpy as np
import cvxpy as cp
from topology import (DCS, DC_REGION, N_DC, ALL_LINKS, LINK_INDEX, N_LINK,
                      INTER_LINKS, link_of_regions)

T = 24
P95 = 0.05
_K95 = max(1, int(np.ceil(0.05 * T)))     # 95 分位 ≈ top-k 均值，k=ceil(0.05T)

# 洲际链路下标（κ 项只计跨洲成本；区域内借贷免费）
INTER_LINK_IDX = [LINK_INDEX[p] for p in INTER_LINKS]


def q95_cvx(flow):
    """95 分位紧凸代理：top-k 均值 sum_largest(flow,k)/k（手册推荐，与真实 95 分位单调）。"""
    return cp.sum_largest(flow, _K95) / _K95

# 洲际 pair → via 路径 traversed 的另两条链路（三角拓扑）
VIA_LINKS = {
    ("AP", "EU"): [("AP", "NA"), ("EU", "NA")],   # 经 NA
    ("EU", "NA"): [("AP", "EU"), ("AP", "NA")],   # 经 AP
    ("AP", "NA"): [("AP", "EU"), ("EU", "NA")],   # 经 EU
}


def _solver():
    # q95_cvx 用 cp.pos（需锥求解器）；优先 Clarabel，否则交给 CVXPY 自动选。
    for s in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            if s in cp.installed_solvers():
                return s
        except Exception:
            pass
    return None


def build_routing_matrix(s):
    """
    给定 direct 分率 s[pair,tau]（inter pair × T），构造 A：把 x(N_DC*N_DC*T)
    线性映射到 phys_est(N_LINK*T)。intra x 走 intra 链路；inter x 按 s 拆 direct/via。
    s 为 None 时退化为「全 direct」（BGP 基线用）。
    """
    A = np.zeros((N_LINK * T, N_DC * N_DC * T))
    if s is None:
        s = {p: np.ones(T) for p in INTER_LINKS}
    for tau in range(T):
        for i in range(N_DC):
            for j in range(N_DC):
                col = (i * N_DC + j) * T + tau
                ri, rj = DC_REGION[DCS[i]], DC_REGION[DCS[j]]
                if ri == rj:
                    # 区域内外溢 → intra 链路
                    link = link_of_regions(ri, rj)   # (ri,ri)
                    row = LINK_INDEX[link] * T + tau
                    A[row, col] = 1.0
                else:
                    p = link_of_regions(ri, rj)       # inter pair
                    sp = s[p][tau]
                    # direct
                    row = LINK_INDEX[p] * T + tau
                    A[row, col] += sp
                    # via
                    for vl in VIA_LINKS[p]:
                        row = LINK_INDEX[vl] * T + tau
                        A[row, col] += (1.0 - sp)
    return A


def overlay_step(d_nb, d_b, cap, kappa, s, joint=True, fb=None, borrow_budget=None):
    """
    解 overlay。返回 dict：x, L, phys, imb, wan95(每链路), wan_metric, bill_money, obj。
    s: direct 分率 dict 或 None（全 direct）。fb: 各链路相对权重(N_LINK,)，默认全 1。
      —— κ 项用「原始 95 分位流」×κ（与 stage1 口径一致：κ 本身是成本标量，
         per-link price 不在此放大 κ 项，只用于 underlay 选路与真实账单指标）。
    borrow_budget: 每时隙跨洲外溢总量上限（对标 stage1 的 |x|<=BORROWABLE）。
      设为 binding 使 imb=0 不可达 → 问题非退化 → 求解器可靠；joint/seq 在
      imb-wan 前沿上分化，GAP 随 κ 上升。
    joint=False → 两段式第一步：只均衡负载，不看 95 计费（seq 基线用）。
    """
    if fb is None:
        fb = np.ones(N_LINK)
    fb = np.asarray(fb, dtype=float).reshape(-1)

    x = cp.Variable((N_DC, N_DC, T), nonneg=True)
    L = d_nb + cp.sum(x, axis=0)                      # L[j,tau] = d_nb[j]+Σ_i x[i,j]
    cons = [cp.sum(x, axis=1) == d_b,                 # 守恒：可借类需求全被服务
            L <= cap[:, None], L >= 0]

    # 跨洲外溢上限（只约束跨洲 x，同洲对不计入预算）
    if borrow_budget is not None:
        cross_pairs = [(i, j) for i in range(N_DC) for j in range(N_DC)
                       if DC_REGION[DCS[i]] != DC_REGION[DCS[j]]]
        for tau in range(T):
            cons.append(sum(x[i, j, tau] for i, j in cross_pairs) <= borrow_budget)

    A = build_routing_matrix(s)
    phys_vec = A @ cp.reshape(x, (N_DC * N_DC * T,), order="C")
    phys = cp.reshape(phys_vec, (N_LINK, T), order="C")

    imb = cp.sum_squares(L - cp.mean(L, axis=0, keepdims=True))
    # 小正则：偏好少跨洲搬运。打破 imb-min 退化面（否则求解器在等优解里乱选，
    # 低 κ 时 joint 反而 wan 更高 → 负 Gap）。joint/seq 同加，保证同口径。
    cross_pairs = [(i, j) for i in range(N_DC) for j in range(N_DC)
                   if DC_REGION[DCS[i]] != DC_REGION[DCS[j]]]
    reg = 1e-3 * sum(x[i, j, t] for i, j in cross_pairs for t in range(T))
    if joint:
        # κ 项只计洲际链路 95 分位（跨洲 WAN 成本；区域内借贷免费）
        wan_cost = sum(float(fb[l]) * q95_cvx(phys[l, :]) for l in INTER_LINK_IDX)
        obj = cp.Minimize(imb + kappa * wan_cost + reg)
    else:
        obj = cp.Minimize(imb + reg)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=_solver())
    if prob.status not in ("optimal", "optimal_inaccurate"):
        for fbk in (cp.CLARABEL, cp.SCS, cp.ECOS):
            try:
                prob.solve(solver=fbk)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue

    x_val = x.value
    if x_val is None:
        raise RuntimeError(f"overlay infeasible/unbounded (status={prob.status})")
    return _settle(x_val, d_nb, d_b, cap, kappa, A)


def _settle(x_val, d_nb, d_b, cap, kappa, A):
    """统一用真实算子结算（同口径）。
    obj = imb + κ·Σ_{inter} q95(phys)   —— 公平目标（fb=1，跨方法一致）
    bill_money = Σ_{inter} price·q95(phys) —— 真实跨洲 95 账单（指标）"""
    from topology import price_vector
    L = d_nb + x_val.sum(axis=0)
    imb_true = float(np.sum((L - L.mean(axis=0, keepdims=True)) ** 2))
    phys = (A @ x_val.reshape(-1, order="C")).reshape(N_LINK, T, order="C")
    wan95 = np.percentile(phys, 100 * (1 - P95), axis=1)        # 每链路 95 分位
    wan_metric = float(np.sum(wan95[INTER_LINK_IDX]))           # 洲际 95 流总和（进 κ 项）
    bill_money = float(np.sum(price_vector()[INTER_LINK_IDX] *
                              wan95[INTER_LINK_IDX]))           # 真实跨洲账单
    obj_true = imb_true + kappa * wan_metric
    return dict(x=x_val, L=L, phys=phys, imb=imb_true, wan95=wan95,
                wan_metric=wan_metric, bill=bill_money, obj=obj_true)


if __name__ == "__main__":
    from workload import gen_demand, split_classes, cap_vector
    import topology as tp
    d = gen_demand()
    d_nb, d_b = split_classes(d)
    cap = cap_vector()
    fb = tp.price_vector()
    # 全 direct 路由下试跑
    r = overlay_step(d_nb, d_b, cap, kappa=2.0, fb=fb, s=None, joint=True)
    print("imb=%.4f  bill95=%.4f  obj=%.4f" % (r["imb"], r["bill"], r["obj"]))
    print("phys row sums:", r["phys"].sum(axis=1).round(3))
