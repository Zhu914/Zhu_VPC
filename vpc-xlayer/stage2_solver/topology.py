"""
topology.py —— 3 洲 × 2 DC 拓扑、RTT 三量级、WAN 链路与 95 分位单价、合规 flag
---------------------------------------------------------------------------
把「跨洲量级差异」写死进数据。RTT 三量级（DC内 <1ms / 区域内 5-50ms / 洲际 100-300ms）
与 95 分位单价的洲际/区域内比值，就是消融实验要扫描的 κ 的物理来源。

■ 红线 —— RTT 三量级与 95 单价的洲际/区域内比值不能被压平；压平则「数量级差异」论点消失。
■ 红线 —— 合规在结构上落地：can_leave_continent 决定业务能否跨洲，违规在优化里不可能发生。
"""
import numpy as np

# ============================================================
# 区域与 DC
# ============================================================
REGIONS = ["AP", "EU", "NA"]
DCS = ["AP1", "AP2", "EU1", "EU2", "NA1", "NA2"]          # 每洲 2 个
DC_REGION = {d: r for d, r in zip(DCS, [r for r in REGIONS for _ in range(2)])}  # 0,1->AP;2,3->EU;4,5->NA
DC_INDEX = {d: i for i, d in enumerate(DCS)}
N_DC = len(DCS)

# ============================================================
# RTT 三量级（毫秒）
# ============================================================
RTT_INTRA_DC = 0.5      # 同 DC 内 <1ms
RTT_INTRA_REGION = 20   # 区域内 5-50ms
RTT_INTER = 180         # 洲际 100-300ms


def rtt(i, j):
    """DC i 到 DC j 的往返时延（毫秒）。i,j 可为下标或名字。"""
    i = DC_INDEX[i] if isinstance(i, str) else i
    j = DC_INDEX[j] if isinstance(j, str) else j
    if i == j:
        return RTT_INTRA_DC
    if DC_REGION[DCS[i]] == DC_REGION[DCS[j]]:
        return RTT_INTRA_REGION
    return RTT_INTER


def rtt_matrix():
    """N_DC×N_DC 时延矩阵。"""
    M = np.zeros((N_DC, N_DC))
    for i in range(N_DC):
        for j in range(N_DC):
            M[i, j] = rtt(i, j)
    return M


# ============================================================
# WAN 链路与 95 分位计费单价
# ----------------------------------------------------------
# 链路按「无序区域对」定义。洲际三条 + 区域内三条。
# 洲际单价远高于区域内 —— 这就是 κ 的成本层级来源。
# ============================================================
INTER_LINKS = [("AP", "EU"), ("EU", "NA"), ("AP", "NA")]
INTRA_LINKS = [("AP", "AP"), ("EU", "EU"), ("NA", "NA")]
ALL_LINKS = INTER_LINKS + INTRA_LINKS
LINK_INDEX = {l: i for i, l in enumerate(ALL_LINKS)}
N_LINK = len(ALL_LINKS)

# 95 分位单价（洲际高、区域内低）
PRICE_95 = {("AP", "EU"): 8.0, ("EU", "NA"): 7.0, ("AP", "NA"): 10.0,
            ("AP", "AP"): 1.0, ("EU", "EU"): 1.0, ("NA", "NA"): 1.0}


def price_vector():
    """各链路 95 单价向量（与 ALL_LINKS 同序）。"""
    return np.array([PRICE_95[l] for l in ALL_LINKS], dtype=float)


def link_of_regions(a, b):
    """两个区域名 → 对应链路（无序对）。同区域返回区域内链路。"""
    if a == b:
        return (a, a)
    return tuple(sorted([a, b], key=lambda x: REGIONS.index(x)))


# 洲际链路容量（归一，体现「稀缺带宽」）；区域内宽松
LINK_CAP = {("AP", "EU"): 2.0, ("EU", "NA"): 2.0, ("AP", "NA"): 2.0,
            ("AP", "AP"): 8.0, ("EU", "EU"): 8.0, ("NA", "NA"): 8.0}


def link_cap_vector():
    return np.array([LINK_CAP[l] for l in ALL_LINKS], dtype=float)


# ============================================================
# 合规 flag（裁剪版二元）
# ----------------------------------------------------------
# can_leave_continent: 业务数据是否可出本洲。
#   数据无关类（可借）= True；数据重耦合类（不可借）= False。
# 违规在结构上不可能：不可借类根本不产生跨洲 x（见 overlay），故合规违规数恒 0。
# ============================================================
def can_leave_continent(tenant):
    return tenant["data_free"]


# 意图分级：时延敏感流走低 RTT 直连路径的强制比例（underlay 用）
INTENT_DIRECT_FRAC = 0.5   # 一半跨洲流为时延敏感，必须走 direct


if __name__ == "__main__":
    print("DCS:", DCS)
    print("links:", ALL_LINKS)
    print("price_95:", price_vector())
    print("rtt matrix:\n", rtt_matrix())
