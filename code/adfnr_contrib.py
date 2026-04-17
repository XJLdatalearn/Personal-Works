# adfnr_contrib.py
import numpy as np
import copy

def similarity(a, x, flag):
    if flag == 0:
        return 1.0 if a == x else 0.0
    else:
        return 1.0 - abs(a - x)

def normalize(x):
    x_max, x_min = np.max(x), np.min(x)
    return (x - x_min) / (x_max - x_min) if x_max != x_min else x

def ADFNR_contrib(data, epsilon):
    """
    返回 (AS, contrib)
      AS      : n×1 异常分数
      contrib : n×m 每个样本每个属性的异常贡献
    """
    data = np.asarray(data, dtype=float)
    n, m = data.shape

    weight1 = np.zeros((n, m))
    weight2 = np.zeros((n, m))
    delta = np.zeros((1, m))
    ID = (data <= 1).all(axis=0) & (data.max(axis=0) != data.min(axis=0))
    delta[0][ID] = 1

    # 预计算所有属性的相似度矩阵
    Set_ori = []
    Set = []
    for col in range(m):
        r = np.eye(n)
        for j in range(n):
            a = data[j, col]
            for k in range(j + 1, n):
                r[j, k] = similarity(a, data[k, col], delta[0, col])
                r[k, j] = r[j, k]
        Set_ori.append(r.copy())
        r_thresh = r.copy()
        r_thresh[r_thresh < epsilon] = 0.0
        Set.append(r_thresh)

    # 计算 Ratio、weight1、weight2
    Ratio = [np.zeros((n, 1)) for _ in range(m)]
    for col in range(m):
        others = [k for k in range(m) if k != col]
        # 其他属性做最小相似度聚合
        tmp = Set_ori[others[0]].copy()
        for k in others[1:]:
            tmp = np.minimum(tmp, Set_ori[k])
        tmp[tmp < epsilon] = 0.0

        # 去重加速
        S_col = Set[col]
        uniqS, inv_idx = np.unique(S_col, axis=0, return_inverse=True)
        for u_idx in range(uniqS.shape[0]):
            row_idx = np.where(inv_idx == u_idx)[0]
            Low_A = np.sum(np.all(tmp <= uniqS[u_idx], axis=1))
            Ratio[col][row_idx, 0] = Low_A / n
            weight1[row_idx, col] = uniqS[u_idx].sum() / n
            weight2[row_idx, col] = 1 - (uniqS[u_idx].sum() / n) ** (1/3)

    # 计算 AS 与贡献矩阵 contrib
    GIA = np.zeros((n, m))
    contrib = np.zeros((n, m))
    AS = np.zeros((n, 1))
    for i in range(n):
        for k in range(m):
            gia = 1 - (Ratio[k][i, 0] / m) * weight1[i, k]
            GIA[i, k] = gia
            contrib[i, k] = gia * weight2[i, k]   # 核心：保存贡献
        AS[i, 0] = np.sum(contrib[i, :]) / m
    return AS, contrib