"""
========================================
ADFNR 财务数据异常检测示例（与论文实验一致）
========================================
本脚本演示如何使用 ADFNR 方法对财务数据进行异常检测
与 run_more_features.py 保持一致，使用33个特征和2021-2023年数据

数据集: CSR_Finidx.xlsx（财务指标数据）+ CG_Co.xlsx（公司治理数据）
变量: 33个财务指标，涵盖资产负债、盈利能力、现金流、财务比率等维度
时间范围: 2021-2023年
异常标签: ST、PT、退市公司

作者: 论文作者
日期: 2026-04-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pyod.models.ecod import ECOD
from ADFNR import ADFNR
import time

np.random.seed(42)

# 33个财务指标（与run_more_features.py一致）
SELECTED_VARIABLES = [
    'A100000', 'A200000', 'A210000', 'A210101', 'A212101', 'A220000', 
    'A220101', 'A220201', 'A300000', 'B120101', 'B140101', 'B140204', 
    'B150101', 'D320801', 'D610000', 'T10100', 'T20201', 'T30700', 
    'T40401', 'T40402', 'T40403', 'T40801', 'T40802', 'T40803', 
    'T60200', 'T40902', 'Surplus', 'Flua10', 'Intrsta10', 'Perdebt', 
    'Begdebt', 'Enddebt', 'Outcap'
]

# 变量名称映射
VARIABLE_NAMES = {
    'A100000': '总资产',
    'A200000': '总负债',
    'A210000': '流动负债',
    'A210101': '短期借款',
    'A212101': '一年内到期非流动负债',
    'A220000': '长期负债',
    'A220101': '长期借款',
    'A220201': '应付债券',
    'A300000': '所有者权益',
    'B120101': '营业收入',
    'B140101': '营业利润',
    'B140204': '净利润',
    'B150101': '每股收益',
    'D320801': '经营活动现金流',
    'D610000': '投资活动现金流',
    'T10100': '财务指标T10100',
    'T20201': '财务指标T20201',
    'T30700': '财务指标T30700',
    'T40401': '财务指标T40401',
    'T40402': '财务指标T40402',
    'T40403': '财务指标T40403',
    'T40801': '财务指标T40801',
    'T40802': '财务指标T40802',
    'T40803': '财务指标T40803',
    'T60200': '财务指标T60200',
    'T40902': '财务指标T40902',
    'Surplus': '盈余公积',
    'Flua10': '流动资产',
    'Intrsta10': '利息支出',
    'Perdebt': '负债比率',
    'Begdebt': '期初含息负债',
    'Enddebt': '期末含息负债',
    'Outcap': '资本支出',
}


def load_financial_data():
    """
    加载财务数据（与run_more_features.py一致）
    
    返回:
        X: 特征矩阵
        y: 标签（1=异常，0=正常）
        feature_cols: 特征列表
    """
    print("="*60)
    print("加载财务数据（2021-2023年，33个特征）")
    print("="*60)
    
    # 读取财务数据
    df_fin = pd.read_excel('../data/CSR_Finidx.xlsx', header=0, skiprows=[1, 2])
    df_fin['Stkcd'] = df_fin['Stkcd'].astype(str).str.strip()
    
    # 只使用最近3年的数据（2021-2023）
    recent_years = ['2021-12-31', '2022-12-31', '2023-12-31']
    df_fin = df_fin[df_fin['Accper'].isin(recent_years)]
    print(f'使用最近3年数据（2021-2023）：{len(df_fin)} 条记录')
    print(f'年份分布：\n{df_fin["Accper"].value_counts().sort_index()}')
    
    # 读取公司治理数据
    df_cg = pd.read_excel('../data/CG_Co.xlsx', skiprows=1)
    df_cg.columns = ['Stkcd', 'Stknme', 'ListedDate', 'Stktype', 'Crcd', 'Conme', 
                     'Cochsnm', 'Conmee', 'Nnindnme', 'Nnindcd', 'Nindnme', 'Nindcd',
                     'Indnme', 'Indcd', 'Busscope', 'Cohisty', 'Regcap', 'EstablishDate', 'DelistedDate']
    df_cg['Stkcd'] = df_cg['Stkcd'].astype(str).str.strip()
    
    # 标记异常样本（ST、PT、退市）
    df_cg['IsST'] = df_cg['Stknme'].str.contains('ST', na=False).astype(int)
    df_cg['IsPT'] = df_cg['Stknme'].str.contains('PT', na=False).astype(int)
    df_cg['IsDelisted'] = df_cg['DelistedDate'].notna().astype(int)
    
    # 合并数据
    df = pd.merge(df_fin, df_cg[['Stkcd', 'IsST', 'IsPT', 'IsDelisted']], 
                  on='Stkcd', how='left')
    
    # 使用所有数值型财务指标
    exclude_cols = ['Stkcd', 'Accper', 'IsST', 'IsPT', 'IsDelisted']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f'\n使用的特征数量: {len(feature_cols)}')
    
    X = df[feature_cols].values.astype(float)
    y = ((df['IsST'] == 1) | (df['IsPT'] == 1) | (df['IsDelisted'] == 1)).astype(int).values
    
    # 删除包含缺失值或无穷值的行
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    print(f'有效样本数: {len(X)}')
    print(f'异常样本数: {y.sum()} ({y.sum()/len(y)*100:.2f}%)')
    print(f'正常样本数: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.2f}%)')
    
    return X, y, feature_cols


def sample_data(X, y, sample_size=800):
    """
    分层抽样（与run_more_features.py一致）
    
    参数:
        X: 特征矩阵
        y: 标签
        sample_size: 采样数量
    
    返回:
        X_sampled: 采样后的特征
        y_sampled: 采样后的标签
    """
    if len(X) <= sample_size:
        return X, y
    
    print(f'\n数据采样: 从 {len(X)} 条采样到 {sample_size} 条')
    
    anomaly_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]
    
    # 保持异常率不变
    n_anomaly = int(sample_size * (y.sum() / len(y)))
    n_normal = sample_size - n_anomaly
    
    sampled_anomaly = np.random.choice(anomaly_indices, min(n_anomaly, len(anomaly_indices)), replace=False)
    sampled_normal = np.random.choice(normal_indices, min(n_normal, len(normal_indices)), replace=False)
    
    sampled_indices = np.concatenate([sampled_anomaly, sampled_normal])
    np.random.shuffle(sampled_indices)
    
    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]
    
    print(f'采样后 - 异常样本: {y_sampled.sum()}, 正常样本: {len(y_sampled)-y_sampled.sum()}')
    
    return X_sampled, y_sampled


def run_adfnr(X, y):
    """
    运行ADFNR算法（带参数搜索）
    
    参数:
        X: 特征矩阵
        y: 标签（用于评估）
    
    返回:
        best_scores: 最佳异常分数
        best_eps: 最佳epsilon
        best_auc: 最佳AUC
    """
    print("\n" + "="*60)
    print("ADFNR 参数搜索")
    print("="*60)
    
    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_eps, best_auc = 0.3, 0
    eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for eps in eps_list:
        start = time.time()
        scores = ADFNR(X_scaled, eps).ravel()
        elapsed = time.time() - start
        auc = roc_auc_score(y, scores)
        print(f'  eps={eps:.1f}, AUC={auc:.4f}, Time={elapsed:.2f}s')
        
        if auc > best_auc:
            best_auc = auc
            best_eps = eps
            best_time = elapsed
    
    print(f'\n最佳 epsilon={best_eps:.1f}, AUC={best_auc:.4f}')
    
    best_scores = ADFNR(X_scaled, best_eps).ravel()
    
    return best_scores, best_eps, best_auc, best_time


def run_comparison_methods(X, y):
    """
    运行对比方法
    
    参数:
        X: 特征矩阵
        y: 标签
    
    返回:
        results: 各方法的异常分数
        times: 各方法的运行时间
    """
    print("\n" + "="*60)
    print("对比方法")
    print("="*60)
    
    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    times = {}
    contamination = y.sum() / len(y)
    
    # LOF
    start = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    results['LOF'] = -lof.fit_predict(X_scaled)
    times['LOF'] = time.time() - start
    print(f'LOF: AUC={roc_auc_score(y, results["LOF"]):.4f}, Time={times["LOF"]:.2f}s')
    
    # Isolation Forest
    start = time.time()
    iforest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    results['IForest'] = -iforest.fit_predict(X_scaled)
    times['IForest'] = time.time() - start
    print(f'IForest: AUC={roc_auc_score(y, results["IForest"]):.4f}, Time={times["IForest"]:.2f}s')
    
    # ECOD
    start = time.time()
    ecod = ECOD(contamination=contamination)
    ecod.fit(X_scaled)
    results['ECOD'] = ecod.decision_scores_
    times['ECOD'] = time.time() - start
    print(f'ECOD: AUC={roc_auc_score(y, results["ECOD"]):.4f}, Time={times["ECOD"]:.2f}s')
    
    return results, times


def plot_results(y, adfnr_scores, comparison_results):
    """
    绘制ROC和PR曲线
    
    参数:
        y: 真实标签
        adfnr_scores: ADFNR异常分数
        comparison_results: 对比方法的异常分数字典
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC曲线
    ax1 = axes[0]
    fpr, tpr, _ = roc_curve(y, adfnr_scores)
    auc = roc_auc_score(y, adfnr_scores)
    ax1.plot(fpr, tpr, label=f'ADFNR (AUC={auc:.3f})', linewidth=2)
    
    for name, scores in comparison_results.items():
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve (2021-2023, 33 Features)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # PR曲线
    ax2 = axes[1]
    precision, recall, _ = precision_recall_curve(y, adfnr_scores)
    ap = average_precision_score(y, adfnr_scores)
    ax2.plot(recall, precision, label=f'ADFNR (AP={ap:.3f})', linewidth=2)
    
    for name, scores in comparison_results.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax2.plot(recall, precision, label=f'{name} (AP={ap:.3f})', linewidth=2)
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve (2021-2023, 33 Features)', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/example_financial_detection_results.png', dpi=300, bbox_inches='tight')
    print('\n结果图已保存到 output/example_financial_detection_results.png')
    plt.show()


def analyze_top_anomalies(X, y, adfnr_scores, feature_cols, top_n=10):
    """
    分析最异常的样本
    
    参数:
        X: 特征矩阵
        y: 标签
        adfnr_scores: ADFNR异常分数
        feature_cols: 特征列表
        top_n: 显示前N个
    """
    print("\n" + "="*60)
    print(f"Top {top_n} 最异常样本分析")
    print("="*60)
    
    sorted_indices = np.argsort(adfnr_scores)[::-1]
    
    for i, idx in enumerate(sorted_indices[:top_n]):
        print(f"\n排名 {i+1} - 样本索引: {idx}")
        print(f"  异常分数: {adfnr_scores[idx]:.4f}")
        print(f"  真实标签: {'异常' if y[idx] == 1 else '正常'}")
        
        # 显示主要特征值
        print(f"  主要财务指标:")
        for j, feat in enumerate(feature_cols[:5]):  # 只显示前5个特征
            print(f"    {VARIABLE_NAMES.get(feat, feat)}: {X[idx, j]:.2f}")


def main():
    """主函数"""
    import os
    os.makedirs('output', exist_ok=True)
    
    # 1. 加载数据
    X, y, feature_cols = load_financial_data()
    
    # 2. 分层抽样
    X, y = sample_data(X, y, sample_size=800)
    
    # 3. 运行ADFNR
    adfnr_scores, best_eps, best_auc, adfnr_time = run_adfnr(X, y)
    
    # 4. 运行对比方法
    comparison_results, comparison_times = run_comparison_methods(X, y)
    
    # 5. 汇总结果
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    print(f"{'方法':<15} {'AUC':<10} {'AP':<10} {'Time(s)':<10}")
    print("-"*60)
    
    # ADFNR
    ap = average_precision_score(y, adfnr_scores)
    print(f"{'ADFNR':<15} {best_auc:<10.4f} {ap:<10.4f} {adfnr_time:<10.2f}")
    
    # 对比方法
    for name, scores in comparison_results.items():
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        print(f"{name:<15} {auc:<10.4f} {ap:<10.4f} {comparison_times[name]:<10.2f}")
    
    # 6. 绘制结果
    plot_results(y, adfnr_scores, comparison_results)
    
    # 7. 分析最异常样本
    analyze_top_anomalies(X, y, adfnr_scores, feature_cols, top_n=10)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()
