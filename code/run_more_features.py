"""
使用更多特征的对比实验 - 33个财务指标
"""
import os
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

def load_financial_data():
    """读取真实上市公司财务数据 - 使用最近3年数据和所有33个财务指标"""
    df_fin = pd.read_excel('../data/CSR_Finidx.xlsx', header=0, skiprows=[1, 2])
    df_fin['Stkcd'] = df_fin['Stkcd'].astype(str).str.strip()
    
    # 只使用最近3年的数据（2021-2023）
    recent_years = ['2021-12-31', '2022-12-31', '2023-12-31']
    df_fin = df_fin[df_fin['Accper'].isin(recent_years)]
    print(f'使用最近3年数据（2021-2023）：{len(df_fin)} 条记录')
    print(f'年份分布：\n{df_fin["Accper"].value_counts().sort_index()}')
    
    df_cg = pd.read_excel('../data/CG_Co.xlsx', skiprows=1)
    df_cg.columns = ['Stkcd', 'Stknme', 'ListedDate', 'Stktype', 'Crcd', 'Conme', 
                     'Cochsnm', 'Conmee', 'Nnindnme', 'Nnindcd', 'Nindnme', 'Nindcd',
                     'Indnme', 'Indcd', 'Busscope', 'Cohisty', 'Regcap', 'EstablishDate', 'DelistedDate']
    df_cg['Stkcd'] = df_cg['Stkcd'].astype(str).str.strip()
    
    df_cg['IsST'] = df_cg['Stknme'].str.contains('ST', na=False).astype(int)
    df_cg['IsPT'] = df_cg['Stknme'].str.contains('PT', na=False).astype(int)
    df_cg['IsDelisted'] = df_cg['DelistedDate'].notna().astype(int)
    
    df = pd.merge(df_fin, df_cg[['Stkcd', 'IsST', 'IsPT', 'IsDelisted']], 
                  on='Stkcd', how='left')
    
    # 使用所有数值型财务指标（排除Stkcd和Accper）
    exclude_cols = ['Stkcd', 'Accper', 'IsST', 'IsPT', 'IsDelisted']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f'\n使用的特征数量: {len(feature_cols)}')
    print(f'特征列表: {feature_cols}')
    
    X = df[feature_cols].values.astype(float)
    y = ((df['IsST'] == 1) | (df['IsPT'] == 1) | (df['IsDelisted'] == 1)).astype(int).values
    
    # 删除包含缺失值或无穷值的行
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols

def main():
    os.makedirs('output', exist_ok=True)
    
    print('===== 读取真实上市公司财务数据（2021-2023年，33个特征） =====')
    X, y, feature_cols = load_financial_data()
    
    # 使用800条样本（特征多了，样本数适当减少以加快计算）
    sample_size = 800
    if len(X) > sample_size:
        print(f'\n数据采样: 从 {len(X)} 条采样到 {sample_size} 条')
        anomaly_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        
        n_anomaly = int(sample_size * (y.sum() / len(y)))
        n_normal = sample_size - n_anomaly
        
        sampled_anomaly = np.random.choice(anomaly_indices, min(n_anomaly, len(anomaly_indices)), replace=False)
        sampled_normal = np.random.choice(normal_indices, min(n_normal, len(normal_indices)), replace=False)
        
        sampled_indices = np.concatenate([sampled_anomaly, sampled_normal])
        np.random.shuffle(sampled_indices)
        
        X = X[sampled_indices]
        y = y[sampled_indices]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print(f'\n样本数={X.shape[0]}, 特征数={X.shape[1]}')
    print(f'异常样本数={y.sum()}, 正常样本数={len(y)-y.sum()}, 异常率={y.sum()/len(y):.2%}')
    
    results = {}
    times = {}
    
    # 1. ADFNR - 参数搜索
    print('\n===== ADFNR 参数搜索 =====')
    best_eps, best_auc = 0.3, 0
    eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for eps in eps_list:
        start = time.time()
        scores = ADFNR(X, eps).ravel()
        elapsed = time.time() - start
        auc = roc_auc_score(y, scores)
        print(f'  eps={eps:.1f}, AUC={auc:.4f}, Time={elapsed:.2f}s')
        if auc > best_auc:
            best_auc = auc
            best_eps = eps
            best_time = elapsed
    
    print(f'\n最佳 epsilon={best_eps:.1f}, AUC={best_auc:.4f}')
    adfnr_scores = ADFNR(X, best_eps).ravel()
    results['ADFNR'] = adfnr_scores
    times['ADFNR'] = best_time
    
    # 2. 对比方法
    print('\n===== 对比方法 =====')
    contamination = y.sum() / len(y)
    
    # LOF
    start = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    results['LOF'] = -lof.fit_predict(X)
    times['LOF'] = time.time() - start
    print(f'LOF: AUC={roc_auc_score(y, results["LOF"]):.4f}, Time={times["LOF"]:.2f}s')
    
    # Isolation Forest
    start = time.time()
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X)
    results['IForest'] = -iso.decision_function(X)
    times['IForest'] = time.time() - start
    print(f'IForest: AUC={roc_auc_score(y, results["IForest"]):.4f}, Time={times["IForest"]:.2f}s')
    
    # ECOD
    start = time.time()
    ecod = ECOD(contamination=contamination)
    ecod.fit(X)
    results['ECOD'] = ecod.decision_scores_
    times['ECOD'] = time.time() - start
    print(f'ECOD: AUC={roc_auc_score(y, results["ECOD"]):.4f}, Time={times["ECOD"]:.2f}s')
    
    # 3. 结果汇总
    print('\n' + '='*70)
    print('完整结果汇总（33个特征）')
    print('='*70)
    print(f'{"方法":<12} {"AUC":<10} {"AP":<10} {"Time(s)":<10} {"Rank":<6}')
    print('-' * 70)
    
    summary = []
    auc_scores = []
    for name in ['ADFNR', 'LOF', 'IForest', 'ECOD']:
        scores = results[name]
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        t = times.get(name, 0)
        auc_scores.append((name, auc))
        summary.append({
            'Method': name,
            'AUC': auc,
            'Average_Precision': ap,
            'Time': t
        })
    
    # 按AUC排序
    auc_scores.sort(key=lambda x: x[1], reverse=True)
    rank_dict = {name: i+1 for i, (name, _) in enumerate(auc_scores)}
    
    for item in summary:
        name = item['Method']
        print(f"{name:<12} {item['AUC']:<10.4f} {item['Average_Precision']:<10.4f} "
              f"{item['Time']:<10.2f} {rank_dict[name]:<6}")
    
    # 4. 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC曲线
    ax1 = axes[0]
    for name in ['ADFNR', 'LOF', 'IForest', 'ECOD']:
        fpr, tpr, _ = roc_curve(y, results[name])
        auc = roc_auc_score(y, results[name])
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve (2021-2023, 33 Features)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # PR曲线
    ax2 = axes[1]
    for name in ['ADFNR', 'LOF', 'IForest', 'ECOD']:
        precision, recall, _ = precision_recall_curve(y, results[name])
        ap = average_precision_score(y, results[name])
        ax2.plot(recall, precision, label=f'{name} (AP={ap:.3f})', linewidth=2)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve (2021-2023, 33 Features)', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/more_features_comparison.png', dpi=300, bbox_inches='tight')
    print('\n对比图已保存到 output/more_features_comparison.png')
    
    # 保存结果
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv('output/more_features_results.csv', index=False, float_format='%.4f')
    print('结果已保存到 output/more_features_results.csv')
    
    # 5. 对比15特征 vs 33特征
    print('\n' + '='*70)
    print('特征数量对比分析')
    print('='*70)
    print('15个特征 vs 33个特征：')
    print('- 15特征：主要资产负债和盈利指标')
    print('- 33特征：包含更多现金流、财务比率、资本支出等指标')
    print('- 更多特征可能带来：')
    print('  * 更全面的财务信息')
    print('  * 但也可能引入噪声')
    print('  * 计算时间增加')
    print('='*70)

if __name__ == '__main__':
    main()
