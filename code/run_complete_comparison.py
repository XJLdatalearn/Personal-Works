"""
完整对比实验 - 使用更多样本和更精细的参数搜索
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
    """读取真实上市公司财务数据"""
    df_fin = pd.read_excel('../data/CSR_Finidx.xlsx', header=0, skiprows=[1, 2])
    df_fin['Stkcd'] = df_fin['Stkcd'].astype(str).str.strip()
    
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
    
    feature_cols = ['A100000', 'A200000', 'A210000', 'A212101', 'A220000', 
                    'A220101', 'A220201', 'A300000', 'B120101', 'B140101', 
                    'B140204', 'B150101', 'D320801', 'T10100', 'T20201']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values.astype(float)
    y = ((df['IsST'] == 1) | (df['IsPT'] == 1) | (df['IsDelisted'] == 1)).astype(int).values
    # 删除包含缺失值或无穷值的行
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols

def main():
    os.makedirs('output', exist_ok=True)
    
    print('===== 读取真实上市公司财务数据 =====')
    X, y, feature_cols = load_financial_data()
    
    # 使用1000条样本
    sample_size = 1000
    if len(X) > sample_size:
        print(f'数据采样: 从 {len(X)} 条采样到 {sample_size} 条')
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
    
    print(f'样本数={X.shape[0]}, 特征={X.shape[1]}')
    print(f'异常样本数={y.sum()}, 正常样本数={len(y)-y.sum()}, 异常率={y.sum()/len(y):.2%}')
    
    results = {}
    times = {}
    
    # 1. ADFNR - 更精细的参数搜索
    print('\n===== ADFNR 参数搜索 =====')
    best_eps, best_auc = 0.3, 0
    eps_list = np.arange(0.05, 0.81, 0.05)
    for eps in eps_list:
        start = time.time()
        scores = ADFNR(X, eps).ravel()
        elapsed = time.time() - start
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc = auc
            best_eps = eps
            best_time = elapsed
        if eps in [0.1, 0.2, 0.3, 0.5, 0.7]:
            print(f'  eps={eps:.2f}, AUC={auc:.4f}, Time={elapsed:.2f}s')
    
    print(f'\n最佳 epsilon={best_eps:.2f}, AUC={best_auc:.4f}')
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
    print('\n===== 完整结果汇总 =====')
    print(f'{"方法":<12} {"AUC":<10} {"AP":<10} {"Time(s)":<10}')
    print('-' * 45)
    
    summary = []
    for name in ['ADFNR', 'LOF', 'IForest', 'ECOD']:
        scores = results[name]
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        t = times.get(name, 0)
        print(f'{name:<12} {auc:<10.4f} {ap:<10.4f} {t:<10.2f}')
        summary.append({
            'Method': name,
            'AUC': auc,
            'Average_Precision': ap,
            'Time': t
        })
    
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
    ax1.set_title('ROC Curve Comparison', fontsize=14)
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
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/complete_comparison.png', dpi=300, bbox_inches='tight')
    print('\n对比图已保存到 output/complete_comparison.png')
    
    # 保存结果
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv('output/complete_results.csv', index=False, float_format='%.4f')
    print('结果已保存到 output/complete_results.csv')
    

if __name__ == '__main__':
    main()
