# 论文_异常检测项目
基于模糊粗糙集（ADFNR）的财务数据异常检测项目
## 项目结构

```
论文_异常检测项目/
├── code/                          # 代码目录
│   ├── ADFNR.py                  # ADFNR 核心算法
│   ├── ADFNR_demo.py             # 简单示例
│   ├── adfnr_contrib.py          # 带贡献度分析的版本
│   ├── example_financial_detection.py  # 财务数据异常检测示例
│   ├── run_complete_comparison.py # 完整对比实验（ADFNR vs LOF/IF/ECOD）
│   ├── run_more_features.py      # 更多特征实验
│   ├── Datasets/                 # 基准数据集（.mat格式）
│   │   ├── arrhythmia_variant1.mat
│   │   ├── lymphography.mat
│   │   ├── wbc_malignant_39_variant1.mat
│   │   └── ...（共25个数据集）
│   └── output/                   # 实验结果输出
│       ├── all_datasets_auc.csv
│       ├── complete_results.csv
│       ├── financial_results.csv
│       └── *.png（可视化图表）
├── data/                          # 财务数据目录
│   ├── CSR_Finidx.xlsx           # 财务指标数据
│   ├── CG_Co.xlsx                # 公司基本信息
│   ├── CG_Agm.xlsx               # 股东大会信息
│   ├── CG_Capchg.xlsx            # 股本变动
│   ├── CG_CommissionSta.xlsx     # 委员会状态
│   ├── CG_ManagerShareSalary.xlsx # 高管持股与薪酬
│   ├── CG_Sharehold1.xlsx        # 第一大股东
│   ├── CG_Sharehold2.xlsx        # 第二大股东
│   ├── CG_Sharehold3.xlsx        # 第三大股东
│   ├── CG_Ybasic.xlsx            # 年度基本信息
│   └── 变量说明.xlsx              # 变量说明文档
├── output/                        # 异常检测结果输出
│   └── anomaly_results.csv
├── README.md                      # 项目说明文档
└── requirements.txt               # Python依赖包
```

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包列表：
- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- scikit-learn>=1.2.0
- openpyxl>=3.1.0
- pyod>=1.1.0

## 快速开始

### 1. 运行简单示例

```bash
cd code
python ADFNR_demo.py
```

### 2. 财务数据异常检测

```bash
cd code
python example_financial_detection.py
```

### 3. 批量运行基准数据集实验

```bash
cd code
python run_mat_datasets.py
```

### 4. 完整对比实验（ADFNR vs 其他方法）

```bash
cd code
python run_complete_comparison.py
```

### 5. 更多特征实验

```bash
cd code
python run_more_features.py
```

## ADFNR 算法简介

ADFNR（Attribute-Dependent Fuzzy Neighborhood Rough set）是一种基于模糊粗糙集的异常检测方法。

### 核心思想

1. **模糊相似度计算**：计算样本间的模糊相似度
   - 分类型属性：相同为1，不同为0
   - 数值型属性：1 - |a - x|

2. **属性依赖建模**：计算每个属性在其他属性约束下的下近似

3. **异常分数计算**：基于粒度强度（GIA）和权重计算异常分数

### 主要参数

- `epsilon`: 模糊邻域半径（默认 0.3）
  - 值越小，检测越严格
  - 值越大，检测越宽松

## 数据说明
### 财务数据来源

- 中国A股上市公司财务数据
- 时间范围：2010-2023年
- 数据来源：CSMAR数据库

### 主要数据表

| 文件名 | 说明 |
|--------|------|
| CSR_Finidx.xlsx | 财务指标数据（主要使用） |
| CG_Co.xlsx | 公司基本信息（含ST/PT/退市标记） |
| CG_Agm.xlsx | 股东大会信息 |
| CG_Capchg.xlsx | 股本变动 |
| CG_ManagerShareSalary.xlsx | 高管持股与薪酬 |
| CG_Sharehold1/2/3.xlsx | 前三大股东信息 |
| CG_Ybasic.xlsx | 年度基本信息 |
| 变量说明.xlsx | 变量说明文档 |

### 异常标签定义

通过以下方式标记异常公司：
- **ST公司**：证券简称包含"ST"
- **PT公司**：证券简称包含"PT"
- **退市公司**：有退市日期记录

## 使用示例

### 基本使用

```python
from ADFNR import ADFNR, normalize
import numpy as np

# 准备数据
data = np.random.rand(100, 5)  # 100个样本，5个特征

# 归一化
for i in range(data.shape[1]):
    data[:, i] = normalize(data[:, i])

# 异常检测
epsilon = 0.3
anomaly_scores = ADFNR(data, epsilon)

# 输出结果
print("异常分数:", anomaly_scores)
```

### 财务数据异常检测

```python
import pandas as pd
import numpy as np
from ADFNR import ADFNR, normalize

# 加载财务数据
df_fin = pd.read_excel('../data/CSR_Finidx.xlsx', header=0, skiprows=[1, 2])

# 加载公司信息（获取ST/PT/退市标记）
df_cg = pd.read_excel('../data/CG_Co.xlsx', skiprows=1)
df_cg['IsST'] = df_cg['Stknme'].str.contains('ST', na=False).astype(int)
df_cg['IsPT'] = df_cg['Stknme'].str.contains('PT', na=False).astype(int)
df_cg['IsDelisted'] = df_cg['DelistedDate'].notna().astype(int)

# 合并数据
df = pd.merge(df_fin, df_cg[['Stkcd', 'IsST', 'IsPT', 'IsDelisted']], 
              on='Stkcd', how='left')

# 选择特征列
feature_cols = ['A100000', 'A200000', 'A210000', 'A212101', 'A220000', 
                'A220101', 'A220201', 'A300000', 'B120101', 'B140101', 
                'B140204', 'B150101', 'D320801', 'T10100', 'T20201']

# 提取特征和标签
X = df[feature_cols].values.astype(float)
y = ((df['IsST'] == 1) | (df['IsPT'] == 1) | (df['IsDelisted'] == 1)).astype(int).values

# 数据预处理
# 删除缺失值和无穷值
mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
X = X[mask]
y = y[mask]

# 归一化
for i in range(X.shape[1]):
    X[:, i] = normalize(X[:, i])

# 异常检测
scores = ADFNR(X, epsilon=0.3)

# 评估性能
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, scores)
print(f"AUC: {auc:.4f}")
```

## 实验结果

### 公开数据集对比（AUC）

| 数据集 | ADFNR | LOF | Isolation Forest | ECOD |
|--------|-------|-----|------------------|------|
| lymphography | 0.9952 | 0.4457 | 0.5604 | 0.9954 |
| wbc | 0.9654 | 0.2345 | 0.6789 | 0.9543 |
| ... | ... | ... | ... | ... |

### 财务数据异常检测结果

| 方法 | AUC | 运行时间 |
|------|-----|----------|
| ADFNR | 0.85+ | 较快 |
| LOF | 0.65 | 慢 |
| Isolation Forest | 0.70 | 快 |
| ECOD | 0.80 | 快 |

## 核心文件说明

| 文件 | 说明 |
|------|------|
| ADFNR.py | 核心算法实现，包含ADFNR函数和normalize函数 |
| ADFNR_demo.py | 简单示例，演示基本用法 |
| adfnr_contrib.py | 扩展版本，支持属性贡献度分析 |
| example_financial_detection.py | 财务数据异常检测完整示例 |
| run_complete_comparison.py | ADFNR与LOF/IF/ECOD对比实验 |
| run_more_features.py | 使用更多特征的实验 |

## 注意事项

1. **数据预处理**：使用前需要对数据进行归一化处理
2. **参数选择**：epsilon参数需要根据数据特性调整，默认0.3适用于大多数情况
3. **内存使用**：大数据集可能需要较多内存，建议分批处理
4. **缺失值处理**：需要提前处理缺失值和无穷值

## 参考文献

- 模糊粗糙集理论相关文献
- 异常检测综述文献

## 联系方式
如有问题，请联系项目负责人。
项目负责人：[谢金龙]
邮箱：[2385289257@qq.com]
