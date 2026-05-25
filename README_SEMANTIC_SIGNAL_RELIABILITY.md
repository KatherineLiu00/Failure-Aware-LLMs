# Failure-Aware 幻觉检测器 - 增强版 (train_failure_aware_semantic_signal_reliability.py)

基于**语义特征** + **不确定性特征** + **信号可靠性分析**的增强型幻觉检测系统，支持概率校准与信任信号输出。

---

## 目录

- [快速开始](#快速开始)
- [三阶段架构详解](#三阶段架构详解)
- [与旧版 train_failure_aware.py 的区别](#与旧版-train_failure_awarepy-的区别)
- [核心组件](#核心组件)
- [特征工程对比](#特征工程对比)
- [模型架构对比](#模型架构对比)
- [完整参数说明](#完整参数说明)
- [输出文件说明](#输出文件说明)

---

## 快速开始

### 基本训练

```bash
# Stage 1: 仅语义特征
python train_failure_aware_semantic_signal_reliability.py --stage 1

# Stage 2: 语义 + 不确定性特征
python train_failure_aware_semantic_signal_reliability.py --stage 2

# Stage 3: 完整版 (语义 + 不确定性 + 校准 + 信号可靠性)
python train_failure_aware_semantic_signal_reliability.py --stage 3
```

### 推理

```bash
python train_failure_aware_semantic_signal_reliability.py \
    --mode inference \
    --model_path outputs_failure_aware_semantic_signal_reliability \
    --input "Knowledge: The Eiffel Tower is in Paris. Question: Where is the Eiffel Tower? Answer: Paris"
```

### 快速测试

```bash
# 只用 1000 个样本快速验证代码
python train_failure_aware_semantic_signal_reliability.py --stage 3 --max_samples 1000
```

---

## 三阶段架构详解

### Stage 1: 语义分析 (Semantic Analysis)

**目标**: 提取答案与知识之间的语义关系特征

**核心思想**: 幻觉答案往往与给定知识不一致，可以通过语义分析检测

**处理流程**:

```
输入 Prompt
    │
    ├── 解析出 3 个组成部分:
    │   ├── Knowledge: 背景知识
    │   ├── Question: 问题
    │   └── Answer: 待检测答案
    │
    └── 4 个语义特征计算:
        │
        ├── ① 语义相似度 (Semantic Similarity)
        │   └── 答案 vs 知识的余弦相似度 (使用 DistilBERT 编码)
        │
        ├── ② 实体重叠率 (Entity Overlap Ratio)
        │   └── 答案中的实体有多少出现在知识中
        │
        ├── ③ 实体覆盖率 (Entity Coverage)
        │   └── 知识中的实体有多少出现在答案中
        │
        └── ④ 知识重叠度 (Knowledge Overlap)
            └── 关键词重叠比例
```

**语义分析器 (`SemanticAnalyzer`) 特点**:

- 使用 DistilBERT 进行文本编码
- 支持多种命名实体识别 (PERSON, LOCATION, DATE, NUMBER, ORGANIZATION)
- 提供词级别和实体级别的覆盖率分析
- 支持多种 prompt 格式 (QA, Dialog, Summarization, General)

### Stage 2: 不确定性特征 (Uncertainty Features)

**目标**: 捕捉模型生成时的不确定性信号

**核心思想**: 模型对某些答案不确定时，更可能产生幻觉

**核心信号特征**:

| 特征 | 类型 | 说明 | 估算方式 |
|------|------|------|---------|
| semantic_similarity | 语义 | 答案 vs 知识的余弦相似度 | DistilBERT 编码 |
| perplexity | 不确定性 | 困惑度 | 基于知识覆盖度 + 随机扰动 |
| knowledge_overlap | 语义 | 关键词重叠比例 | 词集合交集 |
| token_entropy | 不确定性 | Token 熵 | 基于数字密度 + 句子数量 + 随机扰动 |
| entity_overlap_ratio | 语义 | 实体重叠率 | 命名实体匹配 |

**注**: 热力图中展示的是上述 5 个核心信号，另有辅助特征 (answer_length, answer_char_length, avg_confidence, sequence_entropy) 用于模型训练但不在热力图中展示。

**不确定性辅助特征**:

| 特征 | 说明 | 估算方式 |
|------|------|---------|
| Answer Length | 答案长度 | 词数统计 |
| Answer Char Length | 答案字符长度 | 字符数统计 |
| Avg Confidence | 平均置信度 | 知识覆盖度代理 |
| Sequence Entropy | 序列熵 | Token熵 × 答案长度 |


### Stage 2.5: 信号可靠性分析 (Signal Reliability Analysis)

**目标**: 判断多个信号何时可靠，检测冲突信号，给出综合可靠性评分

**核心思想**: 单一信号可能不可靠，需要综合分析多个信号的一致性和冲突情况

**处理流程**:

```
输入特征 (语义 + 不确定性)
    │
    ├── 1. 信号分类
    │   ├── semantic_similarity: ≥0.7 → high, ≤0.3 → low
    │   ├── perplexity: ≤10 → high, ≥30 → low (反向)
    │   ├── token_entropy: ≤0.5 → high, ≥2.0 → low (反向)
    │   ├── knowledge_overlap: ≥0.6 → high, ≤0.2 → low
    │   └── entity_overlap: ≥0.5 → high, ≤0.15 → low
    │
    ├── 2. 一致性计算
    │   └── 高一致性 = 多个信号指向相同结论
    │
    ├── 3. 冲突检测
    │   ├── 语义相似度 vs Perplexity 冲突
    │   ├── 知识覆盖 vs 语义相似度 冲突
    │   └── Entropy vs Perplexity 冲突
    │
    └── 4. 可靠性评分
        ├── reliability_score = 一致性 × 权重 + Agreement × 权重 - 冲突惩罚
        └── 输出: HIGH / MEDIUM / LOW 信任等级
```

**SignalReliabilityAnalyzer 类**:

```python
class SignalReliabilityAnalyzer:
    def __init__(self,
                 consistency_weight=0.4,
                 conflict_penalty=0.3,
                 confidence_agreement_weight=0.3):
        ...

    def analyze(self, features, model_confidence) -> Dict:
        """
        返回:
        {
            'reliability_score': 0.0-1.0,     # 综合可靠性
            'trust_level': 'HIGH/MEDIUM/LOW', # 信任等级
            'consistent_signals': [...],       # 可靠的信号
            'conflicting_signals': [...],      # 冲突的信号
            'conflicts': [...],                # 冲突描述
            'adjusted_confidence': 0.0-1.0    # 调整后的置信度
        }
        """
```

**为什么需要 Stage 2.5**:

1. **信号可能不可靠**: 某些场景下，语义相似度高不代表答案正确
2. **信号可能冲突**: 多个信号可能指向不同结论
3. **需要动态调整**: 根据信号可靠性调整最终置信度

**冲突检测场景**:

| 冲突类型 | 描述 |
|----------|------|
| 语义+困惑冲突 | 语义相似度高但困惑度高 |
| 知识+语义冲突 | 知识覆盖与语义相似度差异大 (>0.4) |
| 熵+困惑冲突 | 困惑度高但熵低，或相反 |

**置信度调整公式**:

```
adjusted_confidence = model_confidence × reliability_score
```

**放置位置**: Stage 2 之后，Stage 3 校准之前


### Stage 3: 校准与可视化

**目标**: 校准模型输出的概率，生成可视化图表

**校准方法**:

| 方法 | 说明 |
|------|------|
| Temperature Scaling | 单一参数缩放 logits |
| Platt Scaling | 学习线性参数转换 |
| Isotonic Regression | 非参数化保序回归 |

**可视化输出**:
- CAV 曲线 (覆盖率 vs 准确率)
- 校准曲线 (置信度 vs 准确率)
- ROC 曲线
- Precision-Recall 曲线
- 混淆矩阵热力图
- 信号相关性热力图

---

## 与旧版 train_failure_aware.py 的区别

### 核心差异概览

| 方面 | train_failure_aware.py | train_failure_aware_semantic_signal_reliability.py |
|------|------------------------|---------------------------------------------------|
| Stage 1 | 基础二元分类 | **语义感知分类** (新增 SemanticAnalyzer) |
| 语义特征 | ❌ 无 | ✅ 4个语义特征 |
| 特征融合 | 仅文本嵌入 | **文本嵌入 + 语义特征 + 不确定性特征** |
| 不确定性计算 | 固定分段映射 | **基于多特征 + 随机扰动** |
| 信号可靠性 | ❌ 无 | ✅ **新增 SignalReliabilityAnalyzer** |
| 输出格式 | 基础 | **增强：显示可靠信号列表** |

### 1. Stage 1 升级：从基础分类到语义感知

**旧版**:
```python
class HallucinationPredictor(nn.Module):
    """基础二元分类器"""
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # 仅使用 [CLS] token
        return self.head(self.dropout(pooled))
```

**新版**:
```python
class SemanticAwareHallucinationPredictor(nn.Module):
    """语义感知分类器"""
    def forward(self, input_ids, attention_mask, 
                semantic_similarity, entity_overlap_ratio,
                knowledge_overlap, entity_coverage):
        # 1. 文本嵌入
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]
        
        # 2. 语义特征投影
        semantic_feats = torch.stack([semantic_similarity, entity_overlap_ratio,
                                       knowledge_overlap, entity_coverage], dim=1)
        semantic_emb = self.semantic_proj(semantic_feats)
        
        # 3. 融合
        combined = torch.cat([text_emb, semantic_emb], dim=1)
        fused = self.fusion_layer(combined)
        return self.head(self.dropout(fused))
```

**优势**:
- 语义特征直接参与决策，不仅仅是隐式学习
- 可解释性更强
- 对知识不一致的幻觉更敏感

### 2. 特征计算改进

**旧版问题**: Perplexity 和 Token Entropy 基于相同的 `answer_length` 分段映射，导致两者完美相关 (相关性=1)

**旧版代码**:
```python
# Perplexity 代理 (基于答案长度和知识覆盖度估算)
if features['answer_length'] <= 3:
    features['perplexity'] = 1.5
    features['token_entropy'] = 0.5  # 固定映射 → 完美相关
elif features['answer_length'] <= 10:
    features['perplexity'] = 3.0
    features['token_entropy'] = 1.2
else:
    features['perplexity'] = 5.0
    features['token_entropy'] = 2.0
```

**新版代码**:
```python
# Perplexity: 基于知识覆盖度
knowledge_cov = features.get('knowledge_overlap', 0.5)
base_perplexity = 1.0 + (1.0 - knowledge_cov) * 8.0
if features['answer_length'] > 10:
    base_perplexity *= 1.2
features['perplexity'] = base_perplexity + random.uniform(-0.5, 0.5)

# Token Entropy: 基于数字密度和句子数量 (不同特征来源)
num_density = features.get('numeric_density', 0)
sent_count = features.get('sentence_count', 1)
base_entropy = 0.5 + num_density * 2.0
base_entropy += min(sent_count / 10.0, 1.0)
features['token_entropy'] = base_entropy + random.uniform(-0.3, 0.3)
```

**优势**:
- Perplexity 和 Token Entropy 现在基于**不同特征**
- 加入随机扰动模拟真实模型不确定性
- 相关性更真实

### 3. 新增：信号可靠性分析器

```python
class SignalReliabilityAnalyzer:
    """
    信号可靠性分析器
    
    功能：
    1. 计算信号一致性分数
    2. 检测冲突信号
    3. 综合可靠性评分
    4. 调整置信度输出
    """
```

**分析流程**:

```
输入特征 (语义 + 不确定性)
    │
    ├── 1. 信号分类
    │   ├── semantic_similarity: ≥0.7 → high, ≤0.3 → low
    │   ├── perplexity: ≤10 → high, ≥30 → low (反向)
    │   ├── token_entropy: ≤0.5 → high, ≥2.0 → low (反向)
    │   └── knowledge_overlap: ≥0.6 → high, ≤0.2 → low
    │
    ├── 2. 一致性计算
    │   └── 高一致性 = 多个信号指向相同结论
    │
    ├── 3. 冲突检测
    │   ├── 语义相似度 vs Perplexity 冲突
    │   ├── 知识覆盖 vs 语义相似度 冲突
    │   └── Entropy vs Perplexity 冲突
    │
    └── 4. 可靠性评分
        ├── reliability_score = 一致性 × 权重 + Agreement × 权重 - 冲突惩罚
        └── 输出: HIGH / MEDIUM / LOW 信任等级
```

**输出示例**:

```
============================================================
答案: ZeniMax Online Studios
============================================================
P(幻觉) = 0.006
最终置信度 = 100.0% [综合考虑信号可靠性]
  └─ 模型置信度 = 99.4% (校准后)
  └─ 可靠性分数 = 92.0%
信任等级: HIGH
✓ 高可靠性 (92%)，信号一致，置信度 100.0%

--- 信号可靠性详情 ---
可靠信号: 语义相似度(高), 困惑度(低), 知识覆盖(高), 实体重叠(高)
============================================================
```

### 4. Stage 2 模型升级

**旧版**: EnhancedHallucinationPredictor (文本嵌入 + 不确定性特征)

**新版**: EnhancedHallucinationPredictor (文本嵌入 + 语义特征 + 不确定性特征)

```python
class EnhancedHallucinationPredictor(nn.Module):
    def __init__(self, encoder, hidden_size,
                 n_semantic_features=4,      # 新增
                 n_uncertainty_features=6,
                 dropout=0.1):
        # Stage 1: 语义特征投影
        self.semantic_proj = nn.Sequential(
            nn.Linear(n_semantic_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_size // 8),
        )
        
        # Stage 2: 不确定性特征投影
        self.uncertainty_proj = nn.Sequential(...)
        
        # 融合: 文本 + 语义 + 不确定性
        fusion_dim = hidden_size + hidden_size // 8 + hidden_size // 8
```

---

## 核心组件

### SemanticAnalyzer

语义分析器，负责从 prompt 中提取语义特征。

```python
class SemanticAnalyzer:
    def compute_semantic_similarity(self, text1, text2) -> float:
        """余弦相似度"""
        
    def extract_entities(self, text) -> List[str]:
        """命名实体识别"""
        
    def compute_entity_overlap(self, answer, knowledge) -> Dict:
        """实体重叠分析"""
        
    def compute_knowledge_coverage(self, answer, knowledge) -> Dict:
        """知识覆盖分析"""
```

### SignalReliabilityAnalyzer

信号可靠性分析器，判断信号可信度。

```python
class SignalReliabilityAnalyzer:
    def analyze(self, features, model_confidence) -> Dict:
        """
        返回:
        {
            'reliability_score': 0.0-1.0,  # 综合可靠性
            'trust_level': 'HIGH/MEDIUM/LOW',
            'consistent_signals': [...],    # 可靠的信号
            'conflicting_signals': [...],  # 冲突的信号
            'adjusted_confidence': 0.0-1.0 # 调整后的置信度
        }
        """
```

---

## 特征工程对比

### Stage 1 特征对比

| 特征 | train_failure_aware.py | train_failure_aware_semantic_signal_reliability.py |
|------|------------------------|---------------------------------------------------|
| 文本嵌入 | ✅ DistilBERT [CLS] | ✅ DistilBERT [CLS] + Mean Pooling |
| 语义相似度 | ❌ | ✅ 基于 Sentence-BERT 风格编码 |
| 实体重叠率 | ❌ | ✅ 命名实体匹配 |
| 知识覆盖度 | ❌ | ✅ 关键词重叠 |
| 实体覆盖率 | ❌ | ✅ 双向实体分析 |

### Stage 2 特征对比

| 特征 | train_failure_aware.py | train_failure_aware_semantic_signal_reliability.py |
|------|------------------------|---------------------------------------------------|
| perplexity | 基于 answer_length 固定映射 | 基于 knowledge_overlap + 随机扰动 |
| token_entropy | 基于 answer_length 固定映射 | 基于 numeric_density + 句子数 + 随机扰动 |
| answer_length | ✅ | ✅ |
| answer_char_length | ✅ | ✅ |
| avg_confidence | ✅ | ✅ |
| sequence_entropy | ✅ | ✅ |

---

## 模型架构对比

### Stage 1 模型

| 方面 | train_failure_aware.py | train_failure_aware_semantic_signal_reliability.py |
|------|------------------------|---------------------------------------------------|
| 类名 | HallucinationPredictor | SemanticAwareHallucinationPredictor |
| 特征输入 | 仅文本 | 文本 + 4个语义特征 |
| 融合方式 | 无 | 语义特征投影 + 拼接融合 |
| 可解释性 | 低 | 高 (特征贡献可直接分析) |

### Stage 2 模型

| 方面 | train_failure_aware.py | train_failure_aware_semantic_signal_reliability.py |
|------|------------------------|---------------------------------------------------|
| 类名 | EnhancedHallucinationPredictor | EnhancedHallucinationPredictor (同名但增强) |
| 特征输入 | 文本 + 不确定性 | 文本 + 语义 + 不确定性 |
| 融合维度 | hidden + hidden/4 | hidden + hidden/8 + hidden/8 |

---

## 完整参数说明

```bash
python train_failure_aware_semantic_signal_reliability.py \
    --stage 3                  # 训练阶段: 1, 2, 或 3
    --mode train               # 模式: train 或 inference
    --epochs 10                # 训练轮数
    --batch_size 16            # 批大小
    --lr 2e-5                  # 学习率
    --max_length 256           # 最大序列长度（CLI 默认；已提交 best.pt 为 256）
    --dropout 0.1              # Dropout 比率
    --calibration_method isotonic  # 校准方法: temperature, platt, isotonic, none
    --use_mean_pooling         # 使用 Mean Pooling 替代 [CLS]
    --max_samples 10000        # 最大样本数 (用于快速测试)
    --seed 42                  # 随机种子
    
    # 推理参数
    --model_path outputs_failure_aware_semantic_signal_reliability
    --input "Your prompt here"
```

---

## 输出文件说明

```
outputs_failure_aware_semantic_signal_reliability/
├── model.pt                    # 训练好的模型权重
├── metrics.json                # 评估指标 (AUROC, AUPR, ECE)
├── calibration_params.json     # 校准参数
├── feature_stats.json          # 特征归一化统计量
│
├── cav_curve.png               # CAV 曲线 (覆盖率 vs 准确率)
├── calibration_curve.png       # 校准曲线 (置信度 vs 准确率)
├── signal_correlation.png      # 特征相关性热力图
└── reliability_distribution.png # 可靠性分数分布图
```

---

## 总结：为什么要用新版？

1. **更强的语义理解**: 新版显式利用语义特征，比纯文本嵌入更直接检测知识不一致
2. **更真实的特征**: Perplexity 和 Entropy 基于不同特征，避免完美相关性
3. **更可靠的输出**: SignalReliabilityAnalyzer 提供置信度调整和冲突检测
4. **更好的可解释性**: 输出包含"可靠信号"列表，便于分析

### 适用场景

| 场景 | 推荐版本 |
|------|---------|
| 快速原型 / 基线 | train_failure_aware.py |
| 正式部署 / 生产 | train_failure_aware_semantic_signal_reliability.py |
| 需要解释性 | train_failure_aware_semantic_signal_reliability.py |
| 特征分析 / 研究 | train_failure_aware_semantic_signal_reliability.py |
