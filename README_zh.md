# CFIF-GF

中文 | [English](README.md)

这是一个基于 PyTorch 的语音情感识别项目，用于复现和扩展论文中的
`WavLM_Att` 与 `CFIF-GF` 方法，支持 IEMOCAP 和 EMODB 数据集。

当前代码包含数据预处理、manifest 读取、数据集自动扫描、10 折
Leave-One-Speaker-Out 交叉验证、训练日志、评价指标，以及 `WavLM_Att` 和
`CFIF-GF` 两个模型结构实现。

## 项目结构

```text
CFIF-GF/
  configs/          YAML 实验配置
  data/             数据集扫描、预处理、fold 划分、dataloader
  models/           模型注册与模型结构
  scripts/          辅助脚本
  utils/            配置、日志、随机种子、指标工具
  train.py          训练与 10 折交叉验证入口
  evaluate.py       评估入口
  requirements.txt  Python 依赖
```

## 环境安装

```bash
pip install -r requirements.txt
```

推荐服务器环境：

- Python 3.9+
- 与服务器 CUDA 驱动匹配的 PyTorch
- 默认支持单卡 GPU 训练
- 首次运行时，HuggingFace `transformers` 会下载 `microsoft/wavlm-base`

## 音频预处理

所有音频在 `data/preprocessing.py` 中统一处理：

- 重采样到 16 kHz
- 每条语音统一为 3 秒
- 短于 3 秒的语音做 zero padding
- 长于 3 秒的语音做截断
- 提取 40 维 MFCC
- MFCC window size 为 40 ms
- MFCC hop length 为 10 ms
- 提取 magnitude spectrogram
- spectrogram FFT 长度为 800
- spectrogram 取前 200 个 FFT bins
- 原始定长 waveform 保留给 WavLM 输入

`SpeechEmotionDataset` 返回：

- `waveform`
- `mfcc`
- `spectrogram`
- `label`
- `speaker_id`
- `file_path`
- `wavlm_features`，默认空；如果 manifest 提供 `wavlm_path`，则读取离线 WavLM 特征

## 数据集

支持的数据集名称：

- `IEMOCAP`
- `EMODB`

数据加载支持两种方式。

### Manifest 方式

将 `dataset.mock` 设为 `false`，并为 LOSO 训练提供 `dataset.all_manifest`。
普通训练/评估也可以使用 `train_manifest` / `test_manifest`。

CSV 格式：

```csv
path,label,speaker_id,wavlm_path
/path/to/audio.wav,angry,Session1_F,
/path/to/audio2.wav,0,Session1_M,/path/to/offline_wavlm.pt
```

`label` 可以是 `dataset.label_names` 中的类别名称，也可以是整数类别编号。
`wavlm_path` 是可选列；提供后，模型会使用离线 WavLM 序列特征，避免训练时运行
WavLM。

### 自动扫描方式

如果 `dataset.mock: false` 且没有配置 `dataset.all_manifest`，代码会尝试扫描标准目录：

- IEMOCAP：读取 `Session*/dialog/EmoEvaluation/*.txt` 和
  `Session*/sentences/wav/**/*.wav`
- EMODB：扫描 `.wav` 文件，并从官方文件名解析 speaker 与 emotion

IEMOCAP 中的 `exc` 会合并到 `happy`，对应常见 4 类设置：
`angry`、`happy`、`neutral`、`sad`。

## 10 折 LOSO 交叉验证

LOSO fold 按 `speaker_id` 划分：

- 每一折留一个 speaker 作为评估/测试集
- 其他 speaker 作为训练集
- 默认期望 10 个 speaker，即 10 折
- 每折保存 `best.pt`、`last.pt`、`metrics.json`、`train_log.csv`
- 不传 `--fold` 时自动训练全部 10 折
- 传 `--fold 0` 时只训练第 0 折
- 所有折结束后，平均 WA、UA、F1 写入 `cross_validation_summary.json`

训练单折：

```bash
python train.py --config configs/iemocap_cfif_gf.yaml --fold 0
```

训练全部折：

```bash
python train.py --config configs/iemocap_cfif_gf.yaml
python train.py --config configs/emodb_cfif_gf.yaml
```

恢复单折训练：

```bash
python train.py --config configs/iemocap_cfif_gf.yaml --fold 0 --resume outputs/IEMOCAP/CFIF-GF/fold_00_<speaker_id>/last.pt
```

输出目录：

```text
outputs/<DATASET>/<MODEL>/
  fold_00_<speaker_id>/
    best.pt
    last.pt
    metrics.json
    train_log.csv
  cross_validation_summary.json
```

## 快速空跑

默认配置使用 `dataset.mock: true`，没有真实音频也能做代码流程检查：

```bash
python train.py --config configs/default.yaml --fold 0
```

## 评价指标

指标在 `utils/metrics.py` 中实现：

- `wa`：Weighted Accuracy，即整体准确率
- `ua`：Unweighted Accuracy，即各类别 recall 的平均值
- `macro_f1`：各类别 F1 的平均值
- `confusion_matrix`：混淆矩阵，行为真实类别，列为预测类别
- `per_class`：每类 emotion 的 precision、recall、F1、support

## 模型

- `WavLM_Att`
- `CFIF-GF`

## WavLM_Att 模型

`models/wavlm_att.py` 实现论文第 3 章模型：

- 原始 `waveform` 输入 HuggingFace `microsoft/wavlm-base`
- WavLM 默认冻结：`model.freeze_wavlm: true`
- 如需 fine-tune WavLM，将 `model.freeze_wavlm` 改为 `false`
- `mfcc` 输入 Bi-LSTM
- MFCC Bi-LSTM hidden size 为 256
- MFCC Bi-LSTM 层数由 `model.mfcc_num_layers` 配置
- `spectrogram` 输入 AlexNet 风格 CNN
- CNN 通道为 64、192、384、256、256
- 第一层卷积核为 11，后续卷积核为 3
- 共同注意力模块使用 MFCC 与 spectrogram 特征生成 WavLM 时间帧注意力权重
- 加权 WavLM 特征、MFCC 特征、spectrogram 特征拼接后分类

模型 forward 返回 logits，用于 `CrossEntropyLoss`。推理概率可调用
`model.predict_proba(...)`，内部会应用 Softmax。

## CFIF-GF 模型

`models/cfif_gf.py` 实现论文第 4 章模型：

- 原始 `waveform` 输入 HuggingFace `microsoft/wavlm-base` 得到 WavLM 序列特征 `X_w`
- WavLM 默认冻结：`model.freeze_wavlm: true`
- `mfcc` 输入 Bi-LSTM 得到 MFCC 序列特征 `X_m`
- `spectrogram` 输入 TFCNN 得到语谱图序列特征 `X_s`
- TFCNN 包含两个并行 Conv2d 分支：
  T-CNN 使用 `5 x 1` 卷积核，F-CNN 使用 `1 x 4` 卷积核
- 每个 TFCNN 分支包含 Conv2d、BatchNorm2d、ReLU、MaxPool2d
- 两个分支 concat 后接 `3 x 3` 卷积块，再通过 AdaptiveAvgPool 和 Linear 得到特征序列
- CFIF 包含两个跨特征交互分支：`MFCC -> WavLM` 与 `Spectrogram -> WavLM`
- 每个 CFIF 分支将 source 与 WavLM 映射到统一 hidden 维度，通过 broadcast add、
  tanh、Linear、softmax 得到 WavLM 序列注意力
- 加权后的 WavLM 特征与 source 序列拼接，再映射到统一维度
- GF 是 gMLP 风格全局融合模块，包含 LayerNorm、Linear、GELU、特征切分、
  LayerNorm + `1 x 1` Conv gating、逐元素乘法、Linear 与残差连接
- 全局融合后的池化特征输入全连接分类器

## 训练命令

训练 WavLM_Att：

```bash
python train.py --config configs/iemocap_wavlm_att.yaml
python train.py --config configs/emodb_wavlm_att.yaml
```

训练 CFIF-GF：

```bash
python train.py --config configs/iemocap_cfif_gf.yaml
python train.py --config configs/emodb_cfif_gf.yaml
```

默认超参数：

- IEMOCAP：learning rate `2e-5`，batch size `32`，epochs `100`，AdamW
- EMODB：learning rate `3e-5`，batch size `64`，epochs `100`，AdamW

Early stopping 在 YAML 的 `train.early_stopping` 中配置，默认监控 `wa`，
patience 为 `10`。

默认使用 CSV 日志。如需 TensorBoard，将配置改为：

```yaml
train:
  log_backend: tensorboard
```

## 评估

评估 best checkpoint：

```bash
python evaluate.py --config configs/iemocap_cfif_gf.yaml --checkpoint outputs/IEMOCAP/CFIF-GF/fold_00_<speaker_id>/best.pt
```

评估脚本会输出 WA、UA、Macro F1、混淆矩阵、每类情感指标，并保存
`<checkpoint>.eval.json`。

## 离线 WavLM 特征

如果 GPU 显存不足，可以提前抽取 WavLM 序列特征：

```bash
python scripts/extract_wavlm_features.py --config configs/iemocap_cfif_gf.yaml --output-dir features/iemocap_wavlm --output-manifest manifests/iemocap_wavlm.csv
```

然后在配置中设置：

```yaml
dataset:
  mock: false
  all_manifest: manifests/iemocap_wavlm.csv
model:
  use_offline_wavlm_features: true
  offline_wavlm_dim: 768
```

manifest 中的 `wavlm_path` 列会被自动读取。启用
`use_offline_wavlm_features: true` 后，训练模型不会加载 WavLM，可降低显存占用。

## 工程化假设

- LOSO 训练中，留出的 speaker 同时作为该折 early stopping 和最终报告的评估/测试集
- IEMOCAP 的 `exc` 合并到 `happy`
- 当论文细节没有完全明确时，代码优先保证 PyTorch 实现可运行，并将维度写入 YAML 便于调整
