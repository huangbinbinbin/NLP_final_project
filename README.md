````md
# NLP Final Project（ZH→EN 翻译）：RNN / Transformer / mT5 复现说明

本仓库包含课程期末翻译项目的完整代码链路：**数据预处理 → 模型训练 → 推理评测 → 批量实验（ablation / sensitivity / low-resource）**。  
实现了三条主线模型并统一了数据格式与评测输出：  
1. **RNN Seq2Seq + Attention**（支持 dot / multiplicative / additive alignment；teacher_forcing / free_running）  
2. **Scratch Transformer NMT**（支持 pos_type / norm_type ablation + 超参敏感性）  
3. **mT5（mt5-small）微调**（训练与推理入口与 Transformer 共享）

---

## 0. 运行前准备

### 0.1 目录约定（相对路径）

本 README 的所有路径均以仓库根目录为基准（相对路径），推荐在仓库根目录执行命令：

```bash
cd NLP_final_project
```

### 0.2 准备 HuggingFace 模型文件（本地离线）

本项目默认以离线方式加载 HuggingFace 模型目录。请将以下模型目录放到仓库下（建议 gitignore，不提交到仓库）：

- `hf_models/opus-mt-zh-en/`（OPUS-MT tokenizer / vocab）
- `hf_models/mt5-small/`（mT5 模型目录，用于微调与推理）

目录示例：

```bash
ls -la hf_models
# opus-mt-zh-en/
# mt5-small/
```

---

## 1. 原始数据

原始数据位于压缩包：

- `AP0004_Midterm&Final_translation_dataset_zh_en/data.zip`

解压示例：

```bash
cd "AP0004_Midterm&Final_translation_dataset_zh_en"
unzip -o data.zip
cd ..
```

解压后通常包含（以实际文件为准）：

- `AP0004_Midterm&Final_translation_dataset_zh_en/train_100k.jsonl`
- `AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl`
- `AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl`

---

## 2. 数据预处理（preprocess_hf_opusmt.py）

预处理脚本：

- `preprocess_hf_opusmt.py`

### 2.1 预处理目标

将原始 zh-en JSONL 转换为 **HF tokenizer ids** 的统一格式，供 RNN/Transformer 的训练与推理直接读取。

### 2.2 预处理输出

默认输出到：

- `out_hf_opusmt_train/processed_train.jsonl`
- `out_hf_opusmt_train/processed_valid.jsonl`
- `out_hf_opusmt_train/processed_test.jsonl`
- `out_hf_opusmt_train/tokenizer_meta.json`
- `out_hf_opusmt_train/preprocess_config.json`

### 2.3 预处理主要流程（概览）

- 读取原始 JSONL（通常包含 `zh`、`en` 字段）
- 文本规范化与清洗（如空白规整、控制字符处理等）
- 使用 OPUS-MT 对应 tokenizer 将 src/tgt 编码为 ids（并对长度做过滤/截断策略）
- 为 src/tgt 末尾追加 EOS
- 写出 `processed_*.jsonl`，并保存 tokenizer 关键元信息 `tokenizer_meta.json`

### 2.4 预处理命令

```bash
python preprocess_hf_opusmt.py \
  --raw_train_jsonl "AP0004_Midterm&Final_translation_dataset_zh_en/train_100k.jsonl" \
  --raw_valid_jsonl "AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl" \
  --raw_test_jsonl  "AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl" \
  --hf_name_or_dir  "hf_models/opus-mt-zh-en" \
  --local_files_only \
  --output_dir      "out_hf_opusmt_train"
```

检查输出：

```bash
ls -la out_hf_opusmt_train
```

---

## 3. 训练脚本（RNN / Transformer / mT5）

### 3.1 脚本清单与功能

| 脚本 | 任务 | 输入 | 输出（典型） |
| --- | --- | --- | --- |
| `train_rnn_nmt_hf.py` | 训练 RNN NMT | `out_hf_opusmt_train/processed_*.jsonl` + `tokenizer_meta.json` | `best.pt` + `train_metrics.json` + `config.json` |
| `train_transformer_nmt_hfc_eos.py` | 训练 scratch Transformer / mT5 微调（共享训练入口） | scratch：processed ids；t5：raw jsonl | `best.pt` + `train_metrics.json` + `config.json` |

### 3.2 RNN 训练概览（train_rnn_nmt_hf.py）

- 读取 processed ids 数据与 tokenizer 元信息
- 构建 Encoder-Decoder RNN（LSTM/GRU）+ Attention（dot/multiplicative/additive）
- 支持训练策略：`teacher_forcing` / `free_running`
- 训练中验证集评估并保存最优 checkpoint `best.pt`

### 3.3 Transformer / mT5 训练概览（train_transformer_nmt_hfc_eos.py）

- scratch Transformer：从零训练 Transformer NMT
- 支持 ablation：`pos_type`（absolute/relative）、`norm_type`（如 layernorm 等）
- mT5：用于 mt5-small 微调（与推理脚本共用 t5 分支）
- 训练中验证集评估并保存最优 checkpoint `best.pt`

---

## 4. 推理脚本（RNN / Transformer / mT5）

### 4.1 脚本清单与功能

| 脚本 | 适用模型 | 支持解码 | 典型输出 |
| --- | --- | --- | --- |
| `inference_rnn_hf_aligned.py` | RNN | greedy / beam | `pred_*.txt` + `details_*.jsonl` + BLEU 打印 |
| `inference_transformer_hf_eos.py` | scratch Transformer / mT5 | greedy / beam | `details_*.jsonl` + 日志 + BLEU/统计 |

---

## 5. 批量实验脚本（run_experiments_*）

### 5.1 脚本清单

| 脚本 | 任务 | 汇总结果文件（典型） |
| --- | --- | --- |
| `run_experiments_rnn_hf.py` | RNN 实验网格（alignment / 超参 / scale / low-resource 等） | `results_rnn_greedy.json`、`results_rnn_beam.json`（以及 free_running 补做版本） |
| `run_experiments_transformer_hf.py` | Transformer 实验网格（scale×bs×lr、pos/norm ablation、low-resource） | `results_transformer_greedy.json`、`results_transformer_beam.json` |
| `run_experiments_t5_hf.py` | mT5 微调 + 推理对比 | `result_t5.json` |

### 5.2 已生成的结果文件（约定）

- RNN：
  - `results_rnn_greedy.json`
  - `results_rnn_beam.json`
  - `results_rnn_freerunning_greedy.json`
  - `results_rnn_freerunning_beam.json`
- Transformer：
  - `results_transformer_greedy.json`
  - `results_transformer_beam.json`
- mT5：
  - `result_t5.json`

---

## 6. 提交要求：best.pt 统一存放（model_best_pt/）

课程要求提供“一键推理”。请将用于推理的 checkpoint 统一放到：

- `model_best_pt/`

### 6.1 重要说明：model_best_pt 过大，使用网盘下载

由于 `model_best_pt/` 目录下的各个 `best.pt` 文件体积较大，不适合直接提交到 GitHub。本项目将该目录通过网盘分享，请按以下信息下载并放回仓库根目录下（保持目录名为 `model_best_pt/`）：

我用夸克网盘给你分享了「model_best_pt」，点击链接或复制整段内容，打开「夸克APP」即可获取。  
/~db5739lD0n~:/  
链接：https://pan.quark.cn/s/4b176eaa02e5?pwd=jBJt  
提取码：jBJt

下载后目录应形如：

```bash
ls -la model_best_pt
# rnn_teacher_forcing_beam_abl_rnn_type=lstm_align=additive_best.pt
# rnn_free_running_beam_abl_rnn_type=lstm_align=additive_best.pt
# transformer_beam_abl_scratch_pos=absolute_norm=layernorm_best.pt
# transformer_beam_abl_scratch_pos=relative_norm=layernorm_best.pt
# mt5small_best.pt
```

### 6.2 文件名约定（建议）

| 模型 | 文件名（放在 `model_best_pt/`） |
| --- | --- |
| RNN（teacher_forcing + beam + lstm + additive） | `rnn_teacher_forcing_beam_abl_rnn_type=lstm_align=additive_best.pt` |
| RNN（free_running + beam + lstm + additive） | `rnn_free_running_beam_abl_rnn_type=lstm_align=additive_best.pt` |
| Transformer（beam，absolute + layernorm） | `transformer_beam_abl_scratch_pos=absolute_norm=layernorm_best.pt` |
| Transformer（beam，relative + layernorm） | `transformer_beam_abl_scratch_pos=relative_norm=layernorm_best.pt` |
| mT5（mt5-small finetune） | `mt5small_best.pt` |

如你的文件名与上述不同，请在对应推理命令中替换 `--ckpt` 路径即可。

---

## 7. 一键推理命令（按模型分别给出）

### 7.0 通用变量与数据路径

```bash
BASE_OUT="test_inference"

PROC_DIR="out_hf_opusmt_train"
VALID_JSONL="${PROC_DIR}/processed_valid.jsonl"
TEST_JSONL="${PROC_DIR}/processed_test.jsonl"
TOKENIZER_META="${PROC_DIR}/tokenizer_meta.json"

HF_NAME_OR_DIR="hf_models/opus-mt-zh-en"
MT5_DIR="hf_models/mt5-small"

mkdir -p "${BASE_OUT}"
```

---

## 7.1 推理 RNN（beam）— additive（teacher_forcing）

### 7.1.1 RNN valid/dev 推理（beam）

```bash
python inference_rnn_hf_aligned.py \
  --ckpt "model_best_pt/rnn_teacher_forcing_beam_abl_rnn_type=lstm_align=additive_best.pt" \
  --data_jsonl "${VALID_JSONL}" \
  --direction zh2en \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 0 \
  --compute_bleu \
  --tgt_lang en \
  --out_path "${BASE_OUT}/rnn_tf_beam_additive/pred_valid.txt" \
  --save_details "${BASE_OUT}/rnn_tf_beam_additive/details_valid.jsonl"
```

### 7.1.2 RNN test 推理（beam）

```bash
python inference_rnn_hf_aligned.py \
  --ckpt "model_best_pt/rnn_teacher_forcing_beam_abl_rnn_type=lstm_align=additive_best.pt" \
  --data_jsonl "${TEST_JSONL}" \
  --direction zh2en \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 0 \
  --compute_bleu \
  --tgt_lang en \
  --out_path "${BASE_OUT}/rnn_tf_beam_additive/pred_test.txt" \
  --save_details "${BASE_OUT}/rnn_tf_beam_additive/details_test.jsonl"
```

---

## 7.2 推理 RNN（beam）— additive（free_running）

### 7.2.1 RNN valid/dev（free_running）推理（beam）

```bash
python inference_rnn_hf_aligned.py \
  --ckpt "model_best_pt/rnn_free_running_beam_abl_rnn_type=lstm_align=additive_best.pt" \
  --data_jsonl "${VALID_JSONL}" \
  --direction zh2en \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 0 \
  --compute_bleu \
  --tgt_lang en \
  --out_path "${BASE_OUT}/rnn_fr_beam_additive/pred_valid.txt" \
  --save_details "${BASE_OUT}/rnn_fr_beam_additive/details_valid.jsonl"
```

### 7.2.2 RNN test（free_running）推理（beam）

```bash
python inference_rnn_hf_aligned.py \
  --ckpt "model_best_pt/rnn_free_running_beam_abl_rnn_type=lstm_align=additive_best.pt" \
  --data_jsonl "${TEST_JSONL}" \
  --direction zh2en \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 0 \
  --compute_bleu \
  --tgt_lang en \
  --out_path "${BASE_OUT}/rnn_fr_beam_additive/pred_test.txt" \
  --save_details "${BASE_OUT}/rnn_fr_beam_additive/details_test.jsonl"
```

---

## 7.3 推理 Transformer（beam）— relative + layernorm

### 7.3.1 Transformer valid/dev 推理（beam）

```bash
python inference_transformer_hf_eos.py \
  --ckpt "model_best_pt/transformer_beam_abl_scratch_pos=relative_norm=layernorm_best.pt" \
  --data_jsonl "${VALID_JSONL}" \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 10 \
  --save_details "${BASE_OUT}/tf_beam_relative_layernorm/details_valid.jsonl" \
  | tee "${BASE_OUT}/tf_beam_relative_layernorm/run_valid.log"
```

### 7.3.2 Transformer test 推理（beam）

```bash
python inference_transformer_hf_eos.py \
  --ckpt "model_best_pt/transformer_beam_abl_scratch_pos=relative_norm=layernorm_best.pt" \
  --data_jsonl "${TEST_JSONL}" \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 10 \
  --save_details "${BASE_OUT}/tf_beam_relative_layernorm/details_test.jsonl" \
  | tee "${BASE_OUT}/tf_beam_relative_layernorm/run_test.log"
```

---

## 7.4 推理 Transformer（beam）— absolute + layernorm

### 7.4.1 Transformer valid/dev 推理（beam）

```bash
python inference_transformer_hf_eos.py \
  --ckpt "model_best_pt/transformer_beam_abl_scratch_pos=absolute_norm=layernorm_best.pt" \
  --data_jsonl "${VALID_JSONL}" \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 10 \
  --save_details "${BASE_OUT}/tf_beam_absolute_layernorm/details_valid.jsonl" \
  | tee "${BASE_OUT}/tf_beam_absolute_layernorm/run_valid.log"
```

### 7.4.2 Transformer test 推理（beam）

```bash
python inference_transformer_hf_eos.py \
  --ckpt "model_best_pt/transformer_beam_abl_scratch_pos=absolute_norm=layernorm_best.pt" \
  --data_jsonl "${TEST_JSONL}" \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --len_norm gnmt \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 10 \
  --save_details "${BASE_OUT}/tf_beam_absolute_layernorm/details_test.jsonl" \
  | tee "${BASE_OUT}/tf_beam_absolute_layernorm/run_test.log"
```

---

## 7.5 推理 mT5（mt5-small 微调）— Valid 集（greedy / beam）

### 7.5.1 Valid：greedy

```bash
python inference_transformer_hf_eos.py \
  --ckpt "model_best_pt/mt5small_best.pt" \
  --data_jsonl "AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl" \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode greedy \
  --infer_batch_size 32 \
  --max_len 80 \
  --min_len 10 \
  --t5_model_dir "${MT5_DIR}" \
  --t5_prompt "translate Chinese to English: " \
  --t5_length_penalty 1.0 \
  --save_details "${BASE_OUT}/mt5/details_valid_greedy.jsonl" \
  | tee "${BASE_OUT}/mt5/run_valid_greedy.log"
```

### 7.5.2 Valid：beam

```bash
python inference_transformer_hf_eos.py \
  --ckpt "model_best_pt/mt5small_best.pt" \
  --data_jsonl "AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl" \
  --tokenizer_meta "${TOKENIZER_META}" \
  --hf_name_or_dir "${HF_NAME_OR_DIR}" \
  --local_files_only \
  --device cuda \
  --decode beam \
  --beam_size 5 \
  --infer_batch_size 1 \
  --max_len 80 \
  --min_len 10 \
  --t5_model_dir "${MT5_DIR}" \
  --t5_prompt "translate Chinese to English: " \
  --t5_length_penalty 1.0 \
  --save_details "${BASE_OUT}/mt5/details_valid_beam.jsonl" \
  | tee "${BASE_OUT}/mt5/run_valid_beam.log"
```

---

## 8. 常见注意事项

### 8.1 关于 `--local_files_only`

若你没有将模型放到 `hf_models/` 并希望在线下载，请去掉 `--local_files_only`，并将 `--hf_name_or_dir` / `--t5_model_dir` 改为可访问的模型来源或本机路径。

### 8.2 关于 tokenizer 一致性

- RNN / Transformer 的训练与推理使用 `out_hf_opusmt_train/tokenizer_meta.json` 对齐 tokenizer 关键 id。
- 请确保 `hf_models/opus-mt-zh-en/` 与 `tokenizer_meta.json` 是一致的一套 tokenizer 设置。

### 8.3 输出文件较大

`details_*.jsonl` 可能较大（逐样本保存推理信息）。如只需要预测文本或 BLEU，可按需关闭 details 输出（以脚本参数为准）。
````
