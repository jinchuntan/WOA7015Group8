
# WOA7015 Med-VQA (VQA-RAD) – Group 8 Codebase - Baseline vs BLIP (+ optional LoRA)

This project trains a ResNet50 + BiLSTM baseline for Medical Visual Question Answering (Med-VQA) and compares it against BLIP-VQA (zero-shot) on the VQA-RAD dataset. It also includes optional BLIP fine-tuning using LoRA adapters.

## Installation
Make sure the environment has been setup. If not, use the command below

```bash
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Use `pip` to install the dependencies.

```bash
pip install -r requirements.txt
```

## Project Structure

```bash
WOA7015_MedVQA/
  README.md
  requirements.txt
  .gitignore
  outputs/
    .gitkeep
  src/
    __init__.py
    data_vqarad.py
    text_utils.py
    metrics.py
    baseline_model.py
    train_baseline.py
    eval_baseline.py
    eval_blip.py
    finetune_blip_lora.py
```

## Usage

### 1) Train baseline (top-K answer classification)

Trains the baseline model using a top-K answer vocabulary (default K = 100). Outputs and checkpoint are saved under outputs/baseline/.

```bash
python -m src.train_baseline --topk 100 --epochs 9
```

Quick debug run (smaller subset):

```bash
python -m src.train_baseline --topk 50 --epochs 1 --batch_size 16
```

### 2) Evaluate baseline

Evaluates the saved checkpoint and writes:

```bash
outputs/baseline_eval/metrics.json
outputs/baseline_eval/predictions.csv
```

```bash
python -m src.eval_baseline --ckpt outputs/baseline/baseline.pt
```

### 3) Evaluate baseline

Runs BLIP-VQA on the VQA-RAD test split and writes:

```bash
outputs/blip_eval/metrics.json
outputs/blip_eval/predictions.csv
```

```bash
python -m src.eval_blip --model_name Salesforce/blip-vqa-base
```

Evaluate on only 100 test samples (debug):

```bash
python -m src.eval_blip --model_name Salesforce/blip-vqa-base --limit_n 100
```

### 4) Fine-tune BLIP with LoRA (optional)

Fine-tunes BLIP-VQA using LoRA adapters and saves them under outputs/blip_lora/.

```bash
python -m src.finetune_blip_lora --model_name Salesforce/blip-vqa-base --epochs 3
```

Small debug fine-tune:

```bash
python -m src.finetune_blip_lora --model_name Salesforce/blip-vqa-base --epochs 1 --train_n 200 --eval_n 100
```

### 5) Evaluate BLIP with LoRA adapters

Loads BLIP base + LoRA adapters and evaluates on the test split.

```bash
python -m src.eval_blip --model_name Salesforce/blip-vqa-base --lora_path outputs/blip_lora
```

### Outputs

All outputs are saved under outputs/:

```bash
outputs/baseline/ baseline checkpoint and training log
outputs/baseline_eval/ baseline evaluation metrics and predictions
outputs/blip_eval/ BLIP evaluation metrics and predictions
outputs/blip_lora/ LoRA adapters and training log```
```

### 6) Generate plots

To generate plots

```bash
python -m src.make_plots --train_history outputs/baseline/train_history.json --baseline_eval_dir outputs/baseline_eval --blip_eval_dir outputs/blip_eval --out_dir outputs/plots
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

