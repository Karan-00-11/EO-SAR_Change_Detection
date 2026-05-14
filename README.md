# EO-SAR Change-Detection

Brief description of the task and approach
- Task: binary change detection between pre-event and post-event imagery.
- Approach: CMCDNet-based model trained on patch pairs, with full-image evaluation by stitching patch predictions.

## Requirements
- Python 3.12
-requirements.txt

## Environment Setup
### Option A: venv
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate:
   - Windows PowerShell:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: conda
1. Create and activate:
   ```bash
   conda create -n cmcdnet python=3.12 -y
   conda activate cmcdnet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure
Place the dataset under the Data directory as follows:

```
Data/
  index/
    patch_metadata.npz (For Filtering Patches, only for Training)
  train/
    pre-event/
    post-event/
    target/
  val/
    pre-event/
    post-event/
    target/
  test/
    pre-event/
    post-event/
    target/
```

## Training
Run training from scratch (uses config.yaml):

```bash
python train.py
```

## Evaluation
Evaluate on validation or test split:

```bash
python eval.py --data_path Data --weights checkpoint/model_checkpoint.pth --split test/val  
```

## Model Weights
- Google Drive: [Model Checkpoint](https://drive.google.com/drive/folders/1bU5siLGXeInKfTmLuC06zJqjyc4iYoKs?usp=sharing)

## Results
Reported metrics:

| Split | Loss | Precision | Recall | F1 | IoU |
|------|------|-----------|--------|----|-----|
| val  | 0.2331 | 0.5014 | 0.7140 | 0.5886 | 0.4175 |
| test | 0.3067 | 0.0657 | 0.0294 | 0.0402 | 0.0207 |

## Citation / References
- CMCDNet architecture: [CODE](https://github.com/CAU-HE/CMCDNet/tree/main) | [PAPER](https://www.sciencedirect.com/science/article/pii/S1569843223000195)
