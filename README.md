# AI for Multimodal RGB-Thermal Pedestrian Detection through Efficient Fusion Strategies

**Team:** D8_TensorTitans
**Hackathon:** Yugma TechFest 2.0 – MedhaDrishti National-Level AI Hackathon
**Challenge:** AI for Object Recognition

---

## 1. Project Overview

This project improves the baseline **Quality-aware RGBT Fusion Detector (QFDet)** for pedestrian
detection on paired RGB-Thermal drone imagery, using the **VTUAV-det** dataset subset provided by
the organizers. The goal is to design an efficient fusion strategy that improves detection
performance — with particular emphasis on small and tiny pedestrians — while keeping the model
computationally lightweight.

Our approach modifies the **decision-level post-processing** stage of the baseline QFDet pipeline,
replacing standard Non-Maximum Suppression (NMS) with **Soft-NMS**, without retraining or altering
the pretrained backbone weights (as required by the hackathon rules).

---

## 2. Repository Structure

```
D8_TensorTitans/
├── source_code/
│   ├── notebook33e0d3b829.ipynb      # Main Kaggle notebook (all stages)
│   ├── make_qfdet_configs.py         # Generates baseline & improved configs
│   ├── verify_qfdet_env.py           # Environment sanity check script
│   ├── compute_stats.py              # Model size, params, inference time, FPS
│   ├── compute_flops.py              # FLOPs computation (dual RGB-Thermal input)
│   ├── find_failures.py              # Identifies worst-performing test images
│   ├── viz_failure.py                # Generates failure case visualization
│   └── results/                      # All generated CSVs, JSONs, and PNG charts
├── trained_model/
│   └── epoch_11_qfdet_vtuav.pth      # Provided pretrained QFDet checkpoint
├── requirements.txt
├── README.md
├── project_report.pdf
├── presentation.pptx
└── demo_video.mp4
```

---

## 3. Environment Setup

**Hardware used:** Kaggle Notebooks (GPU, CUDA 11.1)

**Software:**
- Python 3.8, PyTorch 1.10.0+cu111, Torchvision 0.11.1+cu111
- MMDetection / MMCV (OpenMMLab) — baseline QFDet implementation
- Full dependency list in `requirements.txt`

**Setup steps (reproducible on Kaggle):**

```bash
# 1. Create the conda environment
conda create -n qfdet python=3.8 -y

# 2. Install PyTorch (CUDA 11.1)
conda run -n qfdet pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install project dependencies
conda run -n qfdet pip install -r requirements.txt

# 4. Verify the environment
conda run -n qfdet python verify_qfdet_env.py
```

Verification confirms: PyTorch with CUDA available, MMCV, MMDetection, and successful imports of
`VTUAVdet` (dataset) and `QFDet` (detector).

---

## 4. Dataset

VTUAV-det subset (provided by organizers), only the **Pedestrian** class considered.

| Split | Images | Pedestrians | Avg/Image | Small | Medium | Large | Resolution |
|-------|--------|-------------|-----------|-------|--------|-------|------------|
| Train | 1200   | 8138        | 6.78      | 809   | 5449   | 1880  | 1920×1080  |
| Val   | 300    | 2337        | 7.79      | 423   | 1592   | 322   | 1920×1080  |
| Test  | 200    | 2068        | 10.34     | 529   | 1270   | 269   | 1920×1080  |

Scale definitions follow COCO conventions: Small (<32² px), Medium (32²–96² px), Large (≥96² px).

---

## 5. How to Run

All stages are executed from the main notebook `notebook33e0d3b829.ipynb`, using the
provided pretrained QFDet weights (`epoch_11_qfdet_vtuav.pth`). Training from scratch was not
performed, per the hackathon rules.

**Stage 1 – Dataset Exploration:** Run the dataset analysis cells to reproduce pedestrian
distribution statistics, scale distribution, and annotated RGB-Thermal pair visualizations.

**Stage 2 – Unimodal & Baseline Benchmarking:**
```python
run_eval("baseline_rgb_only_val", "stage2_rgb_only_val")
run_eval("baseline_thermal_only_val", "stage2_thermal_only_val")
run_eval("baseline_fusion_val", "stage2_baseline_fusion_val")
```
Repeat with the `_test` config keys for test-set evaluation.

**Stage 3 – Fusion Strategy (Improved Model):**
```python
run_eval("improved_fusion_test", "stage3_improved_fusion_test")
```
Config difference vs. baseline:
```python
# Baseline
nms = dict(type='nms', iou_threshold=0.5)
# Improved
nms = dict(type='soft_nms', iou_threshold=0.5, min_score=0.001)
```

**Stage 4 – Performance Evaluation:**
```python
python compute_stats.py    # model size, parameters, inference time, FPS
python compute_flops.py    # FLOPs (dual RGB-Thermal input via forward_dummy)
python find_failures.py    # identifies worst missed-detection test images
python viz_failure.py      # generates failure case visualization
```

---

## 6. Results Summary

### Detection Performance (Test Set, Fusion)

| Metric  | Baseline (NMS) | Improved (Soft-NMS) |
|---------|-----------------|-----------------------|
| mAP     | 0.299           | **0.306**             |
| mAP50   | 0.674           | **0.681**              |
| mAP75   | 0.227           | **0.238**              |

### Computational Performance (Improved Model)

| Metric              | Value        |
|----------------------|--------------|
| Model size           | 462.63 MB    |
| Parameters           | 60.63 M      |
| FLOPs                | 130.29 GFLOPs |
| Avg. inference time  | 88.29 ms/image |
| FPS                  | 11.33        |

### Qualitative & Failure Case Analysis

See `results/stage4_qualitative_improved_test.png` for green (ground truth) vs. red (prediction)
box comparisons, and `results/stage4_failure_case_01077.png` for a representative failure case —
a dense nighttime street scene (50 ground-truth pedestrians, 27 detected), showing the model's
main limitation: recovering small, distant pedestrians in crowded scenes.

---

## 7. Fusion Strategy: Description & Justification

We apply a **decision-level fusion improvement** to the baseline QFDet: replacing standard NMS
with **Soft-NMS** during post-processing. Rather than hard-suppressing overlapping detections,
Soft-NMS decays their confidence scores, which helps recover true positives in dense, overlapping
pedestrian clusters — a key challenge noted in the VTUAV-det dataset. This approach:

- Requires **no retraining** (keeps the provided pretrained weights, per hackathon rules)
- Is **computationally lightweight** (single post-processing change)
- Yields a **consistent, measurable improvement** across all COCO metrics on the test set

**Limitation:** Soft-NMS cannot recover pedestrians the model never proposed in the first place.
Remaining failure cases are dominated by very small/distant pedestrians, heavy occlusion, dense
crowds, and thermal background confusion — suggesting future work should target feature-level or
multi-scale fusion improvements rather than post-processing alone.

---

## 8. Team

**D8_TensorTitans**
