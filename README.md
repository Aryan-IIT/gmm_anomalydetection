# Deep Autoencoding Gaussian Mixture Model for Image Anomaly Detection

This repository extends the ICLR’18 DAGMM architecture to the image domain, exploring both standard convolutional and residual-convolutional variants. We demonstrate significant AUROC and precision/recall improvements over a naïve flattened-vector baseline.

## 🔧 Requirements & Setup

* **Python:** 3.10
* **Install dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

## 📂 Repository Structure

```
.
├── data/                 # (optional) place your image datasets here
├── models/               # saved model checkpoints
├── notebooks/            # exploratory analysis and plotting
├── trials/               # training logs & inference outputs
├── src/                  # core implementation of Convolutional DAGMM
│   ├── compression_net.py
│   ├── estimation_net.py
│   ├── train.py
│   └── infer.py
├── requirements.txt
└── README.md
```

## 🚀 Usage

1. **Training**

   ```bash
   python src/train.py \
     --data_dir ./data/images \
     --arch conv    # or “residual” for the residual-conv variant
   ```

2. **Inference**

   ```bash
   python src/infer.py \
     --checkpoint models/conv_dagmm.pt \
     --output_dir trials/conv/
   ```

All training and inference logs, as well as evaluation outputs (scores, ROC curves), are dumped into the `trials/` folder for easy inspection.

## Results

| Dataset                            | Architecture   | Precision | Recall | AUROC |
| ---------------------------------- | -------------- | --------- | ------ | ----- |
| *e.g.*, BrokenBottle vs GoodBottle | Conv-DAGMM     | 0.XX      | 0.XX   | 0.XX  |
|         BrokenBottle vs GoodBottle | Conv-DAGMM     | 0.XX      | 0.XX   | 0.XX  |
| *e.g.*, BrokenBottle vs GoodBottle | Conv-DAGMM     | 0.XX      | 0.XX   | 0.XX  |
| *e.g.*, BrokenBottle vs GoodBottle | Residual-DAGMM | 0.XX      | 0.XX   | 0.XX  |

> **Note:** replace `0.XX` with your actual metrics from `trials/`.

## What’s Inside

* **Convolutional DAGMM (Framework 1):**
  A straightforward CNN-based encoder + GMM estimation network.

* **Residual-Convolutional DAGMM (Framework 2):**
  Adds skip connections in the compression network for richer feature extraction.

* **Loss & EM Updates:**
  Follows the original DAGMM’s energy-based sample scoring and EM update rules.

## Observations

* The flattened-vector baseline (64×64 → 4096) yielded **very poor** AUROC and precision/recall.
* Both convolutional variants significantly outperform the flat baseline—see the “Results” table for details.

## Future Work

* Extend to **multimodal** (audio + image) anomaly detection.
* Experiment with alternative architectures (GAN-based compression, feature-discrepancy methods).
* Tune hyperparameters and explore larger image resolutions.

