# RMT-PR

Official implementation of **RMT-PR: A Reverse-View Mamba-Transformer with Cross-View Adaptive Feature Fusion for LiDAR Place Recognition**.

## Overview

RMT-PR is a single-frame LiDAR place recognition (LPR) framework designed for long-term localization under severe viewpoint changes. Given a single LiDAR scan, RMT-PR constructs range and bird's-eye-view (BEV) projections together with their reverse-view counterparts, learns discriminative features with a Mamba-Transformer encoder, and adaptively fuses complementary descriptors using a NetVLAD-based aggregation strategy.

## News

- The paper has been accepted by *IEEE Transactions on Intelligent Transportation Systems*.

## Environment

The code has been tested with Python 3.7/3.8 and PyTorch. A typical installation is:

```bash
conda create -n rmtpr python=3.8 -y
conda activate rmtpr
pip install -r requirements.txt
```

Please install a PyTorch version compatible with your CUDA environment from the official PyTorch website.

## Data Preparation

The expected preprocessed NCLT data structure is:

```text
RMT-PR/
├── data/
│   └── NCLT/
│       ├── 2012-01-08/ri_bev/000000.npy
│       ├── 2012-06-15/ri_bev/000000.npy
│       ├── 
├── groundtruth/
└── train_set_index/
```

Each `.npy` file is expected to contain the projected range/BEV representations used by RMT-PR. Please update `config/config.yml` according to your local dataset paths.

## Training

```bash
python train/training_RMT_PR.py
```

The default configuration is provided in `config/config.yml`. Trained checkpoints are saved to `checkpoints/` by default.

## Evaluation

First extract descriptors and compute pairwise retrieval distances:

```bash
python test/test_RMT-PR_topn_prepare.py
```

Then compute Top-K recall:

```bash
python test/test_RMT-PR_topn.py
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{luo2026rmtpr,
  title={RMT-PR: A Reverse-View Mamba-Transformer with Cross-View Adaptive Feature Fusion for LiDAR Place Recognition},
  author={Luo, Kan and Yu, Hongshan and Yang, Shuang and Wang, Jingwen and Wang, Yaonan and Civera, Javier and Chen, Xieyuanli},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}
```


## Acknowledgements

We sincerely thank the authors of several excellent open-source LiDAR place recognition and sequence modeling projects, including **[OverlapNet](https://github.com/PRBonn/OverlapNet)**, **[OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer)**, **[CVTNet](https://github.com/BIT-MJY/CVTNet)**, **[OverlapMamba](https://github.com/SCNU-RISLAB/OverlapMamba)**, and Mamba-related PyTorch implementations. Their publicly available code and research provided valuable references for data preprocessing, evaluation protocols, model design, and implementation.
