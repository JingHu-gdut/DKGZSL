# DKGZSL: Dynamic Knowledge-Guided Generative Zero-Shot Learning

A novel generative zero-shot learning framework that leverages dynamic visual-semantic knowledge to unify feature synthesis and refinement, achieving state-of-the-art performance on benchmark datasets.

## 🔍 Overview
DKGZSL addresses the limitations of traditional multi-stage generative zero-shot learning (GZSL) methods by injecting dynamic visual-semantic knowledge directly into the feature synthesis process. It eliminates error propagation across stages and enhances the quality of generated visual features for unseen classes, converting zero-shot learning into a supervised-like task with high generalization capability.

## 🚀 Key Features
- **One-Stage Generation**: Integrates feature synthesis and refinement into a single stage via dynamic knowledge guidance, avoiding cascading errors.
- **Semantic-Oriented Visual Refinement (SOVR)**: Reduces background noise and attention bias with a Feature Denoising Encoder (FDE) and Attribute-Region Attention (ARA).
- **Dynamic Knowledge Transfer Network (KTN)**: Converts semantic information into hierarchical visual knowledge, guiding the generator in real-time.
- **Meta-Fusion Units (MFUs)**: Progressively transmits hierarchical knowledge to the generator, improving inter-class discriminability and intra-class compactness.

<img width="1254" height="715" alt="image" src="https://github.com/user-attachments/assets/7bb4037f-ad27-466b-86d5-edb9585fecc8" />


## 📊 Performance
Outperforms state-of-the-art methods on three benchmark datasets (AWA2, CUB, SUN) under both Conventional ZSL (CZSL) and Generalized ZSL (GZSL) settings, with ResNet-101 and ViT-B/16 backbones:
- **CUB Dataset**: 85.2% (ResNet-101) / 87.2% (ViT-B/16) CZSL accuracy
- **AWA2 Dataset**: 79.2% (ResNet-101) / 83.7% (ViT-B/16) CZSL accuracy
- **SUN Dataset**: 67.7% (ResNet-101) / 73.1% (ViT-B/16) CZSL accuracy

<img width="1228" height="914" alt="image" src="https://github.com/user-attachments/assets/283a0c91-14aa-42e0-a74d-0def664ba186" />


## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/JingHu-gdut/DKGZSL.git
cd DKGZSL

# Install dependencies
pip install -r requirements.txt
```

## 📈 Usage
```python
# Example training command
python train.py --dataset CUB --backbone vit-b/16 --batch_size 128 --epochs 100

# Example evaluation command
python evaluate.py --dataset AWA2 --checkpoint ./checkpoints/dkgzsl_best.pth
```

## 📚 Citation
If you use this work, please cite our paper:
```bibtex
@article{DKGZSL2025,
  title={DKGZSL: Leveraging Dynamic Visual-Semantic Knowledge for Generative Zero-Shot Learning},
  author={},
  journal={},
  year={2025},
  publisher={IEEE}
}
```
