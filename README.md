 #  <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/2b4624f9-8d79-4ca7-99e6-2cda3ce326de" /> DKGZSL: Leveraging Dynamic Knowledge-Guided Generative Zero-Shot Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/) 
If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

A novel generative zero-shot learning framework that leverages dynamic visual-semantic knowledge to unify feature synthesis and refinement, achieving state-of-the-art performance on benchmark datasets.

## 🔍 Overview
DKGZSL addresses the limitations of traditional multi-stage generative zero-shot learning (GZSL) methods by injecting dynamic visual-semantic knowledge directly into the feature synthesis process. It eliminates error propagation across stages and enhances the quality of generated visual features for unseen classes, converting zero-shot learning into a supervised-like task with high generalization capability.

## 🚀 Key Features
- **One-Stage Generation**: Integrates feature synthesis and refinement into a single stage via dynamic knowledge guidance, avoiding cascading errors.
- **Semantic-Oriented Visual Refinement (SOVR)**: Reduces background noise and attention bias with a Feature Denoising Encoder (FDE) and Attribute-Region Attention (ARA).
- **Dynamic Knowledge Transfer Network (KTN)**: Converts semantic information into hierarchical visual knowledge, guiding the generator in real-time.
- **Meta-Fusion Units (MFUs)**: Progressively transmits hierarchical knowledge to the generator, improving inter-class discriminability and intra-class compactness.

## 📑 Framework
Architecture of our proposed DKGZSL for ZSL. DKGZSL consists of Semantic-Oriented Visual Refinement (SOVR) and a Dynamic KnowledgeGuided Generator Network (DKG<sup>2</sup>N). The Knowledge Transfer Network (KTN) in DKG2N converts semantic information into visual knowledge, aligning it with the reshaped visual features from the SOVR. Concurrently, the intermediate visual-semantic knowledge within the KTN is transmitted to the generator via the Meta-Fusion Unit (MFU).
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/7bb4037f-ad27-466b-86d5-edb9585fecc8" />


## 📊 Performance
Outperforms state-of-the-art methods on three benchmark datasets (AWA2, CUB, SUN) under both Conventional ZSL (CZSL) and Generalized ZSL (GZSL) settings, with ResNet-101 and ViT-B/16 backbones.

<table style="margin: 0 auto; text-align: center; width: 80%;">
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th style="padding: 8px; border: 1px solid #ddd;">Methods</th>
      <th style="padding: 8px; border: 1px solid #ddd;">AWA2 (Acc/H)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">CUB (Acc/H)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">SUN (Acc/H)</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #e6f7ff; font-weight: bold;">
      <td style="padding: 8px; border: 1px solid #ddd;">DKGZSL-res</td>
      <td style="padding: 8px; border: 1px solid #ddd;">79.2/72.9</td>
      <td style="padding: 8px; border: 1px solid #ddd;">85.2/75.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">67.7/44.3</td>
    </tr>
    <tr style="background-color: #fff2e6; font-weight: bold;">
      <td style="padding: 8px; border: 1px solid #ddd;">DKGZSL-vit</td>
      <td style="padding: 8px; border: 1px solid #ddd;">83.7/81.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">87.2/76.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.1/51.0</td>
    </tr>
  </tbody>
</table>

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
python CUB.py --dataset CUB --backbone vit-b/16 --batch_size 128 --epochs 100

```

## 📚 Citation
If you use this work, please cite our paper:
```bibtex
@article{DKGZSL2025,
  title={DKGZSL: Leveraging Dynamic Visual-Semantic Knowledge for Generative Zero-Shot Learning},
  author={},
  journal={tcsvt},
  year={2025},
  publisher={IEEE}
}
```

## ⭐ Thanks

Thank you sincerely for taking the time to review our manuscript amid your busy schedule. Your diligent efforts and professional insights are crucial to enhancing the quality of our research.
We look forward to your thoughtful guidance and valuable suggestions. We will carefully consider every comment and make every effort to improve the manuscript, striving to present a more rigorous and high-quality research work. Once again, we would like to express our heartfelt gratitude for your contributions!
