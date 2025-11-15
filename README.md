# DKGZSL: Leveraging Dynamic Knowledge-Guided Generative Zero-Shot Learning

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
- Symbol "$\dagger$" denotes ViT-based methods. 
- Best results based on ViT-Base (red bold) and ResNet-101 (blue bold).

| Methods                | Venue       | AWA2 (Acc/H)       | CUB (Acc/H)         | SUN (Acc/H)         |
|------------------------|-------------|---------------------|---------------------|---------------------|
| **Embedding-based Methods** |             |                     |                     |                     |
| MSDN                   | CVPR’22     | 70.1/67.7           | 76.1/68.1           | 65.8/41.3           |
| TransZero              | AAAI'22     | 70.1/70.2           | 76.8/68.8           | 65.6/40.8           |
| TransZero++            | TPAMI'23    | 72.6/72.5           | 78.3/70.4           | 67.6/42.5           |
| I2MVFormer$^\dagger$   | CVPR'23     | 73.6/73.8           | 42.1/42.8           | --/--               |
| ICIS                   | ICCV'23     | 60.6/56.5           | 64.6/51.6           | 51.8/32.7           |
| DUET$^\dagger$         | AAAI'23     | 69.9/72.7           | 72.3/67.5           | 64.4/45.8           |
| CC-ZSL                 | TCSVT'23    | 68.8/71.1           | 74.3/69.5           | 62.4/40.3           |
| PFRN                   | TCSVT'24    | 71.3/\textcolor{blue}{\textbf{75.6}} | 77.1/73.8       | 66.3/40.9           |
| ZSLViT$^\dagger$       | CVPR'24     | 70.7/74.2           | 78.9/73.6           | 68.3/47.3           |
| DSECN                  | CVPR'24     | 40.0/53.7           | 40.9/45.3           | 49.1/38.5           |
| ZeroMamba              | AAAI'25     | 71.9/76.5           | 80.0/74.2           | 72.4/47.7           |
| SVIP$^\dagger$         | ICCV'25     | 69.8/74.9           | 79.8/75.0           | 71.6/50.7           |
| **Generation-based Methods** |             |                     |                     |                     |
| CE-GZSL                | CVPR'21     | 70.4/70.0           | 77.5/65.3           | 63.3/43.1           |
| FREE                   | ICCV'21     | 64.8/67.1           | 68.8/57.7           | 65.0/41.7           |
| SDGZSL                 | ICCV'21     | 72.1/68.8           | 75.5/63.0           | --/--               |
| ICCE                   | CVPR'22     | 72.7/72.8           | 78.4/66.4           | --/--               |
| TDCSS                  | CVPR'22     | --/66.1             | --/51.9             | --/--               |
| CLSWGAN+DSP            | ICML'23     | --/70.7             | --/56.9             | --/\textcolor{blue}{\textbf{45.5}} |
| VS-Boost               | IJCAI’23    | --/--               | 79.8/68.4           | 62.4/42.5           |
| SHIP$^\dagger$         | ICCV'23     | --/74.7             | --/35.3             | --/--               |
| DGCNet                 | TCSVT’23    | 69.4/62.1           | 71.9/53.2           | 62.8/30.8           |
| VSGMN                  | TNNLS'24    | 71.2/70.3           | 77.8/69.3           | 66.3/40.8           |
| DPCN                   | TCSVT'24    | 70.6/71.4           | 80.1/69.0           | 63.8/43.3           |
| D³GZSL                 | AAAI'24     | --/70.1             | --/67.8             | --/--               |
| VADS$^\dagger$         | CVPR'24     | 82.5/79.3           | 86.8/74.3           | --/--               |
| ViFR                   | IJCV'25     | 73.7/68.0           | 69.1/60.1           | 65.6/40.9           |
| **DKGZSL**             | --          | \textcolor{blue}{\textbf{79.2}}/72.9 | \textcolor{blue}{\textbf{85.2}}/\textcolor{blue}{\textbf{75.4}} | \textcolor{blue}{\textbf{67.7}}/44.3 |
| **DKGZSL$^\dagger$**   | --          | \textcolor{red}{\textbf{83.7}}/\textcolor{red}{\textbf{81.4}} | \textcolor{red}{\textbf{87.2}}/\textcolor{red}{\textbf{76.2}} | \textcolor{red}{\textbf{73.1}}/\textcolor{red}{\textbf{51.0}} |
| **CLIP-based Methods** |             |                     |                     |                     |
| CLIP                   | ICML'21     | --/79.6             | --/29.7             | --/47.8             |
| CoOp                   | IJCV’22     | --/75.0             | --/20.0             | --/49.6             |
| PromptSRC              | ICCV'23     | --/76.8             | --/21.4             | --/48.5             |
| TPR                    | NeurIPS'24  | --/81.6             | --/32.5             | --/47.8             |
| GenZSL                 | ICML'25     | --/87.4             | --/57.4             | --/47.0             |

*Note: Acc = CZSL Top-1 Accuracy; H = GZSL Harmonic Mean (balance of seen/unseen performance); "--" = no available data.*

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
