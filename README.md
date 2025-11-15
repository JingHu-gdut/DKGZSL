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
Outperforms state-of-the-art methods on three benchmark datasets (AWA2, CUB, SUN) under both Conventional ZSL (CZSL) and Generalized ZSL (GZSL) settings, with ResNet-101 and ViT-B/16 backbones:
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/283a0c91-14aa-42e0-a74d-0def664ba186" />

Table 1: Results (%) of the state-of-the-art CZSL and GZSL models on AWA2, CUB, and SUN (generation-based and embedding-based methods). 
- Symbol "$\dagger$" denotes ViT-based methods. 
- Best results based on ViT-Base (red bold) and ResNet-101 (blue bold).

| Methods                | Venue       | AWA2 - CZSL (Acc) | AWA2 - GZSL (U/S/H) | CUB - CZSL (Acc) | CUB - GZSL (U/S/H)   | SUN - CZSL (Acc) | SUN - GZSL (U/S/H)   |
|------------------------|-------------|-------------------|----------------------|------------------|----------------------|-------------------|----------------------|
| **Embedding-based Methods** |             |                   |                      |                  |                      |                   |                      |
| MSDN                   | CVPR’22     | 70.1              | 62.0/74.5/67.7       | 76.1             | 68.7/67.5/68.1       | 65.8              | 52.2/34.2/41.3       |
| TransZero              | AAAI'22     | 70.1              | 61.3/82.3/70.2       | 76.8             | 69.3/68.3/68.8       | 65.6              | 52.6/33.4/40.8       |
| TransZero++            | TPAMI'23    | 72.6              | 64.6/82.7/72.5       | 78.3             | 67.5/73.6/70.4       | 67.6              | 48.6/37.8/42.5       |
| I2MVFormer$^\dagger$   | CVPR'23     | 73.6              | 66.6/82.9/73.8       | 42.1             | 32.4/63.1/42.8       | --                | --                   |
| ICIS                   | ICCV'23     | 60.6              | 45.8/73.7/56.5       | 64.6             | 35.6/93.3/51.6       | 51.8              | 45.2/25.6/32.7       |
| DUET$^\dagger$         | AAAI'23     | 69.9              | 63.7/84.7/72.7       | 72.3             | 62.9/72.8/67.5       | 64.4              | 45.7/45.8/45.8       |
| CC-ZSL                 | TCSVT'23    | 68.8              | 62.2/83.1/71.1       | 74.3             | 73.2/66.1/69.5       | 62.4              | 36.9/44.4/40.3       |
| PFRN                   | TCSVT'24    | 71.3              | 68.6/84.3/**75.6**  | 77.1             | 72.7/75.0/73.8       | 66.3              | 55.5/32.3/40.9       |
| ZSLViT$^\dagger$       | CVPR'24     | 70.7              | 66.1/84.6/74.2       | 78.9             | 69.4/78.2/73.6       | 68.3              | 45.9/48.3/47.3       |
| DSECN                  | CVPR'24     | 40.0              | --/--/53.7          | 40.9             | --/--/45.3           | 49.1              | --/--/38.5           |
| ZeroMamba              | AAAI'25     | 71.9              | 67.9/87.6/76.5       | 80.0             | 72.1/76.4/74.2       | 72.4              | 56.5/41.4/47.7       |
| SVIP$^\dagger$         | ICCV'25     | 69.8              | 65.4/87.7/74.9       | 79.8             | 72.1/78.1/75.0       | 71.6              | 53.7/48.0/50.7       |
| **Generation-based Methods** |             |                   |                      |                  |                      |                   |                      |
| CE-GZSL                | CVPR'21     | 70.4              | 63.1/78.6/70.0       | 77.5             | 63.9/66.8/65.3       | 63.3              | 48.8/38.6/43.1       |
| FREE                   | ICCV'21     | 64.8              | 60.4/75.4/67.1       | 68.8             | 55.7/59.9/57.7       | 65.0              | 47.4/37.2/41.7       |
| SDGZSL                 | ICCV'21     | 72.1              | 64.6/73.6/68.8       | 75.5             | 59.9/66.4/63.0       | --                | --                   |
| ICCE                   | CVPR'22     | 72.7              | 65.3/82.3/72.8       | 78.4             | 67.3/65.5/66.4       | --                | --                   |
| TDCSS                  | CVPR'22     | --                | 59.2/74.9/66.1       | --               | 44.2/62.8/51.9       | --                | --                   |
| CLSWGAN+DSP            | ICML'23     | --                | 60.0/86.0/70.7       | --               | 51.4/63.8/56.9       | --                | 48.3/43.0/**45.5**   |
| VS-Boost               | IJCAI’23    | --                | --/--/--             | 79.8             | 68.0/68.7/68.4       | 62.4              | 49.2/37.4/42.5       |
| SHIP$^\dagger$         | ICCV'23     | --                | 61.2/95.9/74.7       | --               | 22.5/82.2/35.3       | --                | --                   |
| DGCNet                 | TCSVT’23    | 69.4              | 55.9/69.8/62.1       | 71.9             | 50.3/56.4/53.2       | 62.8              | 25.9/38.1/30.8       |
| VSGMN                  | TNNLS'24    | 71.2              | 64.0/77.8/70.3       | 77.8             | 69.6/68.9/69.3       | 66.3              | 50.7/34.1/40.8       |
| DPCN                   | TCSVT'24    | 70.6              | 65.4/78.6/71.4       | 80.1             | 72.7/65.7/69.0       | 63.8              | 48.1/39.4/43.3       |
| D³GZSL                 | AAAI'24     | --                | 64.6/76.7/70.1       | --               | 66.7/69.1/67.8       | --                | --                   |
| VADS$^\dagger$         | CVPR'24     | 82.5              | 75.4/83.6/79.3       | 86.8             | 74.1/74.6/74.3       | --                | --                   |
| ViFR                   | IJCV'25     | 73.7              | 58.4/81.4/68.0       | 69.1             | 57.8/62.7/60.1       | 65.6              | 48.8/35.2/40.9       |
| **DKGZSL**             | --          | **79.2**          | 67.5/79.2/72.9       | **85.2**         | 80.2/71.3/**75.4**   | **67.7**          | 48.2/40.9/44.3       |
| **DKGZSL$^\dagger$**   | --          | $\color{red}{\textbf{83.7}}$ | 79.8/83.0/$\color{red}{\textbf{81.4}}$ | $\color{red}{\textbf{87.2}}$ | 79.0/73.5/$\color{red}{\textbf{76.2}}$ | $\color{red}{\textbf{73.1}}$ | 57.6/45.6/$\color{red}{\textbf{51.0}}$ |
| **CLIP-based Methods** |             |                   |                      |                  |                      |                   |                      |
| CLIP                   | ICML'21     | --                | 77.8/81.7/79.6       | --               | 29.6/29.9/29.7       | --                | 49.5/46.2/47.8       |
| CoOp                   | IJCV’22     | --                | 69.4/81.4/75.0       | --               | 18.2/22.2/20.0       | --                | 49.3/49.9/49.6       |
| PromptSRC              | ICCV'23     | --                | 70.7/84.0/76.8       | --               | 16.3/30.9/21.4       | --                | 49.2/47.8/48.5       |
| TPR                    | NeurIPS'24  | --                | 76.8/87.1/81.6       | --               | 26.9/41.2/32.5       | --                | 45.4/50.5/47.8       |
| GenZSL                 | ICML'25     | --                | 86.1/88.7/87.4       | --               | 53.5/61.9/57.4       | --                | 50.6/43.8/47.0       |

*Note: Gray background rows in the original LaTeX table are omitted here; U=Unseen, S=Seen, H=Harmonic Mean.*

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
