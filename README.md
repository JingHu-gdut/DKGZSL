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
| MSDN                   | CVPR’22     | 70.1/67.7           | 76.1/68.1           | 65.8/41.3           |
| TransZero              | AAAI'22     | 70.1/70.2           | 76.8/68.8           | 65.6/40.8           |
| TransZero++            | TPAMI'23    | 72.6/72.5           | 78.3/70.4           | 67.6/42.5           |
| I2MVFormer             | CVPR'23     | 73.6/73.8           | 42.1/42.8           | --/--               |
| ICIS                   | ICCV'23     | 60.6/56.5           | 64.6/51.6           | 51.8/32.7           |
| DUET                   | AAAI'23     | 69.9/72.7           | 72.3/67.5           | 64.4/45.8           |
| CC-ZSL                 | TCSVT'23    | 68.8/71.1           | 74.3/69.5           | 62.4/40.3           |
| PFRN                   | TCSVT'24    | 71.3/75.6           | 77.1/73.8           | 66.3/40.9           |
| ZSLViT                 | CVPR'24     | 70.7/74.2           | 78.9/73.6           | 68.3/47.3           |
| DSECN                  | CVPR'24     | 40.0/53.7           | 40.9/45.3           | 49.1/38.5           |
| ZeroMamba              | AAAI'25     | 71.9/76.5           | 80.0/74.2           | 72.4/47.7           |
| SVIP                   | ICCV'25     | 69.8/74.9           | 79.8/75.0           | 71.6/50.7           |
| CE-GZSL                | CVPR'21     | 70.4/70.0           | 77.5/65.3           | 63.3/43.1           |
| FREE                   | ICCV'21     | 64.8/67.1           | 68.8/57.7           | 65.0/41.7           |
| SDGZSL                 | ICCV'21     | 72.1/68.8           | 75.5/63.0           | --/--               |
| ICCE                   | CVPR'22     | 72.7/72.8           | 78.4/66.4           | --/--               |
| TDCSS                  | CVPR'22     | --/66.1             | --/51.9             | --/--               |
| CLSWGAN+DSP            | ICML'23     | --/70.7             | --/56.9             | --/45.5             |
| VS-Boost               | IJCAI’23    | --/--               | 79.8/68.4           | 62.4/42.5           |
| SHIP                   | ICCV'23     | --/74.7             | --/35.3             | --/--               |
| DGCNet                 | TCSVT’23    | 69.4/62.1           | 71.9/53.2           | 62.8/30.8           |
| VSGMN                  | TNNLS'24    | 71.2/70.3           | 77.8/69.3           | 66.3/40.8           |
| DPCN                   | TCSVT'24    | 70.6/71.4           | 80.1/69.0           | 63.8/43.3           |
| D³GZSL                 | AAAI'24     | --/70.1             | --/67.8             | --/--               |
| VADS                   | CVPR'24     | 82.5/79.3           | 86.8/74.3           | --/--               |
| ViFR                   | IJCV'25     | 73.7/68.0           | 69.1/60.1           | 65.6/40.9           |
| DKGZSL-res             | --          | 79.2/72.9           |85.2/75.4            | 67.7/44.3           |
| DKGZSL-vit             | --          | 83.7/81.4           |87.2/76.2            | 73.1/51.0           |

*Note: Acc = CZSL Top-1 Accuracy; H = GZSL Harmonic Mean (balance of seen/unseen performance); "--" = no available data.*

<table style="margin: 0 auto; text-align: center; width: 80%;">
  <caption style="text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 10px;">
    Key Results (%) of State-of-the-Art CZSL and GZSL Models (Acc for CZSL, H for GZSL)
  </caption>
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th style="padding: 8px; border: 1px solid #ddd;">Methods</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Venue</th>
      <th style="padding: 8px; border: 1px solid #ddd;">AWA2 (Acc/H)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">CUB (Acc/H)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">SUN (Acc/H)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">MSDN</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR’22</td>
      <td style="padding: 8px; border: 1px solid #ddd;">70.1/67.7</td>
      <td style="padding: 8px; border: 1px solid #ddd;">76.1/68.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">65.8/41.3</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">TransZero</td>
      <td style="padding: 8px; border: 1px solid #ddd;">AAAI'22</td>
      <td style="padding: 8px; border: 1px solid #ddd;">70.1/70.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">76.8/68.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">65.6/40.8</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">TransZero++</td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPAMI'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">72.6/72.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">78.3/70.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">67.6/42.5</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">I2MVFormer</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.6/73.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">42.1/42.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">ICIS</td>
      <td style="padding: 8px; border: 1px solid #ddd;">ICCV'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">60.6/56.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">64.6/51.6</td>
      <td style="padding: 8px; border: 1px solid #ddd;">51.8/32.7</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">DUET</td>
      <td style="padding: 8px; border: 1px solid #ddd;">AAAI'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">69.9/72.7</td>
      <td style="padding: 8px; border: 1px solid #ddd;">72.3/67.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">64.4/45.8</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">CC-ZSL</td>
      <td style="padding: 8px; border: 1px solid #ddd;">TCSVT'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">68.8/71.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">74.3/69.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">62.4/40.3</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">PFRN</td>
      <td style="padding: 8px; border: 1px solid #ddd;">TCSVT'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.3/75.6</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.1/73.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">66.3/40.9</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">ZSLViT</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">70.7/74.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">78.9/73.6</td>
      <td style="padding: 8px; border: 1px solid #ddd;">68.3/47.3</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">DSECN</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">40.0/53.7</td>
      <td style="padding: 8px; border: 1px solid #ddd;">40.9/45.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">49.1/38.5</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">ZeroMamba</td>
      <td style="padding: 8px; border: 1px solid #ddd;">AAAI'25</td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.9/76.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">80.0/74.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">72.4/47.7</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">SVIP</td>
      <td style="padding: 8px; border: 1px solid #ddd;">ICCV'25</td>
      <td style="padding: 8px; border: 1px solid #ddd;">69.8/74.9</td>
      <td style="padding: 8px; border: 1px solid #ddd;">79.8/75.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.6/50.7</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">CE-GZSL</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'21</td>
      <td style="padding: 8px; border: 1px solid #ddd;">70.4/70.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.5/65.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">63.3/43.1</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">FREE</td>
      <td style="padding: 8px; border: 1px solid #ddd;">ICCV'21</td>
      <td style="padding: 8px; border: 1px solid #ddd;">64.8/67.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">68.8/57.7</td>
      <td style="padding: 8px; border: 1px solid #ddd;">65.0/41.7</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">SDGZSL</td>
      <td style="padding: 8px; border: 1px solid #ddd;">ICCV'21</td>
      <td style="padding: 8px; border: 1px solid #ddd;">72.1/68.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">75.5/63.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">ICCE</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'22</td>
      <td style="padding: 8px; border: 1px solid #ddd;">72.7/72.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">78.4/66.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">TDCSS</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'22</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/66.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/51.9</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">CLSWGAN+DSP</td>
      <td style="padding: 8px; border: 1px solid #ddd;">ICML'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/70.7</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/56.9</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/45.5</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">VS-Boost</td>
      <td style="padding: 8px; border: 1px solid #ddd;">IJCAI’23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
      <td style="padding: 8px; border: 1px solid #ddd;">79.8/68.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">62.4/42.5</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">SHIP</td>
      <td style="padding: 8px; border: 1px solid #ddd;">ICCV'23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/74.7</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/35.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">DGCNet</td>
      <td style="padding: 8px; border: 1px solid #ddd;">TCSVT’23</td>
      <td style="padding: 8px; border: 1px solid #ddd;">69.4/62.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.9/53.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">62.8/30.8</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">VSGMN</td>
      <td style="padding: 8px; border: 1px solid #ddd;">TNNLS'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.2/70.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.8/69.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">66.3/40.8</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">DPCN</td>
      <td style="padding: 8px; border: 1px solid #ddd;">TCSVT'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">70.6/71.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">80.1/69.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">63.8/43.3</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">D³GZSL</td>
      <td style="padding: 8px; border: 1px solid #ddd;">AAAI'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/70.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/67.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">VADS</td>
      <td style="padding: 8px; border: 1px solid #ddd;">CVPR'24</td>
      <td style="padding: 8px; border: 1px solid #ddd;">82.5/79.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">86.8/74.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--/--</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 8px; border: 1px solid #ddd;">ViFR</td>
      <td style="padding: 8px; border: 1px solid #ddd;">IJCV'25</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.7/68.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">69.1/60.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">65.6/40.9</td>
    </tr>
    <tr style="background-color: #e6f7ff; font-weight: bold;">
      <td style="padding: 8px; border: 1px solid #ddd;">DKGZSL-res</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--</td>
      <td style="padding: 8px; border: 1px solid #ddd;">79.2/72.9</td>
      <td style="padding: 8px; border: 1px solid #ddd;">85.2/75.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">67.7/44.3</td>
    </tr>
    <tr style="background-color: #fff2e6; font-weight: bold;">
      <td style="padding: 8px; border: 1px solid #ddd;">DKGZSL-vit</td>
      <td style="padding: 8px; border: 1px solid #ddd;">--</td>
      <td style="padding: 8px; border: 1px solid #ddd;">83.7/81.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;">87.2/76.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.1/51.0</td>
    </tr>
  </tbody>
</table>

<p style="text-align: center; margin-top: 10px; font-size: 14px;">
  Note: Acc = CZSL Top-1 Accuracy; H = GZSL Harmonic Mean; "--" = no available data
</p>

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
