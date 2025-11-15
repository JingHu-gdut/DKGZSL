# DKGZSL
Official repository of the DKGZSL paper.

# Abstract
Generative Zero-Shot Learning (GZSL) methods
address the challenge of recognizing unseen classes by synthesizing visual features, thereby converting ZSL into a supervised
learning task. However, existing approaches are predominantly
constrained to two multi-stage strategies: pre-generation prior
knowledge enhancement and post-generation feature refinement.
These paradigms often suffer from error propagation across
stages, ultimately limiting generation quality and representational
fidelity. To overcome these limitations, we propose DKGZSL, a
novel generative framework that injects dynamic visual–semantic
knowledge directly into the feature synthesis process, effectively
unifying generation and refinement into a single cohesive stage.
Specifically, a Knowledge Transfer Network (KTN) is introduced
to convert semantic information into hierarchical visual knowledge representations. To ensure accurate semantic–visual alignment, we further design a Semantic-Oriented Visual Refinement
(SOVR) module that reshapes real visual features into semantically aligned and noise-suppressed representations, providing
precise guidance for the KTN. Moreover, hierarchical knowledge
extracted from each KTN layer is progressively transmitted to
the generator via Meta-Fusion Units (MFUs), enabling dynamic
semantic guidance and improving generation quality. Extensive
experiments on three benchmark datasets demonstrate that
DKGZSL achieves consistent state-of-the-art performance with
both ResNet-101 and ViT-B/16 feature extractors. Comprehensive
ablation studies further confirm the effectiveness and complementarity of each proposed component.
