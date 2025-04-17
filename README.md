# Cora: Predictive Coding with Transformers for Video Prediction

<!-- Add Video Demonstration -->
<p align="center">
  <video src="https://github.com/FeherBalazs/cora/blob/feature_image_reconstruction/assets/videos/cifar10_reconstruction_demo_v0.3.0.mp4" width="80%" autoplay loop muted playsinline>
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>Masked image reconstruction (CIFAR-10) using PC-ViT inference (v0.3.0).</em>
</p>
<!-- End Video Demonstration -->

## Introduction

Cora is a JAX-based library that integrates predictive coding with transformer architectures, initially focusing on image reconstruction and generation, with a long-term goal of video prediction. It is built on top of the [PCX library](https://github.com/liukidar/pcax), which provides a highly configurable framework for developing predictive coding networks. Cora aims to explore the potential advantages of predictive coding with transformer architectures, particularly in terms of generative capabilities derived from energy minimization and local updates.

**Current Status (v0.3.0):** The library now includes a functional Predictive Coding Vision Transformer (PC-ViT) model that demonstrates successful masked image reconstruction (inpainting) on CIFAR-10 subsets, validating the core energy-based generative inference mechanism.

## Installation

Cora is designed to run using a Docker-based setup. Follow these steps to get started:

1. **Clone the Cora repository**:  

   ```bash
   git clone https://github.com/your-username/cora.git
   ```

2. **Navigate to the repository directory**:

      ```bash
      cd cora
      ```

3. **Build and run the Docker image**:

      ```bash
      ./docker/run.sh
      ```
4. **Login to Weights and Biases**:
      ```bash
      wandb login
      ```

5. **Run training**:
      ```bash
      python examples/debug_transformer_wandb.py --config debug_tiny
      ```

## Motivation
Predictive coding is a computational framework inspired by the brain's way of processing information, where higher cortical areas predict sensory input and the errors between predicted and actual sensory inputs are used to update the model. This hierarchical approach has shown promise in neuroscience, and recent advances in deep learning have demonstrated its potential to improve efficiency by potentially reducing the need for global error propagation and focusing on local prediction errors. Cora explores implementing these principles within modern Transformer architectures.

Recent works like I-JEPA and V-JEPA (by Meta) aim to improve unsupervised learning by predicting not pixel-level data, but higher-level latent representations. This aligns with the philosophy of predictive coding, where predictions are made in a compressed latent space at multiple levels of abstraction rather than the raw input. I-JEPA and V-JEPA provide compelling evidence that latent-space predictions can lead to more efficient learning, a direction that this proposal seeks to push further.

This research proposal aims to explore the integration of predictive coding with transformer architectures, particularly focusing on parallel local updates and the potential advantages in video prediction and general learning efficiency. By building on recent trends like I-JEPA and V-JEPA, we seek to push the boundaries of how deep learning models can be made more efficient and biologically plausible, potentially offering a new paradigm for training spatiotemporal models.

By proposing a shift from sequential updates to parallel weight updates at each layer, we hypothesize that this method can outperform traditional transformers in both training speed and memory efficiency. Moreover, we aim to investigate how predictive coding can enhance video prediction models, where traditional transformers struggle with minute pixel-level details. This proposal seeks to implement and test these ideas, contributing to the broader field of generative models, unsupervised learning, and spatiotemporal reasoning.

## Roadmap

Cora is an active research project. My planned roadmap includes:

**Short-Term Goals:**

*   [x] Implement and validate PC-ViT for masked image reconstruction (CIFAR-10 subset). *(v0.3.0)*
*   [ ] Scale unsupervised training to the full CIFAR-10 dataset.
*   [ ] Evaluate learned representations using linear probing on CIFAR-10 classification.
*   [ ] Implement label conditioning mechanisms within the PC-ViT architecture.
*   [ ] Demonstrate label-conditioned image generation on CIFAR-10.

**Mid-Term Goals:**

*   [ ] Scale unsupervised pre-training to ImageNet.
*   [ ] Evaluate learned representations via ImageNet linear probing.
*   [ ] Explore text-conditioning using CLIP embeddings or similar for generation (potentially on LAION).
*   [ ] Begin integration of temporal dynamics for video prediction tasks.

**Long-Term Vision:**

*   [ ] Develop robust PC-Transformer models for video prediction and generation.
*   [ ] Investigate the efficiency (computational, sample) of PC training compared to standard backpropagation for large transformer models.
*   [ ] Explore more advanced predictive coding objectives and dynamics within the transformer framework.

## Citation
If you find Cora useful in your work, please cite the original PCX paper: [arXiv link](https://arxiv.org/abs/2407.01163) and this repo (an OpenReview paper is coming soon).

```bibtex
@article{pinchetti2024benchmarkingpredictivecodingnetworks,
      title={Benchmarking Predictive Coding Networks -- Made Simple}, 
      author={Luca Pinchetti and Chang Qi and Oleh Lokshyn and Gaspard Olivers and Cornelius Emde and Mufeng Tang and Amine M'Charrak and Simon Frieder and Bayar Menzat and Rafal Bogacz and Thomas Lukasiewicz and Tommaso Salvatori},
      year={2024},
      eprint={2407.01163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01163}, 
}
```

## Original Repository
Cora is built on the PCX library, available at https://github.com/liukidar/pcax.