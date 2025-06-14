# Cora: Predictive Coding with Transformers for Video Prediction

<p align="center">
      <img src="assets/videos/cifar10_reconstruction_demo_v0.3.0.gif" alt="PC-ViT reconstruction demo" width="80%">
      <br/>
      <em>Image reconstruction (CIFAR-10) using PC-ViT inference (v0.3.0).</em>
</p>

## Introduction

Cora is a JAX-based library that integrates predictive coding with transformer architectures, initially focusing on image reconstruction and generation, with a long-term goal of video prediction. It is built on top of the [PCX library](https://github.com/liukidar/pcax), which provides a highly configurable framework for developing predictive coding networks. Cora aims to explore the potential advantages of predictive coding with transformer architectures, particularly in terms of generative capabilities derived from energy minimization and local updates.

**Current Status (v0.6.0):** The library now includes a functional Predictive Coding Vision Transformer (PC-ViT) model. Key achievements include successful image reconstruction on CIFAR-10 subsets, scaling unsupervised training to the full CIFAR-10 dataset with a 6-block model achieving ~0.004 MSE. The model now incorporates L1/L2 regularization options for intermediate Vodes, and hyperparameter tuning is systematically performed using Weights & Biases sweeps. These results validate the core energy-based generative inference mechanism and demonstrate the model's capability to learn meaningful representations from larger datasets.

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

This research proposal aims to explore the integration of predictive coding with transformer architectures, particularly focusing on parallel local updates and the potential advantages in video prediction and general learning efficiency. By building on recent trends like I-JEPA and V-JEPA, this research seeks to push the boundaries of how deep learning models can be made more efficient and biologically plausible, potentially offering a new paradigm for training spatiotemporal models.

By proposing a shift from sequential updates to parallel weight updates at each layer, we hypothesize that this method can outperform traditional transformers in both training speed and memory efficiency. Moreover, we aim to investigate how predictive coding can enhance video prediction models, where traditional transformers struggle with minute pixel-level details. This proposal seeks to implement and test these ideas, contributing to the broader field of generative models, unsupervised learning, and spatiotemporal reasoning.

## Roadmap

Cora is an active research project. My planned roadmap includes:

**Short-Term Goals:**

*   [x] Implement and validate PC-ViT for masked image reconstruction (CIFAR-10 subset). *(v0.3.0)*
*   [x] Scale unsupervised training to the full CIFAR-10 dataset. *(v0.4.0 - Successfully trained a 6-block model on the full dataset, achieving ~0.008 MSE)*
*   [x] Evaluate learned representations using linear probing on CIFAR-10 classification. *(v0.5.0 - Initial probing setup)*
*   [x] Add basic regularisation to latents. *(v0.6.0 - Current best linear probing results: 46% on CIFAR-10 on a smaller validation set)*
*   [x] Implement systematic hyperparameter search using W&B sweeps. *(v0.6.0)*
*   [ ] Experiment with methods for creating more meaningful abstract features, informed by linear probing results
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

### Short Term Goals (Next 1-2 Milestones)

*   **Tune SSL Pretraining for Better Downstream Task Performance**: Leverage the linear probing framework to guide hyperparameter tuning and architectural choices for SSL pretraining, aiming to improve the quality of learned representations for tasks like classification.

## Citation
If you find Cora useful in your work, please cite this repo (an OpenReview paper is coming soon).

## Original Repository
Cora is built on the PCX library, available at https://github.com/liukidar/pcax.