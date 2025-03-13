# Cora: Predictive Coding with Transformers for Video Prediction

## Introduction

Cora is a Python JAX-based library that integrates predictive coding with transformer architectures for video prediction and general learning efficiency. It is built on top of the [PCX library](https://github.com/liukidar/pcax), which provides a highly configurable framework for developing predictive coding networks. Cora aims to explore the potential advantages of predictive coding with transformer architectures, particularly in terms of parallel local updates and video prediction.

## Installation

Cora is designed to run on Lambda Labs' GH200 instances using a Docker-based setup. Follow these steps to get started:

1. **Set up a Lambda Labs account** and create a GH200 instance.
2. **Clone the Cora repository**:  
   ```bash
   git clone https://github.com/your-username/cora.git
   ```
3. **Navigate to the repository directory**:
      ```bash
      cd cora
      ```
4. **Build and run the Docker image**:
      ```bash
      ./docker/run.sh
      ```

## Motivation
Predictive coding is a computational framework inspired by the brain’s way of processing information, where higher cortical areas predict sensory input and the errors between predicted and actual sensory inputs are used to update the model. This hierarchical approach has shown promise in neuroscience, and recent advances in deep learning have demonstrated its potential to improve efficiency by reducing the need for global error propagation and focusing on local prediction errors.

Recent works like I-JEPA and V-JEPA (by Meta) aim to improve unsupervised learning by predicting not pixel-level data, but higher-level latent representations. This aligns with the philosophy of predictive coding, where predictions are made in a compressed latent space at multiple levels of abstraction rather than the raw input. I-JEPA and V-JEPA provide compelling evidence that latent-space predictions can lead to more efficient learning, a direction that this proposal seeks to push further.

This research proposal aims to explore the integration of predictive coding with transformer architectures, particularly focusing on parallel local updates and the potential advantages in video prediction and general learning efficiency. By building on recent trends like I-JEPA and V-JEPA, we seek to push the boundaries of how deep learning models can be made more efficient and biologically plausible, potentially offering a new paradigm for training spatiotemporal models.

By proposing a shift from sequential updates to parallel weight updates at each layer, we hypothesize that this method can outperform traditional transformers in both training speed and memory efficiency. Moreover, we aim to investigate how predictive coding can enhance video prediction models, where traditional transformers struggle with minute pixel-level details. This proposal seeks to implement and test these ideas, contributing to the broader field of generative models, unsupervised learning, and spatiotemporal reasoning.

## Citation
If you find Cora useful in your work, please cite the original PCX paper: 

@article{pinchetti2024benchmarkingpredictivecodingnetworks,
      title={Benchmarking Predictive Coding Networks -- Made Simple}, 
      author={Luca Pinchetti and Chang Qi and Oleh Lokshyn and Gaspard Olivers and Cornelius Emde and Mufeng Tang and Amine M'Charrak and Simon Frieder and Bayar Menzat and Rafal Bogacz and Thomas Lukasiewicz and Tommaso Salvatori},
      year={2024},
      eprint={2407.01163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01163}, 
}

## Original Repository
Cora is built on the PCX library, available at https://github.com/liukidar/pcax.