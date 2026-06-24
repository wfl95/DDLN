# DDLN-NoiseRobustFaceRecognition

This repository provides a lightweight PyTorch implementation of a noise-filtering module for noise-robust face recognition training.

The core module, `NoiseFilter`, performs sample filtering directly in cosine-similarity space. It can be plugged into margin-based face recognition losses such as CosFace and ArcFace with minimal changes.

## Overview

Label noise is a common problem in large-scale face recognition datasets, especially when identities are collected or annotated automatically. Incorrect labels can mislead supervised training and degrade recognition performance.

This repository provides a simple plug-and-play noise filtering module. The method tracks similarity statistics during training and progressively filters unreliable samples according to a learned threshold.

## Main Features

- PyTorch implementation
- Plug-and-play design
- Compatible with CosFace and ArcFace
- Noise filtering in cosine-similarity / logit space
- Progressive threshold update during training
- Lightweight computation overhead

## Files

```text
noise_filter_DDLN.py
