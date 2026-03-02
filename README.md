# Demonstrations-CoT-and-Prompting-A-Theoretical-Analysis-of-ICL

This repository contains the official implementation accompanying the paper:

> **Demonstrations, CoT, and Prompting: A Theoretical Analysis of ICL**  
> Xuhan Tong, Yuchen Zeng, Jiawei Zhang  
> arXiv 2026

---

## Overview

This work develops a unified theoretical framework for understanding in-context learning (ICL) under:

- Demonstration variation
- Instruction perturbations
- Chain-of-Thought (CoT) prompting
- Padding and format shifts
- Out-of-distribution (OOD) generalization

We provide both:

- **Formal theoretical analysis**
- **Controlled synthetic experiments** validating the derived bounds

This repository contains the experimental code used to support the theoretical results.

---

## Experiments

The experiments are organized by section of the paper.

### 1. Intrinsic ICL Capability

Key features:

- Demonstration-only supervision (k and b are hidden)
- Relative error evaluation (5% tolerance)
- Controlled distribution shifts on x-range and parameter scale

This setup corresponds to Section 2 of the paper.

---

### 2.  The Quality of Selected Demonstrations

We design two tasks for investigating the effect of demonstration choice. We construct two types of demonstrations: identifying and ambiguous.

- Task 1. Sport Identification
- Task 2. Person Identification

This corresponds to Section 2.

---

### 3. Chain-of-Thought (CoT) Decomposition

We evaluate how multi-step decomposition affects stability and generalization.

We compare:

- Direct prediction
- Single-step CoT
- Multi-step CoT chains

This corresponds to Section 3.

---

### 4. Padding Stability Analysis

Experiments validating the attention stability bound under padding.

See Section 2 and 3.

---

### 5. Ptompting Templates and Instruction Variations

We analyze the effect of 6 different types of prompting templates, consisting:

- Task-consistent instructions
- Misleading instructions
- Other perturbations

This corresponds to Section 4 and 5.

---

## Requirements

The core dependencies include:

- torch
- transformers
- peft
- numpy

Install via:

pip install -r requirements.txt

---

## Citation

If you find this work useful, please cite: 
