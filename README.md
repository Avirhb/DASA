# DASA
Training-free universal ultrasound segmentation (Medical Image Analysis)
**Medical Image Analysis** | Under Review

[Haobo Ling](mailto:chenfang_bme@163.com), Lingyu Chen, Jia Rui Han, Haoxuan Zhang, Yue Wang, Shenghai Xin, Yufei Zhao, Fang Chen*

---
## Overview

DASA is a **training-free** framework for universal ultrasound image segmentation. 
It integrates two complementary pathways:

- **SSP (Static Semantic Pathway)**: Few-shot visual prototype matching with contrastive background suppression.
- **TTBA (Test-Time Background Anchoring)**: Extracts an in-situ background representation from each test image and infers foreground through feature dissimilarity.

The two pathways are fused through **Orthogonal Joint Verification (OJV)**, and the result prompts a frozen SAM for final segmentation — with no parameter updates or gradient backpropagation.

![Pipeline Overview](assets/overview.png)

---
## UUSeg-Bench

We also introduce **UUSeg-Bench**, the first large-scale benchmark for universal ultrasound segmentation:
- 14 datasets | 8 anatomical regions | 9,764 labeled frames | 10 countries
## Code

> **The full source code is being organized and will be released upon paper acceptance.**  
> Please star/watch this repository to be notified when the code is available.

---
## Contact

For questions, please contact: **chenfang_bme@163.com**
