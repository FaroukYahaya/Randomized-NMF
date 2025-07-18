#  Faster-than-Fast NMF via Random Projections and Nesterov Iterations

This repository contains MATLAB code for efficient Non-negative Matrix Factorization (NMF) using **random projections** and **Nesterov's optimal gradient method**. It includes both the standard NMF solver and an accelerated variant based on dimensionality reduction techniques.

The code is based on my paper:

> **F. Yahaya**, M. Puigt, G. Delmaire, and G. Roussel  
> *Faster-than-fast NMF using random projections and Nesterov iterations*  
> arXiv preprint [arXiv:1812.04315](https://arxiv.org/abs/1812.04315)

If you use this code in your research or project, please **cite the paper** above.

Running the `test_nenmf.m` script performs a side-by-side comparison of **standard NMF** vs **randomized NMF** on synthetic data, under a time constraint (e.g. 60 seconds).

The plot below shows the **Relative Reconstruction Error (RRE)** over time. The randomized version converges faster with comparable final error, highlighting the benefit of **sketching-based compression**:

<div align="center">
  <img src="results/rre_plot.png" width="600" alt="RRE vs Time: Standard vs Randomized NMF">
</div>

---

## 📌 Citation

```bibtex
@article{yahaya2018faster,
  title={Faster-than-fast NMF using random projections and Nesterov iterations},
  author={Yahaya, Farouk and Puigt, Matthieu and Delmaire, Guillaume and Roussel, Guillaume},
  journal={arXiv preprint arXiv:1812.04315},
  year={2018}
}
