## Mipmap-GS
### Abstract
```
3D Gaussian Splatting (3DGS) has gained great attention in novel view synthesis (NVS) due to its superior rendering efficiency and high fidelity. However, the trained Gaussians suffer from severe zooming degradation due to non-adjustable representation derived from single-scale training. Though some methods attempt to tackle this problem via post-processing techniques such as selective rendering or filtering technique towards primitives, the specific-scale information is not involved in Gaussians. 
In this paper, we propose a unified optimization method to make Gaussians adaptive for arbitrary scales by self-adjusting the primitive properties (e.g., color, shape and size) and distribution (e.g., position). Inspired by mipmap technique, we design pseudo ground-truth (pseudo-GT) for the target scale and propose a scale-aware guidance loss to introduce scale information into 3D Gaussians in a self-supervised way. Our method is a plug-in module, applicable for any 3DGS models to solve the zoom-in and zoom-out aliasing.
```


# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Mip-Splatting](https://github.com/autonomousvision/mip-splatting.git). The SR model is from [SwinIR](https://github.com/JingyunLiang/SwinIR.git).
We thank all the authors for their great work and repos. 
