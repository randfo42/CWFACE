## Regularization using Noise samples Identified by the Feature norm for Face recognition

![NDR overview](https://github.com/randfo42/NDR-FACE/blob/main/NDR%20overview.png)


> Abstract: Face recognition is a task that involves comparing two images of a face and determining whether they belong to the same person. This task can be applied in a variety of environments, including surveillance systems. However, the performance of the deep learning model used for face recognition can be affected by the quality of the images. Therefore, recent studies on face recognition using deep learning have suggested taking image quality into consideration. Some studies have used feature norms, which is the L2 norm of extracted features from images using a deep learning model, to measure the image quality. However, previous studies have lacked analysis of why the feature norms correspond to image quality. This paper proposes a new hypothesis that suggests that the feature norms of a sample are higher if it is similar to other samples learned by the deep learning model. We also demonstrate that this hypothesis can be used to distinguish noise samples.  Additionally, we introduce a new regularization technique, Noise Direction Regularization, NDR, which uses noise samples to improve face recognition performance in low-resolution environments.
>

changed code in head.py - CoswFace and ArcWFace class


Performance comparison using True Positive Rate (TPR) at specified False Accept Rate (FAR) on the SurvFace dataset, training with CASIA-WebFace dataset.
| Method |  | TPR(%)@FAR |  |  |
| :--- | :--- | :--- | :--- | ---- |
|  | 0.3 | 0.1 | 0.01 | 0.001 |
| CosFace | 45.2 | 23.0 | 6.1 | 1.2 |
| CosFace + NDR | **51.8** | **27.1** | **7.5** | **2.3** |
| Adaface | 46.7 | 24.8 | 6.1 | 2.2 |
| Adaface + NDR | **59.0** | **34.5** | **9.1** | **3.1** |


Comparison of identification rates at Rank 1 and Rank 5 on the TinyFace dataset, training with CASIA-WebFace dataset.
| Method | Rank1 | Rank5 |
| :--- | :--- | :--- |
| CosFace | 56.7 | 62.7 |
| CosFace + NDR | **61.4** | **66.5** |
| Adaface | 61.8 | 66.8 |
| Adaface + NDR | **62.9** | **68.1** |



Code From [Adaface](https://github.com/mk-minchul/AdaFace)

