# USeS-BPCA

## U-Net Semantic Segmentation Enhanced with BPCAPooling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Lucs1590/USeS-BPCA.svg?style=social&label=Star&maxAge=2592000)]

Welcome to the USeS-BPCA – a refined repository that brings forward a deep learning model for semantic image segmentation focusing on U-Net architectures. The key spotlight of this repository is the novel BPCAPooling (Block-based Principal Component Analysis Pooling), an advanced dimensional reduction method developed under my Master's thesis that'll soon be up for grabs.

### Abstract

The evolution of computer vision has had a significant impact on the efficiencies of image segmentation, particularly in the realms of critical medical analysis, scene analysis, and autonomous system projects. The sharp ascent in the dimensions of the AI world, the arrival of artificial neural networks and deep learning techniques that power multiple architectures are driving state-of-art models offering enhanced performance for different scenarios of image segmentation.

However, the challenge of preserving spatiality when reducing dimensionality, especially in the pooling layers of convolutional networks, continues to be a critical concern. This thesis puts an emphasis on preserving the spatiality of samples during dimensionality reduction, proposing a groundbreaking method known as Block-based Principal Component Analysis Pooling (BPCAPooling). It's a PCA-based pooling method aimed at conserving the spatial structure of the samples and ensuring accurate representations of the learned features for the subsequent neural network layers.

This study embraces the application of BPCAPooling in the convolutional neural network architectures for the classification task, with a primary focus on VGG-16. While the BPCAPooling may not outshine traditional methods in terms of performance metrics for classifications such as accuracy and loss, it manifests as a robust alternative especially in the realm of semantic segmentation yielding a mIoU of $0.3333$, accuracy of $86.77%$ and a loss of $0.6659$.

### Architectures

Check out the innovative architectures of U-Nets supplemented with BPCAPooling:

U-Nets with BPCAPooling:
![unet-arch](https://github.com/Lucs1590/USeS-BPCA/blob/4b1e1f338357108eba6e7bb7f31ae515cb8498c5/fixtures/unet-with-bpca.png)

U-Net-Like with BPCAPooling:
![unetlike-arch](https://github.com/Lucs1590/USeS-BPCA/blob/4b1e1f338357108eba6e7bb7f31ae515cb8498c5/fixtures/unet-like-with-bpca-food.png)

### Features

- Harness the power of U-Net and U-Net-Like architectures for semantic segmentation.
- Experience BPCAPooling as a viable alternative to conventional pooling methods.
- Explainability of AI with Xplique.

### Installation

Ready to explore the USeS-BPCA repository? Follow the steps below:

1. Clone the USeS-BPCA repository:

git clone <git@github.com>:Lucs1590/USeS-BPCA.git
2. Install the required packages:

pip install -r requirements.txt
3. Get hands-on with the notebooks folder to run the models.

### Contributing

We welcome your contributions. Feel free to raise issues or create pull requests.

### License

This project is licensed under the terms of the MIT license.

### Citation

If you use this repository in your research, please consider citing it.

### Contact

Got questions? Reach out for any queries via Here put your email.
