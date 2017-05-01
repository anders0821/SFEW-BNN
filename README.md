# SFEW-BNN
Codes for paper "An Efficient Unconstrained Facial Expression Recognition Algorithm based on Stack Binarized Auto-encoders and Binarized Neural Networks"

This project is based on https://github.com/MatthieuCourbariaux/BinaryNet

# Abstract of the Paper
Although deep learning methods have achieved good performances in many pattern recognition tasks,
the over-fitting problem is always a serious issue for training deep networks containing large sets of parameters with limited labeled data.
In this work, Binarized Auto-encoders (BAEs) and Stacked Binarized Auto-encoders (Stacked BAEs) are proposed to learn a kind of domain knowledge from large-scale unlabeled facial datasets.
By transferring the knowledge to another Binarized Neural Networks (BNNs)  based supervised learning task with limited labeled data, the performance of the BNNs can be improved.
A real-world facial expression recognition system is constructed by combining an unconstrained face normalization method, a variant of LBP descriptor, BAEs and BNNs.
into a real-world expression recognition system.
The experiment result shows that the whole system achieves good performance on the Static Facial Expressions in the Wild (SFEW) benchmark with minimal hardware requirements and lower memory and computation costs.