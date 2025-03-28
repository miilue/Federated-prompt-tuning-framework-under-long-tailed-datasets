# A framework for federated prompt learning under long-tailed datasets

<!-- # Don't Only Fine-Tune Majority: Federated Gradient-Balanced Prompt Learning for Long-Tailed Heterogeneous Data          This is an official implementation of the following paper: >Don't Only Fine-Tune Majority: Federated Gradient-Balanced Prompt Learning for Long-Tailed Heterogeneous Data -->


<!-- **Abstract**: Federated prompt learning (FPL) has recently shown great potential by adapting the pre-trained visual-language models to federated learning settings via prompt tuning. However, FPL still suffers from heterogeneous long-tailed class distributions across real-world scenarios. If fine-tuning only favors the majority class, model training will overfit and converge to sub-optimal. To address this issue, we propose a new federated gradient-balanced prompt learning framework, named FedGPA. Specifically, we first analyze and evaluate the gradient disparities between head and tail classes in FPL. We then design a Gradient-Adaptive Prompt Adjuster (GPA) that adaptively adjusts the gradients for each class during client-side prompt learning, according to the classification header weights of the current server-side model. In addition, FedGPA could mitigate the over-discouraged tail classes during prompt tuning, which achieves class-balanced prompt tuning. Extensive experiments on representative datasets (CIFAR-100, OxfordPets, DTD) demonstrate the effectiveness of FedGPA. Compared with state-of-the-art baselines (CLIP2FL, Fed-GraB), FedGPA improve the accuracy of tail classes by 4.4% and 12.7% on the CIFAR-100 dataset with imbalance factors of 50 and 100, respectively. -->

## Dependencies

- Python 3.9.0
- CUDA 11.8 
- PyTorch 2.1.2
- torchvision 0.16.2
- timm 1.0.7

## Parameters

| **parameters**    | **description**                                       |
| ----------------- | ----------------------------------------------------- |
| `rounds`          | Number of rounds                                      |
| `num_users`       | Number of clients                                     |
| `local_ep`        | Number of local epochs                                |
| `local_bs`        | Batch size for local training                         |
| `dataset`         | Dataset, option: `cifar100`, `oxford_pets`, and `dtd` |
| `iid`             | `Action` iid or non iid, option: `store_true`         |
| `alpha_dirichlet` | Parameter for Dirichlet distribution                  |
| `IF`              | Imbalance factor: Min/Max                             |
| `varphi`          | Weight coefficient $\varphi$                          |
| `prompt_token`    | Number of prompt tokens                               |


## Usage

Make sure to add ViT-B/16 pretrained model file `pytorch_model.bin` to the `FedGPA/models` directory before reproducing.

- To train on CIFAR-100 with imbalanced factor 100:

```
python -u fedgpa.py --dataset cifar100 --num_users 10 --local_ep 2 --alpha_dirichlet 0.5 --model ViT --prompt_token 10 --IF 0.01 --gpu 0
```

- To train on OxfordPets with imbalanced factor 100:

```
python -u fedgpa.py --dataset oxford_pets --num_users 10 --local_ep 2 --alpha_dirichlet 0.5 --model ViT --prompt_token 10 --IF 0.01 --gpu 0
```

- To train on DTD with imbalanced factor 100:

```
python -u fedgpa.py --dataset dtd --num_users 10 --local_ep 2 --alpha_dirichlet 0.5 --model ViT --prompt_token 10 --IF 0.01 --gpu 0
```
