# Practicum_Project_On_The_GoodsAD_Dataset
This project deeply explored [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT), a novel anomaly detection model, on the [GoodsAD](https://github.com/jianzhang96/GoodsAD) dataset. Besides, another four wonderful models were also explored. The `Experiment_Results` directory records the results on each model. The code for each model was modified from the below repositories. Many thanks for these great works.

| Model       | Github Code Source       | Implementaion Type |
| ----------- | ------------------------ | ------------------ |
| AnomalyGPT  | CASIA-IVA-Lab/AnomalyGPT | official           |
| WinCLIP     | caoyunkang/WinClip       | unofficial         |
| PromptAD    | FuNz-0/PromptAD          | official           |
| EfficientAD | nelson1425/EfficientAD   | unofficial         |
| DiffusionAD | HuiZhang0812/DiffusionAD | official           |

## 1. Data Preparation

1. GoodsAD: https://github.com/jianzhang96/GoodsAD
2. MVtec-AD: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
3. VisA: https://github.com/amazon-science/spot-diff
4. Pre-training data of PandaGPT: https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/tree/main
5. DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/
6. goodsad-DIS: https://drive.google.com/drive/folders/1r8bUvmqYio65JIFPEiL9RwiuI6_FG2E4?usp=drive_link

## 2. Prepare Weights

### 2.1. Weights For AnomalyGPT

> The detailed instructions for downloading weights can be found in chapters 2.2-2.4 of [this file](https://github.com/CASIA-IVA-Lab/AnomalyGPT/blob/main/README.md).

1. **ImageBind Checkpoint**: download the pre-trained ImageBind model using [this link](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). After downloading, put the downloaded file (imagebind_huge.pth) in `AnomalyGPT/pretrained_ckpt/imagebind_ckpt/` directory.
2. **Vicuna Checkpoint**: To prepare the pre-trained Vicuna model, please follow the instructions provided [[here\]](https://github.com/CASIA-IVA-Lab/AnomalyGPT/blob/main/pretrained_ckpt#1-prepare-vicuna-checkpoint). After downloading, put the downloaded file (imagebind_huge.pth) in `AnomalyGPT/pretrained_ckpt/vicuna_ckpt/7b_v0/ `directory.
3. **Delta Weights of AnomalyGPT**: Download PandaGPT weights from [here](https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024). After downloading, put the downloaded file (imagebind_huge.pth) in `AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/` directory.

### 2.2. Weights For EfficientAD

The weights trained on GoodsAD can be downloaded from [here](https://drive.google.com/file/d/1oPjFpZ-3z0lpO6YNnOpqlWWvmCldw7Fe/view?usp=drive_link). After downloading and unzipping, put the weights in `EfficientAD/output` directory.

### 2.3. Weights For DiffusionAD

The weights trained on GoodsAD can be downloaded from [here](https://drive.google.com/file/d/1TJ3nJhe-VswIg95u9dxVkfOUIcohvEU6/view?usp=drive_link). After downloading and unzipping, put the weights in `DiffusionAD/outputs` directory.

## 3. Run Models
Please follow the `README.md` file of each model.



