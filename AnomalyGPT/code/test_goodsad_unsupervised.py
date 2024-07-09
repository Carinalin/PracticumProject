# unsupervised learning test

import os
import random
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

describles = {}
describles['drink_bottle'] = "This is a photo of a drink bottle for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['drink_can'] = "This is a photo of a drink can for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_bottle'] = "This is a photo of a food bottle for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_box'] = "This is a photo of a food box for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_package'] = "This is a photo of a food package for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['cigarette_box'] = "This is a photo of a cigarette box for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."


# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_goodsad_default/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

def predict(
    input, 
    image_path, 
    normal_img_path, 
    max_length, 
    top_p, 
    temperature,
    history,
    modality_cache,  
):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return response, pixel_output

input = "Is there any anomaly in the image?"
root_dir = '../data/GoodsAD'

mask_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])

CLASS_NAMES = ['drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package', 'cigarette_box']
res = pd.DataFrame(columns=['Right', 'Wrong', 'ACC', 'i_AUROC', 'p_AUROC'], 
                   index = ['drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package', 'cigarette_box'])

for c_name in CLASS_NAMES:
    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'jpg' in file and c_name in file_path and '.ipynb_checkpoints' not in file_path:
                file_paths.append(file_path)
    for file_path in tqdm(file_paths):
        resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, [], 512, 0.1, 1.0, [], [])
        is_normal = 'good' in file_path.split('/')[-2]

        if is_normal:
            img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
        else:
            mask_path = file_path.replace('test', 'ground_truth')
            mask_path = mask_path.replace('.jpg', '.png')
            img_mask = Image.open(mask_path).convert('L')                

        img_mask = mask_transform(img_mask)
        img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
        img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()

        anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

        p_label.append(img_mask)
        p_pred.append(anomaly_map)

        i_label.append(1 if not is_normal else 0)
        i_pred.append(anomaly_map.max())

        position = []

        if 'good' not in file_path and 'Yes' in resp:
            right += 1
        elif 'good' in file_path and 'No' in resp:
            right += 1
        else:
            wrong += 1

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100,2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100,2)
    acc = 100 * right / (right + wrong)
    
    res.loc[c_name, 'ACC'] = acc
    res.loc[c_name, 'Right'] = right
    res.loc[c_name, 'Wrong'] = wrong
    res.loc[c_name, 'i_AUROC'] = i_auroc
    res.loc[c_name, 'p_AUROC'] = p_auroc

    print(c_name)
    print(res.loc[c_name])

print(res)
res.to_csv ('../results/result_of_test_goodsad_default.csv',index=True)
