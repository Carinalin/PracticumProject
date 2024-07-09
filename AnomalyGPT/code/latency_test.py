# test with few-shot and zero-shot via model trained with other datasets

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
from time import time

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)
parser.add_argument("--model", type=str, default="train_supervised")


command_args = parser.parse_args()


print(command_args)



describles = {}
describles['drink_bottle'] = "This is a photo of a drink bottle for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['drink_can'] = "This is a photo of a drink can for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_bottle'] = "This is a photo of a food bottle for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_box'] = "This is a photo of a food box for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_package'] = "This is a photo of a food package for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['cigarette_box'] = "This is a photo of a cigarette box for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

FEW_SHOT = command_args.few_shot

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/'+command_args.model+'/pytorch_model.pt', # train_mvtec & train_visa & train_supervised
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

print(args['anomalygpt_ckpt_path'])
# exit(-1)

model = OpenLLAMAPEFTModel(**args)
print(f'basic model is ready')
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
print(f'load pandagpt_ckpt')
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
print(f'load anomalygpt_ckpt')
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

"""Override Chatbot.postprocess"""


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

CLASS_NAMES = ['cigarette_box', 'drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package']

times = []
for c_name in CLASS_NAMES:
    
    normal_img_folder_path = "../data/GoodsAD/"+c_name+"/train/good/"
    all_imgs = os.listdir(normal_img_folder_path)
    all_imgs = [i for i in all_imgs if i != '.ipynb_checkpoints']
    random_imgs = random.sample(all_imgs, 32)
    normal_img_paths = [normal_img_folder_path+random_imgs[i] for i in range(32)]
    normal_img_paths = normal_img_paths[:command_args.k_shot]
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
    for file_path in tqdm(file_paths[:50]):
        if FEW_SHOT:
            start = time()
            resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, normal_img_paths, 512, 0.1, 1.0, [], [])
            timed = time() - start
        else:
            start = time()
            resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, [], 512, 0.1, 1.0, [], [])
            timed = time() - start
        times.append(timed)

print(len(times))
print(np.mean(times))