# This script is modified from https://github.com/CASIA-IVA-Lab/AnomalyGPT/blob/main/code/datasets/mvtec.py

import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .self_sup_tasks import patch_ex

# The dataset contains 484 different appearance goods divided into 6 categories. 
CLASS_NAMES = {'drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package', 'cigarette_box'}

describles = {}
# the describes were modified to present normal and abnormal semantics in supermarkers goods
describles['drink_bottle'] = "This is a photo of a drink bottle for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['drink_can'] = "This is a photo of a drink can for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_bottle'] = "This is a photo of a food bottle for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_box'] = "This is a photo of a food box for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['food_package'] = "This is a photo of a food package for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['cigarette_box'] = "This is a photo of a cigarette box for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

'''
try to Set parameters for anomaly simulation: 
width_bounds_pct ((float, float), (float, float)): min half-width of patch ((min_dim1, max_dim1), (min_dim2, max_dim2))
skip_background (int, int) or [(int, int),]: optional, assume background color is first and only interpolate patches in areas where dest or src patch has pixelwise MAD < second from background.
intensity_logistic_params (float, float): k, x0 of logitistc map for intensity based label
explanation of other parameters see: 
https://github.com/CASIA-IVA-Lab/AnomalyGPT/blob/main/code/datasets/self_sup_tasks.py

WIDTH_BOUNDS_PCT = {'drink_bottle':((0.03, 0.4), (0.03, 0.2)), 'drink_can':((0.03, 0.4), (0.03, 0.3)), 'food_bottle':((0.03, 0.4), (0.03, 0.3)), 'food_box':((0.03, 0.35), (0.03, 0.35)), 'food_package':((0.03, 0.4), (0.03, 0.4)), 'cigarette_box':((0.03, 0.4), (0.03, 0.3))}
'''

class GoodsDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.transform = transforms.Resize(
                                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )

        self.paths = []
        self.x = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train" in file_path and "good" in file_path and '.jpg' in file and '.ipynb_checkpoints' not in file_path:
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))
        # get a random image idx for training
        self.prev_idx = np.random.randint(len(self.paths))
    
    # return the number of training images
    def __len__(self):
        return len(self.paths)
    
    # 
    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        # class_name = img_path.split('/')[-4]
        path_set = set(img_path.split('/'))
        class_name = (path_set&CLASS_NAMES).pop()
        
        if class_name not in CLASS_NAMES:
            print('no class name was found.')
            print('\n')
            print(path_set)

        self_sup_args={'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                    'intensity_logistic_params': (1/12, 24),
                    'num_patches': 2,
                    'min_object_pct': 0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'shift':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE, 
                    'label_mode':'logistic-intensity',
                    'skip_background': None,
                    'resize_bounds': (.5, 2)
                    }

        x = np.asarray(x)
        origin = x

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)
        # return patchex, label, label_centers
        x, mask, centers = patch_ex(x, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index
        

        # origin：normal image, x: abnormal image with simulated anomaly
        origin = self.norm_transform(origin)
        x = self.norm_transform(x)
        
        # locate the simulated anomaly if exist
        if len(centers) > 0:
            position = []
            for center in centers:
                center_x = center[0] / 224
                center_y = center[1] / 224

                if center_x <= 1/3 and center_y <= 1/3:
                    position.append('top left')
                elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('top')
                elif center_x <= 1/3 and center_y > 2/3:
                    position.append('top right')

                elif center_x <= 2/3 and center_y <= 1/3:
                    position.append('left')
                elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('center')
                elif center_x <= 2/3 and center_y > 2/3:
                    position.append('right')

                elif center_y <= 1/3:
                    position.append('bottom left')
                elif center_y > 1/3 and center_y <= 2/3:
                    position.append('bottom')
                elif center_y > 2/3:
                    position.append('bottom right')

            conversation_normal = []
            conversation_normal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})
            


            conversation_abnormal = []
            conversation_abnormal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})


            # abnormal conversation to tell the locations of anomalies
            if len(centers) > 1:
                abnormal_describe =  "Yes, there are " + str(len(centers)) + " anomalies in the image, they are at the "
                for i in range(len(centers)):
                    if i == 0:
                        abnormal_describe += position[i]

                    elif i == 1 and position[i] != position[i-1]:
                        if i != len(centers) - 1:
                            abnormal_describe += ", "
                            abnormal_describe += position[i]
                        else:
                            abnormal_describe += " and " + position[i] + " of the image."
                    
                    elif i == 1 and position[i] == position[i-1]:
                        if i == len(centers) - 1:
                            abnormal_describe += " of the image."

            else:
                abnormal_describe = "Yes, there is an anomaly in the image, at the " + position[0] + " of the image."

            conversation_abnormal.append({"from":"gpt","value":abnormal_describe})
        # report if no simulated anomaly yield
        else:
            print("no mask")
            conversation_normal = []
            conversation_normal.append({"from":"human","value":describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})

            conversation_abnormal = conversation_normal

        return origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path

    
    # images: [origin images, x]
    # texts: [conversation_normal, conversation_abnormal]
    # class_names: [class_name, class_name]
    # masks: [torch[0], masks]
    # img_paths:[img_path, img_path]
    
    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []
        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[4])
            masks.append(torch.zeros_like(instance[5]))
            img_paths.append(instance[6])

            images.append(instance[2])
            texts.append(instance[3])
            class_names.append(instance[4])
            masks.append(instance[5])
            img_paths.append(instance[6])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )