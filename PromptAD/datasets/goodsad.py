import glob
import os
import random

goodsad_classes = ['cigarette_box', 'drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package']


GOODSAD_DIR = './datasets/GoodsAD'


def load_goodsad(category, k_shot):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '.png') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def load_phase_train(root_path, gt_path, training_ind):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_type = 'good'
        for idx in training_ind:
            img_path = os.path.join(root_path, defect_type, idx+".jpg")
            # print(img_path)
            img_tot_paths.append(img_path)
        gt_tot_paths.extend([0] * len(img_tot_paths))
        tot_labels.extend([0] * len(img_tot_paths))
        tot_types.extend(['good'] * len(img_tot_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    
    assert category in goodsad_classes

    test_img_path = os.path.join(GOODSAD_DIR, category, 'test')
    train_img_path = os.path.join(GOODSAD_DIR, category, 'train')
    ground_truth_path = os.path.join(GOODSAD_DIR, category, 'ground_truth')

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    seed_file = os.path.join('./datasets/seeds_goodsad', category, 'selected_samples_per_run.txt')
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]
            training_indx = strip_line.split(' ')
            
    selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, \
    selected_train_tot_types = load_phase_train(train_img_path, ground_truth_path, training_indx)

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
