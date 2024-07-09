import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    # datasets = ['mvtec', 'visa']
    datasets = ['goodsad']
    # shots = [1, 2, 4]
    shots = [1, 4, 8, 32]

    for shot in shots:
        print("k_shot: ", shot)
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes[:]:
                sh_method = f'python train_seg.py ' \
                            f'--dataset {dataset} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \

                print(sh_method)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




