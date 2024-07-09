import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)  # 

    # datasets = ['mvtec','visa']
    datasets = ['goodsad']
    k = 32 # 0,1,4,8,32
    for dataset in datasets:

        classes = dataset_classes[dataset]

        for cls in classes[:]:

            sh_method = f'python eval_WinCLIP.py ' \
                        f'--dataset {dataset} ' \
                        f'--class-name {cls} ' \
                        f'--k-shot {k}' \
            
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()  # 等待进程结束

