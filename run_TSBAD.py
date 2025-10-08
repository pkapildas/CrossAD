import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from exp.exp_TSBAD import Exp_Anomaly_Detection

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

def run_CrossAD(data_train, data_test, args, id):
    clf = Exp_Anomaly_Detection(args, id)
    clf.train(data_train)
    score = clf.test(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

if __name__ == '__main__':
    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/TSB-AD/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='./dataset/TSB-AD/File_List/TSB-AD-U-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='./test_results/TSB-AD-U/score/')
    parser.add_argument('--save_dir', type=str, default='./test_results/TSB-AD-U/')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--AD_Name', type=str, default='CrossAD')

    # basic config
    parser.add_argument('--configs_path', type=str, default="./configs/TSB-AD-U/", help='')

    parser.add_argument('--data', type=str, default='', help='dataset type')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_(TSB-AD-U)', help='')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    df = pd.read_csv(f"{args.configs_path}/TSB-AD-U_setting_draft.csv")
    setting = dict(zip(df.iloc[:,0], df.iloc[:,1]))

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    
    write_csv = []
    for filename in file_list:
        if os.path.exists(target_dir+'/'+filename.split('.')[0]+'.npy'): continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        args.data = filename
        model_id = setting[filename]
        
        Optimal_Det_HP = {'args': args, 'id': model_id}

        output = run_CrossAD(data_train, data, **Optimal_Det_HP)

        end_time = time.time()
        run_time = end_time - start_time

        if isinstance(output, np.ndarray):
            logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)
        else:
            logging.error(f'At {filename}: '+output)

        ### whether to save the evaluation result
        if args.save:
            try:
                label = label[:len(output)]
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except:
                list_w = [0]*9
            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'Time')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            os.makedirs(args.save_dir, exist_ok = True)
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)