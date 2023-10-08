import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
from utils import save_preds

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

##HACK: add collection
def test(df_test, task, transformer, max_len, batch_size, drop_out, collection):
# def test(df_test, task, transformer, max_len, batch_size, drop_out):
    
    train_dataset = dataset.TransformerDataset_Test(
        text=df_test[config.COLUMN_TEXT].values,
        no_task1 = df_test['NO_value'].values if 'NO_value' in df_test.columns else None,
        max_len=max_len,
        transformer=transformer
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not config.DEVICE or config.DEVICE == 'max' else config.DEVICE
    # model = TransforomerModel(transformer, drop_out, number_of_classes=config.UNITS[task])
    # model.load_state_dict(torch.load(config.LOGS_PATH + '/model_' + task + '_training-dev_' + transfomer + '.pt'))
    # model = torch.load(config.LOGS_PATH + '/model_' + task + '_training-dev_' + transfomer + '.pt')
    # if config.DEVICE == 'max':
    #     model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    device = 'cuda:0'
    #TODO: add diferent models for each task
    #TODO: models must have train data on the file's name
    ##HACK:
    # model = torch.load(config.LOGS_PATH + '/model_' + task + '_training-dev_' + transfomer + '.pt')
    model = torch.load(config.LOGS_PATH + '/model_' + task + '_' + collection + '_' + transfomer + '.pt')
    model.to(device)
    
    # prediction
    no_train, pred_train = engine.test_fn(train_data_loader, model, device)
    #TODO: preds must have train data ans test data on the file's name
    ##HACK:
    # save_preds(no_train, pred_train, df_test, task, '#####', 'test', '#####', transformer)
    save_preds(no_train, pred_train, df_test, task, collection, 'test', '#####', transformer)
    
    return

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    for transfomer in tqdm(config.TRANSFORMERS, desc='TRANSFORMERS', position=0):
        for task in tqdm(config.LABELS, desc='TASKS', position=1):
            ##HACK: add new dev datasets
            for data_for_pred in config.ACIR_DEV_FILE:
                collenction = data_for_pred.split('_')[-1].split('.')[0]
            
                ##HACK: swap config.DATASET_TEST with data_for_pred
                # df_test = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TEST, index_col=None).iloc[:config.N_ROWS]
                df_test = pd.read_csv(config.DATA_PATH + '/' + data_for_pred, index_col=None).iloc[:config.N_ROWS]
                
                if task != 'task1':
                    #TODO: preds must have train data ans test data on the file's name
                    ##HACK:
                    # df_task1 = pd.read_json(config.LOGS_PATH + '/task1_#####_test_#####_' + transfomer + '.json', orient='index')
                    df_task1 = pd.read_json(config.LOGS_PATH + '/task1' + '_' + collenction + '_test_#####_' + transfomer + '.json', orient='index')   
                    df_task1['NO_value'] = df_task1['soft_label'].apply(lambda x: x['NO'])
                    df_task1 = df_task1.reset_index()
            
                    df_test = pd.concat([df_test, df_task1], axis=1)
                    
                tqdm.write(f'\nTask: {task} Data: Test  Transfomer: {transfomer.split("/")[-1]} Max_len: {config.MAX_LEN} Batch_size: {config.BATCH_SIZE}')
                
                test(df_test,
                        task,
                        transfomer,
                        config.MAX_LEN,
                        config.BATCH_SIZE,
                        config.DROPOUT,
                        ##HACK: add collection
                        collenction
                )