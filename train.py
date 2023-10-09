import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
import argparse
from utils import save_preds, eval_preds

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

##HACK: demove df_val
# def train(df_train, df_val, task, epochs, transformer, max_len, batch_size, lr, drop_out, df_results, training_data):
def train(df_train, task, epochs, transformer, max_len, batch_size, lr, drop_out, df_results, training_data):
    
    train_dataset = dataset.TransformerDataset(
        text=df_train[config.COLUMN_TEXT].values,
        target=df_train[config.COLUMN_LABELS + task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )
    
    ##HACK: remove df_val data loading
    # val_dataset = dataset.TransformerDataset(
    #     text=df_val[config.COLUMN_TEXT].values,
    #     target=df_val[config.COLUMN_LABELS + task].values,
    #     max_len=max_len,
    #     transformer=transformer
    # )

    # val_data_loader = torch.utils.data.DataLoader(
    #     dataset=val_dataset, 
    #     batch_size=batch_size, 
    #     num_workers=config.VAL_WORKERS
    # )

    #COMMENT: I may make the number_of_classes simpler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not config.DEVICE or config.DEVICE == 'max' else config.DEVICE
    model = TransforomerModel(transformer, drop_out, number_of_classes=config.UNITS[task]) 
    if config.DEVICE == 'max':
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model.to(device)
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001,},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]

    num_train_steps = int(len(df_train) / batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    # training and evaluation loop
    for epoch in range(1, epochs+1):
        
        no_train, pred_train, _ , loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        ##HACK: 1) ICM-soft is not calculated for training data and 2) save training preds
        # save_preds(no_train, pred_train, df_train, task, training_data, 'training', epoch, transformer)
        # icm_soft_train = eval_preds(task, training_data, 'training', epoch, transformer)
        
        ##HACK: remove df_val preds and evaluation
        # no_val, pred_val, _ , loss_val = engine.eval_fn(val_data_loader, model, device)
        # save_preds(no_val, pred_val, df_val, task, training_data, 'dev', epoch, transformer)
        # icm_soft_val = eval_preds(task, training_data, 'dev', epoch, transformer)
        
        ##HACK: remove icm_soft_train, icm_soft_val and loss
        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            # 'icm_soft_train': icm_soft_train,
                            'loss_train':loss_train,
                            # 'icm_soft_val':icm_soft_val,
                            # 'loss_val':loss_val
                        }, index=[0]
        )
        
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        ##HACK: remove icm_soft_train, icm_soft_val and loss
        # tqdm.write("Epoch {}/{} ICM-soft_training = {:.3f} loss_training = {:.3f} ICM-soft_val = {:.3f}  loss_val = {:.3f}".format(epoch, epochs, icm_soft_train, loss_train, icm_soft_val, loss_val))
        tqdm.write("Epoch {}/{} loss_training = {:.3f}".format(epoch, epochs, loss_train))

        # save models weights
        path_model_save = config.LOGS_PATH + '/model' + '_' + task + '_' + training_data + '_' + transformer.split("/")[-1] + '.pt'
        ##HACK: only sabe model last epoch
        if epoch == epochs:
            torch.save(model, path_model_save)
        # if training_data == 'training-dev' and  epoch == epochs:
        #     # torch.save(model.state_dict(), path_model_save)
        #     torch.save(model, path_model_save)
        # else:
        #     if epoch == 1 or icm_soft_val > df_results['icm_soft_val'][:-1].max():
        #         # torch.save(model.state_dict(), path_model_save)
        #         torch.save(model, path_model_save)

    return df_results


if __name__ == "__main__":
    ##HACK: Now we have fix data for training
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--training_data', type=str, help='Datasets to train the models')
    # args = parser.parse_args()
    
    # if args.training_data == 'training':
    #     datasets = config.DATASET_TRAIN
        
    # elif args.training_data == 'training-dev':
    #     datasets = config.DATASET_TRAIN_DEV
    
    # elif not args.training_data:
    #     print('Specifying --training_data is required')
    #     exit(1)
    
    # else:
    #     print('Specifying --training_data training OR training-dev')
    #     exit(1)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    ##HACK: 1) Use new training datasets and 2) soft_labels_task1 instead of soft_label_task1
    for train_data in tqdm(config.ACIR_TRAIN_FILE, desc='DATASETS', position=0):
    # df_train = pd.read_csv(config.DATA_PATH + '/' + datasets, index_col=None).iloc[:config.N_ROWS]
    # df_train['NO_value'] = df_train['soft_label_task1'].apply(lambda x: eval(x)['NO'])
        df_train = pd.read_csv(config.DATA_PATH + '/EXIST2023_' + train_data + '.csv', index_col=None).iloc[:config.N_ROWS]
        df_train['NO_value'] = df_train['soft_labels_task1'].apply(lambda x: eval(x)['NO'])
        
        ##HACK: df_val is not used
        # df_val = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, index_col=None).iloc[:config.N_ROWS]
        # df_val['NO_value'] = df_val['soft_label_task1'].apply(lambda x: eval(x)['NO']) 

        for transfomer in tqdm(config.TRANSFORMERS, desc='TRANSFORMERS', position=1):
            for task in tqdm(config.LABELS, desc='TASKS', position=2):
                
                ##HACK: epocs
                epochs = config.EPOCHS
                # if args.training_data == 'training-dev':
                #     df_info = pd.read_csv(config.LOGS_PATH + '/training_' + task + '_' + transfomer + '.csv')
                #     epochs = df_info.at[df_info['icm_soft_val'].idxmax(), 'epoch']
                # else:
                #     epochs = config.EPOCHS
                
                ##HACK: remove icm_soft_train, icm_soft_val and loss
                df_results = pd.DataFrame(columns=['task',
                                        'epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        # 'icm_soft_train',
                                        'loss_train',
                                        # 'icm_soft_val',
                                        # 'loss_val'
                                        ])
                ##HACK:
                # tqdm.write(f'\nTask: {task} Data: {args.training_data} Transfomer: {transfomer.split("/")[-1]} Max_len: {config.MAX_LEN} Batch_size: {config.BATCH_SIZE} Dropout: {config.DROPOUT} lr: {config.LR}')
                tqdm.write(f'\nTask: {task} Data: {train_data} Transfomer: {transfomer.split("/")[-1]} Max_len: {config.MAX_LEN} Batch_size: {config.BATCH_SIZE} Dropout: {config.DROPOUT} lr: {config.LR}')
                
                ##HACK:
                df_results = train(df_train,
                                    # df_val,
                                    task,
                                    epochs,
                                    transfomer,
                                    config.MAX_LEN,
                                    config.BATCH_SIZE,
                                    config.LR,
                                    config.DROPOUT,
                                    df_results,
                                    # args.training_data
                                    train_data
                )
                
                ##HACK: change file name
                # df_results.to_csv(config.LOGS_PATH + '/' + args.training_data + '_' + task + '_' + transfomer + '.csv', index=False)
                df_results.to_csv(config.LOGS_PATH + '/' + train_data + '_' + task + '_' + transfomer + '.csv', index=False)