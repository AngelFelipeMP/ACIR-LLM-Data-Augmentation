import config
import pandas as pd

for task in config.LABELS:
    for data_for_pred in config.ACIR_DEV_FILE:
        df_json = pd.read_csv(config.DATA_PATH + '/' + config.DATA + '_' + data_for_pred + '.csv', index_col='id_EXIST')
        
        size_dev_pred = set()
        
        print(f'{task.upper()} | Preds: {data_for_pred}')
        
        for train_data in config.ACIR_TRAIN_FILE:
            df_preds = pd.read_json(config.LOGS_PATH + '/results_send_Damiano/' + task + '_' + train_data + '_' + data_for_pred + '_ensemble' + '.json', orient='index')

            print('     Train data:', train_data)
            
            # check size dev data and size preds
            size_dev_pred.add(len(df_json) == len(df_preds))
            print('     Size_dev_iqual_preds: ', size_dev_pred)
    
            # check ids dev data and ids preds
            are_indices_equal = df_json.index.equals(df_preds.index)
            print('     Index_dev_iqual_preds: ', are_indices_equal)
            
            if task!= 'task3':
                # df_preds['soft_label'] = df_preds['soft_label'].apply(lambda x: eval(x))
                df_preds['sum_to_1'] = df_preds['soft_label'].apply(lambda x: sum(x.values()))
                df_preds['sum_to_1'] = df_preds['sum_to_1'].apply(lambda x: 1 if x > 0.99 else 0)
                print('  sum_to_1: ', df_preds['sum_to_1'].unique())
            print('\n')