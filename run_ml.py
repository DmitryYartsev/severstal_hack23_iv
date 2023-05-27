import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import copy
import torch
import os
import datetime
import psycopg2
import warnings



# Установка сидов
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# Параметры для подключения к БД
warnings.filterwarnings('ignore')

dbname = 'plant_database'
user = 'user'
host = '172.27.0.5'
password = "pass"
port = 5432

db_params = {'dbname':dbname, 'user':user, 'host':host, 'password':password, 'port':port}




# Вспомогательные функции для работы с БД
def make_sql_req(sql_req):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute(sql_req)
        conn.commit()
        count = cursor.rowcount
        cursor.close()
        conn.close()
    except Exception as e: 
        print(e)
        
def create_tab(df, dtypes_dict, table_name):
    dtypes_ = df.dtypes
    dtypes_ = dtypes_.astype(str).map(dtypes_dict).reset_index().values
    sql_req = f'CREATE TABLE {table_name} ('+', '.join([f'"{i[0]}"'+' '+i[1] for i in dtypes_]) + ')'
    make_sql_req(sql_req)

def insert_init_data(df, table_name):
    df = df.fillna('NULL')
    df = df.astype(str)
    cols = '(' + ', '.join([f'"{str(i)}"' for i in df.columns]) + ')'
    vals = ', '.join([', '.join(str(i).split(' ')) for i in df.values])
    vals = vals.replace('[', '(').replace(']', ')')
    vals = vals.replace("'NULL'", 'NULL')
    sql_req = f"INSERT INTO {table_name} {cols} VALUES {vals};"
    make_sql_req(sql_req)
    
def set_primary_key(key_col, table_name):
    sql_req = f"ALTER TABLE {table_name} ADD PRIMARY KEY ({key_col});"
    make_sql_req(sql_req)
    
def set_foreign_key(constraint_name, col_source, col_foreign, table_source, table_foreign):
    sql_req = f"""ALTER TABLE {table_foreign}
    ADD CONSTRAINT {constraint_name} 
    FOREIGN KEY ({col_foreign}) 
    REFERENCES {table_source} ({col_source});"""
    make_sql_req(sql_req)

def get_telemetry_data():
    datetime_end = datetime.datetime.now()
    datetime_start = datetime_end - pd.Timedelta(days=1)
    datetime_end = str(datetime_end)
    datetime_start = str(datetime_start)
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql(f'select * from raw.sensor_telemetry where "DT" between \'{datetime_start}\' and \'{datetime_end}\'', conn)
    conn.close()
    
    data = {}
    for agg_num in df['agg_num'].unique():
        data[agg_num] = {}
        data[agg_num]['data'] = df.loc[df['agg_num']==agg_num].set_index('DT').sort_index().copy()
    del df
    return data


# Чтение real-time данных

# загрузка моделей

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(LogisticRegression, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU() 
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear_2(self.relu(self.linear_1(x))))
        return outputs
    
    
agg_l = [4,5,6,7,8,9]
models_m3 = {}
params_m3 = {}
for i in os.listdir('models_m3'):
    agg_num = int(i.split('_')[-1].split('.')[0])
    models_m3[f'm3_{agg_num}'] = torch.load(f'models_m3/{i}')
    params_m3[agg_num] = {}
    params_m3[agg_num]['x_cols'] = models_m3[f'm3_{agg_num}'].x_cols
    params_m3[agg_num]['means'] = models_m3[f'm3_{agg_num}'].means
    params_m3[agg_num]['std'] = models_m3[f'm3_{agg_num}'].std
    params_m3[agg_num]['thresholds'] = {}
    params_m3[agg_num]['thresholds']['min'] = pd.Series(models_m3[f'm3_{agg_num}'].thresholds_min)
    params_m3[agg_num]['thresholds']['max'] = pd.Series(models_m3[f'm3_{agg_num}'].thresholds_max)
    params_m3[agg_num]['y_proc_columns'] = models_m3[f'm3_{agg_num}'].y_proc_columns
    params_m3[agg_num]['y_raw_columns'] = models_m3[f'm3_{agg_num}'].y_raw_columns


## Функции для генерации фичей
def generate_features_m3(agg_num, data, table = 'data'):
    # используется несколько периода временной агрегации 
    win_sizes = ['15min', '2h']
    x = pd.DataFrame()
    cols = list(data[agg_num]['thresholds']['min'].index)
    for n, ws in enumerate(win_sizes):
        # Сколько фичей за выбранный период агрегации были меньше чем квантиль 0.035
        min_score = (data[agg_num][table][cols]<data[agg_num]['thresholds']['min']).rolling(ws).sum()
        min_score.columns = [f'{i}_{ws}_min_stats' for i in min_score.columns]
        # Максимальный переход между соседними значениями за выбранный период
        diff_score = data[agg_num][table][cols].diff().abs().rolling(ws).max()
        diff_score.columns = [f'{i}_{ws}_diff_stats' for i in diff_score.columns]
        # Std показателя за выбранный период
        std_score = sum_score = data[agg_num][table][cols].rolling(ws).std()
        std_score.columns = [f'{i}_{ws}_std_stats' for i in std_score.columns]
        if n==0:
            x = pd.concat([min_score, diff_score, std_score], axis=1)
        else:
            x = pd.concat([x, min_score, diff_score, std_score], axis=1)
        del min_score, diff_score, std_score

    # Сколько фичей принимали значения nan
    nan_score = data[agg_num][table].isna().sum(axis=1)
    data[agg_num][table] = x
    data[agg_num][table]['nan_score'] = nan_score
    return data[agg_num][table]

def preprocessing_pipeline_m3(data):
    cols_to_drop = ['ТОК РОТОРА 2']
    for agg_num in agg_l:
        data[agg_num]['data'].drop(columns = cols_to_drop, inplace=True)
        data[agg_num]['data'] = generate_features_m3(agg_num, data, table = 'data')
        data[agg_num]['data'] = data[agg_num]['data'][data[agg_num]['x_cols']]
        data[agg_num]['data'] = (data[agg_num]['data'] - data[agg_num]['means'])/data[agg_num]['std']
        median = data[agg_num]['data'].median()
        data[agg_num]['data'] = data[agg_num]['data'].fillna(method='ffill').fillna(median)
    return data

status_m3_dict = {0:'OK', 2:'M3'}
def insert_preds(data):
    for agg_num in agg_l:
        x = torch.Tensor(data[agg_num]['data'][data[agg_num]['x_cols']].values)[-1:]
        pred = models_m3[f'm3_{agg_num}'](x).round().detach().numpy()*2
        pred = pred.astype('uint8')[0]
        status = [status_m3_dict[i] for i in pred]
        pred = pd.DataFrame()
        pred['status'] = status
        pred['tm'] = data[agg_num]['y_proc_columns']
        pred['aggregate_id'] = agg_num
        pred['upd_time'] = datetime.datetime.now()
        conn = psycopg2.connect(**db_params)
        nrows = pd.read_sql(f'select count(*) from ods.m3_tm_status where aggregate_id={agg_num}', conn).values[0][0]
        conn.close()
        if nrows==0:
            insert_init_data(pred, 'ods.m3_tm_status')
        else:
            for n, i in pred.iterrows():
                q = f"""UPDATE ods.m3_tm_status SET status='{i['status']}', upd_time='{str(i['upd_time'])}'  WHERE aggregate_id={i['aggregate_id']} and tm='{i['tm']}';"""
                make_sql_req(q)

        
while True:
    data = get_telemetry_data()
    ## ========================= Предсказания для M3 ==================================
    for agg_num in agg_l:
        for k,v in params_m3[agg_num].items(): data[agg_num][k] = v
    data_m3 = copy.deepcopy(data)
    data_m3 = preprocessing_pipeline_m3(data_m3)
    insert_preds(data_m3)
    ## ========================= Предсказания для M1 ==================================



