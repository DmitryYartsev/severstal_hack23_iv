import pandas as pd
import numpy as np
import os
import psycopg2
import time
import warnings
import gdown
import datetime



warnings.filterwarnings('ignore')

dbname = 'plant_database'
user = 'user'
host = '172.27.0.5'
password = "pass"
port = 5432

db_params = {'dbname':dbname, 'user':user, 'host':host, 'password':password, 'port':port}


## Вспомогательные функции для создание таблиц и загрузки сырых данных
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

def read_sql(sql_req):
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql(f'select * from {sql_req}', conn)
    conn.close()
    return df

dtypes_dict = {'object':'varchar', 'int64':'int', 'int32':'int',\
               'datetime64[ns]':'timestamp', 'float64':'numeric'}



## Создание схем
make_sql_req('DROP SCHEMA raw CASCADE')
make_sql_req('DROP SCHEMA ods CASCADE')
make_sql_req('CREATE SCHEMA raw')
make_sql_req('CREATE SCHEMA ods')
print('======Созданы схемы ODS и RAW======')


# Загрузка сырых данных

# !! Вместо генерации сырых данных, скрипт скачает заготовку с гугл диска. 
# При  необхоимости заготовку для сырых данных можно изменить в закоменитированном коде
# Код для создание заготовок использует данные, обработанные ноутбуком Split_data_0.ipynb в папке dataset_splited
# =============================================================================
# aggregates_list = [4,5,6,7,8,9]
# start_date = pd.to_datetime('2020-06-12')
# end_date = pd.to_datetime('2020-07-13')

# def proc_columns_x(df):
#     cols = pd.DataFrame(df.columns, columns = ['col'])
#     cols.loc[cols['col']!='target', 'num'] = cols.loc[cols['col']!='target', \
#                                                             'col'].str.extract('ЭКСГАУСТЕР (\d)*').astype(int).values
#     cols['col_n'] = cols['col'].str.replace('ЭКСГАУСТЕР (\d).', '', regex=True).str.strip(' ')
#     df['agg_num'] = cols['num'].mean().astype(int)
#     df = df.rename(columns = {i['col']:i['col_n'] for m, i in cols.iterrows()})
#     return df
    
# df_x_executing = pd.DataFrame()
# for n, agg_num in enumerate(aggregates_list):
    
#     df_tab_x = pd.read_parquet(f'ml_backend_service/dataset_train_splited/x_train_{agg_num}.parquet')
#     df_tab_y = pd.read_parquet(f'ml_backend_service/dataset_train_splited/y_train_{agg_num}.parquet')
    
#     df_tab_x = df_tab_x[start_date:end_date]
#     df_tab_y = df_tab_y[start_date:end_date]
    
#     df_tab_x = proc_columns_x(df_tab_x)
#     df_tab_y.columns = df_tab_y.columns.str.replace('Y_ЭКСГАУСТЕР А/М №(\d)_', '', regex=True)\
#                 .str.replace('ЭКСГ.№(\d)', '', regex=True)\
#                 .str.replace('ЭКСГ. №(\d)', '', regex=True)\
#                 .str.replace('ЭКСГ.  №(\d)', '', regex=True)\
#                 .str.replace('А/М №(\d)', '', regex=True)\
#                 .str.replace('А/М№(\d)', '', regex=True)\
#                 .str.strip()
    
#     df_tab_x['agg_num'] = agg_num
#     df_tab_y['agg_num'] = agg_num
#     df_tab_x = df_tab_x.reset_index()
#     df_tab_y = df_tab_y.reset_index()
    
#     if n==0:
#         df_x_executing = df_tab_x.copy()
#         df_y_executing = df_tab_y.copy()
#     else:
#         df_x_executing = pd.concat([df_x_executing, df_tab_x], ignore_index=True)
#         df_y_executing = pd.concat([df_y_executing, df_tab_y], ignore_index=True)
        
# # Shift datetime
# df_x_executing['DT'] += (pd.to_datetime('2023-05-26') - df_x_executing['DT'].min())
# df_y_executing['DT'] += (pd.to_datetime('2023-05-26') - df_x_executing['DT'].min())

# =============================================================================

# Загрузка сырых данных
gdown.download('https://drive.google.com/uc?id=1KqbCw6_-ZoO8iurdy9PZDhOktwHqLYc0')
gdown.download('https://drive.google.com/uc?id=1t14g4aVs7H4-bNF5uvy3xBz6jAJDEOt3')
print('====Данные загружены с гугл диска======')
df_x_executing = pd.read_parquet('dataset_for_demonstration_sensors.parquet')
## TODO - remove
# df_x_executing = df_x_executing[df_x_executing['agg_num'].isin([4,5])]
create_tab(df_x_executing, dtypes_dict, 'raw.sensor_telemetry')
batch_size = 500_000
batches = len(df_x_executing)//batch_size + 1
for i in range(batches):
    df_batch = df_x_executing[i*batch_size:(i+1)*batch_size]
    insert_init_data(df_batch, 'raw.sensor_telemetry')
    print(f'Batch {i+1} / {batches+1} was inserted')
## Нужная только схема таблицы
df_y_executing = pd.read_parquet('dataset_for_demonstration_status.parquet')[:3]
create_tab(df_y_executing, dtypes_dict, 'raw.tm_status')

print('====Сырые данные загружены в БД======')

## Создание таблиц для предиктов модели
q_m1_agg = """CREATE TABLE IF NOT EXISTS ods.m1_agg_status (
   aggregate_id bigint,
   status varchar,
   time_to_downtime bigint,
   tm bigint,
   reason varchar,
   update_time timestamp);"""


q_m3_tm = """CREATE TABLE IF NOT EXISTS ods.m3_tm_status (
   aggregate_id bigint,
   status varchar,
   tm varchar,
   upd_time timestamp);"""

make_sql_req(q_m1_agg)
make_sql_req(q_m3_tm)
print('Созданы таблицы для предиктов модели')