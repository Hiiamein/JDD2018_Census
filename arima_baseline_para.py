#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import datetime
import warnings
import numpy as np
import pandas as pd

from pmdarima.arima import auto_arima
from multiprocessing import Pool
from multiprocessing import cpu_count


def data_process(process_id, city_data_list, date_dt, tmp_df_columns):
    print("process {} start".format(process_id))
    preds_df = pd.DataFrame()
    for city_data in city_data_list:
        district_code = city_data[0]
        sub_df = city_data[1]
        city_code = sub_df['city_code'].iloc[0]
        predict_columns = ['dwell', 'flow_in', 'flow_out']
        tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
        tmp_df['city_code'] = city_code
        tmp_df['district_code'] = district_code

        for column in predict_columns:
            ts_log = np.log(1 + sub_df[column])
            arima_model = auto_arima(ts_log, start_p=1, max_p=9, start_q=1, max_q=9, max_d=5,
                                     start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=5,
                                     m=7, random_state=2018,
                                     trace=False,
                                     seasonal=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True)

            preds = arima_model.predict(n_periods=15)
            preds = pd.Series(preds)
            preds = np.exp(preds) - 1
            tmp_df = pd.concat([tmp_df, preds], axis=1)

        tmp_df.columns = tmp_df_columns
        preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)
    print("process {} finished".format(process_id))
    return preds_df


def multiple_prosess():
    start_time = time.time()
    warnings.filterwarnings('ignore')

    flow_df = pd.read_csv('data/RawData/flow_train.csv')
    flow_df = flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

    date_dt = list()
    init_date = datetime.date(2018, 3, 2)
    for delta in range(15):
        _date = init_date + datetime.timedelta(days=delta)
        date_dt.append(_date.strftime('%Y%m%d'))

    district_code_values = flow_df['district_code'].unique()
    preds_df = pd.DataFrame()

    process_num = cpu_count() - 3
    tmp_df_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    city_data_total = [[district_code, flow_df[flow_df["district_code"] == district_code]] for district_code in
                       district_code_values]
    city_index = np.linspace(0, len(city_data_total), process_num + 1, dtype=np.int32)

    pool = Pool(process_num)
    pool_list = [pool.apply_async(data_process, args=(i + 1, city_data_total[city_index[i]: city_index[i + 1]],
                                                      date_dt, tmp_df_columns))
                 for i in np.arange(len(city_index) - 1)]
    pool.close()
    pool.join()
    results = [p.get() for p in pool_list]

    for result in results:
        preds_df = pd.concat([preds_df, result], axis=0, ignore_index=True)

    preds_df = preds_df.sort_values(by=['date_dt'])

    finish_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    preds_df.to_csv('prediction_{}.csv'.format(finish_time), index=False, header=False)
    end_time = time.time()
    print("used time: {}".format(end_time - start_time))


if __name__ == '__main__':
    multiple_prosess()
