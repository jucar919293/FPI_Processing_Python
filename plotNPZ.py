from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

year = 2021
month = 4
days = [i for i in range(1, 6)] + [n for n in range(7, 8)] + [n for n in range(9, 23) ]+ [n for n in range(25, 27)]
modes = ['CV_MRH_NZK_2', 'IN_MRH_NZK', 'Zenith', 'Aux2', 'West', 'Aux3']


def convert_timesIDL(time):
    year_fun = year
    month_fun = month
    day_fun = day
    hour = math.floor(time)
    minutes = math.floor((time - hour) * 60)
    if hour >= 24:
        day_fun = day_fun + 1
        hour = hour - 24
    date_updated = pd.Timestamp(year_fun, month_fun, day_fun, int(hour), int(minutes))
    return date_updated


def convert_timesPy(time):
    year_fun = time.year
    month_fun = time.month
    day_fun = time.day
    hour = time.hour
    minutes = time.minute

    date_updated = pd.Timestamp(year_fun, month_fun, day_fun, int(hour), int(minutes))
    return date_updated


def preprocess_data_IDL(dframe):
    dframe['times'] = dframe['times'].apply(convert_timesIDL)
    return dframe


def preprocess_data_Py(dframe):
    dframe['times'] = dframe['times'].apply(convert_timesPy)
    return dframe


def get_dateformat(year, month, day, format_str):
    actual_day = datetime(year, month, day)
    temporal_date = actual_day.strftime(format_str)
    return temporal_date


def open_npz(path_npz):
    with np.load(path_npz, allow_pickle=True) as data:
        fpi_results = data['FPI_Results']
        fpi_results = fpi_results.reshape(-1)[0]
        times = fpi_results['sky_times']
        winds = fpi_results['LOSwind']
        temps = fpi_results['T']
        direction = fpi_results['direction']
    return times, winds, temps, direction


if __name__ == '__main__':
    labels = ['temps', 'winds', 'times']
    fig, ax = plt.subplots()
    ax.set_ylim((0, 1300))
    for day in days:
        global day
        pathPy = "results/minime90_mrh_" + get_dateformat(year, month, day, "%Y%m%d") + ".npz"
        timesPy, windsPy, tempsPy, directionPy = open_npz(pathPy)
        # Creating dataframe of Py
        dPy = {'times': timesPy, 'winds': windsPy, 'temps': tempsPy}
        dframePy = pd.DataFrame.from_dict(dPy)
        dframePy = preprocess_data_Py(dframePy)
        dframePy = dframePy[labels]
        for time_stamp in dframePy['times']:
            pointPy = dframePy[(dframePy['times'] == time_stamp)]
            ax.set_ylim(0, 1200)
            if not pointPy.empty:
                ax.scatter(time_stamp, pointPy['temps'], s=5, c='blue')
            print(time_stamp)
    ax.set_ylabel('Temps April - 2021')
    ax.grid(True)
    plt.show()




