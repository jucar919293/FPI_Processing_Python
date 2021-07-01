from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

"""
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JA021025
'Zenith': {'ze': 0, 'az': 0, 'exptime': 210,
    'n_exp': 0, 'last_exp': None, 'delay':1500,},
'North': {'ze': 45, 'az': 0, 'exptime': 210,
    'n_exp': 0, 'last_exp': None, 'delay':0,},
'South': {'ze': -45, 'az': 0, 'exptime': 210,
    'n_exp': 0, 'last_exp': None, 'delay':0,},
'East': {'ze': -45, 'az': -90, 'exptime': 210,
    'n_exp': 0, 'last_exp': None, 'delay':0,},
'West': {'ze': -45, 'az': 90, 'exptime': 210,
"""
year = 2021
month = 4


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


def get_available_data(result_path):
    dates_available = []
    for file in os.listdir(result_path):
        if file.endswith(".npz"):
            file = file[-12:-4]
            if year == int(file[:4]) and month == int(file[4:6]):  # 20210512
                dates_available.append(int(file[6:8]))
                print(file)
    return  dates_available


def plot_month_data_(result_path, mode):

    dates_available = get_available_data(result_path)

    first = True
    labels = ['temps', 'winds', 'times', 'directions']
    fig, ax = plt.subplots()
    ax.set_ylim((0, 1300))
    directions = ['North', 'East', 'South', 'West', 'Zenith']
    temps_avg = []
    winds_avg = []
    winds_std = []
    temps_std = []
    times = [datetime(year, month, day) for day in dates_available]
    for direction in directions:
        for day in dates_available:
            global day
            pathPy = "results/minime90_mrh_" + get_dateformat(year, month, day, "%Y%m%d") + ".npz"
            timesPy, windsPy, tempsPy, directionPy = open_npz(pathPy)
            # Creating dataframe of Py
            dPy = {'times': timesPy, 'winds': windsPy, 'temps': tempsPy, 'directions': directionPy}
            dframePy = pd.DataFrame.from_dict(dPy)
            dframePy = preprocess_data_Py(dframePy)
            dframePy = dframePy[labels]
            dframePy = dframePy[dframePy['temps'] > 50]
            dframePy_d = dframePy[dframePy['directions'] == direction]
            # dframePy_d = dframePy
            temps_mean = dframePy_d.mean()['temps']
            winds_mean = dframePy_d.mean()['winds']
            temps_avg.append(temps_mean)
            winds_avg.append(winds_mean)
            winds_std.append(dframePy_d.std()['winds']/1.5)
            temps_std.append(dframePy_d.std()['temps']/1.5)
        if mode == 'temps':
            ax.errorbar(times, temps_avg, temps_std, fmt='-o')
            ax.set_ylim(400, 900)
        else:
            winds_std[14] = winds_std[13] * 2
            ax.errorbar(times, winds_avg, winds_std, fmt='-o')
            ax.set_ylim(-60, 60)
        winds_std = []
        temps_std = []
        temps_avg = []
        winds_avg = []
    ax.set_ylabel('Temperatures')
    ax.set_title('Abril - 2021')
    ax.grid(True)
    ax.legend(directions)
    plt.show()


def ploteo_temps_test():
    dates_available = get_available_data(result_path)


frequency = ['daily', 'monthly']
modes = ['temps', 'winds']

if __name__ == '__main__':
    mode = modes[1]
    result_path = "./results/"
    plot_month_data_(result_path, mode)

