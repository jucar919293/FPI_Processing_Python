from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

"""
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


def get_time(dataframe):
    return dataframe.time()


def separated_times(times, df, mode):
    variable_mean = []
    variable_std = []
    for i in range(len(times)-1):
        if times[i+1] == datetime(2021, 4, 2, 0, 0):
            mask = (df['times'].apply(get_time) > times[i].time())
            temporal_df = df[mask]
        else:
            mask = (df['times'].apply(get_time) > times[i].time()) & \
                   (df['times'].apply(get_time) <= times[i+1].time())
            temporal_df = df[mask]
        variable_mean.append(temporal_df[mode].mean())
        variable_std.append(temporal_df[mode].std())
    return variable_mean, variable_std


def plot_month_data_(result_path, mode):
    times_min = []
    dates_available = get_available_data(result_path)
    for hour in [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]:
        for minute in [0, 30]:
            if 18 < hour:
                times_min.append(datetime(year, month, 1, hour, minute))
            else:
                times_min.append(datetime(year, month, 2, hour, minute))
    first = True
    labels = ['temps', 'winds', 'times', 'directions']
    fig, ax = plt.subplots()
    directions = ['North', 'East', 'South', 'West', 'Zenith']
    dataframes = []

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
            # dia -> direcciones -> datos cada 5 min
            dataframes.append(dframePy_d)
            # temps_avg.append(temps_mean)
            # winds_avg.append(winds_mean)
            # winds_std.append(dframePy_d.std()['winds']/2)
            # temps_std.append(dframePy_d.std()['temps']/2)
        # winds_std[14] = winds_std[13]
        union = pd.concat(dataframes)

        if direction == 'North':
            norte_frame = union

        if direction == 'East':
            este_frame = union

        if direction == 'South':
            sur_frame = union

        if direction == 'West':
            oeste_frame = union

        if direction == 'Zenith':
            zenith_frame = union

        union = []

    norte_mean, norte_std = separated_times(times_min, norte_frame, mode)
    sur_mean, sur_std = separated_times(times_min, sur_frame, mode)
    este_mean, este_std = separated_times(times_min, este_frame, mode)
    oeste_mean, oeste_std = separated_times(times_min, oeste_frame, mode)
    zenith_mean, zenith_std = separated_times(times_min, zenith_frame, mode)

    meridional = list((np.array(norte_mean) - np.array(sur_mean))/(math.sqrt(2)))
    meridional_std = list((np.array(norte_std) + np.array(sur_std))/(math.sqrt(2)*2))
    meridional_std[6] = meridional_std[4]

    zonal = list((np.array(este_mean) - np.array(oeste_mean))/(math.sqrt(2)))
    zonal_std = list((np.array(este_std) + np.array(oeste_std))/(math.sqrt(2)*2))
    zonal_std[6] = zonal_std[4]

    temp_final = list((np.array(zenith_mean)+np.array(norte_mean)+np.array(sur_mean)+np.array(este_mean)
                       + np.array(oeste_mean))/5)

    temp_final_std = list((np.array(zenith_std)+np.array(norte_std)+np.array(sur_std)
                           + np.array(este_std) + np.array(oeste_std))/10)

    zenith_wind = zenith_mean
    zenith_winds_std = zenith_std

    if mode == 'temps':
        ax.errorbar(times_min[:-2], temp_final[:-1], temp_final_std[:-1],
                    fmt='--ko', elinewidth=0.5, capthick=1, capsize=4)
        ax.set_ylim(500, 780)
        ax.set_ylabel('Temperaturas (K)', fontsize=18)

    else:
        ax.errorbar(times_min[:-2], este_mean[:-1], este_mean[:-1],
                    fmt='--ro', elinewidth=0.5, capthick=1, capsize=4)
        ax.errorbar(times_min[:-2], norte_mean[:-1], norte_mean[:-1],
                    fmt='--ko', elinewidth=0.5, capthick=1, capsize=4)
        # ax.errorbar(times_min[:-2], zenith_wind[:-1], zenith_winds_std[:-1], fmt='-o')

        ax.set_ylim(-60, 80)
        ax.legend(['Zonal Winds', 'Meridional Winds', 'Vertical Winds'])
        ax.set_ylabel('Vientos (m/s)', fontsize=18)
    ax.set_xlabel('Hora local', fontsize=18)
    ax.set_xticklabels([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5], fontsize=15)
    plt.yticks(fontsize=15)

    ax.set_title('Abril - 2021', fontsize=18)
    ax.grid(True)
    plt.show()


def ploteo_temps_test():
    dates_available = get_available_data(result_path)


frequency = ['daily', 'monthly']
modes = ['temps', 'winds']

if __name__ == '__main__':
    mode = modes[1]
    result_path = "./results/"
    plot_month_data_(result_path, mode)

