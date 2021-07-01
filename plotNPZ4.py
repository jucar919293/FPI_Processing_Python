from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import FPI
from scipy import interpolate, signal

import FPIDisplay

MONTHS = [
    'Enero',
    'Febrero',
    'Marzo',
    'Abril',
    'Mayo',
    'Junio',
    'Julio',
    'Agosto',
    'Septiembre',
    'Octubre',
    'Noviembre',
    'Diciembre'
]
results_stub = 'results/'
instr_name = 'minime90'

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
        temps_e = fpi_results['sigma_T']
        direction = fpi_results['direction']
    return times, winds, temps, temps_e, direction, fpi_results


def get_winds(path_npz, direction, reference='laser'):
    with np.load(path_npz, allow_pickle=True) as data:
        FPI_Results = data['FPI_Results']
        FPI_Results = FPI_Results.reshape(-1)[0]
        ref_Dop, e_ref_Dop = FPI.DopplerReference(FPI_Results, reference=reference)
        # Calculate the vertical wind and interpolate it
        ind = FPI.all_indices('Zenith', FPI_Results['direction'])
        w = (FPI_Results['LOSwind'][ind] - ref_Dop[ind])  # LOS is away from instrument
        sigma_w = FPI_Results['sigma_LOSwind'][ind]
        dt = []
        for x in FPI_Results['sky_times'][ind]:
            diff = (x - FPI_Results['sky_times'][0])
            dt.append(diff.seconds + diff.days * 86400.)
        dt = np.array(dt)

        # Remove outliers
        ind = abs(w) < 200.

        if sum(ind) <= 1:
            # No good data, just use all ind
            ind = np.array([True for i in range(len(w))])  # There has to be a clearer way to do this...

        if len(ind) == 0:
            raise Exception('%s: No Zenith look directions' % f)

        # Interpolate
        w2 = interpolate.interp1d(dt[ind], w[ind], bounds_error=False, fill_value=0.0)
        sigma_w2 = interpolate.interp1d(dt[ind], sigma_w[ind], bounds_error=False, fill_value=0.0)
        dt = []
        for x in FPI_Results['sky_times']:
            diff = (x - FPI_Results['sky_times'][0])
            dt.append(diff.seconds + diff.days * 86400.)
        w = w2(dt)
        sigma_w = sigma_w2(dt)
        ind = FPI.all_indices(direction, FPI_Results['direction'])
        if direction == 'Zenith':
            print('Zeni')
            Doppler_Wind = (FPI_Results['LOSwind'][ind] - ref_Dop[ind])
            Doppler_Error = np.sqrt(FPI_Results['sigma_LOSwind'][ind] ** 2)
        else:
            Doppler_Wind = (FPI_Results['LOSwind'][ind] - ref_Dop[ind] - w[ind] * np.cos(
                FPI_Results['ze'][ind] * np.pi / 180.)) / \
                           np.sin(FPI_Results['ze'][ind] * np.pi / 180.)
            Doppler_Error = np.sqrt(FPI_Results['sigma_LOSwind'][ind] ** 2 + sigma_w[ind] ** 2)
        if direction == 'West' or direction == 'South':
            Doppler_Wind = -Doppler_Wind

        # Doppler_Wind = -Doppler_Wind

    return Doppler_Wind, Doppler_Error


def get_available_data(result_path, month):
    dates_available = []
    for file in os.listdir(result_path):
        if file.endswith(".npz"):
            file = file[-12:-4]
            if year == int(file[:4]) and month == int(file[4:6]):  # 20210512
                dates_available.append(int(file[6:8]))
                print(file)
    return dates_available


def get_time(dataframe):
    return dataframe.time()


def separated_times(times, df, mode):
    variable_mean = []
    variable_std = []
    for i in range(len(times) - 1):
        if times[i + 1] == datetime(2021, month, 2, 0, 0):
            mask = (df['times'].apply(get_time) > times[i].time())
            temporal_df = df[mask]
        else:
            mask = (df['times'].apply(get_time) > times[i].time()) & \
                   (df['times'].apply(get_time) <= times[i + 1].time())
            temporal_df = df[mask]
        variable_mean.append(temporal_df[mode].mean())
        if mode == 'temps':
            sigma = 'sigma_T'
        elif mode == 'winds':
            sigma = 'sigma_winds'
        variable_std.append(np.array(temporal_df[mode].std()))
        # variable_std.append(np.array(temporal_df[sigma].mean()))
    return variable_mean, variable_std


def smothResults(dframePy, mode, kernel_size=3):
    data = np.array(dframePy[mode])
    data_filtered = signal.medfilt(data, kernel_size=kernel_size)
    dframePy[mode] = data_filtered
    return dframePy


def plot_month_data_(result_path, mode, month, eliminate, times_min):
    dates_available = get_available_data(result_path, month)
    for el in eliminate:
        try:
            dates_available.remove(el)
        except:
            print('No date available')

    first = True
    labels = ['temps', 'winds', 'sigma_winds', 'times']
    dataframes = []
    directions = ['North', 'East', 'South', 'West']
    times = [datetime(year, month, day) for day in dates_available]
    for direction in directions:
        for day in dates_available:
            global day
            pathPy = "results/minime90_mrh_" + get_dateformat(year, month, day, "%Y%m%d") + ".npz"
            Doppler_Wind, Doppler_Error = get_winds(pathPy, direction)
            times_py, winds_py, temps_py, temps_e_py, direction_py, fpi_results_py = open_npz(pathPy)
            ind = FPI.all_indices(direction, fpi_results_py['direction'])
            dPy = {'times': times_py[ind], 'winds': Doppler_Wind, 'sigma_winds': Doppler_Error,
                   'temps': temps_py[ind], 'sigma_T': temps_e_py[ind]}
            dframePy = pd.DataFrame.from_dict(dPy)
            dframePy = preprocess_data_Py(dframePy)
            dframePy = dframePy[labels]
            dframePy = dframePy[dframePy['temps'] > 1]
            filterdata = smothResults(dframePy, mode, kernel_size=7)
            # dia -> direcciones -> datos cada 5 min
            dataframes.append(filterdata)

        union = pd.concat(dataframes)
        dataframes = []

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

    # oeste_mean, oeste_std = separated_times(times_min, oeste_frame, mode)
    # norte_mean, norte_std = separated_times(times_min, norte_frame, mode)
    # sur_mean, sur_std = separated_times(times_min, sur_frame, mode)
    # este_mean, este_std = separated_times(times_min, este_frame, mode)

    if mode == 'temps':
        total = pd.concat([oeste_frame, este_frame, norte_frame, sur_frame])

        total_c, total_c_std = separated_times(times_min, total, mode)
        # zenith_mean, zenith_std = separated_times(times_min, zenith_frame, mode)

        return total_c, total_c_std
    else:
        zonal = pd.concat([oeste_frame, este_frame])
        meridional = pd.concat([norte_frame, sur_frame])

        meridional_c, meridional_c_std = separated_times(times_min, meridional, mode)
        zonal_c, zonal_c_std = separated_times(times_min, zonal, mode)

        return [meridional_c, meridional_c_std, zonal_c, zonal_c_std]


def plot_day_data_(result_path, mode, month, day, directions, times_min):
    dates_available = get_available_data(result_path, month)

    labels = ['temps', 'winds', 'sigma_winds', 'times', 'sigma_T']
    if day not in dates_available:
        print ('No day available')
    else:
        for direction in directions:
            pathPy = "results/minime90_mrh_" + get_dateformat(year, month, day, "%Y%m%d") + ".npz"
            Doppler_Wind, Doppler_Error = get_winds(pathPy, direction)
            times_py, winds_py, temps_py, temps_e_py, direction_py, fpi_results_py = open_npz(pathPy)
            ind = FPI.all_indices(direction, fpi_results_py['direction'])
            dPy = {'times': times_py[ind], 'winds': Doppler_Wind, 'sigma_winds': Doppler_Error,
                   'temps': temps_py[ind], 'sigma_T': temps_e_py[ind]}
            dframePy = pd.DataFrame.from_dict(dPy)
            dframePy = preprocess_data_Py(dframePy)
            dframePy = dframePy[labels]
            dframePy = dframePy[dframePy['temps'] > 1]
            filterdata = smothResults(dframePy, mode, kernel_size=1)
            # dia -> direcciones -> datos cada 5 min
            union = filterdata

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

        if mode == 'temps':
            sigma = 'sigma_T'
            total = pd.concat([oeste_frame, este_frame, norte_frame, sur_frame])
            total_c, total_c_std = separated_times(times_min, total, mode)
            return [total_c, total_c_std]

        elif mode == 'winds':
            sigma = 'sigma_winds'
            zonal = pd.concat([oeste_frame, este_frame])
            meridional = pd.concat([norte_frame, sur_frame])

            meridional_c, meridional_c_std = separated_times(times_min, meridional, mode)
            zonal_c, zonal_c_std = separated_times(times_min, zonal, mode)
            # zenith_mean, zenith_std = separated_times(times_min, zenith_frame, mode)

            return [meridional_c, zonal_c]


def ploteo_temps_test():
    dates_available = get_available_data(result_path)


# ax.errorbar(times_min[:-2], range(len(times_min[:-2])), alpha=0.0)
# ax.scatter(times_min[:-2], meridional_c[:-1], s=20, c='r', alpha=0.2)
# ax.set_ylim(-100, 150)

if __name__ == '__main__':
    month = 4
    day = 18
    fig = plt.figure(figsize=(2, 2))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    eliminate = [6, 7, 8, 15, 16, 17, 18, 19, 24, 25, 26]
    # eliminate = []
    modes = ['temps', 'winds']
    directions = ['North', 'East', 'South', 'West']
    result_path = "./results/"
    times_min = []
    for hour in [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]:
        for minute in [0, 30]:
            if 18 < hour:
                times_min.append(datetime(year, month, 1, hour, minute))
            else:
                times_min.append(datetime(year, month, 2, hour, minute))

    dates_available = get_available_data(result_path, month)
    size_axis = 10
    size_title = 13

    for day in dates_available:
        data_day_plot = plot_day_data_(result_path, modes[1], month, day, directions, times_min)
        ax1.errorbar(times_min[:-1], range(len(times_min[:-1])), alpha=0.0)
        ax1.scatter(times_min[:-1], data_day_plot[0], alpha=0.4, s=10, c='r')
        ax1.set_xlabel('Hora local', fontsize=size_axis)
        ax1.set_xticklabels([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5], fontsize=size_axis)
        ax1.set_ylim(-100, 100)
        ax1.grid(True)
        ax1.set_title('Vientos meridionales', fontsize=size_title)
        ax1.set_ylabel('Vientos (m/s)', fontsize=size_axis)

        ax2.errorbar(times_min[:-1], range(len(times_min[:-1])), alpha=0.0)
        ax2.scatter(times_min[:-1], data_day_plot[1], alpha=0.4, s=10, c='r')
        ax2.set_xlabel('Hora local', fontsize=size_axis)
        ax2.set_xticklabels([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5], fontsize=size_axis)
        ax2.set_ylim(-60, 150)
        ax2.grid(True)
        ax2.set_title('Vientos zonales', fontsize=size_title)

        ax3.errorbar(times_min[:-1], range(len(times_min[:-1])), alpha=0.0)
        ax3.set_xlabel('Hora local', fontsize=size_axis)
        ax3.set_xticklabels([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5], fontsize=size_axis)
        data_day_plot = plot_day_data_(result_path, modes[0], month, day, directions, times_min)
        ax3.scatter(times_min[:-1], data_day_plot[0], alpha=0.4, s=10, c='r')

    data_month_plot = plot_month_data_(result_path, modes[1], month, eliminate, times_min)

    ax1.errorbar(times_min[:-1], data_month_plot[0], data_month_plot[1],
                 fmt='--ko', elinewidth=0.5, capthick=1, capsize=3, alpha=0.8)

    ax2.errorbar(times_min[:-1], data_month_plot[2], data_month_plot[3],
                 fmt='--ko', elinewidth=0.5, capthick=1, capsize=3, alpha=0.8)

    data_month_plot = plot_month_data_(result_path, modes[0], month, eliminate, times_min)

    ax3.errorbar(times_min[:-1], data_month_plot[0], data_month_plot[1],
                 fmt='--ko', elinewidth=0.5, capthick=1, capsize=3, alpha=0.8)
    ax3.set_ylim(400, 900)
    ax3.grid(True)
    ax3.set_title('Temperatura de vientos', fontsize=size_title)
    ax3.set_ylabel('Temperaturas (K)', fontsize=size_axis)

    # plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.2, wspace=None, hspace=None)
    # plt.set_title(MONTHS[month-1] + '-' + str(year), fontsize=18
    fig.subplots_adjust(hspace=.5)
    plt.show()
