from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
import math
import os
import FPI
from scipy import interpolate, signal


def convert_timesPy(time):
    year_fun = time.year
    month_fun = time.month
    day_fun = time.day
    hour = time.hour
    minutes = time.minute

    date_updated = pd.Timestamp(year_fun, month_fun, day_fun, int(hour), int(minutes))
    return date_updated


def get_dateformat(year, month, day, format_str):
    actual_day = datetime(year, month, day)
    temporal_date = actual_day.strftime(format_str)
    return temporal_date


def open_npz(year, month, day):
    pathPy = "results/minime90_mrh_" + get_dateformat(year, month, day, "%Y%m%d") + ".npz"
    with np.load(pathPy, allow_pickle=True) as data:
        fpi_results = data['FPI_Results']
        fpi_results = fpi_results.reshape(-1)[0]
    return fpi_results

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


if __name__ == '__main__':
    year = 2021
    months = [4, 5, 6]
    day = 18
    result_path = "./results/"
    fig, ax = plt.subplots()
    dataFrame = []
    for month in months:
        dates_available = get_available_data(result_path, month)
        for day in dates_available:
            FPI_Results = open_npz(year, month, day)
            sky_times = FPI_Results['sky_times']
            sky_times = [convert_timesPy(time) for time in sky_times]
            sky_value = FPI_Results['sky_value']
            position = FPI_Results['direction']
            # laser_value = FPI_Results['laser_value']
            winds = FPI_Results['LOSwind']
            temps = FPI_Results['T']
            r = 0.5  # r/rmax, where r is the radius of the radial bin at which to measure the intensity
            I = sky_value['I']
            a1 = sky_value['a1']
            a2 = sky_value['a2']
            Ir = I * (1 + a1 * r + a2 * r ** 2)
            dic_sky = {'sky_times': sky_times, 'winds': winds, 'sky_values': Ir, 'temps': temps,
                       'direction': position}
            frame = pd.DataFrame.from_dict(dic_sky)

            dataFrame.append(frame)

    union = pd.concat(dataFrame)
    union = union[union['direction'] == 'Zenith']
    ax.scatter(union['sky_values'], union['winds'], alpha=0.3, s=30, c='b')
    ax.set_xlabel('Vientos', fontsize=12)
    ax.set_ylabel('Intensidad sky', fontsize=12)
    plt.show()
