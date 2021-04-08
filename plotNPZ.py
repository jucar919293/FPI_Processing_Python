from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

year = 2015
month = 6
days = [i for i in range(1, 7)] + [n for n in range(9, 31)]
modes = ['CV_MRH_NZK_2', 'IN_MRH_NZK', 'Zenith', 'Aux2', 'West', 'Aux3']


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
    fig, ax = plt.subplots()
    ax.set_ylim((0, 1300))
    for day in days:
        path = "results/minime90_mrh_" + get_dateformat(year, month, day, "%Y%m%d") + ".npz"
        times, winds, temps, direction = open_npz(path)
        for mode in modes:
            for i in range(len(times)):
                if mode == direction[i]:
                    ax.scatter(times[i], temps[i], s=5, c='blue')

    ax.legend()
    ax.grid(True)
    plt.show()


