import FPIprocess
import FPIDisplay
from datetime import datetime


def calc_doy(date):
    year = int(date[0:4])
    month = int(date[4:6])
    day = int(date[6:8])
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    return day_of_year


def main(args):
    instr_name = args[0]
    if args[1] == 'today':
        year = datetime.now().year
        doy = datetime.now().timetuple().tm_yday - 1
    else:
        year = int(args[1][0:4])
        doy = calc_doy(args[1])
    print(year)
    print(doy)
    fpi_dir = ''
    results_stub = args[2]
    bw_dir = ''
    x300_dir = ''
    print(results_stub)

    try:
        msg = FPIprocess.process_instr(instr_name, year, doy, fpi_dir=fpi_dir,
                  bw_dir=bw_dir, x300_dir=x300_dir, results_stub=results_stub,
                  send_to_website=False, enable_share=False,
                  send_to_madrigal=False, enable_windfield_estimate=False)
    except:
        print('error')
        print (doy)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
