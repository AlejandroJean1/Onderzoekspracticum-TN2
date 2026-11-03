import os
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel, LinearModel
from nodes.rt_practicum.control import equipment
from lmfit.models import VoigtModel, ConstantModel
from scipy.signal import find_peaks

def init_equipment():
	# Initialize QINC equipment framework.
	equipment.init_equipment("C:\\Users\\TUD277232\\Repositories\\diamondos\\nodes\\rt_practicum\\config\\equipment.conf")
	node = equipment.get_node("rt_practicum")
	node.prepare_rfsource()
	node.prepare_dac()
	rfsource = node.rfsource
	dac = node.dac
	dac.set_scanner_clock_freq(10)
	# dac.set_up_counter()

	return dac, rfsource

def get_count_array(dac, nof_samples):
	return dac.get_counter(nof_samples)[0]

def set_microwave_frequency(rfsource, mw_frequency_hz):
	rfsource.set_frequency(mw_frequency_hz)

def set_microwave_power(rfsource, mw_power_dbm):
	rfsource.set_power(mw_power_dbm)

def fit_two_peaks(dataset, fit_window=50e+6, relative_prominence=0.5):
    x = dataset.mw_frequency.values
    y = dataset.average_counts_normalized.values

    model = ConstantModel()
    params = model.guess(y, x)
    params['c'].set(min=0, max=1)
    peaks, _ = find_peaks(1-y, prominence=(np.max(y) - np.min(y))*relative_prominence)
    peak_heights = (1-y)[peaks]
    peaks_sorted_limited = peaks[np.argsort(peak_heights)][-2:]
    peaks_to_fit = np.sort(peaks_sorted_limited)

    for i, peak in enumerate(peaks_to_fit):
        freq_peak = x[peak]
        peak_slice = dataset.sel(
            mw_frequency=slice(freq_peak-fit_window/2, freq_peak+fit_window/2)
        )
        freqs = peak_slice.mw_frequency.values
        counts = np.clip(1 - peak_slice.average_counts_normalized.values, 0, 1)
        peak_model = VoigtModel(prefix=f"peak_{i}_")
        params.update(peak_model.guess(counts, freqs))
        params[f"peak_{i}_amplitude"].set(min=0)
        model -= peak_model

    result = model.fit(y, params=params, x=x)

    return result

def save_dataset(dataset, name):
    path_to_file = f"nodes/rt_practicum/RP/data/{name}.hdf5"
    if os.path.isfile(path_to_file):
        answer = input(f"File {name}.hdf5 already exists, do you want to override it? [y/n]")

        if answer == "y":
            dataset.to_netcdf(path_to_file, engine="h5netcdf", invalid_netcdf=True)
        else:
            print("File not saved!")
            return
    else:
        dataset.to_netcdf(path_to_file, engine="h5netcdf", invalid_netcdf=True)

def load_dataset(name):
      return xr.load_dataset(f"nodes/rt_practicum/RP/data/{name}.hdf5", engine="h5netcdf")

def store_data(x_dat,y_dat, name):
	folder = r"C:\Users\TUD277232\Repositories\diamondos\Oscar\ODMR_data_magnet"
	filepath = os.path.join(folder,name)
	with open(filepath, "w") as f:
		f.write("# x y\n")
		np.savetxt(f, np.array([x_dat,y_dat]).T)

def store_data_raw(x_dat,y_dat, name):
	folder = r"C:\Users\TUD277232\Repositories\diamondos\Oscar\ODMR_data_magnet_raw"


	filepath = os.path.join(folder,name)

	np.savez(filepath, x = x_dat, y = y_dat)