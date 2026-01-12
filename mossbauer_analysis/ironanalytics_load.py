from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import jv
import xraydb 
from datetime import datetime
import matplotlib.pyplot as plt


import os
import pandas as pd
from dateutil import parser
from dataclasses import dataclass

############################################     CONSTANTS   #########################################################

# Fundamental constants
c = 3e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant in J/K
e = 1.6e-19  # Elementary charge in C
Na = 6.022e23  # Avogadro's number


############################################     FUNCTIONS   #########################################################



@dataclass
class spectrum_data:
    name: str
    description: str
    start_time: str
    total_time: float
    velocity_max: float
    data: np.ndarray
    data_folded: np.ndarray
    data_left :  np.ndarray
    data_right :  np.ndarray
    velocity_list: np.ndarray   
    metadata: dict


def get_ironanalytics_metadata(meta_file): 

    metadata = {}
    with open(meta_file, 'r') as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                if value == "":
                    continue  # Skip keys with no value

                if key in metadata:  #dond't overwrite existing keys (for detectors)
                    key = f"{key}_2"
                metadata[key] = value
    
    return metadata

def print_ironanalytics_metadata(folder_path): 
    for filename in os.listdir(folder_path):
        if filename.endswith("_meta.txt"):
            metadata = get_ironanalytics_metadata(os.path.join(folder_path, filename))
            file_id = filename.replace("_meta.txt", "")
            description = metadata.get("description", "")
            vmax = metadata.get("velocity", "")
            # Extract primary detector info
            volt = metadata.get("voltage", "")
            amp = metadata.get("amplification", "")
            ll = metadata.get("window_LL", "")
            hl = metadata.get("window_HL", "")
            
            print(f"ID: {file_id},\t v_max: {vmax},\t  det: {volt},\t {amp},\t {ll},\t {hl},\t Description: {description}")
            

def fold_spectrum(data: np.ndarray, vmax, offset: int = 0) -> np.ndarray:

    n = len(data)
    center = n // 2 + offset
    left = data[:center][::-1]
    right = data[center:][:len(left)]  # truncate to match length if odd
    left = left[:len(right)]
    folded = left + right
    vlist = np.linspace(-vmax, vmax, 512)[:len(left)]
    return folded, left, right, vlist

def read_ironanalytics_data(folder_path, file_id, offset = 0, plot = False) -> spectrum_data:
    """
    Reads calibration data and metadata from .txt files, returns an instance of CalibrationSpectrum.
    file_id could be something like 'A00043', folder_path is the folder containing the data files.
    """
    if isinstance(file_id, list): #if multiple files, add them up!
        metadata = {}
        data = 0
        for file in file_id:
            meta_file = os.path.join(folder_path, f"{file}_meta.txt")
            data_file = os.path.join(folder_path, f"{file}.txt")
            temp = get_ironanalytics_metadata(meta_file)
            metadata.update(temp)
            data = pd.read_csv(data_file, sep="\t", header=None)[0].to_numpy()
   
        

    elif isinstance(file_id, str):
        meta_file = os.path.join(folder_path, f"{file_id}_meta.txt")
        data_file = os.path.join(folder_path, f"{file_id}.txt")
        metadata = get_ironanalytics_metadata(meta_file)
        data = pd.read_csv(data_file, sep="\t", header=None)[0].to_numpy()
        description = metadata.get("description")


    # Parse times
    start = parser.parse(metadata.get("start"), dayfirst=True)
    stop = parser.parse(metadata.get("stop"), dayfirst=True)
    total_time = (stop - start).total_seconds() 

    # Parse velocity
    vmax = float(metadata.get("velocity"))

    # Read calibration data
    
    data_folded, data_left, data_right, vlist = fold_spectrum(data, vmax, offset=offset)

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(vlist, data_folded/2, label='Folded Spectrum')
        plt.plot(vlist, data_left, label='Left Side', linestyle='--')
        plt.plot(vlist, data_right, label='Right Side', linestyle='--')
        plt.xlabel('Velocity (mm/s)')
        plt.ylabel('Counts')
        plt.title(f'{file_id}, {description}')
        plt.legend()
        plt.grid()
        plt.show()


    return spectrum_data(
        name = file_id,
        description = description,
        start_time = start,
        total_time = total_time,
        velocity_max = vmax,
        data = data,
        data_folded = data_folded,
        data_left = data_left,
        data_right = data_right,
        velocity_list = vlist,
        metadata = metadata
    )




if __name__ == "__main__":
    # Example usage
        # Assuming the directory and file_id are set correctly

    dir = 'C:/Users/magrini/Documents/programming/mossbauer_analysis/data/SAW_spectra/saw_spectra_16mms'
    id = ['A00101','A00102']
    dat = read_ironanalytics_data(dir, id, offset = -3)
    #print_ironanalytics_metadata(dir)
