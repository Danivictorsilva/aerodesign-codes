import numpy as np
import pandas as pd
from StabilityAndControlHelper import StabilityAndControlHelper

aircraft_surfaces_data = pd.read_excel('aircraft-data.xlsx').to_numpy()
surfaces_cl_data = pd.read_excel('surfaces-cL-data.xlsx').to_numpy()

number_of_surfaces = aircraft_surfaces_data.shape[0]

if surfaces_cl_data.shape[1] - 1 != number_of_surfaces:
    raise Exception(
        "Inconsistent data: number of surfaces differ from aircraft data and surfaces lift coefficient data")

stability_and_control_helper = StabilityAndControlHelper()

try:
    for i in range(number_of_surfaces):
        stability_and_control_helper.load_surface_data(
            x_ac=aircraft_surfaces_data[i, 1],
            z_ac=aircraft_surfaces_data[i, 2],
            ar=aircraft_surfaces_data[i, 3],
            mean_chord=aircraft_surfaces_data[i, 4],
            e=aircraft_surfaces_data[i, 5],
            cm_ac=aircraft_surfaces_data[i, 6],
            s=aircraft_surfaces_data[i, 7],
            cl_a=aircraft_surfaces_data[i, 8],
            cl_fn=np.column_stack(
                (surfaces_cl_data[:, 0], surfaces_cl_data[:, i+1]))
        )
    cg_data = stability_and_control_helper.find_cg()
except Exception as e:
    print(e)
else:
    print(cg_data)
