import numpy as np
from timeit import default_timer as timer

class StabilityAndControlHelper:
    def __init__(self):
        self.surfaces_data = []

    def load_surface_data(self, x_ac: float, z_ac: float, ar: float, mean_chord: float, e: float, cm_ac: float, s: float, cl_a: float, cl_fn):
        """
        Loads data of aircraft lifting surfaces for stability calculations. Insert first the reference lifting surface;
        Params:
        - x_ac: x coordinate of surface aerodinamic center [mm];
        - z_ac: z coordinate of surface aerodinamic center [mm];
        - ar: Arpect ratio of surface, calculated as b^2/S [-];
        - mean_chord: surface mean aerodinamic chord [mm];
        - e: Oswald efficiency number [-];
        - cm_ac: moment coefficient of profile over aerodinamic center;
        - s: surface area [m^2];
        - cl_fn: lift coefficient discrete function as alpha vs cl [rad, -];
        - cl_a: lift coefficient derivative with respect to alpha [rad^-1];
        """
        self.surfaces_data.append({
            'x_ac':  x_ac,
            'z_ac': z_ac,
            'ar': ar,
            'mean_chord': mean_chord,
            'e': e,
            'cm_ac': cm_ac,
            's': s,
            'cl_a': cl_a,
            'cl_fn': cl_fn
        })

        if len(self.surfaces_data) == 1:
            self.ref_mean_chord = mean_chord
            self.ref_s = s

    def __calculate_surface_cma(self, surface_index, x_cg):
        surface_info = self.surfaces_data[surface_index]
        x_ac = surface_info['x_ac']
        z_ac = surface_info['z_ac']
        ar = surface_info['ar']
        e = surface_info['e']
        s = surface_info['s']
        cl_a = surface_info['cl_a']
        alpha = surface_info['cl_fn'][:, 0]
        cl = surface_info['cl_fn'][:, 1]

        l_i = x_cg - x_ac
        v_i = s*l_i/(self.ref_s*self.ref_mean_chord)

        return np.mean(v_i*(cl_a*(np.cos(alpha)-z_ac*np.sin(alpha)/l_i) + \
            cl*(- np.sin(alpha)-z_ac*np.cos(alpha)/l_i) + \
            2/(np.pi*ar*e)*cl*cl_a*(np.sin(alpha)+z_ac*np.cos(alpha)/l_i) + \
            1/(np.pi*ar*e)*cl**2*(np.cos(alpha)-z_ac*np.sin(alpha)/l_i)))
        
    def __calculate_aircraft_cma(self, x_cg):
        number_of_surfaces = len(self.surfaces_data)
        cm_a_array = np.zeros(number_of_surfaces)
        for i in range(number_of_surfaces):
            cm_a_array[i] = self.__calculate_surface_cma(i, x_cg)
        return np.sum(cm_a_array)
    
    def find_cg(self, target_static_margin=0.2, neutral_point_initial_x=0., accuracy = 0.01):
        """
        Calculate aircraft center of gravity for a given static margin;
        Params:
        - target_static_margin: final stability static margin of aircraft (default = 0.2) [-];
        - neutral_point_initial_x: assigned value to neutral point start position before iterations (default = 0.) [mm];
        - accuracy: final accuracy of final x_cg value (default = 0.01) [mm];
        """
        start = timer()
        x_np0 = neutral_point_initial_x
        def newton_iteration(dx = 0.01):
            aircraft_cma0 = self.__calculate_aircraft_cma(x_np0)
            aircraft_cma1 = self.__calculate_aircraft_cma(x_np0 + dx)
            dcma_dx = (aircraft_cma1 - aircraft_cma0)/dx
            return x_np0 - aircraft_cma0/dcma_dx
        x_np1 = newton_iteration()
        while abs(x_np1 - x_np0) > accuracy:
            x_np0 = x_np1
            x_np1 = newton_iteration()
        end = timer()
        return dict(x_cg = x_np1 - target_static_margin * self.ref_mean_chord,
                     time_elapsed_in_ms = (end - start)*1e3) 

