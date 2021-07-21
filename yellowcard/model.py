# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

# TODO: imports!
import astropy.coordinates as coord
import numpy as np
from keplerianPlane import LGKepler
from coordinates import LocalGroupHalocentric


def ln_normal(data_val, model_val, variance):
    ''' computes ln normal given a data value and model predicted value '''
    A = 2*np.pi*variance
    B = -1/2 * ( (data_val - model_val)**2 / variance )
    return -1/2 * np.log(A) - B


class TimingArgumentModel:

    def __init__(self, data, data_var):
    # Option 1: def __init__(self, data_y, data_C, unit_system):
    # Option 2: def __init__(self, distance, pm, radial_velocity,
    #                        distance_err, pm_err, rv_err, pm_correlation=0.)

        # TODO: allow passing in the data (distance, proper motion, RV to M31)
        # and uncertainties on the observed position/velocity of M31
        self.data_dist, self.data_pm_alpha_star, self.data_pm_delta, self.data_RV = data
        self.var_dist, self.var_pm_alpha_star, self.var_pm_delta, self.var_RV = data_var


        # TODO: define self._param_info dictionary to store names of parameters
        # (a, e, eta, ...etc)

        # becoming webster
        self._param_info = {}

        # is it okay to set these things as "default"?
        # do they just end up getting updated so it doesn't matter? should i choose dif numbers?
        self._param_info.update({"semi_major_axis": 1000}) # in kpc
        self._param_info['semimajor_axis'] = 1

        self._param_info.update({"eccentricity" : 0.1}) # dimensionless
        self._param_info.update({"eccentric_anomaly": 0}) # radians
        self._param_info.update({"Mtot": 4e12}) # in Msun

# uncomment when you actually use this
#     def unpack_pars(self, par_list):
#         i = 0
#         par_dict = {}
#         for key, par_len in self._param_info.items():
#             if key in self.frozen:
#                 par_dict[key] = self.frozen[key]
#             else:
#                 par_dict[key] = np.squeeze(par_list[i:i+par_len])
#                 i += par_len

#         return par_dict

#     def pack_pars(self, par_dict):
#         parvec = []
#         for i, k in enumerate(self._param_info):
#             if k not in self.frozen:
#                 parvec.append(np.atleast_1d(par_dict[k]))
#         return np.concatenate(parvec)


    def ln_likelihood(self, par_dict):
        # TODO: call observables class to compute x, y, vx, vy given the
        # Keplerian elements / parameters
        inst = LGKepler(**par_dict) # creating keplerian plane with parameters from par_dict
        v_rad_kep, v_tan_kep = inst.vrad_kepler, inst.vtan_kepler # maybe don't need this?

        inst.separation =

        lghc_coord = [r, 0, 0, v_rad_kep, v_tan_kep, 0] # since M31 sits at +x,0,0, (should vrad be negative?) +vtan by def of coordinate frame
        lghc_pos = coord.CartesianRepresentation(x, y, z=0*u.kpc)
        lghc_vel = coord.CartesianDifferential(vx, vy, vz=0*u.km/u.s)

        # TODO: construct a LocalGroupHalocentric instance with x, y, vx, vy
        # (z=vz=0) and transform to ICRS

        # the output from this is a.... matrix? or lghc_coord in ICRS?
        # lghc = LocalGroupHalocentric(lghc_pos.with_differentials(lghc_vel))
        # lghc.transform_to(galactocentric_frame).transform_to(coord.ICRS)
        someOutput = lgcoord.lghalocentric_to_galactocentric(lghc_coord, GalFrame)

        pos = someOutput[0:2]
        vel = someOutput[3:]

        # this is extra not right - need to think about this more
        pred_dist = np.linalg.norm(pos) # (i'm) confused.... shouldn't the norm stay the same if the rotation matrix has det 1?
        pred_pm_alpha_star = 0
        pred_pm_delta = 0
        pred_RV = 0

        # compute log-Normal probability for each model predicted quantity
        # (distance, proper motion, rv to M31)
        ln_dist          = ln_normal(self.data_dist, pred_dist, self.var_dist)
        ln_pm_alpha_star = ln_normal(self.data_pm_alpha_star, pred_pm_alpha_star, self.var_pm_alpha_star)
        ln_pm_delta      = ln_normal(self.data_pm_delta, pred_pm_delta, self.var_pm_delta)
        ln_RV            = ln_normal(self.data_RV, pred_RV, self.var_RV)

        return ln_dist + ln_pm_alpha_star + ln_pm_delta + ln_RV



    def ln_prior(self, par_dict):
        # TODO: need to discuss
        return 0.

    def ln_posterior(self, par_dict):
        # TODO: call ln_likelihood and ln_prior and add the values
        return ln_likelihood(par_dict) + ln_prior(par_dict)

    def __call__(self, par_arr):
        par_dict = self.unpack_pars(par_arr)
        return self.ln_posterior(par_dict)


# Defining __call__ makes this possible:
# model = TimingArgumentModel()
# model([1., 5., 0., 1.])
