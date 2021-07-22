# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

import astropy.coordinates as coord
import numpy as np
from keplerianPlane import LGKepler
from coordinates import LocalGroupHalocentric


def ln_normal(data_val, model_val, variance):
    ''' computes ln normal given a data value and model predicted value '''
    A = 2*np.pi*variance
    B = ( (data_val - model_val)**2 / variance )
    return -1/2 ( np.log(A) + B )


class TimingArgumentModel:

    def __init__(self, distance, pm, radial_velocity,
                       distance_err, pm_err, radial_velocity_err, pm_correlation=0.)
    
        self.dist = distance
        self.pm   = pm
        self.rv   = radial_velocity
        
        self.dist_err = distance_err
        self.pm_err   = pm_err
        self.rv_err   = radial_velocity_err
        self.pm_corr  = pm_correlation
    
        # becoming webster
        self._param_info = {}

        # lengths of each of the parameters
        self._param_info['semiMajorAxis'] = 1
        self._param_info['eccentricity'] = 1
        self._param_info['eccentricAnomaly'] = 1
        self._param_info['totalMass'] = 1


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
        
        inst = LGKepler(**par_dict) # creating keplerian plane with parameters from par_dict
    
        # calculate x,y, and vx, vy in kepler plane
        x, y = inst.xy
        vx, vy = inst.vxy

        lghc_pos = coord.CartesianRepresentation(x, y, z=0*u.kpc)
        lghc_vel = coord.CartesianDifferential(vx, vy, vz=0*u.km/u.s)

        # TODO: construct a LocalGroupHalocentric instance with x, y, vx, vy
        # (z=vz=0) and transform to ICRS

        # define position and velocities in LGHC frame
        lghc = LocalGroupHalocentric(lghc_pos.with_differentials(lghc_vel))
        lghc.transform_to(galactocentric_frame).transform_to(coord.ICRS)
        

        # set these things to 0 for now
        pred_dist = lghc.separation #do i need to pass 0,0,0 or is that default?
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
