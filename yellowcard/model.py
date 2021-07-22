# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

import astropy.coordinates as coord
import numpy as np
from .keplerianPlane import LGKepler
from .coordinates import LocalGroupHalocentric, fiducial_m31_c
from gala.units import UnitSystem

class TimingArgumentModel:

    def __init__(self, distance, pm, radial_velocity, tperi,
                 distance_err, pm_err, radial_velocity_err, tperi_err, 
                 pm_correlation=0., unit_system = None, prior_bounds = None, 
                 galcen_frame = coord.Galactocentric()):
    
        # this is because dictionaries are mutable
        if unit_system is None:
            unit_system = UnitSystem(u.kpc, u.Unit(1e12*u.Msun), u.Gyr, u.radian)
        self.unit_system = unit_system
            
        self.dist  = distance
        self.pm    = pm
        self.rv    = radial_velocity
        self.tperi = tperi
        
        self.dist_err  = distance_err
        self.pm_err    = pm_err
        self.rv_err    = radial_velocity_err
        self.pm_corr   = pm_correlation
        self.tperi_err = tperi_err
        
        self.y = np.array([self.dist.decompose(self.unit_system).value, 
                           *self.pm.decompose(self.unit_system).value,
                           self.rv.decompose(self.unit_system).value,
                           self.tperi.decompose(self.unit_system).value])
        
        # TODO: we're ignoring the pm correlation
        self.Cinv = np.diag([self.dist_err.decompose(self.unit_system).value**-2,
                             *self.pm_err.decompose(self.unit_system).value**-2,
                             self.rv_err.decompose(self.unit_system).value**-2,
                             self.tperi_err.decompose(self.unit_system).value**-2])
    
        # becoming webster
        self._param_info = {}

        # lengths of each of the parameters
        self._param_info['a'] = 1
        self._param_info['ecoseta'] = 1
        self._param_info['esineta'] = 1
        self._param_info['M'] = 1
        self._param_info['Lhatlg'] = 3
        
        self.frozen = {}
        
        # this is because dictionaries are mutable
        if prior_bounds is None:
            prior_bounds = {}
            
        # for now, these values are assumed to be in default unit system    
        prior_bounds.setdefault('a',(500,5000))
        prior_bounds.setdefault('ecoseta',(-1,1))
        prior_bounds.setdefault('esineta',(-1,1))
        prior_bounds.setdefault('M',(1,10))
        prior_bounds.setdefault('Lhatlg', (-1,1))
        
        self.prior_bounds = prior_bounds
        
        self.galcen_frame= galcen_frame

    def unpack_pars(self, par_list):
        i = 0
        par_dict = {}
        for key, par_len in self._param_info.items():
            if key in self.frozen:
                par_dict[key] = self.frozen[key]
            else:
                par_dict[key] = np.squeeze(par_list[i:i+par_len])
                i += par_len

        return par_dict

    def pack_pars(self, par_dict):
        parvec = []
        for i, k in enumerate(self._param_info):
            if k not in self.frozen:
                parvec.append(np.atleast_1d(par_dict[k]))
        return np.concatenate(parvec)


    def ln_likelihood(self, par_dict):
        
        eccentricity = np.sqrt(par_dict['ecoseta']**2 + par_dict['esineta']**2)
        eta = np.arctan2(par_dict['esineta'],par_dict['ecoseta']) 
        
        inst = LGKepler(eccentricity = eccentricity, 
                        eta = eta, 
                        semiMajorAxis = par_dict['a'], 
                        totalMass= par_dict['M']) # creating keplerian plane with parameters from par_dict
    
    
        # calculate x,y, and vx, vy in kepler plane
        r_kep = inst.separation
        x, y = inst.xy
        vx, vy = inst.vxy
        tperiModel = inst.time

        lghc_pos = coord.CartesianRepresentation(x, y, z=0*u.kpc)
        lghc_vel = coord.CartesianDifferential(vx, vy, vz=0*u.km/u.s)
        
        lghc_pole = coord.CartesianRepresentation(*par_dict['Lhatlg'])
        lghc_pole = lghc_pole/lghc_pole.norm() # unit vector
        
        # law of cosines crap
        gamma = fiducial_m31_c.separation(self.galcen_frame.galcen_coord)
        
        # TODO: this is going to have to change when the halo center is offset from the disk center
        sunToMWC S = 8.1*u.kpc # set this for now?
        MWCtoM31 D = r_kep.to(u.kpc) # separation between MWHC and M31 given by model
        
        # TODO: guessing the sign of the radical is positive but check this
        sunToM31 = ( sunToMWC * np.cos(gamma) ) + np.sqrt( (sunToMWC * np.cos(gamma))**2 - sunToMWC**2 + MWCtoM31**2 )
        
        # TODO: define the m31 coord:
        # idk how to do this....
        # m31_coord = SkyCoord(ra = , dec = , dist = sunToM31)?

        # define position and velocities in LGHC frame
        lghc = LocalGroupHalocentric(lghc_pos.with_differentials(lghc_vel),
                                     lg_pole = lghc_pole, m31_coord = m31_coord)
        modelSol = lghc.transform_to(galactocentric_frame).transform_to(coord.ICRS)
        
        
        modely = np.array([modelSol.distance.decompose(self.unit_system).value, 
                           modelSol.pm_ra_cosdec.decompose(self.unit_system).value,
                           modelSol.pm_dec.decompose(self.unit_system).value,
                           modelSol.radial_velocity.decompose(self.unit_system).value,
                           tperiModel.decompose(self.unit_system).value])

        dy = self.y - modely
    
        return -0.5 * (dy.T) @ self.Cinv @ dy



    def ln_prior(self, par_dict):
        # TODO: need to discuss
        
        for name, shape in self._param_info.items():
            if shape == 1:
                if not self.prior_bounds[name][0] < par_dict[name] < self.prior_bounds[name][1]:
                    return -np.inf
            else:
                for value in par_dict[name]:
                    if not self.prior_bounds[name][0] < value < self.prior_bounds[name][1]:
                        return -np.inf
        
        return 0.

    def ln_posterior(self, par_dict):
        # TODO: call ln_likelihood and ln_prior and add the values
        return self.ln_likelihood(par_dict) + self.ln_prior(par_dict)

    def __call__(self, par_arr):
        par_dict = self.unpack_pars(par_arr)
        return self.ln_posterior(par_dict)


# Defining __call__ makes this possible:
# model = TimingArgumentModel()
# model([1., 5., 0., 1.])

# def ln_normal(data_val, model_val, variance):
#     ''' computes ln normal given a data value and model predicted value '''
#     A = 2*np.pi*variance
#     B = ( (data_val - model_val)**2 / variance )
#     return -1/2 * ( np.log(A) + B )
