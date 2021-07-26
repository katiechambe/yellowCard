# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

import astropy.coordinates as coord
import numpy as np
from .keplerianPlane import LGKepler
from .coordinates import LocalGroupHalocentric, fiducial_m31_c
from gala.units import UnitSystem
import astropy.units as u

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
        self._param_info['lnr'] = 1
        self._param_info['ecoseta'] = 1
        self._param_info['esineta'] = 1
        self._param_info['lnM'] = 1
        self._param_info['Lhatlg'] = 3
        
        self.frozen = {}
        
        # this is because dictionaries are mutable
        if prior_bounds is None:
            prior_bounds = {}
            
        # for now, these values are assumed to be in default unit system    
        prior_bounds.setdefault('lnr',(6,9.5))
        prior_bounds.setdefault('ecoseta',(-1,1))
        prior_bounds.setdefault('esineta',(-1,1))
        prior_bounds.setdefault('lnM',(-1,3))
        # prior_bounds.setdefault('Lhatlg', None)
        
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
        
        par_dict = par_dict.copy()
        par_dict['r'] = np.exp(par_dict['lnr'])
        par_dict['M'] = np.exp(par_dict['lnM'])

        a = par_dict['r']/ (1 - par_dict['ecoseta'])
        eccentricity = np.sqrt(par_dict['ecoseta']**2 + par_dict['esineta']**2)
        eta = np.arctan2(par_dict['esineta'],par_dict['ecoseta']) 
        
        inst = LGKepler(eccentricity = eccentricity, 
                        eccentricAnomaly = eta, 
                        semiMajorAxis = a*self.unit_system['length'], 
                        totalMass= par_dict['M']*self.unit_system['mass']) # creating keplerian plane with parameters from par_dict
    
    
        # calculate x,y, and vx, vy in kepler plane
        r_kep = inst.separation
        x, y = inst.xy
        vx, vy = inst.vxy
        vrad_kep, vtan_kep = inst.vrad_kepler, inst.vtan_kepler
        tperiModel = inst.time

        # print(vx.to(u.km/u.s),vy.to(u.km/u.s))

        # lghc_pos = coord.CartesianRepresentation(x, y, 0*u.kpc)
        # lghc_vel = coord.CartesianDifferential(vx, vy, 0*u.km/u.s)

        lghc_pos = coord.CartesianRepresentation( r_kep, 0*u.kpc, 0*u.kpc)
        lghc_vel = coord.CartesianDifferential( vrad_kep, vtan_kep, 0*u.km/u.s)
        
        lghc_pole = coord.CartesianRepresentation(*par_dict['Lhatlg'])
        lghc_pole = lghc_pole/lghc_pole.norm() # unit vector
        
        # law of cosines crap
        gamma = fiducial_m31_c.separation(self.galcen_frame.galcen_coord)
        
        # TODO: this is going to have to change when the halo center is offset from the disk center
        sunToMWC = self.galcen_frame.galcen_distance # set this for now?
        MWCtoM31 = r_kep.to(u.kpc) # separation between MWHC and M31 given by model
        
        # TODO: guessing the sign of the radical is positive but check this
        sunToM31 = ( sunToMWC * np.cos(gamma) ) + np.sqrt( (sunToMWC * np.cos(gamma))**2 - sunToMWC**2 + MWCtoM31**2 )

        # print(sunToM31.to(u.kpc))
        
        # TODO: define the m31 coord:
        # idk how to do this....
        m31_coord = coord.SkyCoord(ra = fiducial_m31_c.ra, dec = fiducial_m31_c.dec, distance = sunToM31)

        # define position and velocities in LGHC frame
        lghc = LocalGroupHalocentric(lghc_pos.with_differentials(lghc_vel),
                                     lg_pole = lghc_pole, m31_coord = m31_coord)
        modelSol = lghc.transform_to(self.galcen_frame).transform_to(coord.ICRS())
        
        
        modely = np.array([modelSol.distance.decompose(self.unit_system).value, 
                           modelSol.pm_ra_cosdec.decompose(self.unit_system).value,
                           modelSol.pm_dec.decompose(self.unit_system).value,
                           modelSol.radial_velocity.decompose(self.unit_system).value,
                           tperiModel.decompose(self.unit_system).value])

        dy = self.y - modely
    
        return -0.5 * dy.T @ self.Cinv @ dy



    def ln_prior(self, par_dict):
        # TODO: need to discuss
        
        for name, shape in self._param_info.items():
            if name not in self.prior_bounds:
                continue 
            
            if shape == 1:
                if not self.prior_bounds[name][0] < par_dict[name] < self.prior_bounds[name][1]:
                    return -np.inf
            else:
                for value in par_dict[name]:
                    if not self.prior_bounds[name][0] < value < self.prior_bounds[name][1]:
                        return -np.inf
        
        lp = 0 
        lp += par_dict['lnr']
        lp += par_dict['lnM']

        lp += ln_normal(par_dict['Lhatlg'], 0, 1).sum()

        return lp

    def ln_posterior(self, par_dict):
        # TODO: call ln_likelihood and ln_prior and add the values
        return self.ln_likelihood(par_dict) + self.ln_prior(par_dict)

    def __call__(self, par_arr):
        par_dict = self.unpack_pars(par_arr)
        try:
            ln_post = self.ln_posterior(par_dict)
        except Exception as e:
            print(f"Step failed: {e!s}")
            return -np.inf
            
        if not np.isfinite(ln_post):
            return -np.inf
        return ln_post


# Defining __call__ makes this possible:
# model = TimingArgumentModel()
# model([1., 5., 0., 1.])

def ln_normal(data_val, model_val, variance):
    ''' computes ln normal given a data value and model predicted value '''
    A = 2*np.pi*variance
    B = ( (data_val - model_val)**2 / variance )
    return -1/2 * ( np.log(A) + B )
