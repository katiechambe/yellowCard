# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

import astropy.coordinates as coord
import numpy as np
from .keplerianPlane import LGKepler
from .coordinates import LocalGroupHalocentric, fiducial_m31_c
from gala.units import UnitSystem
from numpy.linalg import norm
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix

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
        # eParam = 'ln(1-e)'
        self._param_info['lnr'] = 1
        self._param_info['eParam'] = 1
        self._param_info['coseta'] = 1
        self._param_info['sineta'] = 1
        self._param_info['lnM'] = 1
        self._param_info['sinalpha'] = 1
        self._param_info['cosalpha'] = 1

        self.frozen = {}

        # this is because dictionaries are mutable
        if prior_bounds is None:
            prior_bounds = {}

        # for now, these values are assumed to be in default unit system
        prior_bounds.setdefault('lnr',(6,7))
        prior_bounds.setdefault('eParam',(-18,0))
        prior_bounds.setdefault('lnM',(-1,3))

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
    
    def whats_this(self, par_dict):
        ''' takes our original parameter set and transforms them to things we recognize '''
        what_dict = {}
        what_dict['r'] = np.exp(par_dict['lnr'])
        what_dict['e'] = 1 - np.exp(par_dict['eParam'])
        etta = np.arctan2(par_dict['sineta'],par_dict['coseta']) # *u.rad
        what_dict['eta'] = etta%(2*np.pi)
        what_dict['M'] = np.exp(par_dict['lnM'])
        allpha = np.arctan2(par_dict['sinalpha'],par_dict['cosalpha']) # *u.rad
        what_dict['alpha'] = allpha%(2*np.pi)
        return what_dict
    
    def whats_this_mean(self, par_dict):
        ''' takes our original parameter set and transforms them to things we recognize
            differs from whats_this by taking the mean of each recongizable parameter'''
        what_dict = {}
        what_dict['r'] = np.mean(np.exp(par_dict['lnr']))*u.kpc
        what_dict['e'] = np.mean(1 - np.exp(par_dict['eParam']))
        etta = np.arctan2(par_dict['sineta'],par_dict['coseta']) # *u.rad
        what_dict['eta'] = np.mean(etta%(2*np.pi))
        what_dict['M'] = np.mean(np.exp(par_dict['lnM'])*self.unit_system['mass'])
        allpha = np.arctan2(par_dict['sinalpha'],par_dict['cosalpha']) # *u.rad
        what_dict['alpha'] = np.mean(allpha%(2*np.pi))
        return what_dict


    def ln_likelihood(self, par_dict):

        par_dict = par_dict.copy()
        par_dict['r'] = np.exp(par_dict['lnr'])
        par_dict['M'] = np.exp(par_dict['lnM'])
        par_dict['e'] = 1 - np.exp(par_dict['eParam'])


        a = par_dict['r']/ (1 - par_dict['e']*par_dict['coseta'])
        eccentricity = par_dict['e']
        eta = np.arctan2(par_dict['sineta'],par_dict['coseta']) # *u.rad
        if hasattr(eta, 'unit'):
            eta = eta.to_value(u.rad)
        eta = eta % (2*np.pi)
        
        alpha = np.arctan2(par_dict['sinalpha'],par_dict['cosalpha']) # *u.rad
        if hasattr(alpha, 'unit'):
            alpha = alpha.to_value(u.rad)
        alpha = alpha % (2*np.pi)
        
        # creating keplerian plane with parameters from par_dict
        inst = LGKepler(eccentricity = eccentricity, 
                        eccentricAnomaly = eta, 
                        semiMajorAxis = a*self.unit_system['length'],
                        totalMass= par_dict['M']*self.unit_system['mass']) 

        # calculate x,y, and vx, vy in kepler plane
        r_kep = inst.separation
        vrad_kep, vtan_kep = inst.vrad_kepler, inst.vtan_kepler
        tperiModel = inst.time

        lghc_pos = coord.CartesianRepresentation( r_kep, 0*u.kpc, 0*u.kpc)
        lghc_vel = coord.CartesianDifferential( vrad_kep, vtan_kep, 0*u.km/u.s)

        # law of cosines crap
        gamma = fiducial_m31_c.separation(self.galcen_frame.galcen_coord)

        # TODO: this is going to have to change when the halo center is offset from the disk center
        sunToMWC = self.galcen_frame.galcen_distance 
        MWCtoM31 = r_kep.to(u.kpc) # separation between MWHC and M31 given by model

        sunToM31 = ( sunToMWC * np.cos(gamma) ) + np.sqrt( (sunToMWC * np.cos(gamma))**2 - sunToMWC**2 + MWCtoM31**2 )

        m31_coord = coord.SkyCoord(ra = fiducial_m31_c.ra,
                                   dec = fiducial_m31_c.dec,
                                   distance = sunToM31)
        
        m31_galcen = m31_coord.transform_to(self.galcen_frame)
        xhat = m31_galcen.cartesian / m31_galcen.cartesian.norm()
        sph = m31_galcen.represent_as('spherical')
        Rz = rotation_matrix(-sph.lon, 'z')
        Ry = rotation_matrix(sph.lat, 'y')
        Rx = rotation_matrix(alpha, 'x')
        yhat = (Rz @ Ry @ Rx) @ [0, 1, 0]
        zhat = np.cross(xhat.xyz.value, yhat)
        lghc_pole = coord.CartesianRepresentation(*zhat)

        # define position and velocities in LGHC frame
        lghc = LocalGroupHalocentric(lghc_pos.with_differentials(lghc_vel),
                                     lg_pole = lghc_pole,
                                     m31_coord = m31_coord)
        modelSol = lghc.transform_to(self.galcen_frame).transform_to(coord.ICRS())

        modely = np.array([modelSol.distance.decompose(self.unit_system).value,
                           modelSol.pm_ra_cosdec.decompose(self.unit_system).value,
                           modelSol.pm_dec.decompose(self.unit_system).value,
                           modelSol.radial_velocity.decompose(self.unit_system).value,
                           tperiModel.decompose(self.unit_system).value])

        dy = self.y - modely

        return -0.5 * dy.T @ self.Cinv @ dy

    def ln_prior(self, par_dict):
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
        lp += par_dict['eParam']

        lp += ln_normal(par_dict['coseta'], 0, 1)
        lp += ln_normal(par_dict['sineta'], 0, 1)
        
        lp += ln_normal(par_dict['cosalpha'], 0, 1)
        lp += ln_normal(par_dict['sinalpha'], 0, 1)

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
