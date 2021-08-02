# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

import astropy.coordinates as coord
import numpy as np
from .keplerianPlane import LGKepler
from .coordinates import LocalGroupHalocentric
from gala.units import UnitSystem
from numpy.linalg import norm
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.table import QTable


class TimingArgumentModel:

    def __init__(self, distance, pm, radial_velocity, tperi,
                 distance_err, pm_err, radial_velocity_err, tperi_err,
                 pm_correlation=0., unit_system=None, prior_bounds=None,
                 galcen_frame=coord.Galactocentric(), m31_sky_c=None,
                 title=''):

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
        self._param_info['cosalpha'] = 1
        self._param_info['sinalpha'] = 1

        self.frozen = {}

        # this is because dictionaries are mutable
        if prior_bounds is None:
            prior_bounds = {}

        # for now, these values are assumed to be in default unit system
        prior_bounds.setdefault('lnr', (np.log(500), np.log(900)))
        prior_bounds.setdefault('eParam', (-18, 0))
        prior_bounds.setdefault('lnM', (np.log(1), np.log(5)))

        self.prior_bounds = prior_bounds

        self.galcen_frame = galcen_frame

        if m31_sky_c is None:
            m31_sky_c = coord.SkyCoord.from_name('M31')
        self.m31_sky_c = coord.SkyCoord(m31_sky_c)

        self.title = str(title)
        
        self.blobs_dtype = [("vrad",float),("vtan",float),("vscale",float),("sunToM31",float)]

    @classmethod
    def from_dataset(cls, data_file, **kwargs):
        '''
        Reads dataset from file.

        Parameters
        ----------
        data_file : str
            full path to dataset file
        **kwargs
            anything that initializer accepts
        '''
        table = QTable.read(data_file)[0]  # grab first (only) row
        if "ra" not in table.colnames:
            m31_sky_c = None
        else:
            m31_sky_c = coord.SkyCoord(table['ra'], table['dec'])

        kwargs.setdefault("m31_sky_c", m31_sky_c)

        if 'galcen_frame_attrs' in table.meta:
            kwargs.setdefault('galcen_frame', coord.Galactocentric(
                **table.meta['galcen_frame_attrs'])
            )

        kwargs.setdefault('title', table.meta.get('title', ''))

        instance = cls(distance=table['distance'],
                       pm=u.Quantity([table['pm_ra_cosdec'],
                                      table['pm_dec']]),
                       radial_velocity=table['radial_velocity'],
                       tperi=table['tperi'],
                       distance_err=table['distance_err'],
                       pm_err=u.Quantity([table['pm_ra_cosdec_err'],
                                          table['pm_dec_err']]),
                       radial_velocity_err=table['radial_velocity_err'],
                       tperi_err=table['tperi_err'],
                       **kwargs)
        return instance

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

    def transform_pars(self, par_dict):
        '''
        Transforms parametrized mcmc parameters to model params

        Parameters
        ----------
        par_dict: dict
            dictionary containing all sampled parameters
        
        Outputs
        -------
        trans_dict: dict
            dictionary of all model parameters 
        '''
        trans_dict = {}
        trans_dict['r'] = np.exp(par_dict['lnr'])
        trans_dict['e'] = 1 - np.exp(par_dict['eParam'])
        etta = np.arctan2(par_dict['sineta'],par_dict['coseta']) # *u.rad
        trans_dict['eta'] = etta%(2*np.pi)
        trans_dict['M'] = np.exp(par_dict['lnM'])
        allpha = np.arctan2(par_dict['sinalpha'],par_dict['cosalpha']) # *u.rad
        return trans_dict

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

        # creating keplerian plane with parameters from par_dict
        inst = LGKepler(eccentricity = eccentricity,
                        eccentricAnomaly = eta,
                        semiMajorAxis = a*self.unit_system['length'],
                        totalMass= par_dict['M']*self.unit_system['mass'])

        # calculate x,y, and vx, vy in kepler plane
        r_kep = inst.separation
        vrad_kep, vtan_kep = inst.vrad_kepler, inst.vtan_kepler
        vscale_kep = inst.vscale
        tperiModel = inst.time

        lghc_pos = coord.CartesianRepresentation( r_kep, 0*u.kpc, 0*u.kpc)
        lghc_vel = coord.CartesianDifferential( vrad_kep, vtan_kep, 0*u.km/u.s)

        # law of cosines crap
        gamma = self.m31_sky_c.separation(self.galcen_frame.galcen_coord)

        # TODO: this is going to have to change when the halo center is offset from the disk center
        sunToMWC = self.galcen_frame.galcen_distance
        MWCtoM31 = r_kep.to(u.kpc) # separation between MWHC and M31 given by model

        sunToM31 = ( sunToMWC * np.cos(gamma) ) + np.sqrt( (sunToMWC * np.cos(gamma))**2 - sunToMWC**2 + MWCtoM31**2 )

        m31_coord = coord.SkyCoord(ra = self.m31_sky_c.ra,
                                   dec = self.m31_sky_c.dec,
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
        
        blobs = [vrad_kep.decompose(self.unit_system).value,
                 vtan_kep.decompose(self.unit_system).value,
                 vscale_kep.decompose(self.unit_system).value, 
                 sunToM31.decompose(self.unit_system).value]

        return -0.5 * dy.T @ self.Cinv @ dy, blobs# here, need to compute and export 

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
#         lp += ln_normal( par_dict['lnr'], np.log(750), np.log(750)/4)
#         lp += ln_normal( par_dict['lnM'], np.log(4), np.log(4)/4)
        lp += par_dict['eParam'] + np.log( 1 - np.exp(par_dict['eParam']) )

        lp += ln_normal( np.sqrt(par_dict['coseta']**2 + par_dict['sineta']**2), 5, 0.1)
        lp += ln_normal( np.sqrt(par_dict['cosalpha']**2 + par_dict['sinalpha']**2), 5, 0.1)

        return lp

    def ln_posterior(self, par_dict):
        # TODO: call ln_likelihood and ln_prior and add the values
        ll, blobs = self.ln_likelihood(par_dict)
        lp =  self.ln_prior(par_dict)
        return ll + lp, blobs

    def __call__(self, par_arr):
        par_dict = self.unpack_pars(par_arr)
        try:
            ln_post, blobs = self.ln_posterior(par_dict)
        except Exception as e:
            print(f"Step failed: {e!s}")
            return (-np.inf, *np.full(len(self.blobs_dtype), np.nan))
#             return (-np.inf, np.nan, np.nan, np.nan, np.nan)

        if not np.isfinite(ln_post):
            return (-np.inf, *np.full(len(self.blobs_dtype), np.nan))
#             return (-np.inf, np.nan, np.nan, np.nan, np.nan)
        return (ln_post, *blobs)

# Defining __call__ makes this possible:
# model = TimingArgumentModel()
# model([1., 5., 0., 1.])

def ln_normal(data_val, model_val, variance):
    ''' computes ln normal given a data value and model predicted value '''
    A = 2*np.pi*variance
    B = ( (data_val - model_val)**2 / variance )
    return -1/2 * ( np.log(A) + B )


# gaussian prior on mass centered with 4e12 with hard bounds and on separation w center at 700
# make the prior on the angles within an anulus
# make the prior on e proportional to e
# vscale = sqrt gm/a
# vrad
# vtan
# save sun-m31 distance
# blobs
# snippets ?? 