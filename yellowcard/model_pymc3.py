import astropy.coordinates as coord
from astropy.coordinates.builtin_frames.galactocentric import get_matrix_vectors
import astropy.units as u
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from aesara_theano_fallback import tensor as tt
from gala.units import UnitSystem

from .model_mixin import ModelMixin

__all__ = ['TimingArgumentModelPymc3']


class TimingArgumentModelPymc3(ModelMixin):

    def __init__(
        self,
        distance,
        pm,
        radial_velocity,
        tperi,
        distance_err,
        pm_err,
        radial_velocity_err,
        tperi_err,
        pm_correlation=0.0,
        units=None,
        galcen_frame=coord.Galactocentric(),
        m31_sky_c=None,
        title="",
        include_vtravel=True,
        vtravel_mag=32,
        vtravel_mag_err=4,
        **kwargs
    ):

        # this is because dictionaries are mutable
        if units is None:
            units = UnitSystem(
                u.kpc, u.Unit(1e12 * u.Msun), u.Gyr, u.radian
            )
        self.units = units

        self.distance = distance
        self.pm = pm
        self.radial_velocity = radial_velocity
        self.tperi = tperi

        self.distance_err = distance_err
        self.pm_err = pm_err
        self.radial_velocity_err = radial_velocity_err
        self.pm_correlation = pm_correlation
        self.tperi_err = tperi_err

        self.title = str(title)
        self.include_vtravel = include_vtravel
        self.vtravel_mag = vtravel_mag
        self.vtravel_mag_err = vtravel_mag_err

        if m31_sky_c is None:
            m31_sky_c = coord.SkyCoord.from_name("M31")
        self.m31_sky_c = coord.SkyCoord(m31_sky_c)

        self.galcen_frame = galcen_frame

        self.data = {
            'obs_m31_ra': self.m31_sky_c.ra.to_value(u.rad),
            'obs_m31_dec': self.m31_sky_c.dec.to_value(u.rad),
            'obs_distance': self.distance.to_value(units['length']),
            'obs_distance_err': self.distance_err.to_value(units['length']),
            'obs_pmra': self.pm[0].to_value(units['angular speed']),
            'obs_pmra_err': self.pm_err[0].to_value(units['angular speed']),
            'obs_pmdec': self.pm[1].to_value(units['angular speed']),
            'obs_pmdec_err': self.pm_err[1].to_value(units['angular speed']),
            'obs_rv': self.radial_velocity.to_value(units['velocity']),
            'obs_rv_err': self.radial_velocity_err.to_value(units['velocity']),
            'obs_tperi': self.tperi.to_value(units['time']),
            'obs_tperi_err': self.tperi_err.to_value(units['time']),
        }

        import pymc3 as pm
        with pm.Model() as model:
            self._init_kepler(model)
            self._setup_obs(model)

            for k in model.named_vars:
                if k.startswith('obs') and k not in self.data:
                    print(f"{k} missing from data dict!")

            pm.set_data(self.data)

        self.pymc3_model = model

    def _init_kepler(self, model=None):
        with pm.modelcontext(model):
            # TODO: make these customizable
            rlim = (1e2, 1e4)
            Mlim = (0.5, 20)
            r = pm.Bound(pm.Normal, *rlim)('r', 700, 100)  # kpc
            M = pm.Bound(pm.Normal, *Mlim)('M', 4.5, 3)  # 1e12 Msun

            eta = pmx.Angle('eta')  # radians
            sineta = pm.Deterministic('sineta', tt.sin(eta))
            coseta = pm.Deterministic('coseta', tt.cos(eta))

            ln1me = pm.Bound(pm.Uniform, -10, 0)('ln(1-e)', -10, 0)
            e = pm.Deterministic('e', 1 - tt.exp(ln1me))
            pm.Potential('ln(1-e)_prior_factor', ln1me + tt.log(e))

            a = pm.Deterministic('a', r / (1 - e * coseta))

            vscale = pm.Deterministic(
                'vscale',
                tt.sqrt(self.units.get_constant('G') * M / a)
            )
            pm.Deterministic(
                'vrad',
                vscale * (e * sineta) / (1 - e * coseta)
            )
            pm.Deterministic(
                'vtan',
                vscale * tt.sqrt(1 - e**2) / (1 - e * coseta)
            )
            pm.Deterministic(
                'tperi',
                a / vscale * ((eta % (2*np.pi)) - e * sineta)
            )

            m31_ra = pm.Data('obs_m31_ra', np.nan)
            m31_dec = pm.Data('obs_m31_dec', np.nan)

            gamma = tt_angular_separation(
                self.galcen_frame.galcen_coord.ra.radian,
                self.galcen_frame.galcen_coord.dec.radian,
                m31_ra,
                m31_dec
            )
            sun_galcen_dist = self.galcen_frame.galcen_distance.to_value(
                self.units['length'])
            sun_m31_dist = pm.Deterministic(
                name='sun_m31_dist',
                var=(sun_galcen_dist * np.cos(gamma)) + np.sqrt(
                    r**2 - sun_galcen_dist**2 * np.sin(gamma)**2
                )
            )

            pm.Deterministic(
                'm31_icrs_xyz',
                tt_sph_to_xyz(sun_m31_dist, m31_ra, m31_dec)
            )

    def _setup_obs(self, model=None):
        units = self.units
        with pm.modelcontext(model):
            # Matrix to go from ICRS to Galactocentric
            R_I2G, offset_I2G = get_matrix_vectors(
                self.galcen_frame, inverse=False)
            dxyz_I2G = offset_I2G.xyz.to_value(units['length'])
            # dvxyz_I2G = offset_I2G.differentials['s'].d_xyz.to_value(units['velocity'])

            # Matrix to go from Galactocentric to ICRS
            R_G2I, offset_G2I = get_matrix_vectors(
                self.galcen_frame, inverse=True)
            # dxyz_G2I = offset_G2I.xyz.to_value(units['length'])
            dvxyz_G2I = offset_G2I.differentials['s'].d_xyz.to_value(units['velocity'])

            # tangent bases: ra, dec, r
            m31_ra_rad = model.named_vars['obs_m31_ra']
            m31_dec_rad = model.named_vars['obs_m31_dec']
            M = tt.as_tensor([
                -tt.sin(m31_ra_rad),
                tt.cos(m31_ra_rad),
                0.,
                -tt.sin(m31_dec_rad) * tt.cos(m31_ra_rad),
                -tt.sin(m31_dec_rad) * tt.sin(m31_ra_rad),
                tt.cos(m31_dec_rad),
                tt.cos(m31_dec_rad) * tt.cos(m31_ra_rad),
                tt.cos(m31_dec_rad) * tt.sin(m31_ra_rad),
                tt.sin(m31_dec_rad)
            ]).reshape((3, 3))

            # Coordinate system orientation:
            alpha = pmx.Angle('alpha')

            m31_galcen_xyz = tt.dot(R_I2G, model.named_vars['m31_icrs_xyz']) + dxyz_I2G
            m31_galcen_lon = tt.arctan2(m31_galcen_xyz[1], m31_galcen_xyz[0])
            m31_galcen_lat = tt.arcsin(m31_galcen_xyz[2] / model.named_vars['r'])

            xhat = m31_galcen_xyz / model.named_vars['r']

            Rz = tt_rotation_matrix(-m31_galcen_lon, 'z')
            Ry = tt_rotation_matrix(m31_galcen_lat, 'y')
            Rx = tt_rotation_matrix(alpha, 'x')
            yhat = tt.dot(tt.dot(Rz, tt.dot(Ry, Rx)), [0, 1, 0.])
            zhat = tt_cross(xhat, yhat)
            R_LGtoG = tt.stack((xhat, yhat, zhat), axis=1)

            # vtravel things here:
            # TODO: make these variables
            if self.include_vtravel is True:
                vtravel_mag = pm.Normal("vtravel_mag",
                                        (self.vtravel_mag*u.km/u.s).to_value(self.units["velocity"]), 
                                        (self.vtravel_mag_err*u.km/u.s).to_value(self.units["velocity"]))

                # these are in galactocentric 
                vtravel_lon = pm.Normal("vtravel_lon",
                                        (56*u.deg).to_value(u.rad), 
                                        (9*u.deg).to_value(u.rad))

                vtravel_lat = pm.Normal("vtravel_lat",
                                        (-34*u.deg).to_value(u.rad), 
                                        (9.5*u.deg).to_value(u.rad))
                                        
                vtravel_galcen_xyz = tt_sph_to_xyz(vtravel_mag, 
                                                   vtravel_lon, 
                                                   vtravel_lat)

            else:
                vtravel_galcen_xyz = tt.zeros(3)


            # x_LG = tt.as_tensor([
            #     model.named_vars['r'],
            #     0.,
            #     0.
            # ])
            v_LG = tt.as_tensor([
                model.named_vars['vrad'],
                model.named_vars['vtan'],
                0.
            ])

            # x_I = tt.dot(R_G2I, tt.dot(R_LGtoG, x_LG)) + dxyz_G2I
            v_G = tt.dot(R_LGtoG, v_LG) - vtravel_galcen_xyz 
            v_I = tt.dot(R_G2I, v_G) + dvxyz_G2I

            v_I_tangent_plane = tt.dot(M, v_I)  # alpha, delta, radial

            model_pmra = pm.Deterministic(
                'model_pmra',
                v_I_tangent_plane[0] / model.named_vars['sun_m31_dist']
            )
            model_pmdec = pm.Deterministic(
                'model_pmdec',
                v_I_tangent_plane[1] / model.named_vars['sun_m31_dist']
            )
            model_rv = pm.Deterministic(
                'model_rv',
                v_I_tangent_plane[2]
            )

            obs_m31_distance = pm.Data('obs_distance', np.nan)
            obs_m31_distance_err = pm.Data('obs_distance_err', np.nan)
            pm.Normal('LL_distance',
                      model.named_vars['sun_m31_dist'],
                      observed=obs_m31_distance,
                      sd=obs_m31_distance_err)

            obs_tperi = pm.Data('obs_tperi', np.nan)
            obs_tperi_err = pm.Data('obs_tperi_err', np.nan)
            pm.Normal('LL_tperi',
                      model.named_vars['tperi'],
                      observed=obs_tperi,
                      sd=obs_tperi_err)

            obs_m31_pmra = pm.Data('obs_pmra', np.nan)
            obs_m31_pmra_err = pm.Data('obs_pmra_err', np.nan)
            pm.Normal('LL_pmra',
                      model_pmra,
                      observed=obs_m31_pmra,
                      sd=obs_m31_pmra_err)

            obs_m31_pmdec = pm.Data('obs_pmdec', np.nan)
            obs_m31_pmdec_err = pm.Data('obs_pmdec_err', np.nan)
            pm.Normal('LL_pmdec',
                      model_pmdec,
                      observed=obs_m31_pmdec,
                      sd=obs_m31_pmdec_err)

            obs_m31_rv = pm.Data('obs_rv', np.nan)
            obs_m31_rv_err = pm.Data('obs_rv_err', np.nan)
            pm.Normal('LL_radial_velocity',
                      model_rv,
                      observed=obs_m31_rv,
                      sd=obs_m31_rv_err)


def tt_sph_to_xyz(r, lon, lat):
    return tt.as_tensor([
        r * tt.cos(lon) * tt.cos(lat),
        r * tt.sin(lon) * tt.cos(lat),
        r * tt.sin(lat)
    ])


def tt_cross(a, b):
    return tt.as_tensor([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])


def tt_rotation_matrix(angle_rad, axis):
    s = tt.sin(angle_rad)
    c = tt.cos(angle_rad)

    if axis == 'x':
        R = tt.as_tensor([
            1., 0, 0,
            0, c, s,
            0, -s, c
        ])

    elif axis == 'y':
        R = tt.as_tensor([
            c, 0, -s,
            0, 1., 0,
            s, 0, c
        ])

    elif axis == 'z':
        R = tt.as_tensor([
            c, s, 0,
            -s, c, 0,
            0, 0, 1.
        ])

    else:
        raise ValueError('borked')

    return tt.reshape(R, (3, 3))


def tt_angular_separation(lon1, lat1, lon2, lat2):
    sdlon = tt.sin(lon2 - lon1)
    cdlon = tt.cos(lon2 - lon1)
    slat1 = tt.sin(lat1)
    slat2 = tt.sin(lat2)
    clat1 = tt.cos(lat1)
    clat2 = tt.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return tt.arctan2(tt.sqrt(num1**2 + num2**2), denominator)
