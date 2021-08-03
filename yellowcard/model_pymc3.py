import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from aesara_theano_fallback import tensor as tt

# __all__ []


def init_kepler(m31_sky_c, galcen_frame, units,
                rlim=(1e2, 1e4), Mlim=(0.5, 20),
                model=None):
    model = pm.modelcontext(model)

    r = pm.Bound(pm.Normal, *rlim)('r', 700, 100)  # kpc
    M = pm.Bound(pm.Normal, *Mlim)('M', 4.5, 2)  # 1e12 Msun

    eta = pmx.Angle('eta')  # radians
    sineta = pm.Deterministic('sineta', tt.sin(eta))
    coseta = pm.Deterministic('coseta', tt.cos(eta))

    ln1me = pm.Bound(pm.Uniform, -10, 0)('ln(1-e)', -10, 0)
    e = pm.Deterministic('e', 1 - np.exp(ln1me))
    pm.Potential('ln(1-e)_prior_factor', ln1me + tt.log(e))

    a = pm.Deterministic('a', r / (1 - e * coseta))

    vscale = pm.Deterministic(
        'vscale',
        tt.sqrt(units.get_constant('G') * M / a)
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

    gamma = m31_sky_c.separation(galcen_frame.galcen_coord).to_value(u.radian)
    sun_galcen_dist = galcen_frame.galcen_distance.to_value(
        units['length'])
    sun_m31_dist = pm.Deterministic(
        name='sun_m31_dist',
        var=(sun_galcen_dist * np.cos(gamma)) + np.sqrt(
            r**2 - sun_galcen_dist**2 * np.sin(gamma)**2
        )
    )

    pm.Deterministic(
        'm31_icrs_xyz',
        tt_sph_to_xyz(sun_m31_dist, m31_sky_c.ra.radian, m31_sky_c.dec.radian)
    )


def setup_obs(m31_sky_c, galcen_frame, units, model=None):
    model = pm.modelcontext(model)

    # unit helpers
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        v_per_D_to_masyr = (units['velocity'] / units['length']).to(u.mas/u.yr)
        v_to_kms = units['velocity'].to(u.km/u.s)

    # Matrix to go from ICRS to Galactocentric
    R_I2G, offset_I2G = coord.builtin_frames.galactocentric.get_matrix_vectors(
        galcen_frame, inverse=False)
    dxyz_I2G = offset_I2G.xyz.to_value(units['length'])
    # dvxyz_I2G = offset_I2G.differentials['s'].d_xyz.to_value(units['velocity'])

    # Matrix to go from Galactocentric to ICRS
    R_G2I, offset_G2I = coord.builtin_frames.galactocentric.get_matrix_vectors(
        galcen_frame, inverse=True)
    # dxyz_G2I = offset_G2I.xyz.to_value(units['length'])
    dvxyz_G2I = offset_G2I.differentials['s'].d_xyz.to_value(units['velocity'])

    # tangent bases: ra, dec, r
    m31_ra_rad = m31_sky_c.ra.radian
    m31_dec_rad = m31_sky_c.dec.radian
    M = np.array([
        [-np.sin(m31_ra_rad),
         np.cos(m31_ra_rad),
         0.],
        [-np.sin(m31_dec_rad) * np.cos(m31_ra_rad),
         -np.sin(m31_dec_rad) * np.sin(m31_ra_rad),
         np.cos(m31_dec_rad)],
        [np.cos(m31_dec_rad) * np.cos(m31_ra_rad),
         np.cos(m31_dec_rad) * np.sin(m31_ra_rad),
         np.sin(m31_dec_rad)]
    ])

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

    # v_travel!
#     v_travel_galcen_lon = pm.Normal('v_travel_lon', 56., 9.)
#     v_travel_galcen_lat = pm.Normal('v_travel_lat', -34, 10.)
#     v_travel_galcen_mag = pm.Normal(
#         'v_travel_mag',
#         (32. * u.km/u.s).to_value(units['velocity']),
#         (4. * u.km/u.s).to_value(units['velocity'])
#     )
#     v_travel_galcen_xyz = tt_sph_to_xyz(
#         v_travel_galcen_mag,
#         tt.deg2rad(v_travel_galcen_lon),
#         tt.deg2rad(v_travel_galcen_lat)
#     )

#     # APW thinks this is a minus sign...
#     vtmp = tt.dot(R_LGtoG, v_LG)
#     vtmp = vtmp - vtravel.to(units['velocity']).value
    vtmp = tt.dot(R_LGtoG, v_LG)

    # x_I = tt.dot(R_G2I, tt.dot(R_LGtoG, x_LG)) + dxyz_G2I
    v_I = tt.dot(R_G2I, vtmp) + dvxyz_G2I
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
