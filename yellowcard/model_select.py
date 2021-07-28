import astropy.coordinates as coord
import numpy as np
from yellowcard.model import TimingArgumentModel 
from yellowcard.coordinates import fiducial_m31_c,LocalGroupHalocentric
import astropy.units as u

###################################################
# testing VDM 2012 with zero transverse velocity! #
###################################################
def model_testing():
    '''currently changing d_err'''
    galcen_frame = coord.Galactocentric(galcen_distance = 8.29*u.kpc, 
                                        galcen_v_sun = [11.1, 12.24 + 239.3, 7.25] * u.km/u.s, z_sun = 0*u.pc)

    m31_c = coord.SkyCoord(ra = fiducial_m31_c.ra,
                           dec = fiducial_m31_c.dec,
                           distance = 770*u.kpc)
    m31_lgh = m31_c.transform_to(galcen_frame).transform_to(LocalGroupHalocentric())
    pos = m31_lgh.data.represent_as(coord.SphericalRepresentation)
    v_rad_only = coord.SphericalDifferential(0*u.mas/u.yr, 0*u.mas/u.yr, -110*u.km/u.s)
    lgh = LocalGroupHalocentric(pos.with_differentials(v_rad_only))
    v_rad_only_icrs = lgh.transform_to(galcen_frame).transform_to(coord.ICRS())

    # distance to m31
    D, D_err = (770, 5)* u.kpc # in kpc
    D_term = (D_err/D)**2

    model = TimingArgumentModel(distance = D,
                                pm = u.Quantity([v_rad_only_icrs.pm_ra_cosdec, v_rad_only_icrs.pm_dec]),
                                radial_velocity = v_rad_only_icrs.radial_velocity,
                                tperi = 13.75*u.Gyr,
                                distance_err = D_err,
                                pm_err = u.Quantity([1*u.microarcsecond/u.yr, 1*u.microarcsecond/u.yr]),
                                radial_velocity_err = 1 * u.km / u.s,
                                tperi_err = .11 * u.Gyr,
                                galcen_frame = galcen_frame)
    return model

###################################################
# testing VDM 2012 with zero transverse velocity! #
###################################################
def model_vdm_rad():
    galcen_frame = coord.Galactocentric(galcen_distance = 8.29*u.kpc, 
                                        galcen_v_sun = [11.1, 12.24 + 239.3, 7.25] * u.km/u.s, z_sun = 0*u.pc)

    m31_c = coord.SkyCoord(ra = fiducial_m31_c.ra,
                           dec = fiducial_m31_c.dec,
                           distance = 770*u.kpc)
    m31_lgh = m31_c.transform_to(galcen_frame).transform_to(LocalGroupHalocentric())
    pos = m31_lgh.data.represent_as(coord.SphericalRepresentation)
    v_rad_only = coord.SphericalDifferential(0*u.mas/u.yr, 0*u.mas/u.yr, -110*u.km/u.s)
    lgh = LocalGroupHalocentric(pos.with_differentials(v_rad_only))
    v_rad_only_icrs = lgh.transform_to(galcen_frame).transform_to(coord.ICRS())

    # distance to m31
    D, D_err = (770, 40)* u.kpc # in kpc
    D_term = (D_err/D)**2

    model = TimingArgumentModel(distance = D,
                                pm = u.Quantity([v_rad_only_icrs.pm_ra_cosdec, v_rad_only_icrs.pm_dec]),
                                radial_velocity = v_rad_only_icrs.radial_velocity,
                                tperi = 13.75*u.Gyr,
                                distance_err = D_err,
                                pm_err = u.Quantity([1*u.microarcsecond/u.yr, 1*u.microarcsecond/u.yr]),
                                radial_velocity_err = 1 * u.km / u.s,
                                tperi_err = .11 * u.Gyr,
                                galcen_frame = galcen_frame)
    return model
    
############    
# VDM 2012 #
############
def model_vdm():
    galcen_frame = coord.Galactocentric(galcen_distance = 8.29*u.kpc, 
                                        galcen_v_sun = [11.1, 12.24 + 239.3, 7.25] * u.km/u.s, z_sun = 0*u.pc)

    # vw and vn + error
    vW, vW_err = (-125.2, 30.8)* u.km/u.s # in kpc
    vN, vN_err = (-73.8, 28.4)* u.km/u.s # in kpc
    vW_term = (vW_err/vW)**2
    vN_term = (vN_err/vN)**2

    # distance to m31
    D, D_err = (770, 40)* u.kpc # in kpc
    D_term = (D_err/D)**2

    # calculate equiv pms 
    mu_alpha_star, mu_delta = ( ((- vW / D)*u.rad).to(u.microarcsecond/u.yr), ((vN / D)*u.rad).to(u.microarcsecond/u.yr))

    # equiv pm errors
    mu_alpha_star_err = np.sqrt( mu_alpha_star**2 * (vW_term - D_term) ) 
    mu_delta_err = np.sqrt( mu_delta**2 * (vN_term - D_term) ) 

    # print(mu_alpha_star, mu_delta, mu_alpha_star_err, mu_delta_err)

    model = TimingArgumentModel(distance = D,
                                pm = u.Quantity([mu_alpha_star, mu_delta]),
                                radial_velocity = -301*u.km/u.s,
                                tperi = 13.75*u.Gyr,
                                distance_err = D_err,
                                pm_err = u.Quantity([mu_alpha_star_err, mu_delta_err]),
                                radial_velocity_err = 1 * u.km / u.s,
                                tperi_err = .11 * u.Gyr,
                                galcen_frame = galcen_frame)
    return model
    
############    
# fiducial #
############
def model_fid():
    model = TimingArgumentModel(distance = fiducial_m31_c.distance,
                                pm = u.Quantity([fiducial_m31_c.pm_ra_cosdec, fiducial_m31_c.pm_dec]),
                                radial_velocity = fiducial_m31_c.radial_velocity,
                                tperi = 13.7*u.Gyr,
                                distance_err = 10*u.kpc,
                                pm_err = u.Quantity([10*u.microarcsecond/u.yr, 10*u.microarcsecond/u.yr]),
                                radial_velocity_err = 10 * u.km / u.s,
                                tperi_err = .25 * u.Gyr,
                                )
    return model
    
    