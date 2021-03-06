# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates.matrix_utilities import rotation_matrix
from gala.units import UnitSystem

from .coordinates import LocalGroupHalocentric
from .keplerianPlane import LGKepler
from .model_mixin import ModelMixin


class TimingArgumentModel(ModelMixin):

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
        prior_bounds=None,
        galcen_frame=coord.Galactocentric(),
        m31_sky_c=None,
        title="",
    ):

        # this is because dictionaries are mutable
        if units is None:
            units = UnitSystem(
                u.kpc, u.Unit(1e12 * u.Msun), u.Gyr, u.radian
            )
        self.units = units

        self.dist = distance
        self.pm = pm
        self.rv = radial_velocity
        self.tperi = tperi

        self.dist_err = distance_err
        self.pm_err = pm_err
        self.rv_err = radial_velocity_err
        self.pm_corr = pm_correlation
        self.tperi_err = tperi_err

        self.title = str(title)

        if m31_sky_c is None:
            m31_sky_c = coord.SkyCoord.from_name("M31")
        self.m31_sky_c = coord.SkyCoord(m31_sky_c)

        self.galcen_frame = galcen_frame

        # ---

        self.y = np.array([
            self.dist.to_value(self.units['length']),
            *self.pm.to_value(self.units['angular speed']),
            self.rv.to_value(self.units['velocity']),
            self.tperi.to_value(self.units['time'])
        ])

        # TODO: we're ignoring the pm correlation
        errs = np.array([
            self.dist_err.to_value(self.units['length']),
            *self.pm_err.to_value(self.units['angular speed']),
            self.rv_err.to_value(self.units['velocity']),
            self.tperi_err.to_value(self.units['time'])
        ])
        self.Cinv = np.diag(1 / errs ** 2)

        # becoming webster
        self._param_info = {}

        # lengths of each of the parameters
        # eParam = 'ln(1-e)'
        self._param_info["lnr"] = 1
        self._param_info["eParam"] = 1
        self._param_info["coseta"] = 1
        self._param_info["sineta"] = 1
        self._param_info["lnM"] = 1
        self._param_info["cosalpha"] = 1
        self._param_info["sinalpha"] = 1

        self.frozen = {}

        # this is because dictionaries are mutable
        if prior_bounds is None:
            prior_bounds = {}

        # for now, these values are assumed to be in default unit system
        prior_bounds.setdefault("lnr", (np.log(500), np.log(900)))
        prior_bounds.setdefault("eParam", (-18, 0))
        prior_bounds.setdefault("lnM", (np.log(1), np.log(5)))

        self.prior_bounds = prior_bounds

        self.blobs_dtype = [
            ("vrad", float),
            ("vtan", float),
            ("vscale", float),
            ("sunToM31", float),
        ]

    def unpack_pars(self, par_list):
        i = 0
        par_dict = {}
        for key, par_len in self._param_info.items():
            if key in self.frozen:
                par_dict[key] = self.frozen[key]
            else:
                par_dict[key] = np.squeeze(par_list[i : i + par_len])
                i += par_len

        return par_dict

    def pack_pars(self, par_dict):
        parvec = []
        for i, k in enumerate(self._param_info):
            if k not in self.frozen:
                parvec.append(np.atleast_1d(par_dict[k]))
        return np.concatenate(parvec)

    def transform_pars(self, par_dict, with_units=False):
        """
        Transforms parametrized mcmc parameters to model params

        Parameters
        ----------
        par_dict: dict
            dictionary containing all sampled parameters

        Outputs
        -------
        trans_dict: dict
            dictionary of all model parameters
        """
        trans_dict = {}
        trans_dict["r"] = np.exp(par_dict["lnr"])
        trans_dict["e"] = 1 - np.exp(par_dict["eParam"])
        trans_dict["eta"] = np.arctan2(
            np.asarray(par_dict["sineta"]), np.asarray(par_dict["coseta"])
        )
        trans_dict["eta"] %= 2 * np.pi
        trans_dict["M"] = np.exp(par_dict["lnM"])
        trans_dict["alpha"] = np.arctan2(
            np.asarray(par_dict["sinalpha"]), np.asarray(par_dict["cosalpha"])
        )

        if with_units:
            trans_dict["r"] = trans_dict["r"] * self.units["length"]
            trans_dict["eta"] = trans_dict["eta"] * u.rad
            trans_dict["M"] = trans_dict["M"] * self.units["mass"]
            trans_dict["alpha"] = trans_dict["alpha"] * u.rad

        return trans_dict

    def get_orbit_pars(self, par_dict):
        p = par_dict

        out = {}
        out["a"] = p["r"] / (1 - p["e"] * p["coseta"])
        out["vscale"] = np.sqrt(G * p["M"] / out["a"])
        out["vrad"] = (
            out["vscale"] * (p["e"] * p["sineta"]) / (1 - p["e"] * p["coseta"])
        )
        out["vtan"] = (
            out["vscale"]
            * np.sqrt(1 - p["e"] ** 2)
            / (1 - p["e"] * p["coseta"])
        )
        out["tperi"] = (
            out["a"] / out["vscale"]
            * (p["eta"].to_value(u.rad) - p["e"] * p["sineta"])
        )

        # Sun-M31 distance - law of cosines crap
        gamma = self.m31_sky_c.separation(self.galcen_frame.galcen_coord)
        sun_galcen_dist = self.galcen_frame.galcen_distance
        out["sun_m31_dist"] = (sun_galcen_dist * np.cos(gamma)) + np.sqrt(
            p["r"] ** 2 - sun_galcen_dist ** 2 * np.sin(gamma) ** 2
        )

        return out

    def ln_likelihood(self, par_dict):
        p = self.transform_pars(par_dict, with_units=True)
        p.update(par_dict)
        p.update(self.get_orbit_pars(p))

        lghc_pos = coord.CartesianRepresentation(
            p["r"].to(self.units['length']),
            0 * u.kpc,
            0 * u.kpc
        )
        lghc_vel = coord.CartesianDifferential(
            p["vrad"].to(self.units['velocity']),
            p["vtan"].to(self.units['velocity']),
            0 * u.km / u.s
        )

        m31_coord = coord.SkyCoord(
            ra=self.m31_sky_c.ra,
            dec=self.m31_sky_c.dec,
            distance=p["sun_m31_dist"].to(u.kpc),
        )

        m31_galcen = m31_coord.transform_to(self.galcen_frame)
        xhat = m31_galcen.cartesian / m31_galcen.cartesian.norm()
        sph = m31_galcen.represent_as("spherical")
        Rz = rotation_matrix(-sph.lon, "z")
        Ry = rotation_matrix(sph.lat, "y")
        Rx = rotation_matrix(p["alpha"], "x")
        yhat = (Rz @ Ry @ Rx) @ [0, 1, 0]
        zhat = np.cross(xhat.xyz.value, yhat)
        lghc_pole = coord.CartesianRepresentation(*zhat)

        # define position and velocities in LGHC frame
        lghc = LocalGroupHalocentric(
            lghc_pos.with_differentials(lghc_vel),
            lg_pole=lghc_pole,
            m31_coord=m31_coord,
        )
        model_galcen = lghc.transform_to(self.galcen_frame)
        model_icrs = model_galcen.transform_to(coord.ICRS())

        modely = np.array([
            model_icrs.distance.to_value(self.units['length']),
            model_icrs.pm_ra_cosdec.to_value(self.units['angular speed']),
            model_icrs.pm_dec.to_value(self.units['angular speed']),
            model_icrs.radial_velocity.to_value(self.units['velocity']),
            p['tperi'].to_value(self.units['time'])
        ])

        dy = self.y - modely

        blobs = [
            p['vrad'].decompose(self.units).value,
            p['vtan'].decompose(self.units).value,
            p['vscale'].decompose(self.units).value,
            p['sun_m31_dist'].decompose(self.units).value,
        ]

        return -0.5 * dy.T @ self.Cinv @ dy, blobs

    def ln_prior(self, par_dict):
        for name, shape in self._param_info.items():
            if name not in self.prior_bounds:
                continue

            if shape == 1:
                if (
                    not self.prior_bounds[name][0]
                    < par_dict[name]
                    < self.prior_bounds[name][1]
                ):
                    return -np.inf
            else:
                for value in par_dict[name]:
                    if (
                        not self.prior_bounds[name][0]
                        < value
                        < self.prior_bounds[name][1]
                    ):
                        return -np.inf

        lp = 0
        #         lp += ln_normal( par_dict['lnr'], np.log(750), np.log(750)/4)
        #         lp += ln_normal( par_dict['lnM'], np.log(4), np.log(4)/4)
        lp += par_dict["eParam"] + np.log(1 - np.exp(par_dict["eParam"]))

        # Gaussian annuli:
        # lp += ln_normal(
        #     np.sqrt(par_dict["coseta"] ** 2 + par_dict["sineta"] ** 2), 5, 0.1
        # )
        # lp += ln_normal(
        #     np.sqrt(par_dict["cosalpha"] ** 2 + par_dict["sinalpha"] ** 2),
        #     5,
        #     0.1,
        # )

        # Isotropic Gaussians for now...
        lp += -0.5 * (par_dict["coseta"] ** 2 + par_dict["sineta"] ** 2)
        lp += -0.5 * (par_dict["cosalpha"] ** 2 + par_dict["sinalpha"] ** 2)

        return lp

    def ln_posterior(self, par_dict):
        # TODO: call ln_likelihood and ln_prior and add the values
        ll, blobs = self.ln_likelihood(par_dict)
        lp = self.ln_prior(par_dict)
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
    """computes ln normal given a data value and model predicted value"""
    A = 2 * np.pi * variance
    B = (data_val - model_val) ** 2 / variance
    return -1 / 2 * (np.log(A) + B)


# gaussian prior on mass centered with 4e12 with hard bounds and on separation w
# center at 700
# make the prior on the angles within an anulus
# make the prior on e proportional to e
# vscale = sqrt gm/a
# vrad
# vtan
# save sun-m31 distance
# blobs
# snippets ??
