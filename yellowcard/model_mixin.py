import astropy.coordinates as coord
import astropy.units as u
from astropy.table import QTable

__all__ = ['ModelMixin']


class ModelMixin:

    @classmethod
    def from_dataset(cls, data_file, **kwargs):
        """
        Reads dataset from file.

        Parameters
        ----------
        data_file : str
            full path to dataset file
        **kwargs
            anything that initializer accepts
        """
        table = QTable.read(data_file)[0]  # grab first (only) row
        if "ra" not in table.colnames:
            m31_sky_c = None
        else:
            m31_sky_c = coord.SkyCoord(table["ra"], table["dec"])

        kwargs.setdefault("m31_sky_c", m31_sky_c)

        if "galcen_frame_attrs" in table.meta:
            kwargs.setdefault(
                "galcen_frame",
                coord.Galactocentric(**table.meta["galcen_frame_attrs"]),
            )

        kwargs.setdefault("title", table.meta.get("title", ""))

        instance = cls(
            distance=table["distance"],
            pm=u.Quantity([table["pm_ra_cosdec"], table["pm_dec"]]),
            radial_velocity=table["radial_velocity"],
            tperi=table["tperi"],
            distance_err=table["distance_err"],
            pm_err=u.Quantity([table["pm_ra_cosdec_err"], table["pm_dec_err"]]),
            radial_velocity_err=table["radial_velocity_err"],
            tperi_err=table["tperi_err"],
            **kwargs,
        )
        return instance
