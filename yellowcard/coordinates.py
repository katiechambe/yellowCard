# Third-party
import astropy.coordinates as coord
import astropy.units as u

__all__ = ['LocalGroupHalocentric', 'LocalGroupBarycentric']

# Sky coordinates, Distance and RV from Mcconachie 2012
# Proper motions from Table 2: (Blue) https://arxiv.org/pdf/2012.09204.pdf
fiducial_m31_c = coord.SkyCoord(
    ra=10.68470833 * u.deg,
    dec=41.26875 * u.deg,
    distance=731 * u.kpc,
    pm_ra_cosdec=48.98 * u.microarcsecond/u.yr,
    pm_dec=-36.85 * u.microarcsecond/u.yr,
    radial_velocity=-300 * u.km/u.s
)

galcen_frame = coord.Galactocentric()
m31_galcen = fiducial_m31_c.transform_to(galcen_frame)
m31_galcen_pos = m31_galcen.data.without_differentials()
m31_galcen_vel = m31_galcen.velocity
L_mw_m31 = m31_galcen_pos.cross(m31_galcen_vel)
fiducial_lg_pole = L_mw_m31 / L_mw_m31.norm()


class LocalGroupHalocentric(coord.BaseCoordinateFrame):
    """
    Position at the Milky Way Halo barycenter, x-axis toward M31,
    z-axis toward the Local Group angular momentum vector.
    """

    default_representation = coord.CartesianRepresentation
    default_differential = coord.CartesianDifferential

    # Frame attributes
    m31_coord = coord.CoordinateAttribute(
        frame=coord.ICRS,
        default=fiducial_m31_c
    )

    # Pole of Local Group coordinate system at MW Halocenter
    lg_pole = coord.CartesianRepresentationAttribute(
        default=fiducial_lg_pole,
        unit=u.one
    )


class LocalGroupBarycentric(coord.BaseCoordinateFrame):
    """
    Position at the Milky Way—M31 barycenter, x-axis toward M31,
    z-axis toward the Local Group angular momentum vector.
    """

    default_representation = coord.CartesianRepresentation
    default_differential = coord.CartesianDifferential

    # Frame attributes
    m31_coord = coord.CoordinateAttribute(
        frame=coord.ICRS,
        default=fiducial_m31_c
    )

    M_MW_over_M_M31 = coord.Attribute(
        default=0.78
    )

    M_LG = coord.QuantityAttribute(
        default=2.64e12*u.Msun,
        unit=u.Msun
    )


def get_galcen_to_lg_transform(lg_frame, galcen_frame, matrix_only=False,
                               inverse=False):
    """
    This function returns the matrix and (position and velocity) offset
    vectors to transform from a Milky Way Galactocentric reference frame to
    a Local Group Barycenter reference frame.

    Astropy coordinate frames are mostly defined as affine transformations from
    one frame to another. Transformations between inertial frames are given by
    affine transformations, which are defined as A x + b, where A is a matrix
    (typically a rotation matrix: orthogonal matrix with determinant 1),
    x and b are vectors.

    Parameters
    ----------
    lg_frame : `yellowcard.coordinates.LocalGroupHalocentric` instance
    galcen_frame : `astropy.coordinates.Galactocentric` instance
    matrix_only : bool (optional)
        If True, return only the rotation matrix.
    inverse : bool (optional)
        If True, returns the matrix (and offset) for the inverse transform, to
        go from `LocalGroupHalocentric` to `Galactocentric`.

    """
    # shorthand
    lg = lg_frame

    # Get the line connecting M31 to MW center, and angular momentum
    # vector to specify the orientation / rotation around the line
    m31_galcen = lg.m31_coord.transform_to(galcen_frame)
    m31_galcen_pos = m31_galcen.data.without_differentials()
    lg_pole = lg.lg_pole

    # Rotation matrix to align x(Galcen) with the vector to M31 and
    # z(Galcen) with the LG angular momentum vector
    new_x = m31_galcen_pos / m31_galcen_pos.norm()
    new_z = lg_pole
    new_y = - new_x.cross(new_z)
    R = coord.concatenate_representations((new_x, new_y, new_z)).xyz.T

    if matrix_only:
        if inverse:
            return R.T
        else:
            return R

    # TODO: possible broken below here

    # Compute the offset as well: we then need masses and mass ratios
    M_M31 = lg.M_LG / (1 + lg.M_MW_over_M_M31)

    # Positional offset to the barycenter
    # - This is defined already in the LG frame!
    dpos = coord.CartesianRepresentation(lg.m31_coord.distance * [1., 0, 0])

    # Velocity offset to the barycenter
    # - This is defined in the MW frame!
    dvel = - m31_galcen_vel * M_M31 / lg.M_LG

    if inverse:
        inv_dpos = (-dpos).transform(R.T)
        b = inv_dpos.with_differentials(-dvel)
        A = R.T

    else:
        b = dpos.with_differentials(dvel.transform(R))
        A = R

    return A, b


@coord.frame_transform_graph.transform(
    coord.transformations.AffineTransform,
    coord.Galactocentric,
    LocalGroupBarycentric
)
def galactocentric_to_lg(galactocentric_coord, lg_frame):
    return get_galcen_to_lg_transform(lg_frame, galactocentric_coord,
                                      inverse=False, matrix_only=False)


@coord.frame_transform_graph.transform(
    coord.transformations.AffineTransform,
    LocalGroupBarycentric,
    coord.Galactocentric
)
def lg_to_galactocentric(lg_coord, galactocentric_frame):
    return get_galcen_to_lg_transform(lg_coord, galactocentric_frame,
                                      inverse=True, matrix_only=False)


@coord.frame_transform_graph.transform(
    coord.transformations.DynamicMatrixTransform,
    coord.Galactocentric,
    LocalGroupHalocentric
)
def galactocentric_to_lghalocentric(galactocentric_coord, lg_frame):
    return get_galcen_to_lg_transform(lg_frame, galactocentric_coord,
                                      inverse=False, matrix_only=True)


@coord.frame_transform_graph.transform(
    coord.transformations.DynamicMatrixTransform,
    LocalGroupHalocentric,
    coord.Galactocentric
)
def lghalocentric_to_galactocentric(lg_coord, galactocentric_frame):
    return get_galcen_to_lg_transform(lg_coord, galactocentric_frame,
                                      inverse=True, matrix_only=True)
