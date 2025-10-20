"""
The CalcHums module contains a set of functions for calculating humidity
variables. At present, it can only cope with scalars, not arrays.

There are routines for:
specific humidity from dew point temperature and temperature and pressure
vapour pressure from dew point temperature and temperature and pressure
relative humidity from dew point temperature and temperature and pressure
wet bulb temperature from dew point temperature and temperature and pressure
dew point depression from dew point temperature and temperature

There are also routines for:
vapour pressure from specific humidity and pressure and temperature
dew point temperature from vapour pressure and temperature and pressure
relative humidity from vapour pressure and temperature and pressure
wet bulb temperature from vapour pressure and dew point temperature and
temperature (and pressure?)

Where vapour pressure is used as part of the equation a pseudo wet bulb
temperature is calculated. If this is at or below 0 deg C then the ice bulb
equation is used.

ALL NUMBERS ARE RETURNED TO ONE SIGNIFICANT DECIMAL FIGURE.

THIS ROUTINE CANNOT COPE WITH MISSING DATA

THIS ROUTINE HAS a roundit=True/False. The default is True - round to one
decimal place.
Otherwise - set roundit=False

Written by Kate Willett 7th Feb 2016
"""

from __future__ import annotations

import numpy as np

from .auxiliary import isvalid


def vap(td: float, t: float, p: float, roundit: bool = True) -> float:
    """Calculate a vapour pressure scalar or array
    from a scalar or array of dew point temperature and returns it.
    It requires a sea (station actually but sea level ok for marine data)
    level pressure value. This can be a scalar or an array, even if dewpoint
    temperature is an array (CHECK). To test whether to apply the ice or water
    calculation a dry bulb temperature is needed. This allows calculation of a
    pseudo-wet bulb temperature (imprecise) first. If the wet bulb temperature is
    at or below 0 deg C then the ice calculation is used.

    Parameters
    ----------
    td: float
        dew point temperature in degrees C (array or scalar)
    t: float
        dry bulb temperature in degrees C (array or scalar)
    p: float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    roundit: bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        vapour pressure in hPa (array or scalar)

    Notes
    -----
    Buck 1981
    Buck, A. L.: New equations for computing vapor pressure and enhancement factor, J. Appl.
    Meteorol., 20, 1527?1532, 1981.
    Jenson et al. 1990
    Jensen, M. E., Burman, R. D., and Allen, R. G. (Eds.): Evapotranspiration and
    Irrigation Water Requirements: ASCE Manuals and Reports on Engineering Practices No.
    70, American Society of Civil Engineers, New York, 360 pp., 1990.

    TESTED!
    e = vap(10.,15.,1013.)
    e = 12.3

    """
    if not isvalid(td) or not isvalid(t) or not isvalid(p):
        return np.nan

    # Calculate pseudo-e assuming wet bulb to calculate a
    # pseudo-wet bulb (see wb below)
    f = 1 + (7.0 * (10 ** (-4.0))) + ((3.46 * (10 ** (-6.0))) * p)
    e = 6.1121 * f * np.exp(((18.729 - (td / 227.3)) * td) / (257.87 + td))

    a = 0.000066 * p
    b = (409.8 * e) / ((td + 237.3) ** 2)
    w = ((a * t) + (b * td)) / (a + b)

    # Now test for whether pseudo-wetbulb is above or below/equal to zero
    # to establish whether to calculate e with respect to ice or water
    # recalc if ice
    if w <= 0.0:
        f = 1 + (3.0 * (10 ** (-4.0))) + ((4.18 * (10 ** (-6.0))) * p)
        e = 6.1115 * f * np.exp(((23.036 - (td / 333.7)) * td) / (279.82 + td))

    if roundit:
        e = round(e * 10) / 10.0

    return e


def vap_from_sh(sh: float, p: float, roundit: bool = True) -> float:
    """Calculate a vapour pressure scalar or array
    from a scalar or array of specific humidity and pressure and returns it.
    It requires a sea (station actually but sea level ok for marine data)
    level pressure value. This can be a scalar or an array, even if specific humidity
    is an array (CHECK).

    Parameters
    ----------
    sh: float
        specific humidity in g/kg (array or scalar)
    p: float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    roundit: bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        e, vapour pressure in hPa (array or scalar)

    Notes
    -----
    Peixoto & Oort, 1996, Ross & Elliott, 1996
    Peixoto, J. P. and Oort, A. H.: The climatology of relative humidity in the atmosphere, J.
    Climate, 9, 3443?3463, 1996.

    TESTED!
    e = vap_from_sh(7.6,1013.)
    e = 12.3
    """
    e = ((sh / 1000.0) * p) / (0.622 + (0.378 * (sh / 1000.0)))

    if roundit:
        e = round(e * 10.0) / 10.0

    return e


def sh(td: float, t: float, p: float, roundit: bool = True) -> float:
    """Calculate a specific humidity scalar or array
    from a scalar or array of vapour pressure and returns it.
    It requires a sea (station actually but sea level ok for marine data)
    level pressure value. This can be a scalar or an array, even if vapour
    pressure is an array (CHECK).

    Parameters
    ----------
    td : float
        dew point temperature in degrees C (array or scalar)
    t : float
        dry bulb temperature in degrees C (array or scalar)
    p : float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    roundit : bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        specific humidity in g/kg (array or scalar)

    Notes
    -----
    Peixoto & Oort, 1996, Ross & Elliott, 1996
    Peixoto, J. P. and Oort, A. H.: The climatology of relative humidity in the atmosphere, J.
    Climate, 9, 3443?3463, 1996.

    TESTED!
    sh = sh(10.,15.,1013.)
    sh = 7.6
    """
    if not isvalid(td) or not isvalid(t) or not isvalid(p):
        return np.nan

    # Calculate pseudo-e assuming wet bulb to
    # calculate a pseudo-wet bulb (see wb below)
    f = 1 + (7.0 * (10 ** (-4.0))) + ((3.46 * (10 ** (-6.0))) * p)
    e = 6.1121 * f * np.exp(((18.729 - (td / 227.3)) * td) / (257.87 + td))

    a = 0.000066 * p
    b = (409.8 * e) / ((td + 237.3) ** 2)
    w = ((a * t) + (b * td)) / (a + b)

    # Now test for whether pseudo-wetbulb is above or below/equal to zero
    # to establish whether to calculate e with respect to ice or water
    # recalc if ice
    if w <= 0.0:
        f = 1 + (3.0 * (10 ** (-4.0))) + ((4.18 * (10 ** (-6.0))) * p)
        e = 6.1115 * f * np.exp(((23.036 - (td / 333.7)) * td) / (279.82 + td))

    q = 1000.0 * ((0.622 * e) / (p - ((1 - 0.622) * e)))

    if roundit:
        q = round(q * 10.0) / 10.0

    return q


def sh_from_vap(e: float, p: float, roundit: bool = True) -> float:
    """Calculate a specific humidity scalar or array
    from a scalar or array of vapour pressure and returns it.
    It requires a sea (station actually but sea level ok for marine data)
    level pressure value. This can be a scalar or an array, even if vapour
    pressure is an array (CHECK).

    Parameters
    ----------
    e : float
        vapour pressure in hPa (array or scalar)
    p : float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    roundit : bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        specific humidity in g/kg (array or scalar)

    Notes
    -----
    Peixoto & Oort, 1996, Ross & Elliott, 1996
    Peixoto, J. P. and Oort, A. H.: The climatology of relative humidity in the atmosphere, J.
    Climate, 9, 3443?3463, 1996.

    TESTED!
    sh = sh(10.,15.,1013.)
    sh = 7.6
    """
    q = 1000.0 * ((0.622 * e) / (p - ((1 - 0.622) * e)))

    if roundit:
        q = round(q * 10.0) / 10.0

    return q


def rh(td: float, t: float, p: float, roundit: bool = True) -> float:
    """Calculate a relative humidity scalar or array
    from a scalar or array of vapour pressure and temperature and returns
    it. It calculates the saturated vapour pressure from t.
    It requires a sea (station actually but sea level ok for marine data)
    level pressure value. This can be a scalar or an array, even if vapour pressure
    is an array (CHECK). To test whether to apply the ice or water
    calculation a dewpoint and dry bulb temperature are needed. We can assume that the
    dry bulb t is the same as the wet bulb t at saturation. This allows
    calculation of a pseudo-wet bulb temperature (imprecise) first. If the
    wet bulb temperature is at or below 0 deg C then the ice calculation is used.

    Parameters
    ----------
    td : float
        dew point temperature in degrees C (array or scalar)
    t : float
        dry bulb temperature in degrees C (array or scalar)
    p : float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    roundit : bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        relative humidity in %rh (array or scalar)

    Notes
    -----
    Ref:

    TESTED!
    rh = rh(10.,15.,1013.)
    rh = 72.0
    """
    if not isvalid(td) or not isvalid(t) or not isvalid(p):
        return np.nan

    # Calculate pseudo-e assuming wet bulb to calculate a
    # pseudo-wet bulb (see wb below)
    f = 1 + (7.0 * (10 ** (-4.0))) + ((3.46 * (10 ** (-6.0))) * p)
    e = 6.1121 * f * np.exp(((18.729 - (td / 227.3)) * td) / (257.87 + td))

    a = 0.000066 * p
    b = (409.8 * e) / ((td + 237.3) ** 2)
    w = ((a * t) + (b * td)) / (a + b)

    # Now test for whether pseudo-wetbulb is above or below/equal to zero
    # to establish whether to calculate e with respect to ice or water
    # recalc if ice
    if w <= 0.0:
        f = 1 + (3.0 * (10 ** (-4.0))) + ((4.18 * (10 ** (-6.0))) * p)
        e = 6.1115 * f * np.exp(((23.036 - (td / 333.7)) * td) / (279.82 + td))

    # Calculate pseudo-es assuming wet bulb to calculate
    # a pseudo-wet bulb (see wb below)
    # USING t INSTEAD OF td FOR SATURATED VAPOUR PRESSURE
    # (WET BULB T = T AT SATURATION)
    f = 1 + (7.0 * (10 ** (-4.0))) + ((3.46 * (10 ** (-6.0))) * p)
    es = 6.1121 * f * np.exp(((18.729 - (t / 227.3)) * t) / (257.87 + t))

    a = 0.000066 * p
    b = (409.8 * es) / (
        (t + 237.3) ** 2
    )  # "t" here rather than "td" because for es, t==td
    w = ((a * t) + (b * t)) / (
        a + b
    )  # second "t" is "t" here rather than "td" because for ex, t==td

    # Now test for whether pseudo-wetbulb is above or below/equal to zero
    # to establish whether to calculate e with respect to ice or water
    # recalc if ice
    if w <= 0.0:
        f = 1 + (3.0 * (10 ** (-4.0))) + ((4.18 * (10 ** (-6.0))) * p)
        es = 6.1115 * f * np.exp(((23.036 - (t / 333.7)) * t) / (279.82 + t))

    r = (e / es) * 100.0

    if roundit:
        r = round(r * 10.0) / 10.0

    return r


def wb(td: float, t: float, p: float, roundit: bool = True) -> float:
    """Calculate a wet bulb temperature scalar or array
    from a scalar or array of vapour pressure and temperature and
    dew point temperature and returns it.
    It requires a sea (station actually but sea level ok for marine data)
    level pressure value. This can be a scalar or an array, even ifvapour pressure
    is an array (CHECK). To test whether to apply the ice or water
    calculation a dewpoint and dry bulb temperature are needed. This allows
    calculation of a pseudo-wet bulb temperature (imprecise) first. If the
    wet bulb temperature is at or below 0 deg C then the ice calculation is used.

    Parameters
    ----------
    td : float
        dew point temperature in degrees C (array or scalar)
    t : float
        dry bulb temperature in degrees C (array or scalar)
    p : float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    roundit : bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        wet bulb temperature in degrees C (array or scalar)

    Notes
    -----
    Ref:
    Jenson et al. 1990
    Jensen, M. E., Burman, R. D., and Allen, R. G. (Eds.): Evapotranspiration and
    Irrigation Water Requirements: ASCE Manuals and Reports on Engineering Practices No.
    70, American Society of Civil Engineers, New York, 360 pp., 1990.

    TESTED!
    wb = wb(10.,15.,1013)
    wb = 12.2
    """
    if not isvalid(td) or not isvalid(t) or not isvalid(p):
        return np.nan

    # Calculate pseudo-e assuming wet bulb to calculate a pseudo-wet bulb (see wb below)
    f = 1 + (7.0 * (10 ** (-4.0))) + ((3.46 * (10 ** (-6.0))) * p)
    e = 6.1121 * f * np.exp(((18.729 - (td / 227.3)) * td) / (257.87 + td))

    a = 0.000066 * p
    b = (409.8 * e) / ((td + 237.3) ** 2)
    w = ((a * t) + (b * td)) / (a + b)

    # Now test for whether pseudo-wetbulb is above or below/equal to zero
    # to establish whether to calculate e with respect to ice or water
    # recalc if ice
    if w <= 0.0:
        f = 1 + (3.0 * (10 ** (-4.0))) + ((4.18 * (10 ** (-6.0))) * p)
        e = 6.1115 * f * np.exp(((23.036 - (td / 333.7)) * td) / (279.82 + td))

    # Now calculate a slightly better w
    a = 0.000066 * p
    b = (409.8 * e) / ((td + 237.3) ** 2)

    w = ((a * t) + (b * td)) / (a + b)

    if roundit:
        w = round(w * 10.0) / 10.0

    return w


def dpd(td: float, t: float, roundit: bool = True) -> float:
    """Calculate a dew point depression scalar or array
    from a scalar or array of temperature and dew point temperature and returns it.

    Parameters
    ----------
    td : float
        dew point temperature in degrees C (array or scalar)
    t : float
        dry bulb temperature in degrees C (array or scalar)
    roundit : bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        dew point depression in degrees C (array or scalar)

    Notes
    -----
    Ref:

    TESTED!
    dpd = dpd(10..,15.)
    dpd = 5.0
    """
    if not isvalid(td) or not isvalid(t):
        return np.nan

    dp = t - td

    if roundit:
        dp = round(dp * 10.0) / 10.0

    return dp


def td_from_vap(e: float, p: float, t: float, roundit: bool = True) -> float:
    """Calculate a dew point depression scalar or array
    from a scalar or array of vapour pressure and pressure and returns it.
    It also requires temperature to check whether the wet bulb temperature
    is <= 0.0 - if so the ice bulb calculation is used.

    Parameters
    ----------
    e : float
        vapour pressure in hPa (array or scalar)
    p : float
        pressure at observation level in hPa (array or scalar - can be scalar even if others are arrays)
    t : float
        dry bulb temperature in degrees C (array or scalar)
    roundit : bool
        flag to tell function to round to one decimal place, default TRUE

    Returns
    -------
    float
        dew point depression in degrees C (array or scalar)

    Notes
    -----
    Buck 1981
    Buck, A. L.: New equations for computing vapor pressure and enhancement factor, J. Appl.
    Meteorol., 20, 1527?1532, 1981.
    Jenson et al. 1990
    Jensen, M. E., Burman, R. D., and Allen, R. G. (Eds.): Evapotranspiration and
    Irrigation Water Requirements: ASCE Manuals and Reports on Engineering Practices No.
    70, American Society of Civil Engineers, New York, 360 pp., 1990.

    TESTED!
    td = td_from_vap(12.3,1013.,15.)
    td = 10.0
    """
    # First calculate an estimated dew point T
    f = 1 + (7.0 * 10.0 ** (-4.0)) + ((3.46 * 10.0 ** (-6.0)) * p)

    a = 1
    b = (227.3 * np.log(e / (6.1121 * f))) - (18.729 * 227.3)
    c = 257.87 * 227.3 * np.log(e / (6.1121 * f))

    td = (-b - np.sqrt(b**2 - (4 * a * c))) / (2 * a)

    # Now calculate an estimated wet bulb T
    a = 0.000066 * p
    b = (409.8 * e) / ((td + 237.3) ** 2)
    w = ((a * t) + (b * td)) / (a + b)

    # Now test for whether pseudo-wetbulb is above or below/equal to zero
    # to establish whether to calculate td with respect to ice or water
    # recalc if ice
    if w <= 0.0:
        f = 1 + (3.0 * 10.0 ** (-4.0)) + ((4.18 * 10.0 ** (-6.0)) * p)

        a = 1
        b = (333.7 * np.log(e / (6.1115 * f))) - (23.036 * 333.7)
        c = 279.82 * 333.7 * np.log(e / (6.1115 * f))

        td = (-b - np.sqrt(b**2 - (4 * a * c))) / (2 * a)

    if roundit:
        td = round(td * 10.0) / 10.0

    return td
