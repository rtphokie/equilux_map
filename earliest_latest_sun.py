#!/usr/bin/env python3

import datetime

import numpy as np
from mezmorize import Cache
from skyfield import almanac, api

cache = Cache(CACHE_TYPE="filesystem", CACHE_DIR="cache_data")

ts = api.load.timescale()
load = api.Loader("/var/data")
e = load("de430t.bsp")


@cache.memoize()
def find_equinox(year, name="Autumnal"):
    """Return the UTC datetime of the given equinox in `year`."""
    t0 = ts.utc(year, 1, 1)
    t1 = ts.utc(year, 12, 31)
    t, y = almanac.find_discrete(t0, t1, almanac.seasons(e))
    for ti, yi in zip(t, y):
        event_name = almanac.SEASON_EVENTS[yi]
        if name in event_name and "Equinox" in event_name:
            return ti.utc_datetime()
    raise ValueError(f"no {name} equinox found in {year}")


@cache.memoize()
def equilux_by_latitude(
    lat, year=2025, equinox="Autumnal", window_days=8, utc_offset_hours=0.0
):
    """
    Find when the length of the day (sunrise to sunset) is closest to 12
    hours near an equinox, for a given latitude.

    The day *length* depends only on latitude and date, not on longitude -
    so the underlying astronomy only needs to be computed once per latitude
    and can be reused across every longitude at that latitude. But which
    *calendar date* that closest-to-12h day falls on is a local-time
    question: `utc_offset_hours` (the location's approximate civil/solar
    UTC offset, e.g. -5 for US Eastern) is used to translate the sunset
    instant into a local calendar date, and also shifts *which* day is
    closest (the discrete sunrise/sunset samples themselves shift by this
    same offset, since Topos longitude is derived from it). Callers that
    need this for many longitudes should bucket them by rounded UTC offset
    first, rather than calling this once per exact longitude - there are
    only a handful of buckets across a country's width.

    Returns a dict:
        date: the local calendar date (per utc_offset_hours) whose
              sunrise-to-sunset span is closest to 12 hours
        offby_seconds: signed seconds by which that day's length differs
                       from 12h (positive = longer than 12h)
        sunlight_hours: the actual day length on `date`
        equilux_utc: the day length crosses exactly 12 hours at some instant
                     between two consecutive days; this is that instant,
                     linearly interpolated for sub-day precision so nearby
                     latitudes produce a smoothly varying value. This is
                     essentially independent of utc_offset_hours (it's a
                     real physical instant, not a calendar label).
    """
    eq = find_equinox(year, equinox)
    if equinox == "Autumnal":
        # equilux happens in the week(s) after the September equinox
        t0 = ts.utc(eq.year, eq.month, eq.day - 1)
        t1 = ts.utc(eq.year, eq.month, eq.day + window_days - 1)
    elif equinox == "Vernal":
        # equilux happens in the week(s) before the March equinox
        t0 = ts.utc(eq.year, eq.month, eq.day - window_days + 1)
        t1 = ts.utc(eq.year, eq.month, eq.day + 1)
    else:
        raise ValueError(f"unknown equinox {equinox!r}")

    # the reference longitude only needs to be roughly right for the target
    # UTC offset - it fixes which real-time instants the discrete daily
    # rise/set samples fall on, which is what determines which calendar day
    # each one locally belongs to
    # 15 degrees of longitude = 1 hour of solar time - this conversion is
    # fixed and unrelated to whatever bucket width the caller grouped
    # longitudes by (e.g. LON_BUCKET_WIDTH_DEGREES in equilux_map.py)
    obs = api.Topos(latitude_degrees=lat, longitude_degrees=utc_offset_hours * 15.0)
    times, is_rise = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(e, obs))

    sets, durations = [], []
    rise = None
    for t, r in zip(times, is_rise):
        if r:
            rise = t
        elif rise is not None:
            durations.append((t - rise) * 24.0)  # hours; timezone-independent
            sets.append(t)
            rise = None
    durations = np.array(durations)

    idx = int(np.argmin(np.abs(durations - 12.0)))

    # interpolate against whichever neighboring day is on the other side of
    # 12 hours, for a continuous, sub-day-precision estimate
    candidates = [n for n in (idx - 1, idx + 1) if 0 <= n < len(durations)]
    neighbor = min(candidates, key=lambda n: abs(durations[n] - 12.0))
    d0, d1 = durations[idx], durations[neighbor]
    t0_, t1_ = sets[idx], sets[neighbor]
    frac = 0.0 if d0 == d1 else (12.0 - d0) / (d1 - d0)
    equilux_instant = (
        t0_.utc_datetime() + (t1_.utc_datetime() - t0_.utc_datetime()) * frac
    )

    local_set = sets[idx].utc_datetime() + datetime.timedelta(hours=utc_offset_hours)

    return {
        "date": local_set.date(),
        "offby_seconds": (durations[idx] - 12.0) * 3600.0,
        "sunlight_hours": durations[idx],
        "equilux_utc": equilux_instant,
    }
