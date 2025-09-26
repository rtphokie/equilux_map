import json
import time
import unittest
from math import modf
from pprint import pprint

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests_cache
from pytz import timezone
from skyfield import api, almanac
from skyfield.searchlib import find_maxima, find_minima
import datetime, pytz
from mezmorize import Cache

cache = Cache(CACHE_TYPE='filesystem', CACHE_DIR='cache_data')

ts = api.load.timescale()
load = api.Loader("/var/data")

s = requests_cache.CachedSession("tzapicache")
e = load("de430t.bsp")
earth, sun = e["earth"], e["sun"]

now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)



@cache.memoize()
def equilux(lat, lon, tzstr, year=2022, years=1, equinoxes=['Vernal', 'Autumnal']):
    """
    Calculate the equilux (day when day and night are nearly equal) for a given location and year.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        tzstr (str): Timezone string (e.g., 'America/New_York').
        year (int, optional): Year for calculation. Defaults to 2022.
        years (int, optional): Number of years to calculate. Defaults to 1.
        equinoxes (list, optional): List of equinox types to calculate ('Vernal', 'Autumnal').

    Returns:
        dict: Dictionary containing dates, times, daylight duration, and difference from 12 hours
              for each equilux and equinox event.
    """
    tz = timezone(tzstr)

    t0 = ts.utc(year, 1, 1)
    t1 = ts.utc(year, 12, 31)
    # find the solstices and equinoxes
    t, y = almanac.find_discrete(t0, t1, almanac.seasons(e))

    results = {}
    for yi, ti in zip(y, t):
        event_name = almanac.SEASON_EVENTS[yi]
        if "Equinox" in event_name:
            for eq in equinoxes:  # calculate for one or both equinoxes
                if eq in event_name:
                    dt = ti.utc_datetime()
                    results[event_name], _ = ti.astimezone_and_leap_second(tz)
                    obs = api.Topos(lat, lon)
                    if 'Autumnal' in event_name:
                        # equilux happens after the September equinox
                        t0 = ts.utc(dt.year, dt.month, dt.day - 1)
                        t1 = ts.utc(dt.year, dt.month, dt.day + 7)
                    elif 'Vernal' in event_name:
                        # equilux happens before the March equinox
                        t0 = ts.utc(dt.year, dt.month, dt.day - 7)
                        t1 = ts.utc(dt.year, dt.month, dt.day + 1)
                    else:
                        raise ValueError(f"unknown equinox type {event_name}")
                    # find all sunrise and sunset events in this range
                    t2, y2 = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(e, obs))
                    prev = None
                    dates = []
                    delta = []
                    sunlight = []
                    for t2i, y2i in zip(t2, y2):
                        dt2, _ = t2i.astimezone_and_leap_second(tz)
                        hrs = dt2.hour + dt2.minute / 60.0 + dt2.second / 3600.0 + dt2.microsecond / 3600000000
                        if y2i:  # rise
                            prev = hrs
                        elif prev is not None:  # set
                            dates.append(dt2)
                            delta.append(abs(hrs - prev - 12.0))
                            sunlight.append(hrs - prev)
                    idx = delta.index(min(delta))
                    results[event_name], _ = ti.astimezone_and_leap_second(tz)

                    # seconds is sufficient resolution here
                    hrs = int(sunlight[idx])
                    rem = (sunlight[idx] - hrs) * 60.0
                    mins = int(rem)
                    secs = (rem - mins) * 60.0

                    delta = 12 - sunlight[idx]
                    if delta > 0:
                        deltastr = f"{delta * 3600:.1f} seconds shy of"
                    else:
                        deltastr = f"{abs(delta * 3600):.1f} seconds more than"

                    results[event_name.replace("nox", "lux")] = dates[idx]
                    results[f"{event_name.replace('nox', 'lux')} time"] = (
                        f"{hrs} hrs {mins} min {secs:.0f} sec"
                    )
                    results[f"{event_name.replace('nox', 'lux')} delta"] = (
                        deltastr
                    )
                    results[f"{event_name.SEASON_EVENTS[yi].replace('nox', 'lux')} hours"] = (
                        sunlight[idx]
                    )

    return results


