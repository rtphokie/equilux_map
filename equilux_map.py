#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timezonefinder import TimezoneFinder

import datetime
from pprint import pprint
from earliest_latest_sun import equilux
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import Polygon
from shapely.geometry import Point

tf = TimezoneFinder()

markers = {'Thu': 'P', 'Fri': 'X', 'Sat': 'v', 'Sun': '^', 'Mon': '*', 'Tue': 's', 'Wed': 'D'}

pathcolor = {'Total': 'grey', 'Annular': 'darkorange', 'Hybrid': 'red',
             'city_marker': 'lightgrey', 'city_marker_labeled': 'grey', 'city_label': 'grey',
             'world': '#809E7E', 'water': '#8AB4F8', 'coast': '#090eab',
             'state_domestic': '#F7F3E9', 'state_foreign': '#90A686', 'state_highlight': '#fcfad7',
             'state_domestic_edge': 'k', 'state_foreign_edge': 'k',
             'state_highlight_edge': 'k'
             }

todayiso = datetime.datetime.now().isoformat()[:10]

from mezmorize import Cache

cache = Cache(CACHE_TYPE='filesystem', CACHE_DIR='cache_data')


def mapit(dpi=300, pad_factor=0.05, crs='EPSG:9311', step=0.5):
    ''' Create a map of the US with equilux data points plotted.

    :param dpi: resolution of the output image
    :param pad_factor: white (well blue) space around the land
    :param crs: map projection, default is US National Atlas Equal Area
    :param step: degrees between points
    :return:
    '''
    fig = plt.figure()
    fig.patch.set_visible(False)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9), dpi=dpi)
    fig.tight_layout()
    ax.axis('off')
    # see https://catalog.data.gov for US Census Bureau provided shapefiles
    us = gpd.read_file('data/cb_2022_us_all_500k/cb_2022_us_state_500k.shp')
    exclude_states = ["United States Virgin Islands", "Guam", 'American Samoa', 'Puerto Rico',
                      "Commonwealth of the Northern Mariana Islands", "Alaska", "Hawaii"]  # focus on the lower 48
    us = us[~us['NAME'].isin(exclude_states)]
    us_precrs = us.copy()
    min_lon, max_lon, min_lat, max_lat = get_bounds(us, pad_factor=pad_factor)
    us = us.to_crs(crs)  # US National Atlas Equal Area
    xmin, xmax, ymin, ymax = get_bounds(us, pad_factor=pad_factor)
    d = {'geometry': [Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), ])]}
    gdf_water = gpd.GeoDataFrame(d, crs=us.crs)
    gdf_water.plot(ax=ax, color=pathcolor['water'], edgecolor='k', linewidth=2.5)
    us.plot(ax=ax, color='lightgrey', edgecolor='white')
    # fig.savefig(f"equilux_map_base.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)

    # crop to the lower 48 states with some padding
    xmin, xmax, ymin, ymax = get_bounds(us, pad_factor=pad_factor)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # get a list of lat/lon coordinates over land in the US, evenly spaced by step degrees
    coordinates = get_coordinates(max_lat, max_lon, min_lat, min_lon, step, us_precrs)

    # turn that into a Pandas DataFrame with equilux data, this will take a few seconds for points seperated by a few degrees, to several minutes for sub degree spacing
    cities_df = build_dataframe(coordinates)

    # Create a GeoDataFrame from the Pandas DataFrame
    geometry = gpd.points_from_xy(cities_df['longitude'], cities_df['latitude'])
    cities_gdf = gpd.GeoDataFrame(cities_df, geometry=geometry,
                                  crs="EPSG:4326")  # same WGS84 for lat/lon the Census Bureau uses for the base map
    cities_gdf = cities_gdf.to_crs(
        crs)  # ESPG:9311 is US National Atlas Equal Area, what most people are used to seeing, and without the unusually horizontal border with Canada

    # add colored markers indicating the day of the equilux at that point (+ for Thursday, x for Friday, v for Saturday, etc, see above)
    for day in cities_df.Day.unique():
        cities_gdf[cities_gdf['Day'] == day].plot(ax=ax, marker=markers[day], cmap='cool_r', markersize=20,
                                                  zorder=20)  # Plot cities on to2
     # save the resulting map
    fig.savefig(f"equilux_map.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)


@cache.memoize()
def build_dataframe(coordinates):
    data = {'latitude': [], 'longitude': [], 'offby': [], 'Day': []}
    for point in tqdm(coordinates):
        lat, lon = point
        timezone_str = tzname(lat, lon)

        data['latitude'].append(lat)
        data['longitude'].append(lon)

        result = equilux(lat, lon, timezone_str, year=2025, years=1, equinoxes=['Autumnal'])
        data['Day'].append(result['Autumnal Equilux'].strftime('%a'))

        atoms = result['Autumnal Equilux delta'].split(' ')
        sec = abs(float(atoms[0]))
        if atoms[2] == 'shy':
            sec = -sec
        data['offby'].append(sec)
    cities_df = pd.DataFrame(data)
    return cities_df


@cache.memoize()
def get_coordinates(max_lat, max_lon, min_lat, min_lon, step, us_precrs, decimals=2):
    coordinates = []
    lat = round(min_lat, decimals)
    while lat <= max_lat:
        lon = round(min_lon, decimals)
        while lon <= max_lon:
            point = Point(lon, lat)
            land = us_precrs.contains(point).any()
            if land:
                coordinates.append((lat, lon))
            lon += step
        lat += step
    return coordinates


@cache.memoize()
def tzname(lat, lon):
    # Find the timezone name at the given latitude and longitude
    # wrapped in a function to allow caching
    timezone_str = tf.timezone_at(lng=lon, lat=lat)
    return timezone_str


@cache.memoize()
def get_bounds(gdf_focus, pad_factor=0.0):
    """
    Calculate the bounding box (xmin, xmax, ymin, ymax) of a GeoDataFrame,
    optionally expanding the bounds by a given pad_factor.

    Parameters:
        gdf_focus (geopandas.GeoDataFrame): The GeoDataFrame to get bounds from.
        pad_factor (float): Fractional amount to expand the bounds (default is 0.0).

    Returns:
        tuple: (xmin, xmax, ymin, ymax) coordinates of the bounding box.
    """
    focus_bounds = gdf_focus.total_bounds
    xwidth = abs(focus_bounds[2] - focus_bounds[0])
    ywidth = abs(focus_bounds[3] - focus_bounds[1])
    xmin = focus_bounds[0] - xwidth * pad_factor
    xmax = focus_bounds[2] + xwidth * pad_factor
    ymin = focus_bounds[1] - ywidth * pad_factor
    ymax = focus_bounds[3] + ywidth * pad_factor
    return xmin, xmax, ymin, ymax


if __name__ == "__main__":
    mapit(step=5)
