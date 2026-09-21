#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import inspect
from itertools import product

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path
from mezmorize import Cache
from pyproj import Transformer
from shapely.geometry import Polygon
from tqdm import tqdm

from earliest_latest_sun import equilux_by_latitude, find_equinox

BGCOLOR = 'black'
BORDER_COLOR = '#888888'
COUNTRY_BORDER_COLOR = '#3f3f3f'
STATE_BORDER_COLOR = '#555555'
TEXT_COLOR = '#dddddd'

EXCLUDE_STATE_NAMES = ["United States Virgin Islands", "Guam", 'American Samoa', 'Puerto Rico',
                       "Commonwealth of the Northern Mariana Islands", "Alaska", "Hawaii"]  # focus on the lower 48

# sequential yellow -> orange -> red ramp (ColorBrewer YlOrRd), evoking sunlight
# and giving a warm, low-to-high read that fits a day-length map. The equilux
# date is a whole-day (ordinal) quantity, not a continuous one, so this is
# sampled into discrete bands rather than blended.
EQUILUX_CMAP_NAME = 'YlOrRd'

# width, in degrees of longitude, of the buckets columns are grouped into
# almost everywhere. 15 degrees = 1 real hour; using a narrower bucket (e.g.
# 5 degrees = 1/3 hour) gives a closer-to-exact local time at the cost of
# more astronomy calls per latitude row.
LON_BUCKET_WIDTH_DEGREES = 5.0

# finer bucket width used only for latitude rows inside the transition band
# (see lon_refine_step in build_heatmap_grid) - this is what keeps the
# transition line from stair-stepping across longitude.
LON_REFINE_STEP_DEGREES = 0.25

cache = Cache(CACHE_TYPE='filesystem', CACHE_DIR='cache_data')


def equilux_band_colors(n):
    """Pick n evenly-spaced, light->dark steps from the sequential ramp,
    avoiding the near-white end (poor contrast) and the near-black end
    (looks like an outlier/missing-data color)."""
    cmap = plt.get_cmap(EQUILUX_CMAP_NAME)
    return [cmap(p) for p in np.linspace(0.005, 0.9, n)]


def polygon_to_path(geom):
    """Convert a shapely (Multi)Polygon into a matplotlib Path (exterior +
    interior rings, one MOVETO-led subpath each), so it can be used as a
    clip path - this lets a coarse, cheap data grid still render with an
    exact, shapefile-precision coastline instead of a blocky grid-cell edge."""
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    vertices, codes = [], []
    for poly in polys:
        for ring in (poly.exterior, *poly.interiors):
            coords = np.asarray(ring.coords)
            vertices.append(coords)
            ring_codes = np.full(len(coords), Path.LINETO, dtype=Path.code_type)
            ring_codes[0] = Path.MOVETO
            codes.append(ring_codes)
    return Path(np.concatenate(vertices), np.concatenate(codes))


def mapit(dpi=300, pad_factor=0.5, crs='EPSG:9311', step=0.5, refine_step=0.05,
          lon_refine_step=LON_REFINE_STEP_DEGREES, year=0, equinox='Autumnal'):
    ''' Create a map of the US showing, as bands of color, which date each
    latitude's day length (sunrise to sunset) is closest to 12 hours, near
    the given equinox. The date is a whole-day quantity, so it renders as a
    handful of solid bands rather than a smooth gradient

    :param dpi: resolution of the output image
    :param pad_factor: space around the land
    :param crs: map projection, default is US National Atlas Equal Area
    :param step: degrees between sample points
    :param refine_step: if smaller than `step`, latitude bands that contain a
        date transition are resampled at this finer spacing so the boundary
        is drawn sharply without paying that cost everywhere. Set to None or
        >= step to disable.
    :param lon_refine_step: within the latitude rows refine_step identified as
        straddling a transition, longitude is additionally sampled at this
        finer bucket width (instead of LON_BUCKET_WIDTH_DEGREES), since the
        transition latitude drifts smoothly with longitude and would
        otherwise stair-step. Set to None or 0 to disable.
    :param year: year to compute the equilux for
    :param equinox: 'Autumnal' or 'Vernal'
    :return:
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9), dpi=dpi)
    fig.patch.set_facecolor(BGCOLOR)
    fig.tight_layout()
    ax.axis('off')
    ax.set_facecolor(BGCOLOR)

    # see https://catalog.data.gov for US Census Bureau provided shapefiles
    us = gpd.read_file('data/cb_2022_us_all_500k/cb_2022_us_state_500k.shp')
    us = us[~us['NAME'].isin(EXCLUDE_STATE_NAMES)]
    us_precrs = us.copy()
    min_lon, max_lon, min_lat, max_lat = get_bounds(us, pad_factor=pad_factor)
    us = us.to_crs(crs)  # US National Atlas Equal Area
    xmin, xmax, ymin, ymax = get_bounds(us, pad_factor=pad_factor)
    d = {'geometry': [Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), ])]}
    gdf_water = gpd.GeoDataFrame(d, crs=us.crs)
    gdf_water.plot(ax=ax, color=BGCOLOR, edgecolor=BORDER_COLOR, linewidth=1.5, zorder=0)
    us.plot(ax=ax, color=BGCOLOR, edgecolor='none', zorder=1)

    # crop to the lower 48 states with some padding
    xmin, xmax, ymin, ymax = get_bounds(us, pad_factor=pad_factor)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    lon_grid, lat_grid, value_grid, land_mask = build_heatmap_grid(
        us_precrs, min_lat, max_lat, min_lon, max_lon, step, year, equinox,
        refine_step=refine_step, lon_refine_step=lon_refine_step)

    # reproject the sample grid into the same CRS as the base map
    transformer = Transformer.from_crs(us_precrs.crs, crs, always_xy=True)
    x, y = transformer.transform(lon_grid, lat_grid)

    # value_grid holds date ordinals (whole days) - one discrete color band
    # per date actually present on land, boundaries centered between them.
    # (Only land_mask decides which bands make the legend; value_grid itself
    # is unmasked and gets clipped to the coastline visually below, so a
    # coarse `step` never blocks up the coastline.)
    ordinals = sorted(int(o) for o in np.unique(value_grid[land_mask]))
    cmap = ListedColormap(equilux_band_colors(len(ordinals)))
    boundaries = [ordinals[0] - 0.5] + [o + 0.5 for o in ordinals]
    norm = BoundaryNorm(boundaries, cmap.N)

    mesh = ax.pcolormesh(x, y, value_grid, shading='nearest', cmap=cmap, norm=norm, zorder=20)
    # clip the (possibly coarse) color grid to the exact land geometry, so
    # coastline fidelity comes from the shapefile, not from grid resolution
    mesh.set_clip_path(polygon_to_path(us.geometry.union_all()), transform=ax.transData)

    # state and national borders drawn on top of the heatmap fill
    us.boundary.plot(ax=ax, color=STATE_BORDER_COLOR, linewidth=0.5, zorder=25)
    outline = gpd.GeoSeries([us.geometry.union_all()], crs=us.crs)
    outline.boundary.plot(ax=ax, color=COUNTRY_BORDER_COLOR, linewidth=0.6, zorder=26)

    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', ticks=ordinals,
                        fraction=0.04, pad=0.03, shrink=0.5)
    cbar.ax.set_xticklabels([datetime.date.fromordinal(o).strftime('%b %-d') for o in ordinals])
    cbar.set_label(f'date closest to 12 hours of daylight ({equinox.lower()} equinox, {year})', color=TEXT_COLOR)
    cbar.ax.xaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cbar.outline.set_edgecolor(BORDER_COLOR)

    # save the resulting map
    filename = f"equilux_map_{year}_{equinox.lower()}_{step:0.2f}.png"
    fig.savefig(filename, dpi=dpi, facecolor=BGCOLOR, bbox_inches='tight', pad_inches=0.1)


def build_heatmap_grid(us_precrs, min_lat, max_lat, min_lon, max_lon, step, year, equinox,
                        refine_step=None, lon_refine_step=LON_REFINE_STEP_DEGREES):
    """
    Build a lat/lon grid over land, with each land point's value being the
    date (as a Python ordinal day number) whose day length is closest to 12
    hours at that latitude - suitable for pcolormesh (which doesn't require
    a uniform grid, only that X/Y/C share a shape).

    Day *length* only depends on latitude, but which calendar date that
    closest-to-12h day falls on is a local-time question, so longitude
    can't be ignored entirely: columns are grouped into LON_BUCKET_WIDTH_DEGREES-
    wide buckets and the astronomy is computed once per (latitude, bucket)
    pair - still a small, fixed number of astronomy calls, just not quite
    as few as per-latitude-only.

    This grid is refined adaptively in *both* directions, because the two
    axes behave differently:

    - In latitude, the transition is locally confined - most rows are
      clearly on one side or the other, only a narrow band actually
      straddles it. `refine_step`, if smaller than `step`, inserts extra
      latitude samples only within that band.
    - In longitude, the transition latitude drifts *smoothly* across the
      entire width of the country (local calendar date depends on local
      time, which shifts continuously with longitude) - so it's present at
      every longitude, just at a slightly different latitude. Refining
      longitude by "do adjacent buckets ever disagree" would therefore
      refine almost everywhere, which defeats the point. Instead,
      `lon_refine_step` only kicks in for the latitude rows the latitude
      refinement above already identified as being inside the transition
      band, and even then only subdivides whichever single pair of adjacent
      coarse buckets actually disagrees *for that specific row* - since the
      drift is smooth, a row only crosses the transition once, so this
      bounds the extra work to a handful of calls per transition-band row
      rather than a full fine sweep across the country's width.

    Land/water masking is done as a single vectorized operation rather than
    a per-point loop.
    """
    lons = np.arange(min_lon, max_lon + step / 2, step)

    def bucketed(values, width):
        return np.round(values / width) * width

    coarse_ref = bucketed(lons, LON_BUCKET_WIDTH_DEGREES)
    unique_coarse_ref = sorted(set(coarse_ref.tolist()))

    ordinal_cache = {}

    def ordinal_for(lat, ref_lon):
        key = (lat, ref_lon)
        if key not in ordinal_cache:
            ordinal_cache[key] = equilux_by_latitude(
                lat, year=year, equinox=equinox, utc_offset_hours=ref_lon / 15.0
            )['date'].toordinal()
        return ordinal_cache[key]

    coarse_lats = np.arange(min_lat, max_lat + step / 2, step)
    lats = set(coarse_lats.tolist())

    # coarse pass - this is where essentially all the astronomy happens, so
    # it's the one that needs the progress bar
    coarse_combos = list(product(coarse_lats.tolist(), unique_coarse_ref))
    pbar = tqdm(coarse_combos, desc='computing equilux by latitude/longitude (coarse)')
    coarse_vals = {}
    for lat, ref in pbar:
        pbar.set_postfix(lat=f'{lat:5.2f}')
        coarse_vals[(lat, ref)] = ordinal_for(lat, ref)

    # latitude refinement: for each longitude bucket independently, find the
    # coarse latitude interval that straddles its own transition and insert
    # finer samples there. The union of these bands, across every bucket, is
    # exactly the "transition band" used below for longitude refinement.
    fine_lats = set()
    if refine_step and refine_step < step:
        for ref in unique_coarse_ref:
            vals = [coarse_vals[(lat, ref)] for lat in coarse_lats]
            for i in range(len(coarse_lats) - 1):
                if vals[i] != vals[i + 1]:
                    new_rows = np.arange(coarse_lats[i], coarse_lats[i + 1], refine_step)
                    lats.update(new_rows.tolist())
                    fine_lats.update(new_rows.tolist())
    lats = np.array(sorted(lats))

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    land = us_precrs.geometry.union_all()
    land_mask = shapely.contains_xy(land, lon_grid, lat_grid)

    # longitude refinement, scoped per row: within a transition-band row, the
    # smoothly-drifting transition crosses that row's line at exactly one
    # longitude, so only the one pair of adjacent coarse buckets that
    # actually disagree *for this row* needs subdividing - not the whole
    # width of the country. This bounds the extra cost to a handful of
    # calls per transition-band row instead of one full fine sweep per row.
    lon_pairs = list(zip(unique_coarse_ref[:-1], unique_coarse_ref[1:]))

    def fill_row(row, lat, refine_lon):
        coarse_row_vals = {ref: ordinal_for(lat, ref) for ref in unique_coarse_ref}
        for ref in unique_coarse_ref:
            row[coarse_ref == ref] = coarse_row_vals[ref]
        if refine_lon and lon_refine_step:
            for ref_a, ref_b in lon_pairs:
                if coarse_row_vals[ref_a] == coarse_row_vals[ref_b]:
                    continue
                sub_refs = np.arange(ref_a, ref_b + lon_refine_step / 2, lon_refine_step)
                sub_vals = np.array([ordinal_for(lat, r) for r in sub_refs])
                mask = (coarse_ref == ref_a) | (coarse_ref == ref_b)
                nearest = np.abs(lons[mask][:, None] - sub_refs[None, :]).argmin(axis=1)
                row[mask] = sub_vals[nearest]

    value_grid = np.empty(lon_grid.shape, dtype=float)
    fine_indices = [i for i, lat in enumerate(lats) if lat in fine_lats]
    coarse_indices = [i for i in range(len(lats)) if i not in set(fine_indices)]

    for i in coarse_indices:
        fill_row(value_grid[i], lats[i], refine_lon=False)

    if fine_indices:
        pbar = tqdm(fine_indices, desc='computing equilux by latitude/longitude (refining transitions)')
        for i in pbar:
            lat = lats[i]
            pbar.set_postfix(lat=f'{lat:8.3f}')
            fill_row(value_grid[i], lat, refine_lon=True)

    # value_grid is intentionally left unmasked here - the coarser `step` is,
    # the blockier a land/water cutoff baked into the data grid would look
    # against the real coastline. Instead the caller clips the rendered mesh
    # to the exact shapefile geometry (see polygon_to_path in mapit), so
    # coastline fidelity comes from the vector data, not grid resolution.
    # land_mask is still returned so the caller can determine which date
    # bands actually touch land (for the legend), independent of rendering.
    return lon_grid, lat_grid, value_grid, land_mask


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


def closest_equinox(today=None):
    """Return the (year, equinox name) of whichever Vernal/Autumnal equinox,
    among last year's, this year's, and next year's, falls closest to today."""
    today = today or datetime.datetime.now(datetime.timezone.utc)
    candidates = [
        (year, name)
        for year in (today.year - 1, today.year, today.year + 1)
        for name in ('Vernal', 'Autumnal')
    ]
    year, name = min(candidates, key=lambda c: abs((find_equinox(*c) - today).total_seconds()))
    return year, name


if __name__ == "__main__":
    default_year, default_equinox = closest_equinox()
    # pull defaults from mapit()'s own signature so the CLI can't drift out
    # of sync with it (as happened when mapit's default step changed but
    # this argparse default didn't)
    mapit_defaults = {p.name: p.default for p in inspect.signature(mapit).parameters.values()}

    parser = argparse.ArgumentParser(
        description="Map, as bands of color, the date each US latitude's day length "
                     "is closest to 12 hours near a given equinox.")
    parser.add_argument('--year', type=int, default=default_year,
                         help=f"year to compute the equinox/equilux for (default: {default_year}, "
                              "the year of whichever equinox is closest to today)")
    parser.add_argument('--equinox', choices=['Vernal', 'Autumnal'], default=default_equinox,
                         help=f"which equinox to map (default: {default_equinox}, closest to today)")
    parser.add_argument('--step', type=float, default=mapit_defaults['step'],
                         help=f"coarse sample spacing in degrees, both lat and lon "
                              f"(default: {mapit_defaults['step']})")
    parser.add_argument('--refine-step', type=float, default=mapit_defaults['refine_step'],
                         help="finer sample spacing used only within latitude bands that contain a "
                              "date transition, so the boundary is sharp without the cost of using "
                              f"this resolution everywhere; pass 0 to disable "
                              f"(default: {mapit_defaults['refine_step']})")
    parser.add_argument('--lon-refine-step', type=float, default=mapit_defaults['lon_refine_step'],
                         help="within the refined latitude band, longitude is additionally sampled at "
                              "this finer spacing so the transition line doesn't stair-step as it "
                              f"drifts across longitude; pass 0 to disable (default: "
                              f"{mapit_defaults['lon_refine_step']})")
    args = parser.parse_args()

    mapit(year=args.year, equinox=args.equinox, step=args.step,
          refine_step=args.refine_step or None, lon_refine_step=args.lon_refine_step or None)
