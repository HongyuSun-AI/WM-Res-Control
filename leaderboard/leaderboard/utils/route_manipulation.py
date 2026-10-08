#!/usr/bin/env python


import math
import xml.etree.ElementTree as ET
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


def _location_to_gps(lat_ref, lon_ref, location):

    EARTH_RADIUS_EQUA = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}


def location_route_to_gps(route, lat_ref, lon_ref):
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world):
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def downsample_route(route, sample_factor):

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        if prev_option is None:
            ids_to_sample.append(i)
            dist = 0

        elif curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        else:
            curr_location = point[0].location
            prev_location = route[i-1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


def interpolate_trajectory(waypoints_trajectory, hop_resolution=1.0):

    grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), hop_resolution)
    lat_ref, lon_ref = _get_latlon_ref(CarlaDataProvider.get_world())

    route = []
    gps_route = []

    for i in range(len(waypoints_trajectory) - 1):

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp, connection in interpolated_trace:
            route.append((wp.transform, connection))
            gps_coord = _location_to_gps(lat_ref, lon_ref, wp.transform.location)
            gps_route.append((gps_coord, connection))

    return gps_route, route
