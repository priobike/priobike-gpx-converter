import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from mapbox import fetch_city_map
import gpxpy
import gpxpy.gpx
from math import sqrt, pi, ceil, inf
import requests
import json
import scipy

WGS_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
MERCATOR_TO_WGS = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


class Waypoint:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng


def parse_gpx():
    gpx_file = open('test2.gpx', 'r')
    gpx = gpxpy.parse(gpx_file)
    gpx_lats = []
    gpx_lngs = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                gpx_lats.append(point.latitude)
                gpx_lngs.append(point.longitude)
    return WGS_TO_MERCATOR.transform(gpx_lngs, gpx_lats)


def bounding_box(gpx_xs, gpx_ys):
    gpx_lngs, gpx_lats = MERCATOR_TO_WGS.transform(gpx_xs, gpx_ys)

    # Calculate the Web Mercator bounding box
    max_lat = np.array(gpx_lats).max()
    min_lat = np.array(gpx_lats).min()
    max_lng = np.array(gpx_lngs).max()
    min_lng = np.array(gpx_lngs).min()
    min_x, min_y = WGS_TO_MERCATOR.transform(min_lng, min_lat)
    max_x, max_y = WGS_TO_MERCATOR.transform(max_lng, max_lat)

    # Make the bounding box square
    x_diff = max_x - min_x
    y_diff = max_y - min_y
    if x_diff > y_diff:
        min_y -= (x_diff - y_diff) / 2
        max_y += (x_diff - y_diff) / 2
    else:
        min_x -= (y_diff - x_diff) / 2
        max_x += (y_diff - x_diff) / 2

    # Add some padding
    min_x -= 1000
    min_y -= 1000
    max_x += 1000
    max_y += 1000
    return min_x, min_y, max_x, max_y


def calc_angle(x1, y1, x2, y2):
    vector = np.array([x2 - x1, y2 - y1])
    abs_vector = sqrt(pow(vector[0], 2) + pow(vector[1], 2))
    if abs_vector == 0: return 0
    return np.arccos(vector[0] / abs_vector)


def calc_distance(x1, y1, x2, y2):
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def approximate_gpx(gpx_xs, gpx_ys, angle_threshold=(1/20)*pi, distance_threshold1=10000, distance_threshold2=1000):
    approx_xs = []
    approx_ys = []
    last_x = gpx_xs[0]
    last_y = gpx_ys[0]
    last_angle = 0
    approx_xs.append(last_x)
    approx_ys.append(last_y)
    for i in range(len(gpx_xs)):
        x = gpx_xs[i]
        y = gpx_ys[i]
        angle = calc_angle(x, y, last_x, last_y)
        distance = calc_distance(x, y, last_x, last_y)
        if distance > distance_threshold1 or (abs(last_angle - angle) > angle_threshold and distance > distance_threshold2):
            approx_xs.append(x)
            approx_ys.append(y)
            last_x = x
            last_y = y
            last_angle = angle
    approx_xs.append(gpx_xs[-1])
    approx_ys.append(gpx_ys[-1])
    return approx_xs, approx_ys


def get_gh_route(waypoints):
    baseUrl = 'priobike.flow-d.de'
    servicePath = 'drn-graphhopper'
    ghUrl = f'https://{baseUrl}/{servicePath}/route'
    ghUrl += '?type=json'
    ghUrl += '&locale=de'
    ghUrl += '&elevation=true'
    ghUrl += '&points_encoded=false'
    ghUrl += f'&profile=bike2_default'
    # Add the supported details.This must be specified in the GraphHopper config.
    ghUrl += '&details=surface'
    ghUrl += '&details=max_speed'
    ghUrl += '&details=smoothness'
    ghUrl += '&details=lanes'
    ghUrl += '&details=road_class'
    if len(waypoints) == 2:
        ghUrl += '&algorithm=alternative_route'
        ghUrl += '&ch.disable=true'
    for waypoint in waypoints:
        ghUrl += f'&point={waypoint.lat},{waypoint.lng}'
    response = requests.get(ghUrl)
    return response


def reconstruct_route(approx_xs, approx_ys):
    approx_lngs, approx_lats = MERCATOR_TO_WGS.transform(approx_xs, approx_ys)
    max_waypoints = 200  # max number of waypoints for gh is 238
    waypoints = []
    for i in range(len(approx_lngs)):
        waypoints.append(Waypoint(approx_lats[i], approx_lngs[i]))
    response = get_gh_route(waypoints[::ceil(len(waypoints)/max_waypoints)])
    if response.status_code == 200:
        decoded = json.loads(response.text)['paths'][0]
        points = decoded['points']['coordinates']
        return WGS_TO_MERCATOR.transform([row[0] for row in points], [row[1] for row in points])
    raise Exception(response.text)


def main():
    fig, ax = plt.subplots()

    # parse gpx
    gpx_xs, gpx_ys = parse_gpx()

    # map tile
    min_x, min_y, max_x, max_y = bounding_box(gpx_xs, gpx_ys)
    img_min_lng, img_min_lat = MERCATOR_TO_WGS.transform(min_x, min_y)
    img_max_lng, img_max_lat = MERCATOR_TO_WGS.transform(max_x, max_y)
    img = fetch_city_map(img_min_lat, img_min_lng, img_max_lat, img_max_lng)
    ax.imshow(img, extent=[min_x, max_x, min_y, max_y], aspect='1', zorder=0, interpolation='bilinear')

    # gpx
    ax.scatter(gpx_xs, gpx_ys, zorder=1, marker='x', c='orange')

    def cost(params):
        a_thrsh, d1_thrsh, d2_thrsh = params
        # approximated waypoints
        approx_xs, approx_ys = approximate_gpx(
            gpx_xs,
            gpx_ys,
            angle_threshold=a_thrsh,
            distance_threshold1=d1_thrsh,
            distance_threshold2=d2_thrsh
        )
        # reconstructed route
        rec_xs, rec_ys = reconstruct_route(approx_xs, approx_ys)
        # calc distance of reconstructed route from original gpx
        # for each point of reconstructed route calc distance to the closest point of gpx route
        total_d = 0
        for i in range(len(rec_xs)):
            rec_x = rec_xs[i]
            rec_y = rec_ys[i]
            d = inf
            for j in range(len(gpx_xs)):
                gpx_x = gpx_xs[j]
                gpx_y = gpx_ys[j]
                local_d = sqrt(pow(rec_x - gpx_x, 2) + pow(rec_y - gpx_y, 2))
                if local_d < d: d = local_d
            total_d += d
        norm_d = total_d / len(rec_xs)
        return norm_d * len(approx_xs)

    def opt_progress(xk, convergence):
        print(f'best solution so far: {xk}, convergence: {convergence}')

    result = scipy.optimize.differential_evolution(
        func=cost,
        x0=np.array([0.0, 1000.0, 100.0]),
        bounds=[(0.0, pi / 2), (0.0, 10000.0), (0.0, 10000.0)],
        callback=opt_progress
    )

    print(result.x)
    print(cost(result.x))


if __name__ == '__main__':
    main()
