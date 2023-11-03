import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from mapbox import fetch_city_map
import gpxpy
import gpxpy.gpx
from math import sqrt, pi, ceil, inf
import requests
import json

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


def initial_approximate_gpx(gpx_xs, gpx_ys, angle_threshold=(1 / 10) * pi, distance_threshold1=10000.0, distance_threshold2=5000.0):
    approx_xs = []
    approx_ys = []
    gpx_indices = []
    last_x = gpx_xs[0]
    last_y = gpx_ys[0]
    last_angle = 0
    gpx_indices.append(0)
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
            gpx_indices.append(i)
            last_x = x
            last_y = y
            last_angle = angle
    approx_xs.append(gpx_xs[-1])
    approx_ys.append(gpx_ys[-1])
    gpx_indices.append(len(gpx_xs)-1)
    return approx_xs, approx_ys, gpx_indices


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


def cost(gpx_xs, gpx_ys, rec_xs, rec_ys):
    # calc distance of original gpx to reconstructed route
    # for each point of gpx route calc distance to the closest point of reconstructed route
    ds = []
    total_d = 0
    if len(gpx_xs) == 0:
        return ds, total_d
    for j in range(len(gpx_xs)):
        gpx_x = gpx_xs[j]
        gpx_y = gpx_ys[j]
        d = inf
        for i in range(len(rec_xs)):
            rec_x = rec_xs[i]
            rec_y = rec_ys[i]
            local_d = sqrt(pow(rec_x - gpx_x, 2) + pow(rec_y - gpx_y, 2))
            if local_d < d: d = local_d
        ds.append(d)
        total_d += d
    return ds, total_d / len(gpx_xs)


def get_closest(xs, ys, x_s, y_s):
    d = inf
    i = 0
    x_closest = 0
    y_closest = 0
    for j in range(len(xs)):
        x = xs[j]
        y = ys[j]
        local_d = sqrt(pow(x_s - x, 2) + pow(y_s - y, 2))
        if local_d < d:
            d = local_d
            i = j
            x_closest = x
            y_closest = y
    return i, d, x_closest, y_closest


def iteratively_improve_approx(gpx_xs, gpx_ys):
    approx_xs, approx_ys, gpx_indices = initial_approximate_gpx(gpx_xs, gpx_ys)  # [gpx_xs[0], gpx_xs[-1]], [gpx_ys[0], gpx_ys[-1]]
    rec_xs, rec_ys = reconstruct_route(approx_xs, approx_ys)
    cs, total_cost = cost(gpx_xs, gpx_ys, rec_xs, rec_ys)
    # gpx_indices = [0, len(gpx_xs) - 1]
    last_cost = inf
    d_thrsh_global = 100
    d_thrsh_local = 200
    while total_cost > d_thrsh_global and d_thrsh_local >= 100:
        insertions = []
        for i in range(len(approx_xs) - 1):
            rec_xs, rec_ys = reconstruct_route(approx_xs[i:i+2], approx_ys[i:i+2])
            cs, total_cost = cost(gpx_xs[gpx_indices[i]:gpx_indices[i + 1] - 1], gpx_ys[gpx_indices[i]:gpx_indices[i + 1] - 1], rec_xs, rec_ys)
            if len(cs) == 0:
                continue
            cs = np.array(cs)
            i_max = gpx_indices[i] + cs.argmax()
            if total_cost > d_thrsh_local:
                insertions.insert(0, (i+1, i_max, gpx_xs[i_max], gpx_ys[i_max]))
        for insertion in insertions:
            approx_xs.insert(insertion[0], insertion[2])
            approx_ys.insert(insertion[0], insertion[3])
            gpx_indices.insert(insertion[0], insertion[1])
        rec_xs, rec_ys = reconstruct_route(approx_xs, approx_ys)
        cs, total_cost = cost(gpx_xs, gpx_ys, rec_xs, rec_ys)
        if total_cost == last_cost:
            d_thrsh_local = d_thrsh_local / 2
        last_cost = total_cost
        print(f'cost: {total_cost}, #waypoints: {len(approx_xs)}')
    return approx_xs, approx_ys


def main():
    fig, ax = plt.subplots()
    gpx_xs, gpx_ys = parse_gpx()
    approx_xs, approx_ys = iteratively_improve_approx(gpx_xs, gpx_ys)
    rec_xs, rec_ys = reconstruct_route(approx_xs, approx_ys)
    _, total_d = cost(gpx_xs, gpx_ys, rec_xs, rec_ys)
    print(f'#gpx points: {len(gpx_xs)}, #approx points: {len(approx_xs)}, d: {total_d}')
    # plot
    # map tile
    min_x, min_y, max_x, max_y = bounding_box(gpx_xs, gpx_ys)
    img_min_lng, img_min_lat = MERCATOR_TO_WGS.transform(min_x, min_y)
    img_max_lng, img_max_lat = MERCATOR_TO_WGS.transform(max_x, max_y)
    img = fetch_city_map(img_min_lat, img_min_lng, img_max_lat, img_max_lng)
    ax.imshow(img, extent=[min_x, max_x, min_y, max_y], aspect='1', zorder=0, interpolation='bilinear')
    # gpx route
    ax.scatter(gpx_xs, gpx_ys, zorder=1, marker='x', c='orange')
    # approximated route
    ax.scatter(approx_xs, approx_ys, zorder=3, facecolors='none', edgecolors='r', linewidths=2, s=160)
    # reconstructed route
    ax.scatter(rec_xs, rec_ys, zorder=2, facecolors='none', edgecolors='m', linewidths=2, s=160)
    plt.show()


if __name__ == '__main__':
    main()
