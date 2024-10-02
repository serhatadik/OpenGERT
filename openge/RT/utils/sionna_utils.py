import geopy

def lonlat_to_sionna_xy(lat, lon, min_lon, min_lat, max_lon, max_lat):
    # Calculate the center of the scene
    center_scene_lat = (min_lat + max_lat) / 2.0
    center_scene_lon = (min_lon + max_lon) / 2.0

    # Define coordinates for latitude and longitude distance calculations
    coords_y1 = (lat, lon)
    coords_y2 = (center_scene_lat, lon)  # Point directly above or below on the same longitude

    coords_x1 = (lat, lon)
    coords_x2 = (lat, center_scene_lon)  # Point directly left or right on the same latitude

    # Calculate distances using geopy, but return signed distances
    distance_x = geopy.distance.distance(coords_x1, coords_x2).m
    distance_y = geopy.distance.distance(coords_y1, coords_y2).m

    # Adjust the sign of the distance based on the latitude and longitude difference
    if lat < center_scene_lat:
        distance_y = -distance_y  # Negative if latitude is south of the center
    if lon < center_scene_lon:
        distance_x = -distance_x  # Negative if longitude is west of the center

    return (distance_x, distance_y)
