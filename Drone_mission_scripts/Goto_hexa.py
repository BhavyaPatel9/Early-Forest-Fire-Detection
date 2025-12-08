import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import argparse

# -----------------------------
# Connect to Vehicle
# -----------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--connect' , default = '/dev/ttyACM0')
args = parser.parse_args()
vehicle = connect(args.connect, wait_ready=True)

# -----------------------------
# Helper Functions
# -----------------------------

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to target altitude.
    """
    print("Basic pre-arm checks...")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f} m")
        if alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)


def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobalRelative object moved by dNorth and dEast meters.
    """
    earth_radius = 6378137.0  # Radius of "spherical" earth
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, original_location.alt)


def hexagon_vertices(start_location, side_length):
    """
    Returns 6 vertices (LocationGlobalRelative) of a hexagon.
    First vertex = start_location.
    """
    vertices = [start_location]
    for i in range(1, 6):
        angle_deg = 60 * i
        angle_rad = math.radians(angle_deg)
        dNorth = side_length * math.cos(angle_rad)
        dEast = side_length * math.sin(angle_rad)
        new_point = get_location_metres(start_location, dNorth, dEast)
        vertices.append(new_point)
    return vertices


def goto_with_delay(location, delay_time=5):
    print(f"Going to: Lat {location.lat:.6f}, Lon {location.lon:.6f}")
    vehicle.simple_goto(location)
    time.sleep(delay_time)  # Wait at the vertex


# -----------------------------
# Mission Start
# -----------------------------
print("Taking off to 10m altitude")
arm_and_takeoff(10)

home_location = vehicle.location.global_frame
print(f"Home set at: {home_location.lat:.6f}, {home_location.lon:.6f}")

vehicle.airspeed = 3

# Given target coordinate (first vertex)
point1 = LocationGlobalRelative(21.1600971, 72.7867148, 10)
print("Going to first vertex (given coordinate)")
vehicle.simple_goto(point1)
time.sleep(10)

# Generate hexagon vertices (side = 5 m)
vertices = hexagon_vertices(point1, 5)

# Visit all vertices and wait 5 seconds each
print("Starting hexagon path...")
for i, vertex in enumerate(vertices):
    print(f"Visiting vertex {i+1}")
    goto_with_delay(vertex, delay_time=5)

# -----------------------------
# Return to Launch (RTL)
# -----------------------------
print("Hexagon completed. Returning to home...")
vehicle.mode = VehicleMode("RTL")

# Wait until disarmed
while vehicle.armed:
    print(" Waiting for drone to land and disarm...")
    time.sleep(2)

print("Mission completed. Closing vehicle object.")
vehicle.close()
