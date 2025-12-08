import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--connect' , default = '/dev/ttyACM0')
args = parser.parse_args()
vehicle = connect(args.connect, wait_ready=True)

def set_servo(vehicle, servo_number, pwm_value):
	pwm_value_int = int(pwm_value)
	msg = vehicle.message_factory.command_long_encode(
		0, 0, 
		mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
		0,
		servo_number,
		pwm_value_int,
		0,0,0,0,0
		)
	vehicle.send_mavlink(msg)  

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

arm_and_takeoff(10)
Home = vehicle.location.global_frame
Home.alt = 10
print("Set default/target airspeed to 3")
vehicle.airspeed = 3

print("Going towards point B")
point1 = LocationGlobalRelative(21.1600971, 72.7867148, 10)
vehicle.simple_goto(point1)
time.sleep(10)
vehicle.mode = VehicleMode("LAND")
time.sleep(10)



while vehicle.armed:
    print(" Waiting for disarming...")
    time.sleep(5)
    
set_servo(vehicle,6,1900)
time.sleep(5)


arm_and_takeoff(10)
vehicle.home_location = Home

print("Now RTL to initial home")
vehicle.mode = VehicleMode("RTL")

# Close vehicle object before exiting script
print("Close vehicle object")
vehicle.close()

