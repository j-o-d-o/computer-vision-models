from data.carla import Params
import os
import sys
import random
import time
import cv2
import numpy as np

# Importing carla
carla_egg_file = os.path.join(Params.CARLA_BASE_PATH, 'PythonAPI/carla/dist/carla-0.9.11-py3.7-%s.egg' % ('win-amd64' if os.name == 'nt' else 'linux-x86_64'))
if not os.path.isfile(carla_egg_file):
    print("WARNING: Carla egg file not found at %s" % carla_egg_file)
sys.path.append(carla_egg_file)
import carla


def show_rgb(image):
    img = np.array(image.raw_data).astype('uint8')
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]
    print("---")
    print("Update!")
    cv2.imshow("Camera RGB", img)
    cv2.waitKey(1)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    print("Available Maps:")
    print(client.get_available_maps())
    client.load_world('Town05')
    client.reload_world()
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Create vehicle
    vehicle_bp = blueprint_library.filter('vehicle.citroen.c3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Make sure the spectator camera is following our vehicle by spawning a dummy actor
    dummy_sensor_bp = blueprint_library.find('sensor.other.collision')
    dummy_sensor_transform = carla.Transform(carla.Location(x=-6, z=3))
    dummy_sensor = world.spawn_actor(dummy_sensor_bp, dummy_sensor_transform, attach_to=ego_vehicle)
    spectator = world.get_spectator()

    # Attach sensors to vehicle
    cam_rgb_bp = blueprint_library.find('sensor.camera.rgb')
    cam_rgb_transform = carla.Transform(carla.Location(x=-1, z=1.2))
    camera_rgb = world.spawn_actor(cam_rgb_bp, cam_rgb_transform, attach_to=ego_vehicle)
    # camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame
    camera_rgb.listen(show_rgb)
    
    # spawn somewhere
    # spawn_points = world.get_map().get_spawn_points()

    # spawn at random waypoint and go to next waypoint
    # Find next waypoint 2 meters ahead.
    # waypoint = random.choice(waypoint.next(2.0))
    # Drive there
    # vehicle.set_transform(waypoint.transform)

    # Debug test, just drive forward
    ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.02))
    while True:
        timestamp = world.wait_for_tick()
        spectator.set_transform(dummy_sensor.get_transform())

if __name__ == "__main__":
    main()
