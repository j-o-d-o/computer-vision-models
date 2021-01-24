import carla
import random

class World():
    def __init__(self, client: carla.Client):
        print("Available Maps:")
        print(client.get_available_maps())
        client.load_world('Town04')
        client.reload_world()
        self.carla_world: carla.World = client.get_world()
        blueprint_library: carla.BlueprintLibrary = self.carla_world.get_blueprint_library()

        # Create vehicle
        vehicle_bp = blueprint_library.filter('vehicle.citroen.c3')[0]
        spawn_point = random.choice(self.carla_world.get_map().get_spawn_points())
        self.ego_vehicle: carla.Vehicle = self.carla_world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle.set_autopilot(True)

    def destroy(self):
        self.ego_vehicle.destroy()
