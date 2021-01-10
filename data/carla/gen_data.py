from data.carla import Params
import os
import sys

# Importing carla
carla_egg_file = os.path.join(Params.CARLA_BASE_PATH, 'PythonAPI/carla/dist/carla-0.9.11-py3.7-%s.egg' % ('win-amd64' if os.name == 'nt' else 'linux-x86_64'))
if not os.path.isfile(carla_egg_file):
    print("WARNING: Carla egg file not found at %s" % carla_egg_file)
sys.path.append(carla_egg_file)
import carla


