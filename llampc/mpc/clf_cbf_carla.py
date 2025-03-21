#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA auto control.

"""
from __future__ import print_function

"""	Nonlinear MPC using true Dynamic bicycle model.
"""

__author__ = 'Tianhao Wu'
__email__ = 'twu4@andrew.cmu.edu'


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import matplotlib.pyplot as plt
try:
    sys.path.append('/home/dvij/bayesrace')
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Imports from Bayesrace ----------------------------------------------------
# ==============================================================================
import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from llampc.params import ORCA, CarlaParams
from llampc.models import Dynamic
from llampc.tracks import ETHZ, CarlaRace
from llampc.mpc.planner import ConstantSpeed
from llampc.mpc.nmpc import setupNLP

# ==============================================================================
# -- Imports from carla --------------------------------------------------------
# ==============================================================================


import carla

import argparse
import logging
import math
from scipy.interpolate import interp1d
from carla_utils import *

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


#####################################################################
# default settings


SAMPLING_TIME = 0.02
MANUAL_CONTROL = False


#####################################################################
# load vehicle parameters

params = CarlaParams(control='pwm')
model = Dynamic(**params)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        world = client.load_world('Town10HD')
        
        sim_world = client.get_world()

        if args.sync:
            print("In synchronous mode")
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.02
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)

        controller = KeyboardControl(world, args.autopilot)
        
        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        _control = carla.VehicleControl()
        
        while True:
            
            if args.sync:
                sim_world.tick()
                clock.tick_busy_loop(60)
            
            location = world.player.get_location()
            velocity = world.player.get_velocity()
            vx = velocity.x
            vy = velocity.y
            acc = world.player.get_acceleration()
            w = world.player.get_angular_velocity().z*math.pi/180.
            yaw = world.player.get_transform().rotation.yaw*math.pi/180.
            roll = world.player.get_transform().rotation.roll
            pitch = world.player.get_transform().rotation.pitch
            states[0,itr] = location.x
            states[1,itr] = location.y
            states[2,itr] = yaw
            states[3,itr] = vx*math.cos(yaw) + vy*math.sin(yaw)
            states[4,itr] = -vx*math.sin(yaw) + vy*math.cos(yaw)
            states[5,itr] = w
            # print(states[:,itr+1])
            # states[6,itr+1] = (inputs[1,itr]-states[6,itr])/dt
            
            
            if MANUAL_CONTROL :
                if controller.parse_events(client, world, clock, args.sync):
                    return
            else :
                pass
             
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.ford.mustang")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        default=True,
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
