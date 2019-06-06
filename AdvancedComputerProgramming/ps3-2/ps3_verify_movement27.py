# 6.0002 Problem Set 3:
# Edited Fall 2016

import math
import random
import ps3_visualize
import pylab


def test_robot_movement(robot_type, room_type):
    # check if room is furnished room
    is_furnished = str(room_type).find('FurnishedRoom') > 0
    
    room = room_type(5, 5, 4)
    if is_furnished:
        room.add_furniture_to_room()
    robots = [robot_type(room, 1, 1)]
    coverage = 0
    time_steps = 0
    min_coverage = 1.0
    if is_furnished:
        anim = ps3_visualize.RobotVisualization(1, 5, 5, room.furniture_tiles) 
    else:
        anim = ps3_visualize.RobotVisualization(1, 5, 5, [])  
    while coverage < min_coverage:
        time_steps += 1 
        for robot in robots:
            robot.update_position_and_clean()
            anim.update(room, robots)
            coverage = float(room.get_num_cleaned_tiles())/room.get_num_tiles()
    anim.done()
