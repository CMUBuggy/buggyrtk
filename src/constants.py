'''
This module defines constant expressions that are used across multiple modules.
'''
import os
from collections import namedtuple

DATA_DIR = os.environ['DATA_DIR']

G = 9.8106 # m/s/s

T = 'timestamp'

X = 'x'
Y = 'y'
Z = 'z'
STD_X = 'std_x'
STD_Y = 'std_y'
STD_Z = 'std_y'
DX = 'dx'
DY = 'dy'
DZ = 'dz'

SPEED = 'speed'
STD_SPEED = 'std_speed'

FIX_TYPE = 'rtktype'

UtmCoordinate = namedtuple('UtmCoordinate', 'x y')
STARTING_LINE = UtmCoordinate(589756, 4477312) # approximate
TRANSITION_1_2 = UtmCoordinate(589719, 4477200) # approximate
TOP_OF_HILL_2 = UtmCoordinate(589703, 4477163) # approximate
START_OF_FREEROLL = UtmCoordinate(589668, 4477146) # based on freeroll analysis
START_OF_BACK_HILLS = UtmCoordinate(589314, 4477289) # based on freeroll analysis
TRANSITION_3_4 = UtmCoordinate(589381, 4477271) # approximate
TRANSITION_4_5 = UtmCoordinate(589513, 4477237) # approximate
FINISH_LINE = UtmCoordinate(589681, 4477193) # approximate

# Refer to the Transition Zone definition in the bylaws
TRANSITION_ZONE_LENGTH = 13.716 # meters or 45 feet

HILL_1 = 'Hill 1'
HILL_2 = 'Hill 2'
HILL_3 = 'Hill 3'
HILL_4 = 'Hill 4'
HILL_5 = 'Hill 5'

HillDefinition = namedtuple('HillDefinition', 'label start end')
HILLS = {
    HILL_1: HillDefinition(HILL_1, STARTING_LINE, TRANSITION_1_2),
    HILL_2: HillDefinition(HILL_2, TRANSITION_1_2, START_OF_FREEROLL),
    HILL_3: HillDefinition(HILL_3, START_OF_BACK_HILLS, TRANSITION_3_4),
    HILL_4: HillDefinition(HILL_4, TRANSITION_3_4, TRANSITION_4_5),
    HILL_5: HillDefinition(HILL_5, TRANSITION_4_5, FINISH_LINE),
}