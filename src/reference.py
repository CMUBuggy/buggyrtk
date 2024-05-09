import os
from track import Track
from constants import DATA_DIR

default_reference_file = f'{DATA_DIR}/parsed/yyyy-mm-dd_hhmmss_SonOfThunderpuppy_0.csv'

def load(file=None):
    '''
    Loads a GPS Track and sets it up to act as reference track
    '''
    file = file or default_reference_file
    track = Track('reference', file)
    sampling_distance = 5 # meters
    # TODO maybe cache this to a file?
    splits, distance = track.get_splits_every(sampling_distance)
    track.fine_splits = splits
    track.split_xs = track.pos_x(splits)
    track.split_ys = track.pos_y(splits)
    track.split_dxs = track.vel_x(splits)
    track.split_dys = track.vel_y(splits)
    return distance, track
