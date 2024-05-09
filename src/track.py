import os
import pandas as pd
import numpy as np

from functools import cached_property
from scipy.integrate import cumulative_simpson
from scipy.interpolate import splprep, splev, BSpline, splrep
from scipy.optimize import minimize

from constants import *

class Track:
    def __init__(self, label, data_path, reference_track=None, dataframe=None, **kwargs):
        self.label = label
        self.data_path = data_path
        self.full_dataframe = dataframe if dataframe is not None else pd.read_csv(data_path, comment='#')
        self.reference_track = reference_track
        self.additional_args = kwargs
        
    def __str__(self):
        return self.label
    
    @property
    def filename(self):
        return os.path.splitext(os.path.basename(self.data_path))[0]
    
    @property
    def date(self):
        return '2024-03-21' # TODO: cleanup
        return self.filename.split('_')[0]
    
    @property
    def buggy_name(self):
        return self.filename.split('_')[2]
    
    @property
    def roll_number(self):
        return self.filename.split('_')[3]
    
    def resample(self):
        '''
        Returns a track with data resampled according to recorded standard deviations
        '''
        df = self.full_dataframe.copy(deep=True)
        
        rescale = self.additional_args.get('resample_scaling', 1.0)
        df[X] += np.random.normal(scale=df[STD_X]) * rescale
        df[Y] += np.random.normal(scale=df[STD_Y]) * rescale
        df[Z] += np.random.normal(scale=df[STD_Z]) * rescale
        df[SPEED] += np.random.normal(scale=df[STD_SPEED]) * rescale
        
        new_args = dict(**self.additional_args)
        padding = self.additional_args.get('resample_padding', 2)
        new_args['padding_start'] = np.random.normal()*padding
        new_args['padding_end'] = np.random.normal()*padding
        
        return Track(self.label, 'resampled', self.reference_track, dataframe=df, **new_args)
    
    def coarse_split(self, x, y):
        '''
        Split index of closest approach to x, y
        x = utm easting
        y = utm northing
        '''
        delta = (self.full_dataframe[X] - x)**2 + (self.full_dataframe[Y] - y)**2
        return delta.idxmin()
    
    @property
    def start_of_freeroll(self):
        '''
        Returns the index of the approximate beginning of the freeroll
        As detected by a north/south threshold
        '''
        df = self.full_dataframe
        return df[df[Y] < START_OF_FREEROLL.y].index[0]
    
    @property
    def end_of_freeroll(self):
        '''
        Returns the index of the approximate end of the chute turn
        Detected as the last point east of a threshold value.
        '''
        df = self.full_dataframe
        return df[(df[X] < START_OF_BACK_HILLS.x)].index[-1]
    
    @property
    def freeroll(self):
        '''
        Slice of full_dataframe that will be used for modeling the freeroll
        Assumes one roll per track
        '''
        df = self.full_dataframe
        start_idx = self.start_of_freeroll
        end_idx = self.end_of_freeroll
        padding_start = self.additional_args.get('padding_start', 0)
        padding_end = self.additional_args.get('padding_end', 0)
        after_start = df[T] > df[T][start_idx] - padding_start
        before_end = df[T] < df[T][end_idx] + padding_end
        mask = after_start & before_end
        return df[mask]
    
    @property
    def back_hills(self):
        '''
        Slice of full_dataframe with any data that falls within the bounds of the back hills
        '''
        df = self.full_dataframe
        after_start = df[X] > START_OF_BACK_HILLS.x
        before_end = df[X] < FINISH_LINE.x
        north_of_flagstaff = df[Y] > FINISH_LINE.y - 10
        mask = after_start & before_end & north_of_flagstaff
        return df[mask]
    
    @property
    def front_hills(self):
        '''
        Slice of full_dataframe with any data that falls within the bounds of the front hills
        '''
        df = self.full_dataframe
        east_enough = df[X] > FINISH_LINE.x # + threshold?
        north_enough = df[Y] > START_OF_FREEROLL.y
        mask = east_enough & north_enough
        return df[mask]
    
    def _hill_split(self, hill, buffer):
        PROGRESS = 'hill_progress'
        DELTA = 'delta_hill_progress'
        EPSILON = 0.01
        
        df = self.full_dataframe.copy()
        
        xy0 = np.array(hill.start)
        xy1 = np.array(hill.end)
        direction = xy1 - xy0
        norm = np.linalg.norm(direction)
        direction = direction / norm
        buffer = buffer / norm
        df[PROGRESS] = np.sum((df[[X,Y]] - xy0) * direction, axis=1) / norm
        df[DELTA] = df[PROGRESS].diff()
        
        mask = (df[DELTA] > 0) & (df[PROGRESS] > 0.0 - buffer) & (df[PROGRESS] < 1.0 + buffer)
        while mask.any():
            start_idx = df[mask].index[0]
            idx = start_idx
            while idx in df.index and mask[idx]:
                mask[idx] = False
                idx +=1
            end_idx = idx - 1
            start, end = df[T][start_idx], df[T][end_idx]
            if df[PROGRESS][start_idx] < EPSILON and df[PROGRESS][end_idx] > 1 - EPSILON:
                yield hill.label, start, end
            else:
                print(f'Skipped partial hill {hill.label, start, end}')
                
    def get_all_hill_splits(self):
        for hill in HILLS.values():
            for label, start, end in self._hill_split(hill, TRANSITION_ZONE_LENGTH):
                yield label, start, end

    def time_slice(self, start, end):
        df = self.full_dataframe
        mask = (df[T] >= start) & (df[T] <= end)
        return df[mask]

    
    def get_knot_vector(self, count_key, default=50):
        '''
        Returns a knot vector for use with splrep
        The beggining and ending knots are removed per the splrep documentation
        "These should be interior knots as knots on the ends will be added automatically."
        Freeroll only
        '''
        t = self.freeroll[T]
        count = self.additional_args.get(count_key, default)
        return np.linspace(t.min(), t.max(), count)[1:-1]
        
    @cached_property
    def speed(self):
        '''
        Absolute speed as a function of time
        Freeroll only
        '''
        degree = self.additional_args.get('speed_spline_degree', 5)
        knots = self.get_knot_vector('speed_spline_knots', 15)
        df = self.freeroll
        tck = splrep(
            df[T].to_numpy(),
            df[SPEED].to_numpy(),
            w=(1/df[STD_SPEED]),
            k=degree, t=knots,
        )
        return BSpline(*tck, extrapolate=False)
    
    @cached_property
    def acceleration(self):
        '''
        Signed forward component of acceleration as a function of time
        Freeroll only
        '''
        return self.speed.derivative()
    
    @cached_property
    def pos_x(self):
        '''
        X coordinate of position as a function of time
        Freeroll only
        '''
        degree = self.additional_args.get('x_spline_degree', 5)
        knots = self.get_knot_vector('x_spline_knots', 50)
        df = self.freeroll
        tck = splrep(
            df[T].to_numpy(),
            df[X].to_numpy(),
            w=(1/df[STD_X]),
            k=degree, t=knots,
        )
        return BSpline(*tck, extrapolate=False)
    
    @cached_property
    def pos_y(self):
        '''
        Y coordinate of position as a function of time
        Freeroll only
        '''
        degree = self.additional_args.get('y_spline_degree', 5)
        knots = self.get_knot_vector('y_spline_knots', 50)
        df = self.freeroll
        tck = splrep(
            df[T].to_numpy(),
            df[Y].to_numpy(),
            w=(1/df[STD_Y]),
            k=degree, t=knots,
        )
        return BSpline(*tck, extrapolate=False)
        
    @cached_property
    def pos_z(self):
        '''
        Z coordinate of position as a function of time
        Freeroll only
        '''
        degree = self.additional_args.get('z_spline_degree', 5)
        knots = self.get_knot_vector('z_spline_knots', 15)
        df = self.freeroll
        tck = splrep(
            df[T].to_numpy(),
            df[Z].to_numpy(),
            w=(1/df[STD_Z]),
            k=degree, t=knots,
        )
        return BSpline(*tck, extrapolate=False)
    
    @cached_property
    def vel_x(self):
        '''
        X component of velocity as a function of time
        Freeroll only
        '''
        return self.pos_x.derivative()
    
    @cached_property
    def vel_y(self):
        '''
        Y component of velocity as a function of time
        Freeroll only
        '''
        return self.pos_y.derivative()
    
    @cached_property
    def vel_z(self):
        '''
        Z component of velocity as a function of time
        Freeroll only
        '''
        return self.pos_z.derivative()
    
    @cached_property
    def fine_splits(self):
        '''
        Fine split times relative to the configured reference curve
        Freeroll only
        '''
        splits = np.zeros_like(self.reference_track.fine_splits)
        for idx in range(len(splits)):
            x0 = self.reference_track.split_xs[idx]
            y0 = self.reference_track.split_ys[idx]
            dx = self.reference_track.split_dxs[idx]
            dy = self.reference_track.split_dys[idx]
            def error(time):
                projection = (self.pos_x(time) - x0) * dx + (self.pos_y(time) - y0) * dy
                return projection * projection
            nearest_sample = self.coarse_split(x0, y0)
            guess = self.full_dataframe[T][nearest_sample]
            result = minimize(error, guess)
            splits[idx] = result.x[0]
        return splits
    
    def drag(self, time):
        '''
        Drag is a force that acts to slow down the buggy.
        Freeroll only
        
        We don't know the mass of the buggy, so we normalize this as an acceleration measured in Gs.
        Drag is computed as the difference between the idealized and observed acceleration.
        
        The idealized accleration is the acceleration due to gravity for a frictionless
        buggy sliding on an inclined plane in a vacuum. 
        
        F = m * a = m * G * sin(theta)
        sin(theta) = change in altitude / distance traveled
        
        Reference:
        https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/mechanics/dynamics/inclined-planes-mechanics.html
        '''
        speed = self.speed(time)
        dz = self.vel_z(time)
        accl_ideal = -G * dz / speed
        accl_observed = self.acceleration(time)
        loss = (accl_ideal - accl_observed) / G
        return loss
    
    def curvature_2d(self, t):
        '''
        Signed curvature in X,Y as a function of time
        Freeroll only
        
        Curvature is inversly proportional to the radius of the circle of best fit.
        We are using a right handed coordinate system.
        Therefore positive curvature is "concave up" and turns left as time progresses.
        Negative curvature is "concave down" and turns right as time progresses.
        '''
        d1x = self.vel_x(t)
        d2x = self.vel_x.derivative()(t)
        d1y = self.vel_y(t)
        d2y = self.vel_y.derivative()(t)
        
        return (d1x*d2y - d1y*d2x) / (d1x*d1x + d1y*d1y) ** 1.5
        
    def kinetic_energy(self, t):
        '''
        Mass normalized kinetic energy as a function of time
        Freeroll only
        '''
        v = self.speed(t)
        return 0.5 * v ** 2
    
    def potential_energy(self, t):
        '''
        Mass normalized potential energy as a function of time
        Freeroll only
        Potential is measured relative to the lowest point of the sample
        Note that variation between may change the location of z.min()
        '''
        z = self.pos_z(t)
        return G * (z - z.min())
        
    def total_energy(self, t):
        '''
        Mass normalized total energy of the vehicle
        Freeroll only
        '''
        return self.kinetic_energy(t) + self.potential_energy(t)
    
    def get_splits_every(self, interval, N=int(1e6), tolerance=1e-3):
        '''
        Get split times for every `interval` meters of travel in the freeroll.
        Asserts that the computed distance is within `tolerance` of `interval`.
        N: number of samples to take when approximating distance
        tolerance: required precision of the computed distance
        '''
        df = self.freeroll
        start = df[T].min()
        end = df[T].max()
        t = np.linspace(start, end, N)
        dx = self.vel_x(t)
        dy = self.vel_y(t)
        dz = self.vel_z(t)
        arclength = cumulative_simpson(
            np.sqrt(dx**2 + dy**2 + dz**2),
            x=t
        )
        arclength = np.concatenate([[0.0], arclength])
        
        splits = []
        distances = []
        target = 0.0
        for idx in range(N):
            distance = arclength[idx]
            if distance >= target:
                assert (distance - target < tolerance), f'Exceeded tolerance at {dist}'
                splits.append(t[idx])
                distances.append(distance)
                target += interval
        return splits, distances