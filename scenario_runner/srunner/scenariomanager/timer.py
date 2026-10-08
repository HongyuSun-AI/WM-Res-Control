#!/usr/bin/env python


import datetime
import operator
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class GameTime(object):


    _current_game_time = 0.0
    _carla_time = 0.0
    _last_frame = 0
    _platform_timestamp = 0
    _init = False

    @staticmethod
    def on_carla_tick(timestamp):
        if GameTime._last_frame < timestamp.frame:
            frames = timestamp.frame - GameTime._last_frame if GameTime._init else 1
            GameTime._current_game_time += timestamp.delta_seconds * frames
            GameTime._last_frame = timestamp.frame
            GameTime._platform_timestamp = datetime.datetime.now()
            GameTime._init = True
            GameTime._carla_time = timestamp.elapsed_seconds

    @staticmethod
    def restart():
        GameTime._current_game_time = 0.0
        GameTime._carla_time = 0.0
        GameTime._last_frame = 0
        GameTime._init = False

    @staticmethod
    def get_time():
        return GameTime._current_game_time

    @staticmethod
    def get_carla_time():
        return GameTime._carla_time

    @staticmethod
    def get_wallclocktime():
        return GameTime._platform_timestamp

    @staticmethod
    def get_frame():
        return GameTime._last_frame


class SimulationTimeCondition(py_trees.behaviour.Behaviour):


    def __init__(self, timeout, comparison_operator=operator.gt, name="SimulationTimeCondition"):
        super(SimulationTimeCondition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._timeout_value = timeout
        self._start_time = 0.0
        self._comparison_operator = comparison_operator

    def initialise(self):
        self._start_time = GameTime.get_time()
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):

        elapsed_time = GameTime.get_time() - self._start_time

        if not self._comparison_operator(elapsed_time, self._timeout_value):
            new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class TimeOut(SimulationTimeCondition):


    def __init__(self, timeout, name="TimeOut"):
        super(TimeOut, self).__init__(timeout, name=name)
        self.timeout = False

    def update(self):

        new_status = super(TimeOut, self).update()

        if new_status == py_trees.common.Status.SUCCESS:
            self.timeout = True

        return new_status


class RouteTimeoutBehavior(py_trees.behaviour.Behaviour):
    MIN_TIMEOUT = 300
    TIMEOUT_ROUTE_PERC = 10

    def __init__(self, ego_vehicle, route, debug=False, name="RouteTimeoutBehavior"):
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._ego_vehicle = ego_vehicle
        self._route = route
        self._debug = debug

        self._start_time = None
        self._timeout_value = self.MIN_TIMEOUT
        self.timeout = False

        self._wsize = 3
        self._current_index = 0

        self._route_length = len(self._route)
        self._route_transforms, _ = zip(*self._route)

        self._route_accum_meters = []
        prev_loc = self._route_transforms[0].location
        for i, tran in enumerate(self._route_transforms):
            loc = tran.location
            d = loc.distance(prev_loc)
            accum = 0 if i == 0 else self._route_accum_meters[i - 1]

            self._route_accum_meters.append(d + accum)
            prev_loc = loc

    def initialise(self):
        self._start_time = GameTime.get_time()
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        ego_location = CarlaDataProvider.get_location(self._ego_vehicle)
        if ego_location is None:
            return new_status

        new_index = self._current_index

        for index in range(self._current_index, min(self._current_index + self._wsize + 1, self._route_length)):
            route_transform = self._route_transforms[index]
            route_veh_vec = ego_location - route_transform.location
            if route_veh_vec.dot(route_transform.get_forward_vector()) > 0:
                new_index = index

        if new_index > self._current_index:
            dist = self._route_accum_meters[new_index] - self._route_accum_meters[self._current_index]
            max_speed = self._ego_vehicle.get_speed_limit() / 3.6
            timeout_speed = max_speed * self.TIMEOUT_ROUTE_PERC / 100
            self._timeout_value += dist / timeout_speed
            self._current_index = new_index

        elapsed_time = GameTime.get_time() - self._start_time
        if elapsed_time > self._timeout_value:
            new_status = py_trees.common.Status.SUCCESS
            self.timeout = True

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status
