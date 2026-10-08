import numpy as np


def serialize_route(road,route):
    result=[]
    for t,_ in route:
        w=road.get_waypoint(t.location)
        result.append(dict(x=t.location.x,y=t.location.y,
            lane_id=[w.road_id,w.section_id,w.lane_id],junction=w.is_junction))
    return result


class PlannedLane:
    def __init__(self,points):
        self.points=points;self.xy=np.array([[p['x'],p['y']] for p in points]);self.index=0
        if len(points)<2:raise ValueError('Empty planned route')

    def target(self,state):
        lo=max(0,self.index-15);hi=min(len(self.points),self.index+81)
        self.index=lo+int(((self.xy[lo:hi]-[state['x'],state['y']])**2).sum(1).argmin())
        p=self.points[self.index]
        if p['junction']:raise ValueError('Planned lane junction')
        return p['lane_id']
