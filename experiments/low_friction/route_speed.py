import math
import numpy as np


class RouteSpeedPlan:
    def __init__(self,points,target_kmh):
        xy=np.array([[p['x'],p['y']] for p in points],dtype=float)
        xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-6]]
        if len(xy)<2:raise ValueError('Route too short')
        self.xy=xy;self.delta=np.diff(xy,axis=0);self.length=np.linalg.norm(self.delta,axis=1)
        self.arc=np.r_[0,np.cumsum(self.length)];self.index=0;self.target_kmh=target_kmh

    def observe(self,state):
        pos=np.array([state['x'],state['y']]);lo=max(0,self.index-5);hi=min(len(self.delta),self.index+81)
        delta=self.delta[lo:hi];a=self.xy[lo:hi]
        u=np.clip(((pos-a)*delta).sum(1)/(self.length[lo:hi]**2),0,1)
        points=a+u[:,None]*delta;j=int(((points-pos)**2).sum(1).argmin());self.index=lo+j
        remaining=float(self.arc[-1]-self.arc[self.index]-u[j]*self.length[self.index])
        cap=min(self.target_kmh/3.6,math.sqrt(2*.8*max(0.,remaining-5.)))
        if remaining<=5.2:cap=0.
        stopped=remaining<=6 and np.linalg.norm(pos-self.xy[-1])<=8 and math.hypot(state['vx'],state['vy'])<.15
        if stopped:cap=0.
        return dict(remaining_route_m=remaining,commanded_speed_kmh=cap*3.6,
                    phase='arrival' if cap<self.target_kmh/3.6 else 'cruise',stopped=bool(stopped))
