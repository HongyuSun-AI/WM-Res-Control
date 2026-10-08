import math
from numbers import Integral
from control_readback import check_readback


def index_readbacks(records):
    result={}
    for row in records:
        f=row['command_frame'];observed=row['observed_frame']
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<0 for v in (f,observed)):
            raise ValueError('Invalid control frame ID')
        if f in result:raise ValueError('Duplicate command frame')
        for control in (row['requested'],row['actual']):
            if not all(math.isfinite(control[k]) for k in ('steer','throttle','brake')):
                raise ValueError('Nonfinite control')
            if not (-1<=control['steer']<=1 and 0<=control['throttle']<=1 and 0<=control['brake']<=1):
                raise ValueError('Control out of bounds')
            if min(control['throttle'],control['brake'])>1e-4:raise ValueError('Overlapping pedals')
        checked=check_readback(f,observed,row['requested'],row['actual'],tolerance=1e-5)
        if not checked['matched'] or row.get('matched') is not True:raise ValueError('Readback mismatch')
        result[f]=checked
    return result


def verified_action(row,readbacks,available_frame=None):
    f=row['state']['frame']
    if f not in readbacks:raise ValueError('Missing readback for command frame '+str(f))
    rb=readbacks[f]
    if available_frame is not None and rb['observed_frame']>available_frame:
        raise ValueError('Readback unavailable at policy time')
    if any(abs(rb['requested'][k]-row['control_final'][k])>1e-7 for k in ('steer','throttle','brake')):
        raise ValueError('Readback/reference control mismatch')
    actual=rb['actual']
    return [actual['steer'],actual['throttle']-actual['brake']]
