import math


def control_values(control):
    return {k:float(getattr(control,k)) for k in ('steer','throttle','brake')}


def check_readback(command_frame,observed_frame,requested,actual,tolerance=1e-5):
    if observed_frame!=command_frame+1:raise ValueError('Next-frame readback required')
    if not math.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid tolerance')
    delta={k:abs(actual[k]-requested[k]) for k in ('steer','throttle','brake')}
    if not all(math.isfinite(v) for d in (requested,actual) for v in d.values()):raise ValueError('Nonfinite control')
    return dict(command_frame=command_frame,observed_frame=observed_frame,requested=requested,
                actual=actual,abs_delta=delta,matched=all(v<=tolerance for v in delta.values()),tolerance=tolerance)
