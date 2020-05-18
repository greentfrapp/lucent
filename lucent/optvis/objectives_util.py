
"""Utility functions for Objectives."""

def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg

def _extract_act_pos(acts, x=None, y=None):
    shape = acts.shape
    x = shape[2] // 2 if x is None else x
    y = shape[3] // 2 if y is None else y
    return acts[:, :, y:y+1, x:x+1]

def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        if isinstance(batch, int):
            return t[batch:batch+1]
        else:
            return t
    return T2
