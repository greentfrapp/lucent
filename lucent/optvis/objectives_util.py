def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg

def _extract_act_pos(acts, x=None, y=None):
  shape = acts.shape
  x_ = shape[2] // 2 if x is None else x
  y_ = shape[3] // 2 if y is None else y
  return acts[:, :, y_:y_+1, x_:x_+1]
