def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg
