"""Bounded delta debugging, requiring an explicit deterministic failure oracle."""


def minimize(candidates, fails, *, max_attempts=1000):
  """Find a 1-minimal subsequence while preserving the supplied failure.

  This never executes unknown code by itself or rewrites the user's source.
  fails(tuple) must return bool, and identical inputs must give identical
  results. Budget exhaustion is reported rather than called a minimal case.
  """
  if type(max_attempts) is not int or max_attempts < 2:
    raise ValueError("max_attempts must be at least two")
  remaining = tuple(candidates)
  observations, attempts = {}, 0

  def check(value):
    nonlocal attempts
    result = fails(value)
    attempts += 1
    if type(result) is not bool:
      raise TypeError("failure oracle must return bool")
    key = tuple(id(v) for v in value)
    if key in observations and observations[key] != result:
      raise ValueError("failure oracle is nondeterministic")
    observations[key] = result
    return result

  if not check(remaining) or not check(remaining):
    raise ValueError("initial case does not reproduce the failure")
  width = max(1, len(remaining) // 2)
  while remaining and attempts < max_attempts:
    changed = False
    exhausted = False
    for start in range(0, len(remaining), width):
      trial = remaining[:start] + remaining[start + width :]
      if check(trial):
        if attempts < max_attempts and not check(trial):
          raise ValueError("failure oracle is nondeterministic")
        remaining, changed = trial, True
        break
      if attempts >= max_attempts:
        exhausted = start + width < len(remaining)
        break
    if not changed:
      if width == 1:
        return {"case": remaining, "attempts": attempts, "minimal": not exhausted}
      width = max(1, width // 2)
  return {"case": remaining, "attempts": attempts, "minimal": not remaining}
