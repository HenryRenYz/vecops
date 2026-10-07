"""Strict metadata selection shared by complete DSL benchmark catalogs."""


def validate_catalog_filters(parser, cases, filters, *, allow_empty=False):
  """Reject unknown values and silently dropped nonempty filter choices.

  ``filters`` contains (CLI flag, requested values, case getter) tuples. An
  empty metadata listing is useful and allowed, but even listings reject
  values absent from the catalog. Runtime runs also require every requested
  choice to have at least one case under the other requested filters.
  Partition/resume/limit happen later and may legitimately leave zero work.
  """
  filters = tuple((flag, choices, getter) for flag, choices, getter in filters if choices)
  for flag, choices, getter in filters:
    available = {getter(case) for case in cases}
    unknown = set(choices) - available
    if unknown:
      parser.error(f"{flag} contains values absent from the catalog: {', '.join(sorted(unknown))}")
  matches = [case for case in cases if all(getter(case) in choices for _, choices, getter in filters)]
  if not allow_empty:
    for flag, choices, getter in filters:
      compatible = {
        getter(case)
        for case in cases
        if all(
          other_getter(case) in other_choices
          for other_flag, other_choices, other_getter in filters
          if other_flag != flag
        )
      }
      unsupported = set(choices) - compatible
      if unsupported:
        parser.error(f"{flag} has no catalog case with the other requested filters: {', '.join(sorted(unsupported))}")
    if not matches:
      parser.error("requested filters select no catalog cases")
  return set(matches)


def parse_partition(parser, value):
  try:
    part, count = map(int, value.split("/"))
  except ValueError:
    parser.error("partition requires INDEX/COUNT integers")
  if count < 1 or not 0 <= part < count:
    parser.error("partition requires 0 <= INDEX < COUNT")
  return part, count
