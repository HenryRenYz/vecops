# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Build shared-page compiler profiles from checked, matched rank traces."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def _read(directory, route, warmup):
  sequences = defaultdict(dict)
  sizes = {}
  for path in sorted(Path(directory).glob('*.samples.*.jsonl')):
    per_process = defaultdict(dict)
    occurrences = defaultdict(int)
    for line in path.read_text().splitlines():
      row = json.loads(line)
      if row['route'] != route:
        continue
      case, rank, size = row['case'], row['rank'], row['size']
      elapsed = row['elapsed_us']
      if not 0 <= rank < size or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f'invalid rank timing in {path}')
      if sizes.setdefault(case,size) != size:
        raise ValueError(f'inconsistent rank count for case {case}')
      site = row.get('graph',''),row['ordinal']
      occurrence = occurrences[case,rank,site]
      occurrences[case,rank,site] += 1
      per_process[case,rank][(*site,occurrence)] = elapsed
    for (case,rank),samples in per_process.items():
      if rank in sequences[case]:
        raise ValueError(f'multiple runs for rank {rank} in case {case}')
      sequences[case][rank] = samples
  result = {}
  for case, ranks in sequences.items():
    if len(ranks) != sizes[case] or any(ranks[0].keys() != x.keys() for x in ranks.values()):
      continue
    values = {site:max(rank[site] for rank in ranks.values()) for site in list(ranks[0])[warmup:]}
    if len(values) >= 3:
      result[case] = values
  return result


def build_profile(copy, source=None, destination=None, *, warmup=1):
  """Use complete rank sets and equal-size batches of real native intervals."""
  timings = {'copy':_read(copy,'copy',warmup)}
  for route,directory in (('source',source),('destination',destination)):
    if directory is not None:
      timings[route] = _read(directory,route,warmup)
  cases = {}
  for case, copy_values in timings['copy'].items():
    matched = {route:values[case] for route,values in timings.items() if case in values}
    if len(matched) == 1:
      continue
    common = set.intersection(*(set(values) for values in matched.values()))
    sites = [site for site in copy_values if site in common]
    chunk = len(sites)//3
    if not chunk:
      continue
    # Use the same producer-consumer occurrences on each measured route.
    cases[case] = {route:[statistics.mean(values[site] for site in sites[i*chunk:(i+1)*chunk])
                         for i in range(3)] for route,values in matched.items()}
  return {'schema':'vecops-shared-page/1','cases':cases}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--copy',type=Path,required=True)
  parser.add_argument('--source',type=Path)
  parser.add_argument('--destination',type=Path)
  parser.add_argument('--warmup',type=int,default=1)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  if args.warmup < 0 or (args.source is None and args.destination is None):
    parser.error('provide a shared route and a nonnegative warmup count')
  profile = build_profile(args.copy,args.source,args.destination,warmup=args.warmup)
  if not profile['cases']:
    parser.error('no matched case with at least three complete rank samples per path')
  args.output.write_text(json.dumps(profile,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__':
  main()
