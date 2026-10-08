# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU loop/pipeline policy for matmul, independent of the C++ printer.

Prefetch is opt-in unless a calibrated target profile enables it. Vector/SME
lookahead uses two individually named operand banks; AMX keeps its eight-tile
budget and maps lookahead to bounded panel hints, not extra tile registers.
"""

from dataclasses import replace
from types import MappingProxyType

from ...ir.iteration import LoopSchedule, ReductionSpec
from ...ir.schedule import MatmulSchedule, ScheduleAction, matrix_loop
from ...passes.manager import register_pass


@register_pass("cpu-schedule-matmul", stage="cpu-memory", after=("cpu-plan-matmul",))
def schedule_matmul(prepared, context):
  costs = context.environment.costs
  mode = context.options.get("cpu_matmul_pipeline", "auto")
  if mode not in {"auto", "off", "prefetch", "staged"}:
    raise ValueError("cpu_matmul_pipeline must be auto, off, prefetch or staged")
  policy = costs.policy("matmul_pipeline", 0)
  if policy not in (0, 1, 2):
    raise ValueError("target matmul_pipeline policy must be 0, 1 or 2")
  selected = ("off", "prefetch", "staged")[int(policy)] if mode == "auto" else mode
  if selected not in {"off", "prefetch", "staged"}:
    raise ValueError("target matmul_pipeline policy must be off, prefetch or staged")
  distance = context.options.get("cpu_matmul_prefetch_distance", costs.policy("matmul_prefetch_distance", 2))
  lines = context.options.get("cpu_matmul_prefetch_lines", costs.policy("matmul_prefetch_lines", 2))
  if type(distance) is not int or not 1 <= distance <= 16:
    raise ValueError("cpu_matmul_prefetch_distance must be in [1,16]")
  if type(lines) is not int or not 1 <= lines <= 16:
    raise ValueError("cpu_matmul_prefetch_lines must be in [1,16]")
  plans = {}
  for key, plan in prepared.matmuls.items():
    dtype = plan.expression.type.dtype
    reduction = ReductionSpec("dot", dtype, ordering="sequential")

    def action(name, effects="none"):
      return ScheduleAction(name, effects)

    compute = (action("operand-load", "read"), action("multiply-accumulate", "state"))
    hint = (action("future-panel-prefetch", "hint"),) if selected != "off" else ()
    k_schedule = LoopSchedule(1, selected, distance)
    native = []
    for family, m_step, n_step, k_step in (
      ("vector", "MR", "VL", "one"),
      ("sme", "SMR", "SNR", "SKR"),
      ("amx", "AMR", "ANR", "AKR"),
      ("gemv", "one", "columns", "k_group"),
    ):
      fold = matrix_loop(
        f"{family}.k",
        "K",
        k_step,
        (*hint, *compute),
        schedule=k_schedule,
        recurrence=reduction,
        bound="full-tiles" if family == "amx" else "extent",
      )
      if selected == "staged" and family in {"vector", "sme"}:
        fold = replace(
          fold,
          bound="lookahead",
          prologue=(action("operand-bank-prime", "read"),),
          body=(
            action("next-operand-load", "read"),
            action("multiply-accumulate", "state"),
            action("rotate-operand-banks", "state"),
          ),
          finalization=(action("operand-pipeline-drain", "state"),),
        )
      # Initialization/finalization are explicit siblings of the K recurrence.
      n_loop = matrix_loop(
        f"{family}.n",
        "N",
        n_step,
        (action("accumulator-init", "read/state"), fold, action("epilogue-store", "write")),
        start="n_start" if family in {"amx", "gemv"} else None,
      )
      native.append(matrix_loop(f"{family}.m", "M", m_step, (n_loop,), start="m_start" if family == "amx" else None))
    native.append(
      matrix_loop(
        "amx.outer",
        "outer_extent",
        "outer_step",
        (matrix_loop("amx.inner", "inner_extent", "inner_step", (action("amx-spatial-leaf", "read/write"),)),),
      )
    )
    mc = matrix_loop("cache.m", "M", "BM", (action("pack-A", "read/write"), action("native-leaf", "read/write")))
    kc = matrix_loop(
      "cache.k",
      "K",
      "BK",
      (*hint, action("pack-B", "read/write"), mc),
      schedule=k_schedule,
      recurrence=reduction,
      bound="nonempty",
    )
    nc = matrix_loop("cache.n", "N", "BN", (kc,))
    schedule = MatmulSchedule((nc, *native), selected, lines)
    plans[key] = replace(plan, schedule=schedule, pass_trace=(*plan.pass_trace, "cpu-schedule-matmul"))
  return replace(prepared, matmuls=MappingProxyType(plans))
