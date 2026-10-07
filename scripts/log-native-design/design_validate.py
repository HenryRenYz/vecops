#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Offline design + validation for the native (promotion-free) SVE log kernels.

Develops the constants for include/vecops/vec/details/sve/math/Log.h and
bit-simulates the exact SVE op sequences in long double:

f32 Strict (target: ULP bit distance <= 1 vs the correctly rounded result)
  reduction : u = bits(x) - 0x3f2aaaab; k = s32(u) >> 23;
              z = bits((u & 0x7fffff) + off); r = z - 1 (exact).
  polynomial: Q(r) ~ (log_base(1+r) - r) / r^2, degree 13, Estrin.
  assembly  : kflo = kf * lo            (rounds in a tiny binade)
              r2   = r * r
              pf   = fma(r2, Q, kflo)   (one rounding, |pf| <= ~0.075)
              t    = r + pf             (one rounding, |t| <= ~0.41)
              y    = fma(kf, hi, t)     (final rounding; kf*hi exact inside
                                         the fma since hi keeps >= 9 trailing
                                         zero mantissa bits)
  Subnormal lanes: prescale x by 2^23 (exact) and fold -23 into off, so the
  special path reruns the identical op sequence.

f16 Fast / Estimate (target: relative error <= 2^-10 / 2^-7)
  Same skeleton in pure f16 arithmetic; Q degree 6 (Fast) / 2 (Estimate),
  Estrin-lite / 2-step Horner, window candidates 0x39aa and 0x39a8.

Reference: long-double log, which is effectively exact for these budgets.
"""

import struct
import sys

import numpy as np

LD = np.longdouble
F32 = np.float32
F16 = np.float16

LN2 = LD("0.6931471805599453094172321214581766")
LN10 = LD("2.3025850929940456840179914546843642")
LOG10_2 = LD("0.3010299956639811952137388947244930")

LN_OF_BASE = {"e": LD(1.0), "2": LN2, "10": LN10}

BASES = {
    "e": {"kconst": LN2, "inv": LD(1.0)},
    "2": {"kconst": LD(1.0), "inv": LD(1.0) / LN2},
    "10": {"kconst": LOG10_2, "inv": LD(1.0) / LN10},
}


def r32(x):
    return np.asarray(x).astype(F32).astype(LD)


def r16(x):
    return np.asarray(x).astype(F16).astype(LD)


def f32_bits(x):
    return np.ascontiguousarray(np.asarray(x, dtype=LD).astype(F32)).view(np.uint32)


def f32_from_bits(u):
    return u.copy().view(F32).astype(LD)


def f16_bits(x):
    return np.ascontiguousarray(np.asarray(x, dtype=LD).astype(F16)).view(np.uint16)


def f16_from_bits(u):
    return u.copy().view(F16).astype(LD)


def split_hi_lo(v, keep_bits):
    """Quantize v (an exact constant) to keep_bits significant mantissa bits.

    Returns (hi, lo) with hi carrying 23+1-keep_bits trailing zero fraction
    bits so that k*hi stays exact for |k| < 2^(23+1-keep_bits).
    """
    bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    masked = bits & (~((1 << (24 - keep_bits)) - 1))
    hi = LD(struct.unpack("<f", struct.pack("<I", masked))[0])
    lo = v - hi
    return hi, lo


def frac_trailing_zeros(v, fmt):
    if fmt == F32:
        u = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        frac = u & 0x7FFFFF
    else:
        u = struct.unpack("<H", struct.pack("<e", float(v)))[0]
        frac = u & 0x3FF
    if frac == 0:
        return 24 if fmt == F32 else 11
    return (frac & -frac).bit_length() - 1


# ---------------------------------------------------------------------------
# shared f32 machinery
# ---------------------------------------------------------------------------

OFF32 = np.uint32(0x3F3504F3)  # fl32(1/sqrt2), window [0.70710677, 1.4142135)


def reduce_f32(x, off, kbias=0):
    u = f32_bits(x)
    t = u - off
    k = (t.view(np.int32) >> 23) - kbias
    kf = k.astype(LD)
    z = f32_from_bits((t & np.uint32(0x7FFFFF)) + off)
    r = r32(z - LD(1.0))
    return kf, r


def fma32(a, b, c):
    return r32(a * b + c)


def estrin_q(r, d, rnd, fma):
    """Deg-13 Estrin over 14 coefficients, matching the C++ op order:
    A..G = d{2i} + r*d{2i+1}; AB/CD/EF at r2; ABCD/EFG at r4; Q at r8."""
    assert len(d) == 14
    A = fma(r, d[1], np.full_like(r, d[0]))
    B = fma(r, d[3], np.full_like(r, d[2]))
    C = fma(r, d[5], np.full_like(r, d[4]))
    D = fma(r, d[7], np.full_like(r, d[6]))
    E = fma(r, d[9], np.full_like(r, d[8]))
    F = fma(r, d[11], np.full_like(r, d[10]))
    G = fma(r, d[13], np.full_like(r, d[12]))
    r2 = rnd(r * r)
    AB = fma(B, r2, A)
    CD = fma(D, r2, C)
    EF = fma(F, r2, E)
    r4 = rnd(r2 * r2)
    ABCD = fma(CD, r4, AB)
    EFG = fma(G, r4, EF)
    r8 = rnd(r4 * r4)
    Q = fma(EFG, r8, ABCD)
    return Q, r2


def sim_f32_strict(x, fam, off=OFF32, kbias=0):
    kf, r = reduce_f32(x, off, kbias)
    Q, r2 = estrin_q(r, fam["d"], r32, fma32)
    kflo = r32(kf * fam["lo"])
    pf = fma32(r2, Q, kflo)
    t = fma32(r, fam["alpha"], pf)
    y = fma32(kf, fam["hi"], t)
    return y


def ulp_dist_bits(eb, ab):
    return np.where(eb > ab, eb - ab, ab - eb).astype(np.int64)


def measure_f32(x, sim, lnb=LD(1.0), exact_override=None, **kw):
    y = sim(x, **kw)
    exact = exact_override(x) if exact_override is not None else np.log(x) / lnb
    eb = f32_bits(exact)
    ab = f32_bits(y)
    same_sign = np.signbit(eb) == np.signbit(ab)
    dist = np.where(same_sign, ulp_dist_bits(eb, ab), 1 << 20)
    return dist, y, exact


# ---------------------------------------------------------------------------
# shared f16 machinery
# ---------------------------------------------------------------------------

def fma16(a, b, c):
    return r16(a * b + c)


def sim_f16_fast(x, fam, off, kbias=0):
    kf, r = reduce_f16(x, off, kbias)
    d = fam["d"]
    assert len(d) == 7
    A = fma16(r, d[1], np.full_like(r, d[0]))
    B = fma16(r, d[3], np.full_like(r, d[2]))
    C = fma16(r, d[5], np.full_like(r, d[4]))
    r2 = r16(r * r)
    AB = fma16(B, r2, A)
    CD = fma16(d[6], r2, C)
    r4 = r16(r2 * r2)
    Q = fma16(CD, r4, AB)
    kflo = r16(kf * fam["lo"])
    pf = fma16(r2, Q, kflo)
    t = fma16(r, fam["alpha"], pf)
    y = fma16(kf, fam["hi"], t)
    return y


def sim_f16_est(x, fam, off, kbias=0):
    kf, r = reduce_f16(x, off, kbias)
    d = fam["d"]
    assert len(d) == 3
    inner = fma16(r, d[2], np.full_like(r, d[1]))
    Q = fma16(inner, r, np.full_like(r, d[0]))
    r2 = r16(r * r)
    kflo = r16(kf * fam["lo"])
    pf = fma16(r2, Q, kflo)
    t = fma16(r, fam["alpha"], pf)
    y = fma16(kf, fam["hi"], t)
    return y


def reduce_f16(x, off, kbias=0):
    u = f16_bits(x)
    t = u - off
    k = (t.view(np.int16) >> 10) - kbias
    kf = k.astype(LD)
    z = f16_from_bits((t & np.uint16(0x03FF)) + off)
    r = r16(z - LD(1.0))
    return kf, r


# ---------------------------------------------------------------------------
# coefficient generation
# ---------------------------------------------------------------------------

def poly_affine(e, a, b):
    """Coefficients of q(r) = p(a + b*r) from the coefficients e of p."""
    q = np.zeros(len(e), dtype=LD)
    for c in e[::-1]:
        # q = q * (a + b r) + c
        nxt = np.zeros(len(e), dtype=LD)
        nxt[1:] += q[:-1] * b
        nxt += q * a
        nxt[0] += LD(c)
        q = nxt
    return q


def fit_q(rmin, rmax, deg, inv):
    """Minimax-style fit of (log_b(1+r) - alpha*r) / r^2 over [rmin, rmax],
    with alpha = log_b(e) = inv and base b selected by inv (1, 1/ln2, 1/ln10).

    Chebyshev fit in the normalized variable s in [-1,1] (well conditioned),
    converted to the power basis in s, then affinely recomposed into r in
    long double. The assembly adds the alpha*r lead back separately:
    y = k*log_b(2) + alpha*r + r^2*Q(r) = k*c + log_b(z).
    """
    n = 240001
    rmid = (rmin + rmax) / 2
    rhalf = (rmax - rmin) / 2
    j = np.arange(n, dtype=LD)
    theta = np.pi * (j + LD(0.5)) / n
    rr = rmid + rhalf * np.cos(theta)
    ss = (rr - rmid) / rhalf
    f = inv * (np.log1p(rr) - rr) / (rr * rr)
    c = np.polynomial.chebyshev.chebfit(
        ss.astype(np.float64), f.astype(np.float64), deg)
    e = np.polynomial.chebyshev.cheb2poly(c).astype(LD)
    a = -rmid / rhalf
    b = LD(1.0) / rhalf
    return poly_affine(e, a, b)


def bias_correct(d_rounded, rmin, rmax, inv, rnd):
    """One pass: absorb the mean rounding bias of the coefficient set into d0."""
    rr = np.linspace(rmin, rmax, 400001, dtype=LD)
    target = inv * (np.log1p(rr) - rr) / (rr * rr)
    got = np.polyval(d_rounded[::-1].astype(LD), rr)
    bias = float(np.mean(got - target))
    d0 = rnd(LD(d_rounded[0]) - LD(bias))
    out = d_rounded.copy()
    out[0] = d0
    return out


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def gen_f32_family(base, bias=False):
    b = BASES[base]
    if base == "2":
        hi, lo = LD(1.0), LD(0.0)
    else:
        hi, lo = split_hi_lo(b["kconst"], keep_bits=15)
    alpha = r32(b["inv"])
    d = fit_q(LD(-0.2928933), LD(0.4142136), 13, b["inv"])
    d32 = r32(d)
    if bias:
        d32 = bias_correct(d32, LD(-0.2928933), LD(0.4142136), b["inv"], lambda v: r32(v))
    return {"hi": r32(hi), "lo": r32(lo), "alpha": alpha, "d": d32, "base": base}


def gen_f16_family(base, rmin, rmax, deg):
    b = BASES[base]
    if base == "2":
        hi, lo = LD(1.0), LD(0.0)
    elif base == "e":
        hi = LD(0.6875)
        lo = LN2 - hi
    else:
        hi = LD(0.3046875)
        lo = LOG10_2 - hi
    d = fit_q(rmin, rmax, deg, b["inv"])
    return {"hi": r16(hi), "lo": r16(lo), "alpha": r16(b["inv"]),
            "d": r16(d), "base": base}


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def f32_normal_inputs(chunk=1 << 19):
    rng = np.random.default_rng(0xC0FFEE)
    edges = np.array(
        [0, 1, 2, 0x2AAAA9, 0x2AAAAA, 0x2AAAAB, 0x2AAAAC, 0x2AAAAD,
         0x555554, 0x555555, 0x555556, 0x7FFFFE, 0x7FFFFF], dtype=np.uint32)
    for e in range(1, 255):
        head = (np.uint32(e) << 23) | edges
        if e in (125, 126, 127):
            mants = np.arange(0, 1 << 23, dtype=np.uint32)
        else:
            mants = rng.integers(0, 1 << 23, size=1 << 17, dtype=np.uint32)
        body = (np.uint32(e) << 23) | mants
        for piece in (head, body):
            for i in range(0, len(piece), chunk):
                yield piece[i:i + chunk]


def f32_subnormal_inputs():
    rng = np.random.default_rng(0xBAD5EED)
    edges = np.array([1, 2, 3, 0x7FFFFE, 0x7FFFFF], dtype=np.uint32)
    mants = rng.integers(1, 1 << 23, size=1 << 21, dtype=np.uint32)
    u = np.concatenate([edges, mants])
    x = f32_from_bits(u)
    return r32(x * LD(1.0) * LD(2.0) ** 23)  # exact prescale


def run_f32_strict(fam, label, bias_applied=False):
    worst = -1
    worst_u = 0
    for u in f32_normal_inputs():
        x = f32_from_bits(u)
        dist, _, _ = measure_f32(x, sim_f32_strict, lnb=LN_OF_BASE[fam["base"]], fam=fam)
        i = int(np.argmax(dist))
        if dist[i] > worst:
            worst = int(dist[i])
            worst_u = int(u[i])
    # subnormal lanes through the folded-off special path
    xs = f32_subnormal_inputs()
    # xs are already prescaled by 2^23; reference is log of the original.
    exact_corr = lambda v: (np.log(v) - LD(23) * LN2) / LN_OF_BASE[fam["base"]]
    dist, _, _ = measure_f32(
        xs, sim_f32_strict, lnb=LN_OF_BASE[fam["base"]], fam=fam, off=OFF32,
        kbias=23, exact_override=exact_corr)
    i = int(np.argmax(dist))
    if dist[i] > worst:
        worst = int(dist[i])
        worst_u = int(f32_bits(xs)[i])
    tag = " (+bias)" if bias_applied else ""
    print(f"  f32 strict base {fam['base']}{tag}: max ULP = {worst} at bits 0x{worst_u:08x}")
    return worst


def run_f16(fam, off, tier, label):
    u = np.arange(0, 1 << 16, dtype=np.uint16)
    x = f16_from_bits(u)
    kernel_lanes = (u >= np.uint16(0x0400)) & (u < np.uint16(0x7C00))
    sim = sim_f16_fast if tier == "fast" else sim_f16_est
    y = sim(x, fam, off)
    exact = np.log(x) / LN_OF_BASE[fam["base"]]
    rel = np.where(kernel_lanes & (x != 1), np.abs(y / exact - LD(1.0)), 0)
    sign_bad = int(np.sum(kernel_lanes & (np.signbit(y) != np.signbit(exact))))
    at_one = y[np.searchsorted(u, 0x3C00)]  # x = 1.0
    one_ok = at_one == 0 and not np.signbit(at_one)
    i = int(np.argmax(rel))
    bound = 9.765625e-4 if tier == "fast" else 7.8125e-3
    print(f"  f16 {tier} base {fam['base']} [{label}]: max rel = {float(rel[i]):.4e} "
          f"(bound {bound:.1e}) at 0x{u[i]:04x}; sign bad {sign_bad}; "
          f"log(1)=+0: {bool(one_ok)}")
    return float(rel[i]), sign_bad, bool(one_ok)


def run_f16_subnormal(fam, off, tier):
    u = np.arange(1, 0x0400, dtype=np.uint16)
    x = f16_from_bits(u)
    scaled = r16(x * LD(1024.0))
    sim = sim_f16_fast if tier == "fast" else sim_f16_est
    y = sim(scaled, fam, off, kbias=10)
    exact = np.log(x) / LN_OF_BASE[fam["base"]]
    rel = np.abs(y / exact - LD(1.0))
    i = int(np.argmax(rel))
    print(f"    subnormals: max rel = {float(rel[i]):.4e} at 0x{u[i]:04x}")
    return float(rel[i])


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what in ("all", "f32"):
        print("== f32 Strict native ==")
        for base in ("e", "2", "10"):
            fam = gen_f32_family(base)
            tz = frac_trailing_zeros(fam["hi"], F32)
            print(f"  base {base}: hi={float(fam['hi']):.9g} "
                  f"(trailing frac zeros {tz}), lo={float(fam['lo']):.9g}")
            assert tz >= 9, "k*hi would not stay exact"
            w = run_f32_strict(fam, "main")
            if w > 1:
                fam = gen_f32_family(base, bias=True)
                run_f32_strict(fam, "bias", bias_applied=True)

    if what in ("all", "f16"):
        print("== f16 native ==")
        windows = {
            "sqrt2": (np.uint16(0x39A8), LD(-0.2929688), LD(0.4140625)),
        }
        for wname, (off, rmin, rmax) in windows.items():
            print(f" window {wname} off=0x{off:04x}")
            for base in ("e", "2", "10"):
                fam_fast = gen_f16_family(base, rmin, rmax, 6)
                run_f16(fam_fast, off, "fast", wname)
                run_f16_subnormal(fam_fast, off, "fast")
                fam_est = gen_f16_family(base, rmin, rmax, 2)
                run_f16(fam_est, off, "est", wname)
                run_f16_subnormal(fam_est, off, "est")


if __name__ == "__main__":
    main()
