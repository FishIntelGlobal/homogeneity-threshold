#!/usr/bin/env python3
"""
homogeneity_simulation.py
=========================
Agent-based simulation for Paper II: The Homogeneity Threshold.

Implements all four simulation suites described in Sections 5.2-5.5:
  1. Core simulation (baseline parameters)
  2. Regime-switching fundamentals
  3. Heterogeneous agent parameters
  4. Robustness suite (31 configurations)

Plus the contrarian analysis sweep (Section 5.5).

Author:  Jason Gething
Affiliation: FishIntel Global Ltd.
Paper:   Gething (2026b), "The Homogeneity Threshold"
License: MIT
Repository: https://github.com/FishIntelGlobal/homogeneity-threshold

Usage:
    python homogeneity_simulation.py              # Run all suites
    python homogeneity_simulation.py --suite core  # Core only
    python homogeneity_simulation.py --quick       # Fast mode

Requirements:
    numpy >= 1.21
    matplotlib >= 3.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("homogeneity_sim")


# ===================================================================
# CONFIGURATION
# ===================================================================

@dataclass
class SimConfig:
    """All simulation parameters. Defaults match Paper II Section 5.1."""

    n_agents: int = 1_000
    n_periods: int = 10_000

    tau_p: float = 1.0
    tau_0: float = 2.0
    gamma: float = 2.0
    r: float = 0.3

    shock_rate: float = 0.02
    shock_scale: float = 2.0

    h_star: float = 0.5
    cascade_sigma: float = 3.0
    rolling_window: int = 200

    tau_P0: float = 10.0
    delta: float = 2.0
    tau_R: float = 5.0
    kappa: float = 0.5

    phi: float = 0.0

    regime_switching: bool = False
    sigma_normal: float = 1.0
    sigma_high: float = 3.0
    regime_transition_prob: float = 0.01

    heterogeneous_gamma: bool = False
    gamma_log_mean: float = 0.6931
    gamma_log_std: float = 0.3

    seed: int = 42
    output_dir: str = "figures"
    verbose: bool = True


# ===================================================================
# MATHEMATICAL CORE
# ===================================================================

def h_of_q(q: float, cfg: SimConfig) -> float:
    """Behavioural homogeneity h(q) = tau_0*q^gamma / (tau_0*q^gamma + (1-r)*tau_p).

    Uses (1-r), NOT (1-r)^2. The squared form appears only in rho(q).
    """
    if q <= 0.0:
        return 0.0
    tau_ai = cfg.tau_0 * q ** cfg.gamma
    return tau_ai / (tau_ai + (1.0 - cfg.r) * cfg.tau_p)


def h_of_q_array(q_arr: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Vectorised h(q)."""
    tau_ai = cfg.tau_0 * np.maximum(q_arr, 0.0) ** cfg.gamma
    return tau_ai / (tau_ai + (1.0 - cfg.r) * cfg.tau_p)


def rho_of_q(q: float, cfg: SimConfig) -> float:
    """Pairwise correlation rho(q) = tau_AI / (tau_AI + (1-r)^2 * tau_p)."""
    if q <= 0.0:
        return 0.0
    tau_ai = cfg.tau_0 * q ** cfg.gamma
    return tau_ai / (tau_ai + (1.0 - cfg.r) ** 2 * cfg.tau_p)


def phi_star(q: float, cfg: SimConfig) -> float:
    """Critical diversity threshold phi*(q) = 1 - sqrt(h*/h(q))."""
    hq = h_of_q(q, cfg)
    if hq <= cfg.h_star:
        return 0.0
    return 1.0 - np.sqrt(cfg.h_star / hq)


def h_tilde(q: float, phi: float, cfg: SimConfig) -> float:
    """Effective homogeneity with contrarians: h_tilde(q,phi) = (1-phi)^2 * h(q)."""
    return (1.0 - phi) ** 2 * h_of_q(q, cfg)


def propagation_time(q: float, cfg: SimConfig) -> float:
    """AI-mediated propagation time tau_P(q) = tau_P0 * (1-q)^delta."""
    return cfg.tau_P0 * max(1.0 - q, 1e-12) ** cfg.delta


def safeguard_works(q: float, cfg: SimConfig) -> bool:
    """Safeguards work when tau_P(q) >= kappa * tau_R."""
    return propagation_time(q, cfg) >= cfg.kappa * cfg.tau_R


# ===================================================================
# SIMULATION ENGINE
# ===================================================================

@dataclass
class SimResult:
    """Container for simulation output."""

    config: SimConfig
    label: str = ""

    q_series: np.ndarray = field(default_factory=lambda: np.array([]))
    h_series: np.ndarray = field(default_factory=lambda: np.array([]))
    price_series: np.ndarray = field(default_factory=lambda: np.array([]))
    volatility_series: np.ndarray = field(default_factory=lambda: np.array([]))

    cascade_times: List[int] = field(default_factory=list)
    cascade_magnitudes: List[float] = field(default_factory=list)
    cascade_q_values: List[float] = field(default_factory=list)
    safeguard_engaged: List[bool] = field(default_factory=list)
    safeguard_success: List[bool] = field(default_factory=list)

    prop1_monotonicity: bool = False
    prop2_speed: bool = False
    prop3_opacity: bool = False

    @property
    def n_cascades(self) -> int:
        return len(self.cascade_times)

    @property
    def avg_magnitude(self) -> float:
        if not self.cascade_magnitudes:
            return 0.0
        return float(np.mean(self.cascade_magnitudes))

    def quintile_stats(self) -> List[Dict]:
        """Per-quintile cascade counts and safeguard success rates."""
        stats = []
        for i in range(5):
            q_lo = i * 0.2
            q_hi = (i + 1) * 0.2
            h_lo = h_of_q(q_lo, self.config)
            h_hi = h_of_q(q_hi, self.config)

            mask = [(q_lo <= q < q_hi) for q in self.cascade_q_values]
            n_cas = sum(mask)
            n_sg = sum(s for s, m in zip(self.safeguard_success, mask) if m)
            sg_rate = (n_sg / n_cas * 100) if n_cas > 0 else 0.0

            stats.append({
                "quintile": i + 1,
                "q_range": (q_lo, q_hi),
                "h_range": (h_lo, h_hi),
                "cascades": n_cas,
                "safeguard_rate": sg_rate,
            })
        return stats


def run_simulation(cfg: SimConfig, label: str = "") -> SimResult:
    """Run a single agent-based simulation.

    Each period t:
        1. Draw fundamental theta_t (with optional regime switching)
        2. Compute AI quality q(t) = t / T
        3. Each agent forms posterior via Bayesian update
        4. Compute market price and returns
        5. Detect cascades (returns > cascade_sigma * rolling sigma)
        6. Check safeguard engagement
    """
    rng = np.random.default_rng(cfg.seed)
    result = SimResult(config=cfg, label=label)

    T = cfg.n_periods
    N = cfg.n_agents

    # Per-agent gamma (heterogeneous or uniform)
    if cfg.heterogeneous_gamma:
        agent_gammas = rng.lognormal(
            mean=cfg.gamma_log_mean, sigma=cfg.gamma_log_std, size=N
        )
    else:
        agent_gammas = np.full(N, cfg.gamma)

    # Contrarian agents
    n_contrarian = int(cfg.phi * N)
    is_contrarian = np.zeros(N, dtype=bool)
    if n_contrarian > 0:
        contrarian_idx = rng.choice(N, size=n_contrarian, replace=False)
        is_contrarian[contrarian_idx] = True

    regime = 0
    prices = np.zeros(T)
    returns = np.zeros(T)
    q_values = np.zeros(T)
    h_values = np.zeros(T)
    rolling_vol = np.zeros(T)
    price = 100.0

    for t in range(T):
        q = t / T
        q_values[t] = q

        # Regime switching
        if cfg.regime_switching:
            if rng.random() < cfg.regime_transition_prob:
                regime = 1 - regime
            sigma_t = cfg.sigma_high if regime == 1 else cfg.sigma_normal
        else:
            sigma_t = 1.0

        theta = rng.normal(0.0, sigma_t)

        # Agent signal weights
        tau_ai_per_agent = cfg.tau_0 * np.maximum(q, 1e-15) ** agent_gammas
        weights = tau_ai_per_agent / (
            tau_ai_per_agent + (1.0 - cfg.r) * cfg.tau_p
        )
        weights[is_contrarian] = 0.0

        # Private signals
        private_signals = theta + rng.normal(
            0.0, 1.0 / np.sqrt(cfg.tau_p), size=N
        )

        # AI signal (shared)
        if q > 0:
            ai_precision = cfg.tau_0 * q ** cfg.gamma
            ai_signal = theta + rng.normal(
                0.0, 1.0 / np.sqrt(max(ai_precision, 1e-12))
            )
        else:
            ai_signal = 0.0

        # Posterior estimates
        estimates = weights * ai_signal + (1.0 - weights) * private_signals
        market_price_change = np.mean(estimates)

        # Exogenous shock
        shock = 0.0
        if rng.random() < cfg.shock_rate:
            shock = rng.exponential(cfg.shock_scale) * rng.choice([-1, 1])

        price_change = market_price_change + shock * h_of_q(q, cfg)
        price += price_change
        prices[t] = price

        if t > 0:
            returns[t] = price_change

        # Effective homogeneity
        if cfg.heterogeneous_gamma and not cfg.phi > 0:
            conformist_mask = ~is_contrarian
            h_values[t] = float(np.mean(weights[conformist_mask])) if np.any(conformist_mask) else 0.0
        else:
            h_eff = h_of_q(q, cfg)
            if cfg.phi > 0:
                h_eff = (1.0 - cfg.phi) ** 2 * h_eff
            h_values[t] = h_eff

        # Rolling volatility
        if t >= cfg.rolling_window:
            window = returns[t - cfg.rolling_window:t]
            rolling_vol[t] = np.std(window) if np.std(window) > 0 else 1e-6
        else:
            rolling_vol[t] = np.std(returns[:max(t, 1)]) if t > 0 else 1.0

    result.q_series = q_values
    result.h_series = h_values
    result.price_series = prices
    result.volatility_series = rolling_vol

    # Cascade detection
    for t in range(cfg.rolling_window, T):
        if rolling_vol[t] > 0 and abs(returns[t]) > cfg.cascade_sigma * rolling_vol[t]:
            q_t = q_values[t]
            mag = abs(returns[t]) / rolling_vol[t]

            result.cascade_times.append(t)
            result.cascade_magnitudes.append(mag)
            result.cascade_q_values.append(q_t)

            sg_can_respond = safeguard_works(q_t, cfg)
            h_below = h_of_q(q_t, cfg) < cfg.h_star
            sg_success = sg_can_respond and h_below

            result.safeguard_engaged.append(sg_can_respond)
            result.safeguard_success.append(sg_success)

    # Proposition checks
    q_test = np.linspace(0.001, 1.0, 1000)
    h_test = h_of_q_array(q_test, cfg)
    result.prop1_monotonicity = bool(np.all(np.diff(h_test) > 0))

    q_S = 1.0 - (cfg.kappa * cfg.tau_R / cfg.tau_P0) ** (1.0 / cfg.delta)
    result.prop2_speed = 0 < q_S < 1.0

    prop_times = [propagation_time(qi, cfg) for qi in q_test]
    result.prop3_opacity = all(
        prop_times[i] >= prop_times[i + 1] for i in range(len(prop_times) - 1)
    )

    return result


# ===================================================================
# SUITE 1: CORE SIMULATION
# ===================================================================

def run_core_suite(cfg: Optional[SimConfig] = None) -> SimResult:
    """Run baseline core simulation (Section 5.2)."""
    if cfg is None:
        cfg = SimConfig()

    log.info("=" * 65)
    log.info("SUITE 1: Core Simulation")
    log.info("=" * 65)
    log.info(f"  N={cfg.n_agents}, T={cfg.n_periods}, seed={cfg.seed}")

    t0 = time.time()
    result = run_simulation(cfg, label="Core (Baseline)")
    elapsed = time.time() - t0

    log.info(f"  Completed in {elapsed:.1f}s")
    log.info(f"  Total cascades: {result.n_cascades}")
    log.info(f"  Avg magnitude: {result.avg_magnitude:.2f}s")

    stats = result.quintile_stats()
    for s in stats:
        log.info(
            f"    Q{s['quintile']} (q: {s['q_range'][0]:.2f}-{s['q_range'][1]:.2f}, "
            f"h: {s['h_range'][0]:.3f}-{s['h_range'][1]:.3f}): "
            f"{s['cascades']} cascades, safeguard {s['safeguard_rate']:.1f}%"
        )

    log.info(f"  P1 monotonicity: {'PASS' if result.prop1_monotonicity else 'FAIL'}")
    log.info(f"  P2 speed:        {'PASS' if result.prop2_speed else 'FAIL'}")
    log.info(f"  P3 opacity:      {'PASS' if result.prop3_opacity else 'FAIL'}")

    return result


# ===================================================================
# SUITE 2: REGIME-SWITCHING
# ===================================================================

def run_regime_switching_suite(cfg: Optional[SimConfig] = None) -> SimResult:
    """Run regime-switching simulation (Section 5.3)."""
    if cfg is None:
        cfg = SimConfig()

    cfg.regime_switching = True
    cfg.sigma_normal = 1.0
    cfg.sigma_high = 3.0
    cfg.regime_transition_prob = 0.01

    log.info("")
    log.info("=" * 65)
    log.info("SUITE 2: Regime-Switching Fundamentals")
    log.info("=" * 65)

    t0 = time.time()
    result = run_simulation(cfg, label="Regime-Switching")
    elapsed = time.time() - t0

    log.info(f"  Completed in {elapsed:.1f}s")
    log.info(f"  Total cascades: {result.n_cascades}")
    log.info(f"  Avg magnitude: {result.avg_magnitude:.2f}s")
    log.info(f"  P1 monotonicity: {'PASS' if result.prop1_monotonicity else 'FAIL'}")

    return result


# ===================================================================
# SUITE 3: HETEROGENEOUS GAMMA
# ===================================================================

def run_heterogeneous_suite(cfg: Optional[SimConfig] = None) -> SimResult:
    """Run heterogeneous agent simulation (Section 5.3)."""
    if cfg is None:
        cfg = SimConfig()

    cfg.heterogeneous_gamma = True
    cfg.gamma_log_mean = np.log(2.0)
    cfg.gamma_log_std = 0.3

    log.info("")
    log.info("=" * 65)
    log.info("SUITE 3: Heterogeneous Agent Parameters")
    log.info("=" * 65)

    t0 = time.time()
    result = run_simulation(cfg, label="Heterogeneous gamma")
    elapsed = time.time() - t0

    log.info(f"  Completed in {elapsed:.1f}s")
    log.info(f"  Total cascades: {result.n_cascades}")
    log.info(f"  P1 monotonicity: {'PASS' if result.prop1_monotonicity else 'FAIL'}")

    return result


# ===================================================================
# SUITE 4: ROBUSTNESS (31 CONFIGURATIONS)
# ===================================================================

def get_robustness_configs() -> List[Tuple[str, SimConfig]]:
    """Generate 31 parameter configurations for robustness testing.

    Varies: N (100-5000), r (0.0-0.7), gamma (0.5-4.0),
            phi (0.0-0.30), shock_scale (1.0-8.0),
            plus combined scenarios.
    """
    configs = []
    base_seed = 42

    for i, n in enumerate([100, 250, 500, 2_000, 5_000]):
        configs.append((f"N={n}", SimConfig(n_agents=n, seed=base_seed + i)))

    for i, r_val in enumerate([0.0, 0.1, 0.2, 0.5, 0.7]):
        configs.append((f"r={r_val}", SimConfig(r=r_val, seed=base_seed + 10 + i)))

    for i, g_val in enumerate([0.5, 1.0, 1.5, 3.0, 4.0]):
        configs.append((f"gamma={g_val}", SimConfig(gamma=g_val, seed=base_seed + 20 + i)))

    for i, phi_val in enumerate([0.05, 0.10, 0.15, 0.20, 0.30]):
        configs.append((f"phi={phi_val}", SimConfig(phi=phi_val, seed=base_seed + 30 + i)))

    for i, ss in enumerate([1.0, 3.0, 5.0, 8.0]):
        configs.append((f"shock={ss}", SimConfig(shock_scale=ss, seed=base_seed + 40 + i)))

    configs.append(("regime-switch", SimConfig(regime_switching=True, seed=base_seed + 50)))
    configs.append(("het-gamma", SimConfig(heterogeneous_gamma=True, seed=base_seed + 51)))
    configs.append(("regime+phi=0.20", SimConfig(regime_switching=True, phi=0.20, seed=base_seed + 52)))
    configs.append(("het-gamma+phi=0.15", SimConfig(heterogeneous_gamma=True, phi=0.15, seed=base_seed + 53)))
    configs.append(("regime+het+phi=0.10", SimConfig(
        regime_switching=True, heterogeneous_gamma=True, phi=0.10, seed=base_seed + 54
    )))
    configs.append(("extreme-convergence", SimConfig(n_agents=100, r=0.7, gamma=4.0, seed=base_seed + 55)))
    configs.append(("conservative", SimConfig(n_agents=5_000, r=0.0, gamma=0.5, seed=base_seed + 56)))

    assert len(configs) == 31, f"Expected 31 configs, got {len(configs)}"
    return configs


def run_robustness_suite(quick: bool = False) -> List[Tuple[str, SimResult]]:
    """Run all 31 robustness configurations (Section 5.4)."""
    log.info("")
    log.info("=" * 65)
    log.info("SUITE 4: Robustness (31 configurations)")
    log.info("=" * 65)

    configs = get_robustness_configs()
    results = []
    n_pass = 0

    for i, (label, cfg) in enumerate(configs):
        if quick:
            cfg.n_periods = 2_000
        result = run_simulation(cfg, label=label)
        mono = result.prop1_monotonicity
        n_pass += int(mono)
        status = "PASS" if mono else "FAIL"
        log.info(f"  [{i+1:2d}/31] {label:25s} -> {result.n_cascades:3d} cascades, P1: {status}")
        results.append((label, result))

    log.info(f"\n  Result: {n_pass}/31 maintain Proposition 1 monotonicity")
    return results


# ===================================================================
# CONTRARIAN ANALYSIS SWEEP (Section 5.5)
# ===================================================================

def run_contrarian_sweep(cfg: Optional[SimConfig] = None) -> Dict:
    """Run targeted contrarian sweep at key q values."""
    if cfg is None:
        cfg = SimConfig()

    log.info("")
    log.info("=" * 65)
    log.info("CONTRARIAN ANALYSIS SWEEP")
    log.info("=" * 65)

    q_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results: Dict = {
        "q_values": q_values,
        "phi_star_theoretical": [],
        "h_values": [],
    }

    for q_val in q_values:
        ps = phi_star(q_val, cfg)
        hq = h_of_q(q_val, cfg)
        sweep_results["phi_star_theoretical"].append(ps)
        sweep_results["h_values"].append(hq)
        log.info(f"  q={q_val:.1f}: h(q)={hq:.4f}, phi*={ps:.4f} ({ps*100:.1f}%)")

    return sweep_results


# ===================================================================
# FIGURE GENERATION
# ===================================================================

_BG = "#080E1A"
_FG = "#E8ECF0"
_GRID = "#1A2235"
_AMBER = "#F7B731"
_TEAL = "#2ED8A3"
_BLUE = "#5BA4F5"
_RED = "#F0483E"
_MUTED = "#6B7D99"


def _style_ax(ax: plt.Axes, title: str = "") -> None:
    """Apply dark-theme styling."""
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.spines["bottom"].set_color(_GRID)
    ax.spines["left"].set_color(_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    if title:
        ax.set_title(title, color=_FG, fontsize=10, fontweight="bold", pad=8)


def generate_core_figure(result: SimResult, output_path: str) -> str:
    """Generate Figure 1: Core simulation results (4-panel)."""
    fig = plt.figure(figsize=(14, 10), facecolor=_BG)
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)
    cfg = result.config

    # Panel A: h(q)
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "A. Behavioural Homogeneity h(q)")
    q = np.linspace(0, 1, 500)
    h = h_of_q_array(q, cfg)
    ax1.fill_between(q, cfg.h_star, 1.0, alpha=0.08, color=_RED)
    ax1.fill_between(q, 0, cfg.h_star, alpha=0.04, color=_TEAL)
    ax1.plot(q, h, color=_BLUE, linewidth=2.5, label="h(q)")
    ax1.axhline(cfg.h_star, color=_AMBER, linewidth=1, linestyle="--", label=f"h* = {cfg.h_star}")
    q_c = q[np.argmin(np.abs(h - cfg.h_star))]
    ax1.axvline(q_c, color=_RED, linewidth=0.8, linestyle=":", alpha=0.7)
    ax1.annotate(f"q_c = {q_c:.2f}", xy=(q_c, cfg.h_star),
                 xytext=(q_c + 0.08, cfg.h_star + 0.08),
                 color=_RED, fontsize=8, arrowprops=dict(arrowstyle="->", color=_RED, lw=0.8))
    ax1.set_xlabel("AI Quality q")
    ax1.set_ylabel("h(q)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.85)
    ax1.legend(loc="upper left", fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG)

    # Panel B: Cascades by quintile
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "B. Cascade Events by AI Quintile")
    stats = result.quintile_stats()
    labels = [f"Q{s['quintile']}" for s in stats]
    counts = [s["cascades"] for s in stats]
    colors = [_TEAL, _TEAL, _AMBER, _RED, _RED]
    bars = ax2.bar(labels, counts, color=colors, alpha=0.85)
    ax2.set_ylabel("Cascade Count")
    ax2.set_xlabel("AI Quality Quintile")
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", color=_FG, fontsize=9)

    # Panel C: Safeguard success
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "C. Safeguard Success Rate")
    sg_rates = [s["safeguard_rate"] for s in stats]
    ax3.bar(labels, sg_rates, color=_AMBER, alpha=0.85)
    ax3.set_ylabel("Success Rate (%)")
    ax3.set_xlabel("AI Quality Quintile")
    ax3.set_ylim(0, 110)
    for i, rate in enumerate(sg_rates):
        ax3.text(i, rate + 2, f"{rate:.1f}%", ha="center", va="bottom", color=_FG, fontsize=9)

    # Panel D: Volatility
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "D. Rolling Volatility")
    t = np.arange(len(result.volatility_series))
    ax4.plot(t, result.volatility_series, color=_BLUE, linewidth=0.5, alpha=0.7)
    if result.cascade_times:
        cas_t = np.array(result.cascade_times)
        cas_vol = result.volatility_series[cas_t]
        ax4.scatter(cas_t, cas_vol, color=_RED, s=8, alpha=0.6, zorder=5)
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Rolling sigma")
    ax4.set_xlim(0, cfg.n_periods)

    fig.suptitle("The Homogeneity Threshold -- Core Simulation Results",
                 color=_FG, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             f"N={cfg.n_agents:,}  T={cfg.n_periods:,}  Total cascades: {result.n_cascades}",
             ha="center", color=_MUTED, fontsize=8)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    log.info(f"  Saved: {output_path}")
    return output_path


def generate_contrarian_figure(sweep: Dict, cfg: SimConfig, output_path: str) -> str:
    """Generate Figure 2: Contrarian diversity analysis (2-panel)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=_BG)

    _style_ax(ax1, "A. Critical Diversity Threshold phi*(q)")
    q = np.linspace(0.01, 1.0, 500)
    phi_vals = np.array([phi_star(qi, cfg) for qi in q])
    ax1.plot(q, phi_vals * 100, color=_TEAL, linewidth=2.5)
    ax1.fill_between(q, 0, phi_vals * 100, alpha=0.08, color=_TEAL)
    cal_q = [0.6, 0.7, 0.8, 0.9, 1.0]
    cal_phi = [phi_star(qi, cfg) * 100 for qi in cal_q]
    ax1.scatter(cal_q, cal_phi, color=_AMBER, s=50, zorder=5, edgecolors="white", linewidth=0.5)
    for qi, pi in zip(cal_q, cal_phi):
        ax1.annotate(f"{pi:.1f}%", xy=(qi, pi), xytext=(qi - 0.05, pi + 1.5),
                     color=_AMBER, fontsize=8, ha="center")
    ax1.set_xlabel("AI Quality q")
    ax1.set_ylabel("Required Diversity phi* (%)")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 22)

    _style_ax(ax2, "B. Effective Homogeneity h_tilde(q, phi)")
    phi_scenarios = [0.0, 0.05, 0.10, 0.15, 0.20]
    colors_b = [_RED, "#E87D50", _AMBER, _TEAL, "#4ADE80"]
    for phi_val, col in zip(phi_scenarios, colors_b):
        h_eff = np.array([h_tilde(qi, phi_val, cfg) for qi in q])
        label_str = f"phi = {phi_val:.0%}" if phi_val > 0 else "phi = 0%"
        ax2.plot(q, h_eff, color=col, linewidth=1.8, label=label_str)
    ax2.axhline(cfg.h_star, color=_AMBER, linewidth=1, linestyle="--",
                label=f"h* = {cfg.h_star}", alpha=0.7)
    ax2.set_xlabel("AI Quality q")
    ax2.set_ylabel("Effective Homogeneity")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc="upper left", fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, ncol=2)

    fig.suptitle("The Homogeneity Threshold -- Contrarian Analysis",
                 color=_FG, fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    log.info(f"  Saved: {output_path}")
    return output_path


def generate_robustness_figure(results: List[Tuple[str, SimResult]], output_path: str) -> str:
    """Generate Figure 3: Robustness suite summary (horizontal bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 12), facecolor=_BG)
    _style_ax(ax, "Robustness Suite: 31 Configurations")

    labels = [label for label, _ in results]
    cascades = [r.n_cascades for _, r in results]
    passed = [r.prop1_monotonicity for _, r in results]
    colors = [_TEAL if p else _RED for p in passed]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, cascades, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cascade Count")
    ax.invert_yaxis()

    for i, (c, p) in enumerate(zip(cascades, passed)):
        symbol = "PASS" if p else "FAIL"
        ax.text(c + 1, i, f"{symbol} ({c})", va="center", color=_FG, fontsize=8)

    n_pass = sum(passed)
    ax.text(0.98, 0.02, f"P1 Monotonicity: {n_pass}/{len(results)} pass",
            transform=ax.transAxes, ha="right", va="bottom",
            color=_AMBER, fontsize=11, fontweight="bold")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    log.info(f"  Saved: {output_path}")
    return output_path


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    """Run all simulation suites and generate figures."""
    parser = argparse.ArgumentParser(
        description="Paper II: The Homogeneity Threshold -- Simulation Suite"
    )
    parser.add_argument("--suite",
                        choices=["core", "regime", "heterogeneous", "robustness", "contrarian", "all"],
                        default="all")
    parser.add_argument("--quick", action="store_true", help="Reduced T for testing")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 65)
    log.info("  THE HOMOGENEITY THRESHOLD")
    log.info("  Paper II Simulation Suite")
    log.info("  Gething (2026b) | FishIntel Global Ltd.")
    log.info("=" * 65)

    t_total = time.time()

    if args.suite in ("core", "all"):
        core_result = run_core_suite(SimConfig(seed=args.seed))
        generate_core_figure(core_result, os.path.join(args.output_dir, "figure_1_core_results.png"))

    if args.suite in ("regime", "all"):
        run_regime_switching_suite(SimConfig(seed=args.seed))

    if args.suite in ("heterogeneous", "all"):
        run_heterogeneous_suite(SimConfig(seed=args.seed))

    if args.suite in ("robustness", "all"):
        rob_results = run_robustness_suite(quick=args.quick)
        generate_robustness_figure(rob_results, os.path.join(args.output_dir, "figure_3_robustness.png"))

    if args.suite in ("contrarian", "all"):
        sweep = run_contrarian_sweep(SimConfig(seed=args.seed))
        generate_contrarian_figure(sweep, SimConfig(seed=args.seed),
                                   os.path.join(args.output_dir, "figure_2_contrarian_analysis.png"))

    elapsed = time.time() - t_total
    log.info("")
    log.info("=" * 65)
    log.info(f"  COMPLETE -- {elapsed:.1f}s total")
    log.info(f"  Figures: {args.output_dir}/")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
