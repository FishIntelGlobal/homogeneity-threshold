#!/usr/bin/env python3
"""
Paper II — Hero Image Generator (Elite Edition)
================================================

Generates a publication-quality figure for
"The Homogeneity Threshold: How AI Precision Creates Systemic Fragility"

Design Language:
    DARK EDITORIAL OBSERVATORY
    - Midnight ink base with atmospheric depth layers
    - Amber/gold as danger signal — teal as safety
    - One dominant narrative curve — everything else supports it
    - Mathematical elegance, not chart clutter
    - Strategic negative space, asymmetric composition
    - Bloomberg Terminal authority meets Nature cover drama

Typography:
    - Title: Georgia / serif with tracked spacing
    - Data labels: DejaVu Sans / clean sans-serif
    - Math: Serif italic, precise sizing

Color Tokens:
    Midnight:   #080E1A → #0F1829 (gradient base)
    Ink Panel:  #131D33 (chart backgrounds)
    Amber:      #F7B731 (danger / threshold / h*)
    Teal:       #2ED8A3 (safety / diversity / φ*)
    Ice Blue:   #5BA4F5 (main h(q) curve)
    Red Alert:  #F0483E (cascade / critical)
    Bone White: #E4E8EE (primary text)
    Slate:      #6B7D99 (secondary text)
    Deep Slate: #3A4D6B (muted / grid)

Usage:
    python generate_paper_ii_hero.py

Output:
    paper_ii_hero.png  (300 DPI, ~1MB, publication-ready)

Author:  Jason Gething / FishIntel Global Ltd.
Repo:    github.com/FishIntelGlobal/homogeneity-threshold
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import os
from typing import Tuple, List

# ============================================================
# § 1.  DESIGN SYSTEM — single source of truth
# ============================================================

class DS:
    """Design System tokens. Every colour, size, weight lives here."""

    # ── Palette ──────────────────────────────────────────────
    BG_DEEP        = "#080E1A"     # deepest background
    BG_MID         = "#0F1829"     # mid-layer
    BG_PANEL       = "#131D33"     # chart panel fill
    BG_PANEL_EDGE  = "#1C2B4A"     # panel border

    GRID_FINE      = "#1A2744"     # minor grid
    GRID_MAJOR     = "#243556"     # major grid

    # Accent colours (maximally distinct roles)
    AMBER          = "#F7B731"     # danger / h* threshold
    AMBER_GLOW     = "#F7B73118"   # amber halo
    AMBER_DIM      = "#C4901F"     # amber de-emphasised

    TEAL           = "#2ED8A3"     # safety / φ*
    TEAL_GLOW      = "#2ED8A318"
    TEAL_DIM       = "#1FA67B"

    ICE            = "#5BA4F5"     # primary curve h(q)
    ICE_BRIGHT     = "#8DC4FF"     # highlight
    ICE_GLOW       = "#5BA4F514"

    RED            = "#F0483E"     # cascade / critical
    RED_DIM        = "#C03830"
    RED_GLOW       = "#F0483E14"

    # Neutrals
    BONE           = "#E4E8EE"     # primary text
    SILVER         = "#A3B1C6"     # secondary text
    SLATE          = "#6B7D99"     # muted text
    DEEP_SLATE     = "#3A4D6B"     # very muted
    WHITE          = "#FFFFFF"

    # ── Typography ───────────────────────────────────────────
    FONT_DISPLAY   = "DejaVu Serif"            # display title
    FONT_SERIF     = "DejaVu Serif"           # math / formal
    FONT_SANS      = "DejaVu Sans"            # labels / body
    FONT_MATH      = "DejaVu Serif"           # equations
    FONT_MONO      = "Latin Modern Mono"      # code / URLs

    # Sizes (points)
    SZ_HERO_TITLE  = 32
    SZ_SUBTITLE    = 14
    SZ_PANEL_TITLE = 14
    SZ_LABEL       = 11
    SZ_TICK        = 9
    SZ_ANNOTATION  = 10
    SZ_SMALL       = 8.5
    SZ_TINY        = 7.5

    # ── Layout constants ─────────────────────────────────────
    CURVE_LW       = 3.0
    GLOW_LW        = 9
    GLOW_ALPHA     = 0.12
    SPINE_LW       = 0.7


# ============================================================
# § 2.  MATHEMATICAL MODEL — Paper II core equations
# ============================================================

def h_of_q(q: np.ndarray,
           tau0: float = 2.0, gamma: float = 2.0,
           r: float = 0.3,   tau_p: float = 1.0) -> np.ndarray:
    """Behavioural homogeneity h(q) = τ₀q^γ / (τ₀q^γ + (1−r)τₚ)"""
    tau_ai = tau0 * np.power(q, gamma)
    return tau_ai / (tau_ai + (1 - r) * tau_p)


def phi_star(q: np.ndarray, h_star: float = 0.5, **kw) -> np.ndarray:
    """Critical diversity threshold φ∗(q, h∗) = 1 − √(h∗ / h(q))"""
    hq = h_of_q(q, **kw)
    return np.where(hq > h_star, 1.0 - np.sqrt(h_star / hq), 0.0)


def rho_of_q(q: np.ndarray,
             tau0: float = 2.0, gamma: float = 2.0,
             r: float = 0.3,   tau_p: float = 1.0) -> np.ndarray:
    """Pairwise correlation ρ(q) — conservative upper bound."""
    tau_ai = tau0 * np.power(q, gamma)
    return tau_ai / (tau_ai + (1 - r)**2 * tau_p)


# ============================================================
# § 3.  AXIS STYLING — every panel gets this treatment
# ============================================================

def style_panel(ax: plt.Axes, title: str, title_color: str = DS.BONE) -> None:
    """Apply dark observatory styling to a single panel."""
    ax.set_facecolor(DS.BG_PANEL)

    # Panel title — left-aligned, tracked, bold
    ax.set_title(title, fontsize=DS.SZ_PANEL_TITLE,
                 fontfamily=DS.FONT_SANS, fontweight="bold",
                 color=title_color, loc="left", pad=12)

    # Grid layers
    ax.grid(True, which="major", color=DS.GRID_MAJOR,
            linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", color=DS.GRID_FINE,
            linewidth=0.3, alpha=0.2)
    ax.minorticks_on()

    # Ticks
    ax.tick_params(axis="both", which="major",
                   colors=DS.SLATE, labelsize=DS.SZ_TICK, length=4)
    ax.tick_params(axis="both", which="minor",
                   colors=DS.DEEP_SLATE, length=2)

    # Spines
    for sp in ax.spines.values():
        sp.set_color(DS.BG_PANEL_EDGE)
        sp.set_linewidth(DS.SPINE_LW)


def axis_label(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    """Consistent axis label styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=DS.SZ_LABEL,
                      fontfamily=DS.FONT_SANS, color=DS.SILVER, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=DS.SZ_LABEL,
                      fontfamily=DS.FONT_SANS, color=DS.SILVER, labelpad=8)


# ============================================================
# § 4.  HELPER — glow-traced curve
# ============================================================

def glow_curve(ax: plt.Axes, x, y, color: str, lw: float = DS.CURVE_LW,
               label: str = "", zorder: int = 5, **kw):
    """Plot a curve with an outer glow halo for visual drama."""
    # Halo
    ax.plot(x, y, color=color, linewidth=DS.GLOW_LW,
            alpha=DS.GLOW_ALPHA, zorder=zorder - 1)
    # Core line with dark outline
    ax.plot(x, y, color=color, linewidth=lw, zorder=zorder,
            label=label,
            path_effects=[pe.withStroke(linewidth=lw + 2,
                                        foreground=DS.BG_PANEL)], **kw)


# ============================================================
# § 5.  BUILD THE FIGURE
# ============================================================

def generate(output: str = "paper_ii_hero.png", dpi: int = 300) -> None:
    """Construct the full 5-panel hero image."""

    fig = plt.figure(figsize=(18, 11), facecolor=DS.BG_DEEP)

    # ── Grid layout ──────────────────────────────────────────
    #   Row 0 (hero):  [ h(q) wide panel      ] [ φ* panel  ]
    #   Row 1 (detail):[ cascade ] [ trap      ] [ key stats ]

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.15, 1.0, 0.8],
        height_ratios=[1.0, 0.82],
        hspace=0.30, wspace=0.24,
        left=0.055, right=0.970, top=0.815, bottom=0.065,
    )

    ax_hero  = fig.add_subplot(gs[0, 0:2])   # spans 2 cols
    ax_phi   = fig.add_subplot(gs[0, 2])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_trap  = fig.add_subplot(gs[1, 1])
    ax_stats = fig.add_subplot(gs[1, 2])

    q = np.linspace(0.001, 1.0, 600)
    H_STAR = 0.5

    # =========================================================
    # PANEL A — HERO:  h(q) homogeneity curve
    # =========================================================
    ax = ax_hero
    style_panel(ax, "Behavioural Homogeneity   h(q)")

    hq = h_of_q(q)
    rq = rho_of_q(q)

    # ── Danger zone gradient (above h*) ──
    n_bands = 80
    for i in range(n_bands):
        lo = H_STAR + i * (1.0 - H_STAR) / n_bands
        hi = lo + (1.0 - H_STAR) / n_bands
        ax.axhspan(lo, hi, color=DS.RED, alpha=0.008 + 0.05 * (i / n_bands)**2.2)

    # ── Safe zone tint ──
    ax.axhspan(0, H_STAR, color=DS.TEAL, alpha=0.018)

    # ── h* threshold dashed line ──
    ax.axhline(H_STAR, color=DS.AMBER, lw=2.0, ls="--", alpha=0.85, zorder=3)
    ax.text(0.015, H_STAR + 0.016, "h∗ = 0.50   cascade threshold",
            transform=ax.get_yaxis_transform(),
            color=DS.AMBER, fontsize=DS.SZ_ANNOTATION,
            fontfamily=DS.FONT_SANS, fontweight="bold")

    # ── Correlation bound ρ(q) ──
    ax.fill_between(q, hq, rq, alpha=0.06, color=DS.ICE_BRIGHT, zorder=1)
    ax.plot(q, rq, color=DS.ICE_BRIGHT, lw=1.0, ls=":", alpha=0.45, zorder=2,
            label="ρ(q)  correlation bound")

    # ── Main h(q) curve — the hero ──
    glow_curve(ax, q, hq, DS.ICE, label="h(q)  homogeneity")

    # ── Critical crossing point q_c ──
    q_c = q[np.argmin(np.abs(hq - H_STAR))]
    ax.plot(q_c, H_STAR, "o", color=DS.AMBER, ms=10, zorder=7,
            markeredgecolor=DS.BG_DEEP, markeredgewidth=2.5)
    ax.annotate(f"q_c ≈ {q_c:.2f}",
                xy=(q_c, H_STAR),
                xytext=(q_c - 0.16, H_STAR - 0.08),
                color=DS.AMBER, fontsize=DS.SZ_ANNOTATION,
                fontfamily=DS.FONT_MATH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DS.AMBER, lw=1.5),
                zorder=8)

    # ── h(1) endpoint marker ──
    h1 = h_of_q(np.array([1.0]))[0]
    ax.plot(1.0, h1, "s", color=DS.RED, ms=9, zorder=7,
            markeredgecolor=DS.BG_DEEP, markeredgewidth=2.5)
    ax.annotate(f"h(1) = {h1:.3f}",
                xy=(1.0, h1),
                xytext=(0.76, h1 + 0.07),
                color=DS.RED, fontsize=DS.SZ_ANNOTATION,
                fontfamily=DS.FONT_MATH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DS.RED, lw=1.5),
                zorder=8)

    # ── Zone watermarks ──
    ax.text(0.14, 0.24, "STABLE", color=DS.TEAL, fontsize=24,
            fontfamily=DS.FONT_SANS, fontweight="bold", alpha=0.10,
            ha="center", va="center")
    ax.text(0.82, 0.72, "CASCADE\nZONE", color=DS.RED, fontsize=18,
            fontfamily=DS.FONT_SANS, fontweight="bold", alpha=0.08,
            ha="center", va="center", linespacing=1.3)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.85)
    axis_label(ax, "AI Model Quality   q", "Homogeneity   h(q)")
    ax.legend(loc="upper left", fontsize=DS.SZ_SMALL,
              facecolor=DS.BG_PANEL, edgecolor=DS.GRID_MAJOR,
              labelcolor=DS.SILVER, framealpha=0.92)

    # =========================================================
    # PANEL B — DIVERSITY THRESHOLD  φ*(q)
    # =========================================================
    ax = ax_phi
    style_panel(ax, "Diversity Threshold   φ∗(q)", DS.TEAL)

    phi = phi_star(q) * 100  # percentage

    # Fill
    ax.fill_between(q, 0, phi, alpha=0.12, color=DS.TEAL)

    # Curve with glow
    glow_curve(ax, q, phi, DS.TEAL)

    # Calibration dots
    cal = [(0.6, 0.7), (0.7, 7.4), (0.8, 12.1), (0.9, 15.4), (1.0, 17.8)]
    for qv, pv in cal:
        ax.plot(qv, pv, "o", color=DS.TEAL, ms=6, zorder=7,
                markeredgecolor=DS.BG_DEEP, markeredgewidth=1.5)

    # Annotate bookends
    ax.annotate("φ∗ = 7.4%", xy=(0.7, 7.4), xytext=(0.53, 16),
                color=DS.TEAL, fontsize=DS.SZ_SMALL,
                fontfamily=DS.FONT_MATH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DS.TEAL, lw=1.2),
                zorder=8)
    ax.annotate("φ∗ = 17.8%", xy=(1.0, 17.8), xytext=(0.82, 24),
                color=DS.TEAL, fontsize=DS.SZ_SMALL,
                fontfamily=DS.FONT_MATH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DS.TEAL, lw=1.2),
                zorder=8)

    # Directional note
    ax.text(0.55, 2.0, "↗  more AI quality = more diversity needed",
            color=DS.AMBER_DIM, fontsize=DS.SZ_TINY,
            fontfamily=DS.FONT_SANS, fontstyle="italic", alpha=0.6)

    ax.set_xlim(0.50, 1.05)
    ax.set_ylim(0, 27)
    axis_label(ax, "AI Quality   q", "Required Diversity   φ∗  (%)")

    # =========================================================
    # PANEL C — CASCADE SEVERITY BY QUINTILE
    # =========================================================
    ax = ax_bar
    style_panel(ax, "Cascade Events by AI Quintile")

    labels = ["Q1\n0–0.2", "Q2\n0.2–0.4", "Q3\n0.4–0.6",
              "Q4\n0.6–0.8", "Q5\n0.8–1.0"]
    events = [23, 18, 24, 19, 22]
    safeguard = [100, 100, 20.8, 0, 0]

    palette = [DS.TEAL, DS.TEAL, DS.AMBER, DS.RED, DS.RED]
    x = np.arange(len(labels))

    bars = ax.bar(x, events, width=0.62, color=palette, alpha=0.82,
                  edgecolor=DS.BG_DEEP, linewidth=1.4, zorder=3)

    # Glow rectangles behind bars
    for b, c in zip(bars, palette):
        ax.bar(b.get_x() + b.get_width() / 2, b.get_height(),
               width=b.get_width() + 0.08, alpha=0.06, color=c, zorder=2)

    # Safeguard annotations
    for i, (s, e) in enumerate(zip(safeguard, events)):
        c = DS.TEAL if s == 100 else (DS.AMBER if s > 0 else DS.RED)
        lbl = f"{s:.0f}%" if s == int(s) else f"{s}%"
        ax.text(i, e + 0.7, f"↑ {lbl}", ha="center", va="bottom",
                color=c, fontsize=DS.SZ_SMALL,
                fontfamily=DS.FONT_SANS, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=DS.SZ_TINY,
                       fontfamily=DS.FONT_SANS, color=DS.SLATE)
    ax.set_ylim(0, 33)
    axis_label(ax, ylabel="Cascade Events")

    ax.text(0.97, 0.97, "safeguard success ↑", transform=ax.transAxes,
            ha="right", va="top", fontsize=DS.SZ_TINY,
            color=DS.DEEP_SLATE, fontfamily=DS.FONT_SANS)

    # =========================================================
    # PANEL D — THE CONVERGENCE TRAP  F(q)
    # =========================================================
    ax = ax_trap
    style_panel(ax, "The Convergence Trap   F(q)", DS.RED)

    # Component curves (stylised for visual narrative)
    c_homo   = h_of_q(q)
    c_speed  = np.clip(1 - 0.3 * (1 - q)**1.5 / 0.15, 0, 1)
    c_opac   = np.clip(1 - (0.8 * (1 - 0.7 * q)) / 0.4, 0, 1)
    F_compound = c_homo * c_speed * c_opac

    # Faint component traces
    ax.plot(q, c_homo,  color=DS.ICE,   lw=1.0, ls="--", alpha=0.35, label="h(q)")
    ax.plot(q, c_speed, color=DS.AMBER, lw=1.0, ls="--", alpha=0.35, label="Speed failure")
    ax.plot(q, c_opac,  color=DS.RED_DIM, lw=1.0, ls="--", alpha=0.35, label="Opacity failure")

    # Danger fill
    ax.fill_between(q, 0, F_compound, alpha=0.10, color=DS.RED)

    # Compound F(q) — bright white
    glow_curve(ax, q, F_compound, DS.WHITE, label="F(q)  compound")

    # TRAP watermark
    ax.text(0.80, 0.78, "TRAP", color=DS.RED, fontsize=26,
            fontfamily=DS.FONT_SANS, fontweight="bold", alpha=0.07,
            ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.62)
    axis_label(ax, "AI Quality   q", "Fragility   F(q)")
    ax.legend(loc="upper left", fontsize=DS.SZ_TINY,
              facecolor=DS.BG_PANEL, edgecolor=DS.GRID_MAJOR,
              labelcolor=DS.SILVER, framealpha=0.92)

    # =========================================================
    # PANEL E — KEY RESULTS (data card)
    # =========================================================
    ax = ax_stats
    ax.set_facecolor(DS.BG_PANEL)
    for sp in ax.spines.values():
        sp.set_color(DS.BG_PANEL_EDGE)
        sp.set_linewidth(DS.SPINE_LW)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.50, 0.96, "KEY RESULTS", ha="center", va="top",
            fontsize=DS.SZ_PANEL_TITLE, fontfamily=DS.FONT_SANS,
            fontweight="bold", color=DS.BONE)

    # Rule
    ax.plot([0.08, 0.92], [0.915, 0.915], color=DS.AMBER, lw=2.5, alpha=0.55)

    rows: List[Tuple[str, str, str]] = [
        ("31 / 31",  "robustness configs\npass monotonicity",  DS.TEAL),
        ("0.741",    "h(1) — maximum\nhomogeneity at q = 1",  DS.ICE),
        ("17.8 %",   "φ∗ at max AI\ndiversity required",      DS.AMBER),
        ("106",      "cascade events in\nbaseline simulation", DS.RED),
        ("0 %",      "safeguard success\nwhen q > 0.6",       DS.RED),
    ]

    y_vals = np.linspace(0.85, 0.10, len(rows))
    for (val, desc, color), y in zip(rows, y_vals):
        ax.text(0.10, y, val, ha="left", va="center",
                fontsize=24, fontfamily=DS.FONT_SANS,
                fontweight="bold", color=color)
        ax.text(0.56, y, desc, ha="left", va="center",
                fontsize=DS.SZ_SMALL, fontfamily=DS.FONT_SANS,
                color=DS.SILVER, linespacing=1.35)

    # =========================================================
    # HEADER — Title / Subtitle / Equations / Branding
    # =========================================================

    # Main title
    fig.text(0.055, 0.975,
             "The Homogeneity Threshold",
             fontsize=DS.SZ_HERO_TITLE, fontfamily=DS.FONT_DISPLAY,
             fontweight="bold", color=DS.BONE, va="top")

    # Subtitle
    fig.text(0.055, 0.930,
             "How AI Precision Creates Systemic Fragility",
             fontsize=DS.SZ_SUBTITLE, fontfamily=DS.FONT_SANS,
             color=DS.AMBER, va="top", fontstyle="italic")

    # Attribution — right aligned
    fig.text(0.970, 0.975, "Paper II  ·  Gething (2026)",
             fontsize=DS.SZ_LABEL, fontfamily=DS.FONT_SANS,
             color=DS.SLATE, ha="right", va="top")
    fig.text(0.970, 0.950, "FishIntel Global Ltd.",
             fontsize=DS.SZ_SMALL, fontfamily=DS.FONT_SANS,
             color=DS.DEEP_SLATE, ha="right", va="top")

    # Equation banner — centred between title and charts
    fig.text(0.50, 0.885,
             "h(q) = τ₀q^γ / (τ₀q^γ + (1−r)τₚ)"
             "              "
             "φ∗(q, h∗) = 1 − √(h∗ / h(q))",
             fontsize=DS.SZ_LABEL + 1, fontfamily=DS.FONT_MATH,
             color=DS.ICE_BRIGHT, ha="center", va="center",
             alpha=0.65,
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor=DS.BG_MID,
                       edgecolor=DS.GRID_MAJOR,
                       alpha=0.80))

    # Thin decorative rule under equation
    fig.patches.append(FancyBboxPatch(
        (0.20, 0.859), 0.60, 0.001,
        boxstyle="round,pad=0.0005",
        facecolor=DS.AMBER, alpha=0.18,
        transform=fig.transFigure, zorder=0))

    # Top accent line — full width amber stroke
    fig.patches.append(FancyBboxPatch(
        (0.0, 0.995), 1.0, 0.005,
        boxstyle="square,pad=0",
        facecolor=DS.AMBER, alpha=0.65,
        transform=fig.transFigure, zorder=10))

    # Footer
    fig.text(0.055, 0.018,
             "github.com/FishIntelGlobal/homogeneity-threshold",
             fontsize=DS.SZ_TINY, fontfamily=DS.FONT_MONO,
             color=DS.DEEP_SLATE, va="bottom")
    fig.text(0.970, 0.018,
             "© 2026 FishIntel Global Ltd.  All rights reserved.",
             fontsize=DS.SZ_TINY, fontfamily=DS.FONT_SANS,
             color=DS.DEEP_SLATE, ha="right", va="bottom")

    # =========================================================
    # SAVE
    # =========================================================
    fig.savefig(output, dpi=dpi, facecolor=DS.BG_DEEP,
                edgecolor="none", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    size_bytes = os.path.getsize(output)
    print(f"✅  Hero image saved → {output}")
    print(f"    Resolution: {dpi} DPI")
    print(f"    File size:  {size_bytes:,} bytes  ({size_bytes / 1024:.0f} KB)")


# ============================================================
# § 6.  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate()
