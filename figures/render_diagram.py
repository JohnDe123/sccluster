#!/usr/bin/env python3
"""
Render the scCluster pipeline diagram as a publication-ready figure.

Output: pipeline_diagram.pdf (vector) and pipeline_diagram.png (raster)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Rectangle, Polygon
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# ── Colors (soft, academic palette) ─────────────────────
C_GRAY_BG     = "#f5f4f1"
C_GRAY_BORDER = "#e0ded8"
C_GRAY_TEXT   = "#6b6560"
C_GRAY_DARK   = "#8c8279"
C_GRAY_ICON   = "#e8e3da"

C_BLUE_BG     = "#eef4f9"
C_BLUE_BORDER = "#d0dde9"
C_BLUE_TEXT   = "#3a5f8a"
C_BLUE_MED    = "#4a7cb5"
C_BLUE_LIGHT  = "#a0bcd8"
C_BLUE_PALE   = "rgba(160,200,240,0.4)"  # handled in matplotlib

C_GREEN_BG     = "#eef5ef"
C_GREEN_BORDER = "#c8dcc9"
C_GREEN_TEXT   = "#3d6b40"
C_GREEN_MED    = "#5a9a5e"
C_GREEN_LIGHT  = "#a0b8a2"

C_ORANGE_BG     = "#fdf5f0"
C_ORANGE_BORDER = "#f0d8c0"
C_ORANGE_TEXT   = "#b85a2e"
C_ORANGE_MED    = "#d4784a"
C_ORANGE_LIGHT  = "#d4a878"

C_ARROW    = "#c5bfb8"
C_LINE     = "#d5cfc7"
C_BG       = "#ffffff"
C_DARK     = "#1a1816"
C_BODY     = "#5c5550"
C_CAPTION  = "#9a938b"
C_PANEL_BG = "#fdfdfc"
C_PANEL_BORDER = "#e8e5e0"

# Cluster colors
CLUSTER_COLORS = ["#5a9a5e", "#e8a840", "#c570a0", "#5a8ec0", "#b090c8",
                  "#d4784a", "#7a9a5e", "#c8a060", "#a570b0", "#6a8eb0"]

# ── Font setup ───────────────────────────────────────────
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", ".claude", "plugins", "cache",
                        "anthropic-agent-skills", "claude-api", "f458cee31a75",
                        "skills", "canvas-design", "canvas-fonts")

# Try to find WorkSans font
WORK_SANS_REG = None
WORK_SANS_BOLD = None
for base in [FONT_DIR,
             r"C:\Users\25605\.claude\plugins\cache\anthropic-agent-skills\claude-api\f458cee31a75\skills\canvas-design\canvas-fonts"]:
    if os.path.exists(base):
        reg = os.path.join(base, "WorkSans-Regular.ttf")
        bold = os.path.join(base, "WorkSans-Bold.ttf")
        if os.path.exists(reg):
            WORK_SANS_REG = reg
            WORK_SANS_BOLD = bold
            break

if WORK_SANS_REG:
    from matplotlib.font_manager import fontManager
    fontManager.addfont(WORK_SANS_REG)
    fontManager.addfont(WORK_SANS_BOLD)
    FP_REG = FontProperties(fname=WORK_SANS_REG, size=8)
    FP_BOLD = FontProperties(fname=WORK_SANS_BOLD, size=8)
    FP_TITLE = FontProperties(fname=WORK_SANS_BOLD, size=18)
    FP_SECTION = FontProperties(fname=WORK_SANS_BOLD, size=10)
    FP_SMALL = FontProperties(fname=WORK_SANS_REG, size=7)
    FP_TINY = FontProperties(fname=WORK_SANS_REG, size=6.5)
    FP_MONO = FontProperties(fname=WORK_SANS_REG, size=8)
else:
    FP_REG = {"family": "sans-serif", "size": 8}
    FP_BOLD = {"family": "sans-serif", "weight": "bold", "size": 8}
    FP_TITLE = {"family": "sans-serif", "weight": "bold", "size": 18}
    FP_SECTION = {"family": "sans-serif", "weight": "bold", "size": 10}
    FP_SMALL = {"family": "sans-serif", "size": 7}
    FP_TINY = {"family": "sans-serif", "size": 6.5}
    FP_MONO = {"family": "monospace", "size": 8}


def _fp(size=8, bold=False, mono=False, tiny=False, section=False, title=False):
    """Get font properties with given size."""
    if title:
        sz = 18
    elif section:
        sz = 10
    elif tiny:
        sz = 6.5
    elif mono:
        sz = size - 1 if size > 7 else size
    else:
        sz = size

    if WORK_SANS_REG:
        f = WORK_SANS_BOLD if (bold or section or title) else WORK_SANS_REG
        return FontProperties(fname=f, size=sz)
    else:
        kw = {"family": "sans-serif", "size": sz}
        if bold or section or title:
            kw["weight"] = "bold"
        if mono:
            kw["family"] = "monospace"
        return kw


def _zorder(top=False):
    """Return zorder values for proper layering."""
    return 100 if top else 10


def _setup_figure():
    """Create the master figure canvas."""
    fig = plt.figure(figsize=(26, 17), facecolor=C_BG, dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 17)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_facecolor(C_BG)
    return fig, ax


def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, radius=0.15, lw=0.5, z=None):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=z or _zorder(),
    )
    ax.add_patch(box)


def draw_arrow_h(ax, x1, x2, y, color=C_ARROW, lw=0.8, z=None):
    """Draw horizontal arrow."""
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            connectionstyle="arc3,rad=0",
        ),
        zorder=z or _zorder(top=True),
    )


def draw_number_badge(ax, x, y, num, bg_color):
    """Draw a small numbered circle badge."""
    circle = Circle((x, y), 0.14, facecolor=bg_color, edgecolor="none",
                     zorder=_zorder(top=True))
    ax.add_patch(circle)
    ax.text(x, y, str(num), fontproperties=_fp(size=8, bold=True),
            color="white", ha="center", va="center", zorder=_zorder(top=True)+1)


def draw_module_header(ax, x, y, step_num, title, badge_color, text_color):
    """Draw a module's step badge and title."""
    draw_number_badge(ax, x + 0.05, y - 0.05, step_num, badge_color)
    ax.text(x + 0.35, y, title, fontproperties=_fp(section=True),
            color=text_color, va="center", ha="left",
            zorder=_zorder(top=True))


# ═══════════════════════════════════════════════════════════
# DRAWING FUNCTION
# ═══════════════════════════════════════════════════════════

def draw_diagram(fig, ax):
    """Draw the complete pipeline diagram."""

    # ── Layout coordinates ──────────────────────────────
    # Module 1: Preprocessing
    M1_X, M1_W = 0.5, 3.8
    # Module 2: scVI
    M2_X, M2_W = 4.8, 4.6
    # Module 3: Clustering
    M3_X, M3_W = 9.9, 3.8
    # Module 4: Markers
    M4_X, M4_W = 14.2, 4.6

    MODULE_Y = 5.8
    MODULE_H = 9.8

    ARROW_W = 0.5
    CENTER_Y = MODULE_Y + MODULE_H / 2  # ~10.7

    # ── Title ───────────────────────────────────────────
    ax.text(0.5, 16.2, "scCluster: Hybrid Single-Cell Clustering Workflow",
            fontproperties=_fp(title=True), color=C_DARK, va="bottom", ha="left")
    ax.text(25.5, 16.2, "Thyroid Epithelial Cell Analysis\nscVI + Leiden + Wilcoxon",
            fontproperties=_fp(size=8), color=C_CAPTION, va="bottom", ha="right")

    # ═══════════════════════════════════════════════════
    # MODULE 1: Input & Preprocessing (Gray)
    # ═══════════════════════════════════════════════════
    draw_rounded_box(ax, M1_X, MODULE_Y, M1_W, MODULE_H,
                     C_GRAY_BG, C_GRAY_BORDER)
    draw_module_header(ax, M1_X, MODULE_Y + MODULE_H - 0.22,
                       1, "Input & Preprocessing", C_GRAY_DARK, C_GRAY_TEXT)

    # Heatmap
    hm_x, hm_y = M1_X + 0.25, MODULE_Y + 5.5
    hm_w, hm_h = 3.3, 1.6
    draw_rounded_box(ax, hm_x, hm_y, hm_w, hm_h, "#faf9f7", C_LINE, radius=0.08, lw=0.35)
    # Heatmap cells - denser grid with realistic expression pattern
    rng = np.random.default_rng(42)
    hm_data = rng.random((32, 56))
    # Create correlated rows (gene modules)
    for block_start in [0, 8, 16, 24]:
        base = rng.random(56)
        for i in range(block_start, min(block_start + 8, 32)):
            hm_data[i] = base * 0.6 + hm_data[i] * 0.4
    # Draw cells
    for i in range(32):
        for j in range(56):
            v = hm_data[i, j] * 0.48 + 0.02
            rect = Rectangle(
                (hm_x + 0.06 + j * 0.055, hm_y + hm_h - 0.11 - i * 0.045),
                0.051, 0.041,
                facecolor=(0.50, 0.46, 0.40, v), edgecolor="none",
                zorder=_zorder(top=True),
            )
            ax.add_patch(rect)
    ax.text(hm_x + hm_w / 2, hm_y + hm_h + 0.08, "Expression Matrix",
            fontproperties=_fp(size=8, bold=True), color=C_GRAY_TEXT,
            ha="center", va="bottom")
    ax.text(hm_x + hm_w / 2, hm_y - 0.15, "Cells × Genes",
            fontproperties=_fp(size=7), color=C_CAPTION,
            ha="center", va="top")

    # QC / Norm / HVG icons
    icon_y = hm_y + 4.8
    icon_w, icon_h = 0.7, 0.55
    for i, (label, detail, ico_x) in enumerate([
        ("Quality Control", "min_genes · MT%", M1_X + 0.3),
        ("Normalization", "log1p · 10K/cell", M1_X + 1.55),
        ("HVG Selection", "Top 2,000 genes", M1_X + 2.8),
    ]):
        draw_rounded_box(ax, ico_x, icon_y, icon_w, icon_h, C_GRAY_ICON, C_LINE, radius=0.06, lw=0.3)
        # Icon content
        if i == 0:
            # Bar chart icon (QC)
            bar_w = 0.08
            bars = [(ico_x + 0.12 + j * 0.13, icon_y + 0.1, bar_w, 0.15 + rng.random() * 0.25) for j in range(4)]
            for bx, by, bw, bh in bars:
                rect = Rectangle((bx, by), bw, bh, facecolor=C_GRAY_DARK, alpha=0.7,
                                 edgecolor="none", zorder=_zorder(top=True))
                ax.add_patch(rect)
        elif i == 1:
            # Normalization icon
            rect = Rectangle((ico_x + 0.15, icon_y + 0.1), 0.4, 0.35,
                             facecolor="none", edgecolor=C_GRAY_DARK, lw=0.6,
                             zorder=_zorder(top=True))
            ax.add_patch(rect)
            ax.plot([ico_x + 0.15, ico_x + 0.55], [icon_y + 0.3, icon_y + 0.3],
                    color=C_GRAY_DARK, lw=0.4, linestyle="dashed", zorder=_zorder(top=True))
        else:
            # HVG scatter
            xs = ico_x + 0.15 + rng.random(30) * 0.38
            ys = icon_y + 0.12 + rng.random(30) * 0.32
            ax.scatter(xs, ys, s=5, c=C_GRAY_DARK, alpha=0.5, edgecolor="none", zorder=_zorder(top=True))

        ax.text(ico_x + icon_w / 2, icon_y + icon_h + 0.06, label,
                fontproperties=_fp(size=8, bold=True), color=C_BODY,
                ha="center", va="bottom")
        ax.text(ico_x + icon_w / 2, icon_y - 0.06, detail,
                fontproperties=_fp(size=6.5), color=C_CAPTION,
                ha="center", va="top")

    # Output note
    ax.text(M1_X + M1_W / 2, icon_y - 0.65, "Filtered & Normalized →",
            fontproperties=_fp(size=8, bold=True), color=C_BLUE_MED,
            ha="center")

    # ═══════════════════════════════════════════════════
    # ARROW 1 → 2
    # ═══════════════════════════════════════════════════
    draw_arrow_h(ax, M1_X + M1_W, M2_X, CENTER_Y)

    # ═══════════════════════════════════════════════════
    # MODULE 2: scVI Deep Representation Learning (Blue)
    # ═══════════════════════════════════════════════════
    draw_rounded_box(ax, M2_X, MODULE_Y, M2_W, MODULE_H,
                     C_BLUE_BG, C_BLUE_BORDER)
    draw_module_header(ax, M2_X, MODULE_Y + MODULE_H - 0.22,
                       2, "Deep Representation Learning", C_BLUE_MED, C_BLUE_TEXT)

    # Dashed VAE bounding box with subtle fill
    vae_x, vae_y = M2_X + 0.2, MODULE_Y + 0.8
    vae_w, vae_h = M2_W - 0.4, 7.5
    rect_vae_bg = FancyBboxPatch((vae_x, vae_y), vae_w, vae_h,
                                  boxstyle="round,pad=0.1",
                                  facecolor=(0.94, 0.96, 0.99, 0.4),
                                  edgecolor="none",
                                  zorder=_zorder()-2)
    ax.add_patch(rect_vae_bg)
    rect_vae = Rectangle((vae_x, vae_y), vae_w, vae_h,
                          facecolor="none", edgecolor=(0.55, 0.70, 0.84, 0.35),
                          linestyle="dashed", lw=0.5, zorder=_zorder())
    ax.add_patch(rect_vae)
    ax.text(vae_x + vae_w / 2, vae_y + vae_h + 0.08, "Variational Autoencoder (scVI)",
            fontproperties=_fp(size=7.5, bold=True), color="#6a90b8",
            ha="center", va="bottom")

    # Encoder stacked layers with neuron-like interior dots
    enc_x = vae_x + 0.25
    enc_y = vae_y + 2.2
    enc_w = 0.85
    layer_specs = [
        (0.18, 6, 0.72),   # (height, n_neurons, alpha)
        (0.22, 5, 0.62),
        (0.26, 4, 0.52),
        (0.30, 3, 0.42),
    ]
    layer_y = enc_y
    for h, n_neurons, alpha in layer_specs:
        rect = FancyBboxPatch((enc_x, layer_y), enc_w, h,
                               boxstyle="round,pad=0.02",
                               facecolor=C_BLUE_MED, alpha=alpha,
                               edgecolor="none", zorder=_zorder(top=True))
        ax.add_patch(rect)
        # Neuron dots inside
        for ni in range(n_neurons):
            nx = enc_x + enc_w * (ni + 1) / (n_neurons + 1)
            ax.scatter(nx, layer_y + h / 2, s=5, c="white", alpha=0.4,
                       edgecolor="none", zorder=_zorder(top=True)+1)
        layer_y += h + 0.1
    ax.text(enc_x + enc_w / 2, layer_y + 0.08, "Encoder",
            fontproperties=_fp(size=8, bold=True), color=C_BLUE_TEXT, ha="center")

    # Decoder stacked layers (mirror of encoder)
    dec_x = vae_x + vae_w - 0.25 - enc_w
    dec_y = vae_y + 2.2
    layer_y = dec_y
    for h, n_neurons, alpha in reversed(layer_specs):
        rect = FancyBboxPatch((dec_x, layer_y), enc_w, h,
                               boxstyle="round,pad=0.02",
                               facecolor=C_BLUE_MED, alpha=alpha,
                               edgecolor="none", zorder=_zorder(top=True))
        ax.add_patch(rect)
        for ni in range(n_neurons):
            nx = dec_x + enc_w * (ni + 1) / (n_neurons + 1)
            ax.scatter(nx, layer_y + h / 2, s=5, c="white", alpha=0.4,
                       edgecolor="none", zorder=_zorder(top=True)+1)
        layer_y += h + 0.1
    ax.text(dec_x + enc_w / 2, layer_y + 0.08, "Decoder",
            fontproperties=_fp(size=8, bold=True), color=C_BLUE_TEXT, ha="center")

    # Encoder → Latent arrows
    latent_cx = vae_x + vae_w / 2
    latent_cy = enc_y + 1.8
    arr1_y = enc_y + 1.1
    ax.annotate("", xy=(latent_cx - 0.3, latent_cy), xytext=(enc_x + enc_w, arr1_y),
                arrowprops=dict(arrowstyle="->", color=C_BLUE_LIGHT, lw=0.5),
                zorder=_zorder(top=True))
    # Latent → Decoder arrows
    ax.annotate("", xy=(dec_x, arr1_y), xytext=(latent_cx + 0.3, latent_cy),
                arrowprops=dict(arrowstyle="->", color=C_BLUE_LIGHT, lw=0.5),
                zorder=_zorder(top=True))

    # Latent space (cloud + z) - multi-layer for depth
    for radius, alpha_face, alpha_edge in [(0.62, 0.08, 0.12), (0.50, 0.18, 0.25), (0.38, 0.32, 0.42)]:
        cloud = Circle((latent_cx, latent_cy), radius,
                       facecolor=(0.58, 0.76, 0.90, alpha_face),
                       edgecolor=(0.50, 0.68, 0.84, alpha_edge),
                       linestyle="dashed", lw=0.4, zorder=_zorder())
        ax.add_patch(cloud)
    # Scatter points inside latent space - denser toward center
    for _ in range(35):
        angle = rng.random() * np.pi * 2
        radius = rng.random()**0.6 * 0.44  # bias toward center
        px = latent_cx + np.cos(angle) * radius
        py = latent_cy + np.sin(angle) * radius
        ax.scatter(px, py, s=3.5, c=C_BLUE_MED, alpha=0.45 + 0.3 * (1 - radius / 0.44),
                   edgecolor="none", zorder=_zorder(top=True))
    # z label with subtle glow
    ax.text(latent_cx, latent_cy + 0.02, "z",
            fontproperties=_fp(size=14, bold=True, mono=True), color=C_BLUE_MED,
            ha="center", va="center", zorder=_zorder(top=True)+2,
            alpha=0.95)

    # Labels
    ax.text(enc_x + enc_w / 2, vae_y + 0.5, "Count Matrix",
            fontproperties=_fp(size=7), color=C_BLUE_TEXT, ha="center")
    ax.text(dec_x + enc_w / 2, vae_y + 0.5, "Reconstructed\nCounts",
            fontproperties=_fp(size=7), color=C_BLUE_TEXT, ha="center")
    ax.text(dec_x - 0.15, enc_y + 0.5, "ZINB",
            fontproperties=_fp(size=7), color="#8aabc5", ha="right")

    # VAE description
    ax.text(latent_cx, enc_y + 0.0, "30-dim latent\nz ~ N(0, I)",
            fontproperties=_fp(size=7), color=C_BLUE_TEXT, ha="center",
            style="italic")

    # scVI embedding goes downward
    emb_y = vae_y + 0.2
    ax.annotate("", xy=(latent_cx, emb_y - 0.3), xytext=(latent_cx, latent_cy - 0.55),
                arrowprops=dict(arrowstyle="->", color=C_BLUE_MED, lw=0.8),
                zorder=_zorder(top=True))
    ax.text(latent_cx + 0.08, (latent_cy - 0.55 + emb_y - 0.3) / 2,
            "scVI\nEmbedding", fontproperties=_fp(size=7, bold=True),
            color=C_BLUE_MED, ha="left", va="center")

    # ═══════════════════════════════════════════════════
    # ARROW 2 → 3
    # ═══════════════════════════════════════════════════
    draw_arrow_h(ax, M2_X + M2_W, M3_X, CENTER_Y)

    # ═══════════════════════════════════════════════════
    # MODULE 3: Graph-Based Clustering (Green)
    # ═══════════════════════════════════════════════════
    draw_rounded_box(ax, M3_X, MODULE_Y, M3_W, MODULE_H,
                     C_GREEN_BG, C_GREEN_BORDER)
    draw_module_header(ax, M3_X, MODULE_Y + MODULE_H - 0.22,
                       3, "Graph-Based Clustering", C_GREEN_MED, C_GREEN_TEXT)

    # kNN graph sub-panel
    knn_x, knn_y = M3_X + 0.2, MODULE_Y + 6.2
    knn_w, knn_h = M3_W - 0.4, 1.8
    draw_rounded_box(ax, knn_x, knn_y, knn_w, knn_h, "#f8faf7", C_GREEN_BORDER, radius=0.08, lw=0.3)
    ax.text(knn_x + knn_w / 2, knn_y + knn_h + 0.08, "k-Nearest Neighbor Graph",
            fontproperties=_fp(size=8, bold=True), color=C_GREEN_TEXT, ha="center")
    ax.text(knn_x + knn_w / 2, knn_y - 0.12, "k = 15, metric = Euclidean",
            fontproperties=_fp(size=7), color="#6b8c6e", ha="center")

    # Draw kNN nodes and edges inside the sub-panel
    nodes_pos = []
    for _ in range(35):
        nx = knn_x + 0.2 + rng.random() * (knn_w - 0.4)
        ny = knn_y + 0.2 + rng.random() * (knn_h - 0.4)
        nc = CLUSTER_COLORS[rng.integers(0, 5)]
        nodes_pos.append((nx, ny, nc))
    # Edges (connect nearby nodes)
    for i in range(len(nodes_pos)):
        for j in range(i + 1, len(nodes_pos)):
            d = np.sqrt((nodes_pos[i][0] - nodes_pos[j][0])**2 +
                        (nodes_pos[i][1] - nodes_pos[j][1])**2)
            if d < 0.55:
                ax.plot([nodes_pos[i][0], nodes_pos[j][0]],
                        [nodes_pos[i][1], nodes_pos[j][1]],
                        color=C_GREEN_LIGHT, lw=0.25, alpha=1 - d/0.55,
                        zorder=_zorder())
    # Nodes
    for nx, ny, nc in nodes_pos:
        ax.scatter(nx, ny, s=10, c=nc, alpha=0.75, edgecolor="none", zorder=_zorder(top=True))

    # Arrow down
    arr3_y = knn_y - 0.08
    ax.annotate("", xy=(M3_X + M3_W / 2, arr3_y - 0.5),
                xytext=(M3_X + M3_W / 2, arr3_y),
                arrowprops=dict(arrowstyle="->", color=C_GREEN_LIGHT, lw=0.6),
                zorder=_zorder(top=True))

    # Leiden box - more prominent
    lei_x, lei_y = M3_X + 0.5, arr3_y - 1.15
    lei_w, lei_h = M3_W - 1.0, 0.55
    draw_rounded_box(ax, lei_x, lei_y, lei_w, lei_h, "#d8e8d9", C_GREEN_MED, radius=0.06, lw=0.5)
    # Inner highlight
    draw_rounded_box(ax, lei_x + 0.02, lei_y + 0.02, lei_w - 0.04, lei_h - 0.04,
                     "#d8e8d9", "none", radius=0.04, lw=0)
    ax.text(lei_x + lei_w / 2, lei_y + lei_h / 2, "Leiden Algorithm",
            fontproperties=_fp(size=9, bold=True), color=C_GREEN_TEXT,
            ha="center", va="center")
    ax.text(lei_x + lei_w / 2, lei_y - 0.1, "Modularity optimization · Resolution = 1.0",
            fontproperties=_fp(size=7), color="#5a8a5d", ha="center")

    # Arrow down
    arr4_y = lei_y - 0.08
    ax.annotate("", xy=(M3_X + M3_W / 2, arr4_y - 0.35),
                xytext=(M3_X + M3_W / 2, arr4_y),
                arrowprops=dict(arrowstyle="->", color=C_GREEN_LIGHT, lw=0.6),
                zorder=_zorder(top=True))

    # UMAP scatter sub-panel
    umap_x, umap_y = M3_X + 0.2, arr4_y - 3.4
    umap_w, umap_h = M3_W - 0.4, 3.0
    draw_rounded_box(ax, umap_x, umap_y, umap_w, umap_h, "#fafcfa", C_GREEN_BORDER, radius=0.08, lw=0.3)
    ax.text(umap_x + umap_w / 2, umap_y + umap_h + 0.06, "UMAP Projection",
            fontproperties=_fp(size=7.5, bold=True), color=C_GREEN_TEXT, ha="center")

    # Cluster scatter points
    centers = [(umap_x + 0.6, umap_y + 2.0), (umap_x + 2.2, umap_y + 0.6),
               (umap_x + 2.0, umap_y + 2.2), (umap_x + 0.8, umap_y + 0.8)]
    for ci, (cx, cy) in enumerate(centers[:4]):
        for _ in range(50):
            angle = rng.random() * np.pi * 2
            radius = rng.random() * 0.45
            px = cx + np.cos(angle) * radius
            py = cy + np.sin(angle) * radius * 0.7
            if umap_x + 0.08 < px < umap_x + umap_w - 0.08 and umap_y + 0.08 < py < umap_y + umap_h - 0.08:
                ax.scatter(px, py, s=4, c=CLUSTER_COLORS[ci], alpha=0.65,
                           edgecolor="none", zorder=_zorder(top=True))

    # Cluster legend below UMAP
    leg_y = umap_y - 0.3
    for ci in range(4):
        ax.scatter(umap_x + 0.15 + ci * 1.0, leg_y, s=15, c=CLUSTER_COLORS[ci],
                   alpha=0.8, edgecolor="none", zorder=_zorder(top=True))
        ax.text(umap_x + 0.25 + ci * 1.0, leg_y, f"C{ci}",
                fontproperties=_fp(size=7), color=C_BODY, va="center")
    ax.text(umap_x + umap_w / 2, leg_y - 0.18, "n clusters found",
            fontproperties=_fp(size=7), color=C_CAPTION, ha="center")

    # ═══════════════════════════════════════════════════
    # ARROW 3 → 4
    # ═══════════════════════════════════════════════════
    draw_arrow_h(ax, M3_X + M3_W, M4_X, CENTER_Y)

    # ═══════════════════════════════════════════════════
    # MODULE 4: Marker Gene Identification (Orange)
    # ═══════════════════════════════════════════════════
    draw_rounded_box(ax, M4_X, MODULE_Y, M4_W, MODULE_H,
                     C_ORANGE_BG, C_ORANGE_BORDER)
    draw_module_header(ax, M4_X, MODULE_Y + MODULE_H - 0.22,
                       4, "Marker Gene Identification", C_ORANGE_MED, C_ORANGE_TEXT)

    # Wilcoxon comparison panel
    wil_x, wil_y = M4_X + 0.2, MODULE_Y + 6.8
    wil_w, wil_h = M4_W - 0.4, 1.4
    draw_rounded_box(ax, wil_x, wil_y, wil_w, wil_h, "#fefaf7", C_ORANGE_BORDER, radius=0.08, lw=0.3)
    ax.text(wil_x + wil_w / 2, wil_y + wil_h + 0.08, "Wilcoxon Rank-Sum Test",
            fontproperties=_fp(size=8, bold=True), color=C_ORANGE_TEXT, ha="center")

    # Target vs Rest visual
    targ_x = wil_x + 0.3
    draw_rounded_box(ax, targ_x, wil_y + 0.2, 0.7, 0.5, "#e8c8a8", C_ORANGE_BORDER, radius=0.04, lw=0.2)
    ax.text(targ_x + 0.35, wil_y + 0.45, "Target", fontproperties=_fp(size=7, bold=True),
            color="#8b4513", ha="center", va="center")
    ax.text(targ_x + 1.1, wil_y + 0.45, "vs.", fontproperties=_fp(size=10),
            color=C_ORANGE_TEXT, ha="center", va="center")
    rest_x = targ_x + 1.6
    draw_rounded_box(ax, rest_x, wil_y + 0.2, 0.7, 0.5, "#f0e0d0", C_ORANGE_BORDER, radius=0.04, lw=0.2)
    ax.text(rest_x + 0.35, wil_y + 0.45, "Rest", fontproperties=_fp(size=7, bold=True),
            color="#8b6b4a", ha="center", va="center")
    # H0 hypothesis
    ax.text(wil_x + wil_w / 2, wil_y + 0.08,
            "H$_0$: expression$_{cluster}$ = expression$_{rest}$",
            fontproperties=_fp(size=7), color="#9a6b4a", ha="center")

    # Arrow down
    ax.annotate("", xy=(M4_X + M4_W / 2, wil_y - 0.08),
                xytext=(M4_X + M4_W / 2, wil_y),
                arrowprops=dict(arrowstyle="->", color=C_ORANGE_LIGHT, lw=0.5),
                zorder=_zorder(top=True))

    # Volcano plot (left side)
    vol_x, vol_y = M4_X + 0.25, wil_y - 2.2
    vol_w, vol_h = 2.0, 1.7
    draw_rounded_box(ax, vol_x, vol_y, vol_w, vol_h, "#fefcf9", C_ORANGE_BORDER, radius=0.06, lw=0.2)
    ax.text(vol_x + vol_w / 2, vol_y + vol_h + 0.05, "Volcano Plot",
            fontproperties=_fp(size=7.5, bold=True), color=C_ORANGE_TEXT, ha="center")

    # Volcano points
    vol_cx = vol_x + vol_w / 2
    vol_cy = vol_y + vol_h / 2 + 0.1
    for _ in range(70):
        px = vol_cx + (rng.random() - 0.5) * 1.5
        py = vol_cy + rng.random() * 0.7
        dist = abs(px - vol_cx) / 0.75
        h = (py - (vol_cy - 0.1)) / 0.7
        is_sig = dist > 0.45 and h > 0.35
        c = C_ORANGE_MED if (is_sig and px > vol_cx) else (C_BLUE_MED if (is_sig and px < vol_cx) else "#d5cdc4")
        alpha = 0.7 if is_sig else 0.3
        ax.scatter(px, py, s=4, c=c, alpha=alpha, edgecolor="none", zorder=_zorder(top=True))

    # Threshold lines (confined to volcano area)
    ax.plot([vol_x + 0.1, vol_x + vol_w - 0.1], [vol_cy + 0.18, vol_cy + 0.18],
            color=C_ORANGE_LIGHT, lw=0.3, linestyle="dashed", zorder=_zorder(top=True))
    ax.text(vol_x + vol_w - 0.1, vol_y + 0.08, "p$_{adj}$ < 0.05\n|logFC| > 1.0",
            fontproperties=_fp(size=6.5), color="#9a6b4a", ha="right", va="bottom")

    # Gene result table (right side)
    tbl_x, tbl_y = vol_x + vol_w + 0.2, vol_y + 0.1
    tbl_w, tbl_h = 2.0, 1.5
    draw_rounded_box(ax, tbl_x, tbl_y, tbl_w, tbl_h, "#fefcf9", C_ORANGE_BORDER, radius=0.06, lw=0.2)
    ax.text(tbl_x + tbl_w / 2, tbl_y + tbl_h + 0.05, "Top Markers",
            fontproperties=_fp(size=7.5, bold=True), color=C_ORANGE_TEXT, ha="center")
    genes_data = [("TG", "3.2", "***"), ("TPO", "2.8", "***"),
                  ("TSHR", "2.1", "**"), ("PAX8", "1.9", "**"),
                  ("NKX2-1", "1.5", "**"), ("KRT19", "1.2", "*")]
    for gi, (gene, logfc, sig) in enumerate(genes_data):
        yp = tbl_y + tbl_h - 0.25 - gi * 0.2
        ax.text(tbl_x + 0.12, yp, gene, fontproperties=_fp(size=7, mono=True),
                color="#4a3f38", va="center")
        ax.text(tbl_x + 0.85, yp, f"logFC {logfc}", fontproperties=_fp(size=7),
                color="#6b4a30", va="center")
        ax.text(tbl_x + 1.7, yp, sig, fontproperties=_fp(size=7, bold=True),
                color=C_ORANGE_MED, va="center")

    # FDR annotation
    ax.text(M4_X + M4_W / 2, vol_y - 0.22, "Benjamini-Hochberg FDR correction",
            fontproperties=_fp(size=7), color=C_CAPTION, ha="center")

    # ═══════════════════════════════════════════════════
    # VALIDATION PANEL (Bottom)
    # ═══════════════════════════════════════════════════
    val_y = 0.4
    val_h = 5.0
    draw_rounded_box(ax, 0.5, val_y, 18.8, val_h, C_PANEL_BG, C_PANEL_BORDER, radius=0.12, lw=0.4)
    ax.text(0.8, val_y + val_h - 0.22, "Validation & Visualization",
            fontproperties=_fp(size=9, bold=True), color=C_GRAY_TEXT, va="center")

    # Separator line above validation panel
    ax.plot([0.5, 19.3], [val_y + val_h + 0.05, val_y + val_h + 0.05],
            color=C_PANEL_BORDER, lw=0.4, zorder=_zorder())

    # Left: Larger UMAP
    vumap_x, vumap_y = 0.85, val_y + 0.35
    vumap_w, vumap_h = 8.0, 4.2
    draw_rounded_box(ax, vumap_x, vumap_y, vumap_w, vumap_h, "#fafcfa", "#e0ded6", radius=0.08, lw=0.3)
    ax.text(vumap_x + 0.2, vumap_y + vumap_h - 0.15, "UMAP: scVI Latent Space",
            fontproperties=_fp(size=8.5, bold=True), color=C_BODY)

    # Large UMAP scatter
    vcenters = [(vumap_x + 2.0, vumap_y + 3.2), (vumap_x + 6.5, vumap_y + 3.0),
                (vumap_x + 4.5, vumap_y + 1.0), (vumap_x + 1.5, vumap_y + 1.2),
                (vumap_x + 5.5, vumap_y + 0.8)]
    cluster_names = ["Follicular 1", "Follicular 2", "C-cells", "Hürthle", "Progenitor"]
    for ci, (cx, cy) in enumerate(vcenters):
        for _ in range(85):
            angle = rng.random() * np.pi * 2
            radius = rng.random() * 1.1
            px = cx + np.cos(angle) * radius
            py = cy + np.sin(angle) * radius * 0.65
            if vumap_x + 0.15 < px < vumap_x + vumap_w - 0.15 and vumap_y + 0.15 < py < vumap_y + vumap_h - 0.9:
                ax.scatter(px, py, s=3, c=CLUSTER_COLORS[ci], alpha=0.6,
                           edgecolor="none", zorder=_zorder(top=True))

    # Legend UMAP
    leg_start_x = vumap_x + 0.2
    leg_start_y = vumap_y + 0.18
    for ci, name in enumerate(cluster_names):
        ax.scatter(leg_start_x + ci * 1.55, leg_start_y, s=12, c=CLUSTER_COLORS[ci],
                   alpha=0.8, edgecolor="none", zorder=_zorder(top=True))
        ax.text(leg_start_x + ci * 1.55 + 0.1, leg_start_y, name,
                fontproperties=_fp(size=6.5), color="#7a7570", va="center")

    # Axis labels
    ax.text(vumap_x + vumap_w / 2, vumap_y + 0.06, "UMAP 1", fontproperties=_fp(size=7),
            color=C_CAPTION, ha="center")
    ax.text(vumap_x + 0.04, vumap_y + vumap_h / 2, "UMAP 2", fontproperties=_fp(size=7),
            color=C_CAPTION, va="center", rotation=90)

    # Right: Dotplot
    dot_x, dot_y = vumap_x + vumap_w + 0.5, val_y + 0.35
    dot_w, dot_h = 9.15, 4.2
    draw_rounded_box(ax, dot_x, dot_y, dot_w, dot_h, "#fafcfa", "#e0ded6", radius=0.08, lw=0.3)
    ax.text(dot_x + 0.2, dot_y + dot_h - 0.15, "Marker Gene Expression",
            fontproperties=_fp(size=8.5, bold=True), color=C_BODY)
    ax.text(dot_x + dot_w - 0.2, dot_y + dot_h - 0.15, "Dot size = % expressed · Color = mean expr.",
            fontproperties=_fp(size=6.5), color=C_CAPTION, ha="right")

    # Dotplot matrix
    d_genes = ["TG", "TPO", "TSHR", "PAX8", "NKX2-1", "KRT19", "CALCA", "CHGA", "SLC5A5", "DIO1", "IYD", "FCGBP"]
    d_clusters = ["C0", "C1", "C2", "C3", "C4"]
    # Pre-defined expression data
    d_data = {
        "C0": {"TG": (0.88, 2.8), "TPO": (0.82, 2.5), "TSHR": (0.75, 2.0), "PAX8": (0.70, 1.8),
               "NKX2-1": (0.65, 1.5), "KRT19": (0.45, 0.9), "CALCA": (0.05, 0.2), "CHGA": (0.08, 0.2),
               "SLC5A5": (0.60, 1.3), "DIO1": (0.55, 1.1), "IYD": (0.50, 1.0), "FCGBP": (0.40, 0.8)},
        "C1": {"TG": (0.75, 1.9), "TPO": (0.70, 1.7), "TSHR": (0.55, 1.1), "PAX8": (0.60, 1.3),
               "NKX2-1": (0.50, 0.9), "KRT19": (0.85, 2.6), "CALCA": (0.10, 0.2), "CHGA": (0.05, 0.1),
               "SLC5A5": (0.35, 0.7), "DIO1": (0.30, 0.6), "IYD": (0.25, 0.5), "FCGBP": (0.55, 1.2)},
        "C2": {"TG": (0.02, 0.1), "TPO": (0.01, 0.0), "TSHR": (0.05, 0.1), "PAX8": (0.08, 0.2),
               "NKX2-1": (0.10, 0.2), "KRT19": (0.30, 0.7), "CALCA": (0.90, 3.0), "CHGA": (0.88, 2.8),
               "SLC5A5": (0.05, 0.0), "DIO1": (0.02, 0.0), "IYD": (0.04, 0.1), "FCGBP": (0.15, 0.3)},
        "C3": {"TG": (0.15, 0.3), "TPO": (0.10, 0.2), "TSHR": (0.12, 0.2), "PAX8": (0.15, 0.3),
               "NKX2-1": (0.30, 0.7), "KRT19": (0.40, 0.9), "CALCA": (0.05, 0.1), "CHGA": (0.20, 0.5),
               "SLC5A5": (0.10, 0.2), "DIO1": (0.08, 0.1), "IYD": (0.18, 0.4), "FCGBP": (0.35, 0.8)},
        "C4": {"TG": (0.08, 0.1), "TPO": (0.05, 0.1), "TSHR": (0.20, 0.5), "PAX8": (0.35, 0.8),
               "NKX2-1": (0.45, 1.0), "KRT19": (0.50, 1.1), "CALCA": (0.05, 0.1), "CHGA": (0.10, 0.2),
               "SLC5A5": (0.08, 0.1), "DIO1": (0.05, 0.1), "IYD": (0.10, 0.2), "FCGBP": (0.55, 1.3)},
    }

    cell_w = 0.68
    cell_h = 0.32
    d_start_x = dot_x + 2.8
    d_start_y = dot_y + dot_h - 0.7
    MAX_SIZE = 8
    MAX_EXPR = 3.2

    # Column headers
    for gi, g in enumerate(d_genes):
        ax.text(d_start_x + gi * cell_w + cell_w / 2, d_start_y + 0.25, g,
                fontproperties=_fp(size=6.5, mono=True), color="#4a3f38",
                ha="center", rotation=45)

    # Row labels
    for ci, cl in enumerate(d_clusters):
        ax.text(d_start_x - 0.15, d_start_y - ci * cell_h, cl,
                fontproperties=_fp(size=7.5, bold=True), color=C_BODY,
                ha="right", va="center")

    # Dots
    for ci, cl in enumerate(d_clusters):
        for gi, g in enumerate(d_genes):
            pct, expr = d_data[cl].get(g, (0, 0))
            cx = d_start_x + gi * cell_w + cell_w / 2
            cy = d_start_y - ci * cell_h
            sz = 1.5 + (pct / 1.0) * (MAX_SIZE - 1.5)
            intensity = max(0.06, expr / MAX_EXPR)
            ax.scatter(cx, cy, s=sz**2, c=[(0.71, 0.35, 0.18, intensity)],
                       edgecolor="none", zorder=_zorder(top=True))

    # Legend for dotplot
    dleg_x = dot_x + dot_w - 1.0
    dleg_y = dot_y + 0.3
    for vi, v in enumerate([0.2, 0.5, 0.8, 1.0]):
        sz = 1.5 + v * (MAX_SIZE - 1.5)
        ax.scatter(dleg_x, dleg_y + vi * 0.28, s=sz**2,
                   c=[(0.71, 0.35, 0.18, 0.1 + v * 0.7)], edgecolor="none",
                   zorder=_zorder(top=True))
        ax.text(dleg_x + 0.18, dleg_y + vi * 0.28, f"{int(v*100)}%",
                fontproperties=_fp(size=6), color=C_CAPTION, va="center")

    # ── Caption at very bottom ──────────────────────────
    ax.text(9.9, 0.18,
            "scCluster Pipeline · scVI (Lopez et al., Nature Methods 2018) · "
            "Leiden (Traag et al., Scientific Reports 2019) · "
            "Scanpy (Wolf et al., Genome Biology 2018)",
            fontproperties=_fp(size=7), color=C_CAPTION, ha="center")

    return fig, ax


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig, ax = _setup_figure()
    fig, ax = draw_diagram(fig, ax)

    # Save as PDF (vector)
    pdf_path = os.path.join(output_dir, "pipeline_diagram.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor=C_BG,
                edgecolor="none", pad_inches=0.3)
    print(f"PDF saved to: {pdf_path}")

    # Save as PNG (raster)
    png_path = os.path.join(output_dir, "pipeline_diagram.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=C_BG,
                edgecolor="none", pad_inches=0.3)
    print(f"PNG saved to: {png_path}")

    plt.close(fig)
    print("Done! Publication-ready diagram created.")


if __name__ == "__main__":
    main()
