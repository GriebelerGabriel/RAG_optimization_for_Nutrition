"""Generate a visual flowchart of the experimental pipeline."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 26))
ax.set_xlim(0, 20)
ax.set_ylim(0, 26)
ax.axis("off")

# Colors
C_HEADER = "#1a237e"
C_BASELINE = "#2e7d32"
C_ABLATION = "#e65100"
C_NO_RAG = "#c62828"
C_COMPARE = "#00695c"
C_ARROW = "#555555"

y = 25.2

# ── Title ──
ax.text(10, y, "RAG Optimization — Experimental Pipeline", fontsize=22, fontweight="bold",
        ha="center", va="center", color=C_HEADER)
y -= 0.5
ax.text(10, y, "Flat Ablation: each config varies ONE parameter from the baseline",
        fontsize=12, ha="center", va="center", color="#666")
y -= 0.9


# ── Helpers ──
def draw_box(ax, x, y, w, h, title, configs, color, bg_color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=bg_color, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 0.35, y + h - 0.38, title, fontsize=11, fontweight="bold", color=color)
    for i, cfg in enumerate(configs):
        ax.text(x + 0.35, y + h - 0.78 - i * 0.33, cfg, fontsize=9.5, family="monospace", color="#333")


def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my + 0.18, label, fontsize=8.5, ha="center",
                color=color, style="italic")


# ── Baseline ──
bx, bw, bh = 4, 12, 1.3
rect = FancyBboxPatch((bx, y - bh), bw, bh, boxstyle="round,pad=0.15",
                       facecolor="#e8f5e9", edgecolor=C_BASELINE, linewidth=2.5)
ax.add_patch(rect)
ax.text(bx + 0.5, y - 0.4, "BASELINE", fontsize=13, fontweight="bold", color=C_BASELINE)
ax.text(bx + 0.5, y - 0.85,
        "llama_naive:  cs=512  ov=50  top_k=5  naive  no rerank  temp=0.0  Llama3.1:8b",
        fontsize=10, family="monospace", color="#333")
y -= bh + 0.6

# ── Column layout ──
L_x, L_w = 0.5, 9       # left column
R_x, R_w = 10.5, 9      # right column

# ══════════════════════════════════════════════
# LEFT COLUMN — Ablation Groups (flat, all from baseline)
# ══════════════════════════════════════════════

# Group 1: Chunk Size
g1_y = y - 2.3
g1_h = 2.3
draw_box(ax, L_x, g1_y, L_w, g1_h,
         "CHUNK SIZE", [
             "llama_cs256     cs=256   (baseline=512)",
             "llama_cs1024    cs=1024  (baseline=512)",
         ], C_ABLATION, "#fff3e0")
arrow(ax, 10, y + 0.1, 10, g1_y + g1_h, "vary chunk_size", C_ABLATION)

# Group 2: Overlap
g2_y = g1_y - 2.5
g2_h = 1.8
draw_box(ax, L_x, g2_y, L_w, g2_h,
         "OVERLAP", [
             "llama_ov0       ov=0    (baseline=50)",
             "llama_ov100     ov=100  (baseline=50)",
         ], C_ABLATION, "#fff3e0")
arrow(ax, 10, g1_y, 10, g2_y + g2_h, "vary overlap", C_ABLATION)

# Group 3: Retrieval Strategy
g3_y = g2_y - 2.5
g3_h = 1.8
draw_box(ax, L_x, g3_y, L_w, g3_h,
         "RETRIEVAL STRATEGY", [
             "llama_hyde      HyDE query expansion",
             "llama_reranker  CrossEncoder reranking",
         ], C_ABLATION, "#fff3e0")
arrow(ax, 10, g2_y, 10, g3_y + g3_h, "vary retrieval", C_ABLATION)

# Group 4: Top-K + Temperature
g4_y = g3_y - 3.3
g4_h = 2.8
draw_box(ax, L_x, g4_y, L_w, g4_h,
         "TOP-K + TEMPERATURE", [
             "llama_topk3     top_k=3   temp=0.0",
             "llama_topk8     top_k=8   temp=0.0",
             "llama_temp030   top_k=5   temp=0.3  (baseline=0.0)",
         ], C_ABLATION, "#fff3e0")
arrow(ax, 10, g3_y, 10, g4_y + g4_h, "vary top_k / temp", C_ABLATION)

# ══════════════════════════════════════════════
# RIGHT COLUMN — Fair Comparison + No-RAG
# ══════════════════════════════════════════════

# LLM Comparison
llm_y = g2_y
llm_h = 1.7
draw_box(ax, R_x, llm_y, R_w, llm_h,
         "LLM COMPARISON  (same RAG params)", [
             "fair_llama8b       Llama 3.1 8B  (local)",
             "fair_gpt35_turbo   GPT-3.5-turbo (cloud)",
         ], C_COMPARE, "#e0f2f1")

# No-RAG Baselines
nrag_y = llm_y - 2.8
nrag_h = 1.7
draw_box(ax, R_x, nrag_y, R_w, nrag_h,
         "NO-RAG BASELINES  (parametric knowledge only)", [
             "no_rag_llama8b     Llama 3.1 8B, no retrieval",
             "no_rag_gpt35_turbo GPT-3.5-turbo, no retrieval",
         ], C_NO_RAG, "#ffebee")

# ══════════════════════════════════════════════
# Legend + Run command
# ══════════════════════════════════════════════

legend_y = 1.0
legend_items = [
    (C_BASELINE, "Baseline"),
    (C_ABLATION, "Ablation (1 param)"),
    (C_COMPARE, "LLM comparison"),
    (C_NO_RAG, "No-RAG"),
]
for i, (color, label) in enumerate(legend_items):
    xp = 1.5 + i * 4.5
    rect = FancyBboxPatch((xp, legend_y), 0.4, 0.3, boxstyle="round,pad=0.05",
                           facecolor="white", edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(xp + 0.55, legend_y + 0.15, label, fontsize=9.5, va="center", color="#333")

cmd_y = 2.0
ax.text(10, cmd_y + 0.5, "Run all experiments:", fontsize=12, fontweight="bold",
        ha="center", color=C_HEADER)
ax.text(10, cmd_y,
        "make benchmark-all   →   14 configs × N QA per config",
        fontsize=11, ha="center", family="monospace", color="#333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#ccc"))

plt.tight_layout()
Path("results").mkdir(parents=True, exist_ok=True)
fig.savefig("results/experiment_flow.png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig("results/experiment_flow.pdf", bbox_inches="tight", facecolor="white")
print("Saved to results/experiment_flow.png and .pdf")
