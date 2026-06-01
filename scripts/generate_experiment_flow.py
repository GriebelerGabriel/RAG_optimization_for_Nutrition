"""Generate a visual flowchart of the experimental pipeline."""
"""Generate a visual flowchart of the experimental pipeline."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis("off")

# Colors
C_HEADER = "#1a237e"
C_BASELINE = "#2e7d32"
C_ABLATION = "#e65100"
C_INTERACTION = "#7b1fa2"
C_NO_RAG = "#c62828"
C_COMPARE = "#00695c"
C_COMBINED = "#1565c0"
C_ARROW = "#555555"

y = 27.2

# ── Title ──
ax.text(10, y, "RAG Optimization — Experimental Pipeline", fontsize=22, fontweight="bold",
        ha="center", va="center", color=C_HEADER)
y -= 0.5
ax.text(10, y, "Controlled Ablation: vary ONE parameter at a time from baseline",
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
        "fair_llama8b:  cs=512  ov=50  top_k=5  naive  no rerank  temp=0.1  Llama3.1:8b",
        fontsize=10, family="monospace", color="#333")
y -= bh + 0.6

# ── Column layout ──
L_x, L_w = 0.5, 9       # left column
R_x, R_w = 10.5, 9      # right column

# ══════════════════════════════════════════════
# LEFT COLUMN — Ablation Phases (sequential)
# ══════════════════════════════════════════════

# Phase 1: Chunk Size
p1_y = y - 3.2
p1_h = 2.3
draw_box(ax, L_x, p1_y, L_w, p1_h,
         "PHASE 1 — Chunk Size", [
             "llama_cs256     cs=256   ov=50  topk=5  naive",
             "llama_cs512     cs=512   ov=50  topk=5  naive  (= baseline)",
             "llama_cs1024    cs=1024  ov=50  topk=5  naive",
         ], C_ABLATION, "#fff3e0")
arrow(ax, 10, y + 0.1, 10, p1_y + p1_h, "vary chunk_size", C_ABLATION)

# Phase 2: Overlap
p2_y = p1_y - 3.0
p2_h = 1.8
draw_box(ax, L_x, p2_y, L_w, p2_h,
         "PHASE 2 — Overlap  (best cs fixed)", [
             "llama_ov0       cs=best  ov=0    topk=5  naive",
             "llama_ov100     cs=best  ov=100  topk=5  naive",
         ], C_ABLATION, "#fff3e0")
arrow(ax, L_x + L_w / 2, p1_y, L_x + L_w / 2, p2_y + p2_h, "best cs → fix", C_ABLATION)

# Phase 3: Retrieval Strategy
p3_y = p2_y - 3.5
p3_h = 2.8
draw_box(ax, L_x, p3_y, L_w, p3_h,
         "PHASE 3 — Retrieval Strategy  (best cs+ov fixed)", [
             "llama_hyde      naive → HyDE          (query expansion)",
             "llama_reranker  naive + CrossEncoder   (reranking only)",
             "llama_hybrid    hybrid + CrossEncoder  (stacked)",
         ], C_ABLATION, "#fff3e0")
arrow(ax, L_x + L_w / 2, p2_y, L_x + L_w / 2, p3_y + p3_h, "best ov → fix", C_ABLATION)

# Phase 4: Top-K + Temperature
p4_y = p3_y - 3.3
p4_h = 2.8
draw_box(ax, L_x, p4_y, L_w, p4_h,
         "PHASE 4 — Top-K + Temperature  (best retrieval fixed)", [
             "llama_topk3     top_k=3   temp=0.1",
             "llama_topk8     top_k=8   temp=0.1",
             "llama_temp000   top_k=best temp=0.0",
             "llama_temp030   top_k=best temp=0.3",
         ], C_ABLATION, "#fff3e0")
arrow(ax, L_x + L_w / 2, p3_y, L_x + L_w / 2, p4_y + p4_h, "best retrieval → fix", C_ABLATION)

# ══════════════════════════════════════════════
# RIGHT COLUMN — Other experiments
# ══════════════════════════════════════════════

# Interaction: Chunk Size × HyDE
int_y = p2_y
int_h = 2.3
draw_box(ax, R_x, int_y, R_w, int_h,
         "INTERACTION — Chunk Size × HyDE", [
             "llama_hyde_cs256   HyDE + cs=256",
             "llama_hyde         HyDE + cs=512   (= phase 3)",
             "llama_hyde_cs1024  HyDE + cs=1024",
         ], C_INTERACTION, "#f3e5f5")
arrow(ax, L_x + L_w, p1_y + p1_h / 2, R_x, int_y + int_h / 2 + 0.5,
      "cross HyDE × chunk", C_INTERACTION)

# LLM Comparison
llm_y = int_y - 3.0
llm_h = 1.7
draw_box(ax, R_x, llm_y, R_w, llm_h,
         "LLM COMPARISON  (same RAG params)", [
             "fair_llama8b       Llama 3.1 8B",
             "fair_gpt35_turbo   GPT-3.5-turbo",
         ], C_COMPARE, "#e0f2f1")

# No-RAG Baselines
nrag_y = llm_y - 2.8
nrag_h = 1.7
draw_box(ax, R_x, nrag_y, R_w, nrag_h,
         "NO-RAG BASELINES  (parametric knowledge only)", [
             "no_rag_llama8b     Llama, no retrieval",
             "no_rag_gpt35_turbo GPT-3.5-turbo, no retrieval",
         ], C_NO_RAG, "#ffebee")

# ══════════════════════════════════════════════
# BOTTOM — Combined Best
# ══════════════════════════════════════════════

best_y = p4_y - 2.0
bw2, bh2 = 14, 1.3
bx2 = (20 - bw2) / 2
rect2 = FancyBboxPatch((bx2, best_y), bw2, bh2, boxstyle="round,pad=0.15",
                        facecolor="#e3f2fd", edgecolor=C_COMBINED, linewidth=2.5)
ax.add_patch(rect2)
ax.text(bx2 + 0.5, best_y + 0.85, "COMBINED BEST", fontsize=13, fontweight="bold", color=C_COMBINED)
ax.text(bx2 + 0.5, best_y + 0.35,
        "tune_llama_best:  cs=256  ov=100  hybrid + reranker  temp=0.1",
        fontsize=10, family="monospace", color="#333")
arrow(ax, L_x + L_w / 2, p4_y, bx2 + bw2 / 2, best_y + bh2,
      "combine best of each phase", C_COMBINED)

# ══════════════════════════════════════════════
# Legend + Run command
# ══════════════════════════════════════════════

legend_y = 1.0
legend_items = [
    (C_BASELINE, "Baseline"),
    (C_ABLATION, "Ablation (1 param)"),
    (C_INTERACTION, "Interaction"),
    (C_COMPARE, "LLM comparison"),
    (C_NO_RAG, "No-RAG"),
    (C_COMBINED, "Combined best"),
]
for i, (color, label) in enumerate(legend_items):
    xp = 0.8 + i * 3.2
    rect = FancyBboxPatch((xp, legend_y), 0.4, 0.3, boxstyle="round,pad=0.05",
                           facecolor="white", edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(xp + 0.55, legend_y + 0.15, label, fontsize=9.5, va="center", color="#333")

cmd_y = 2.0
ax.text(10, cmd_y + 0.5, "Run all experiments:", fontsize=12, fontweight="bold",
        ha="center", color=C_HEADER)
ax.text(10, cmd_y,
        "make benchmark-all   →   18 configs × 100 QA = 1800 evaluations",
        fontsize=11, ha="center", family="monospace", color="#333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#ccc"))

plt.tight_layout()
Path("results").mkdir(parents=True, exist_ok=True)
fig.savefig("results/experiment_flow.png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig("results/experiment_flow.pdf", bbox_inches="tight", facecolor="white")
print("Saved to results/experiment_flow.png and .pdf")
