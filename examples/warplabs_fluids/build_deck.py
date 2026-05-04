"""
Build a 5-slide NVIDIA-style tech deck for the warplabs-fluids Phase 1 results.

Run from examples/warplabs_fluids/:
  python build_deck.py
Outputs: warplabs_fluids_deck.pptx
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

HERE   = Path(__file__).parent
BENCH  = HERE / "benchmarks"
OUT    = HERE / "warplabs_fluids_deck.pptx"

# ── NVIDIA palette ────────────────────────────────────────────────────────────
GREEN   = RGBColor(0x76, 0xB9, 0x00)   # NVIDIA green
BLACK   = RGBColor(0x00, 0x00, 0x00)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x1A, 0x1A)   # slide background
PANEL   = RGBColor(0x2A, 0x2A, 0x2A)   # content card
GRAY    = RGBColor(0xA0, 0xA0, 0xA0)
GREEN_D = RGBColor(0x4A, 0x7A, 0x00)   # darker green for accents

SW, SH = Inches(13.333), Inches(7.5)   # 16:9 widescreen

# ── helpers ───────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def fill_bg(slide, color=DARK):
    """Fill slide background with a solid colour."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, left, top, width, height, bg=None, border=None):
    """Add a filled rectangle (no text)."""
    shape = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    shape.line.fill.background()
    if bg:
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
    else:
        shape.fill.background()
    if border:
        shape.line.color.rgb = border
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def label(slide, text, left, top, width, height,
          size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT,
          italic=False):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def multiline(slide, lines, left, top, width, height,
              default_size=16, default_color=WHITE, default_bold=False):
    """lines = list of (text, size, color, bold) or just strings."""
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, str):
            text, size, color, bold = item, default_size, default_color, default_bold
        else:
            text = item[0]
            size  = item[1] if len(item) > 1 else default_size
            color = item[2] if len(item) > 2 else default_color
            bold  = item[3] if len(item) > 3 else default_bold

        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return txb


def green_bar(slide, top=0.0, height=0.08):
    """Thin NVIDIA green top bar."""
    box(slide, 0, top, 13.333, height, bg=GREEN)


def bottom_bar(slide):
    box(slide, 0, 7.3, 13.333, 0.2, bg=PANEL)
    label(slide, "NVIDIA  ·  Warp  ·  Confidential",
          0.2, 7.32, 8, 0.18, size=8, color=GRAY)
    label(slide, "warplabs-fluids  |  Phase 1",
          10.5, 7.32, 2.6, 0.18, size=8, color=GRAY, align=PP_ALIGN.RIGHT)


def image(slide, path, left, top, width):
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def section_chip(slide, text, left, top, width=2.6):
    """Small green pill label."""
    box(slide, left, top, width, 0.28, bg=GREEN)
    label(slide, text, left + 0.08, top + 0.02, width - 0.1, 0.26,
          size=10, bold=True, color=BLACK)


# ── Slide 1 — Title ───────────────────────────────────────────────────────────
def slide_title(prs):
    s = blank_slide(prs)
    fill_bg(s)
    green_bar(s)

    # large green accent stripe on left
    box(s, 0, 0.08, 0.55, 7.42, bg=GREEN)

    # rotated vertical text "WARP" — approximated as stacked label
    label(s, "WARP", 0.04, 2.8, 0.48, 1.6, size=22, bold=True, color=BLACK,
          align=PP_ALIGN.CENTER)

    # title block
    label(s, "Porting JaxFluids to NVIDIA Warp",
          0.75, 1.6, 11.5, 1.2, size=38, bold=True, color=WHITE)
    label(s, "Phase 1  ·  1-D Compressible Euler",
          0.75, 2.85, 11.5, 0.6, size=22, bold=False, color=GREEN)

    # divider
    box(s, 0.75, 3.55, 10.5, 0.04, bg=GRAY)

    # sub-bullets
    multiline(s, [
        ("Scheme:  WENO3 reconstruction  +  HLLC Riemann solver  +  SSP-RK2",        16, GRAY,  False),
        ("Validated against exact Riemann solution  ·  4-backend V&V",                16, GRAY,  False),
        ("Benchmark:  JAX CPU  /  JAX CUDA  /  Warp CPU  /  Warp CUDA",              16, GRAY,  False),
        ("Result:  Warp CUDA  2.4 – 1.6×  faster than JAX CUDA (fused kernels)",     16, GREEN, True),
    ], 0.75, 3.7, 11.5, 2.5, default_size=16)

    # bottom meta
    bottom_bar(s)
    label(s, "RTX 5000 Ada  ·  Warp 1.12.1  ·  JAX 0.6.2  ·  WSL2 Ubuntu 22.04",
          0.75, 6.95, 10, 0.3, size=9, color=GRAY)


# ── Slide 2 — Strategy & Approach ────────────────────────────────────────────
def slide_strategy(prs):
    s = blank_slide(prs)
    fill_bg(s)
    green_bar(s)

    label(s, "Strategy & Approach", 0.4, 0.15, 10, 0.6, size=26, bold=True, color=WHITE)
    section_chip(s, "METHODOLOGY", 0.4, 0.82)
    bottom_bar(s)

    # LEFT column ── Why
    box(s, 0.35, 1.18, 3.9, 5.75, bg=PANEL)
    label(s, "Why Warp?", 0.55, 1.28, 3.5, 0.45, size=17, bold=True, color=GREEN)
    multiline(s, [
        ("• Python-first GPU kernels — no CUDA C required",            14, WHITE, False),
        ("",),
        ("• @wp.func inlines to registers; @wp.kernel marks global-memory boundaries", 14, WHITE, False),
        ("",),
        ("• Kernel fusion collapses 6 launches → 2 per timestep",     14, GREEN, True),
        ("",),
        ("• Memory pool (cudaMallocAsync) — O(1) allocs inside loops", 14, WHITE, False),
        ("",),
        ("• Targets production physics workloads at NVIDIA",           14, WHITE, False),
    ], 0.55, 1.78, 3.55, 4.8, default_size=14)

    # MIDDLE column ── Scheme
    box(s, 4.45, 1.18, 4.2, 5.75, bg=PANEL)
    label(s, "Numerical Scheme", 4.65, 1.28, 3.8, 0.45, size=17, bold=True, color=GREEN)
    multiline(s, [
        ("Equations",    13, GRAY,  True),
        ("1-D compressible Euler  [ρ, ρu, E]", 13, WHITE, False),
        ("",),
        ("Reconstruction",  13, GRAY, True),
        ("WENO3 on primitive variables (Jiang-Shu 1996)", 13, WHITE, False),
        ("",),
        ("Riemann solver",  13, GRAY, True),
        ("HLLC  (Toro 2009, Einfeldt wave speeds)", 13, WHITE, False),
        ("",),
        ("Time integration",  13, GRAY, True),
        ("SSP-RK2  (2 stages, CFL=0.4)", 13, WHITE, False),
        ("",),
        ("Ghost cells",  13, GRAY, True),
        ("ng=2 embedded in state array", 13, WHITE, False),
        ("",),
        ("Precision",  13, GRAY, True),
        ("float32  (matches JAX default)", 13, WHITE, False),
    ], 4.65, 1.78, 3.9, 4.8, default_size=13)

    # RIGHT column ── Kernel arch
    box(s, 8.85, 1.18, 4.1, 5.75, bg=PANEL)
    label(s, "Kernel Architecture", 9.05, 1.28, 3.7, 0.45, size=17, bold=True, color=GREEN)
    multiline(s, [
        ("Fused kernel (current)",  13, GREEN, True),
        ("",),
        ("1 launch per RK stage  →  2 / step", 13, WHITE, False),
        ("",),
        ("Thread i computes both interface", 12, GRAY, False),
        ("fluxes in registers (5-cell stencil):", 12, GRAY, False),
        ("",),
        ("  F_left  = WENO3+HLLC(i-2..i+1)", 12, WHITE, False),
        ("  F_right = WENO3+HLLC(i-1..i+2)", 12, WHITE, False),
        ("",),
        ("No global flux array allocated.", 12, GRAY, False),
        ("BC handled inline (clamp / modulo).", 12, GRAY, False),
        ("",),
        ("Unfused baseline:  6 launches / step", 12, GRAY, True),
        ("  bc  +  flux  +  update  ×  2 stages", 12, GRAY, False),
    ], 9.05, 1.78, 3.75, 4.8, default_size=12)


# ── Slide 3 — Correctness ─────────────────────────────────────────────────────
def slide_correctness(prs):
    s = blank_slide(prs)
    fill_bg(s)
    green_bar(s)

    label(s, "Correctness  ·  Sod Shock Tube V&V", 0.4, 0.15, 12, 0.6,
          size=26, bold=True, color=WHITE)
    section_chip(s, "EXACT RIEMANN SOLUTION", 0.4, 0.82, width=3.2)
    bottom_bar(s)

    # profile image — spans most of the width
    img = BENCH / "sod_profiles.png"
    if img.exists():
        image(s, img, 0.35, 1.15, 8.9)

    # right panel — explanation + numbers
    box(s, 9.45, 1.15, 3.55, 5.75, bg=PANEL)
    label(s, "What is Sod?", 9.65, 1.25, 3.2, 0.4, size=15, bold=True, color=GREEN)
    multiline(s, [
        ("Membrane at x=0.5 bursts at t=0.", 12, WHITE, False),
        ("Three waves emerge:", 12, WHITE, False),
        ("  ← rarefaction fan", 12, GRAY, False),
        ("  → contact discontinuity", 12, GRAY, False),
        ("  → shock wave", 12, GRAY, False),
        ("Exact Riemann solution available", 12, GRAY, False),
        ("(Toro 2009) — ideal for V&V.", 12, GRAY, False),
    ], 9.65, 1.7, 3.2, 2.0, default_size=12)

    label(s, "L1 errors  (N=512, t=0.2)", 9.65, 3.75, 3.2, 0.35,
          size=13, bold=True, color=GREEN)

    # error table rows
    rows = [
        ("Backend",     "L1(ρ)",    "L1(u)",    "L1(p)"),
        ("JAX CPU",     "1.73e-3",  "3.19e-3",  "1.18e-3"),
        ("JAX CUDA",    "1.73e-3",  "3.19e-3",  "1.18e-3"),
        ("Warp CPU",    "1.73e-3",  "3.19e-3",  "1.18e-3"),
        ("Warp CUDA",   "1.73e-3",  "3.19e-3",  "1.18e-3"),
    ]
    col_x = [9.65, 10.55, 11.28, 12.01]
    col_w = [0.88,  0.71,  0.71,  0.71]
    for r_idx, row in enumerate(rows):
        y = 4.15 + r_idx * 0.38
        bg_r = PANEL if r_idx % 2 == 0 else RGBColor(0x32, 0x32, 0x32)
        box(s, 9.65, y, 3.2, 0.36, bg=bg_r)
        for c_idx, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
            clr = GREEN if r_idx == 0 else (WHITE if c_idx == 0 else GRAY)
            sz  = 10 if r_idx == 0 else 10
            label(s, cell, cx + 0.04, y + 0.04, cw, 0.3, size=sz,
                  bold=(r_idx == 0), color=clr)

    label(s, "All four backends produce identical results.",
          9.65, 6.12, 3.2, 0.35, size=11, bold=True, color=GREEN)


# ── Slide 4 — Performance & Throughput ───────────────────────────────────────
def slide_performance(prs):
    s = blank_slide(prs)
    fill_bg(s)
    green_bar(s)

    label(s, "Performance  ·  Throughput Scaling", 0.4, 0.15, 12, 0.6,
          size=26, bold=True, color=WHITE)
    section_chip(s, "GPU THROUGHPUT", 0.4, 0.82)
    bottom_bar(s)

    img = BENCH / "plot_throughput.png"
    if img.exists():
        image(s, img, 0.35, 1.15, 8.9)

    # right callout panel
    box(s, 9.45, 1.15, 3.55, 5.75, bg=PANEL)
    label(s, "Key Results", 9.65, 1.25, 3.2, 0.4, size=15, bold=True, color=GREEN)

    multiline(s, [
        ("Warp CUDA  vs  JAX CUDA",        13, WHITE,  True),
        ("482 vs 298 Mcell/s at 131K cells", 12, GREEN, False),
        ("→  1.62× faster",                 13, GREEN,  True),
        ("",),
        ("Kernel fusion impact",            13, WHITE,  True),
        ("6 launches/step → 2/step",        12, GRAY,   False),
        ("~3× speedup on Warp CUDA",        12, GREEN,  False),
        ("",),
        ("GPU crossover vs CPU",            13, WHITE,  True),
        ("Warp CUDA > JAX CPU at N~2K",     12, GRAY,   False),
        ("Warp CUDA > JAX CUDA at N~2K",    12, GRAY,   False),
        ("",),
        ("Both GPUs still scaling",         13, WHITE,  True),
        ("at 131K cells — not yet",         12, GRAY,   False),
        ("bandwidth-limited",               12, GRAY,   False),
        ("",),
        ("CPU backends flat",               13, WHITE,  True),
        ("Warp: single-threaded LLVM",      12, GRAY,   False),
        ("JAX: saturates ~30 Mcell/s",      12, GRAY,   False),
    ], 9.65, 1.72, 3.2, 5.0, default_size=12)


# ── Slide 5 — Memory + Next Steps ────────────────────────────────────────────
def slide_memory_roadmap(prs):
    s = blank_slide(prs)
    fill_bg(s)
    green_bar(s)

    label(s, "Memory Footprint  ·  Roadmap", 0.4, 0.15, 12, 0.6,
          size=26, bold=True, color=WHITE)
    section_chip(s, "MEMORY + NEXT STEPS", 0.4, 0.82, width=3.0)
    bottom_bar(s)

    img = BENCH / "plot_memory.png"
    if img.exists():
        image(s, img, 0.35, 1.15, 7.3)

    # memory insight panel
    box(s, 7.85, 1.15, 2.15, 2.85, bg=PANEL)
    label(s, "Memory", 8.0, 1.25, 1.85, 0.35, size=14, bold=True, color=GREEN)
    multiline(s, [
        ("Warp: 32 MiB flat",       12, WHITE, True),
        ("cudaMallocAsync pool —",   11, GRAY,  False),
        ("pre-allocated once,",      11, GRAY,  False),
        ("O(1) reuse.",              11, GRAY,  False),
        ("",),
        ("JAX: grows linearly",      12, WHITE, True),
        ("Matches theoretical min.", 11, GRAY,  False),
        ("(with PREALLOCATE=false)", 11, GRAY,  False),
        ("",),
        ("Crossover ~3M cells.",     11, GREEN, False),
    ], 8.0, 1.65, 1.9, 2.25, default_size=11)

    # Roadmap
    box(s, 7.85, 4.1, 5.15, 2.8, bg=PANEL)
    label(s, "Roadmap", 8.05, 4.2, 4.7, 0.4, size=15, bold=True, color=GREEN)

    phases = [
        ("✓", "Phase 1  COMPLETE",
         "1-D Euler  ·  WENO3-HLLC-RK2  ·  float32\nKernel fusion  ·  4-backend V&V  ·  scaling benchmarks",
         GREEN),
        ("2", "Phase 2  NEXT",
         "2-D Euler  ·  Strang dimensional splitting\nKelvin-Helmholtz V&V  ·  2-D N-scaling",
         WHITE),
        ("3", "Phase 3  PLANNED",
         "3-D compressible Navier-Stokes\nTaylor-Green vortex V&V  ·  full turbulence benchmark",
         GRAY),
    ]
    for i, (num, title, detail, tcol) in enumerate(phases):
        y = 4.68 + i * 0.72
        # number bubble
        box(s, 8.0, y, 0.32, 0.32, bg=GREEN if num == "✓" else PANEL,
            border=GREEN)
        label(s, num, 8.0, y, 0.32, 0.32, size=12, bold=True,
              color=BLACK if num == "✓" else GREEN, align=PP_ALIGN.CENTER)
        label(s, title, 8.4, y, 4.3, 0.28, size=12, bold=True, color=tcol)
        label(s, detail, 8.4, y + 0.3, 4.3, 0.38, size=10, color=GRAY)

    # Note at bottom right
    label(s, "Phase 3 target: 3-D turbulence at GPU scale",
          7.85, 6.9, 5.1, 0.3, size=9, color=GREEN, italic=True)


# ── assemble ──────────────────────────────────────────────────────────────────
prs = new_prs()
slide_title(prs)
slide_strategy(prs)
slide_correctness(prs)
slide_performance(prs)
slide_memory_roadmap(prs)

prs.save(str(OUT))
print(f"Saved -> {OUT}")
