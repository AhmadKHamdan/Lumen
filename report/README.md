# Lumen — Final Report (LaTeX)

Compile with **XeLaTeX + biber** (fontspec/polyglossia require XeLaTeX; the Arabic
abstract uses the Amiri font). On Overleaf: Menu → Compiler → XeLaTeX. Locally:

```
xelatex main && biber main && xelatex main && xelatex main
```

## Before submission — checklist

1. **`pictures/bzu_logo.png`** — copy from the February project (title page needs it).
2. **`[FILL: ...]` markers** (rendered red) — measured numbers only you can provide:
   all in Chapter 7 (end-to-end task tallies, hardware spec, FPS/latency figures).
   Search for `\tofill`.
3. **`[VERIFY ...]` entries in `references.bib`** — four entries where only your
   February `references.bib` has the true source (martinez2014cognitive,
   hossain2010cognitive, chen2015wearable, liu2019spatial, malizia2022design).
   Replace them with your original entries; all other entries are real works but
   deserve a once-over.
4. **Figures to add (optional but recommended)** — the report compiles without them;
   drop into `pictures/` and add `\includegraphics` where useful:
   - phone screenshot: session UI during Object Allocation
   - phone screenshot: navigation scan in progress
   - door detector training curves (from `door_training/runs/`)
   - a photo of a live trial (door approach / obstacle scene)
5. **Arabic abstract** — written by us; have a native reviewer confirm tone.

## Layout

```
main.tex                 preamble, title page, front matter, includes
references.bib           bibliography (IEEE style, biber backend)
chapters/
  abstract.tex           English + Arabic abstracts
  abbreviations.tex
  ch1_introduction.tex   motivation, problem, methodology, contributions, scope
  ch2_background.tex     perception, speech, browser sensing, FSM control
  ch3_related_work.tex   + summary & research gap
  ch4_system_design.tex  design evolution, requirements, architecture, FSM, safety
  ch5_task_families.tex  Object Allocation, Reach Guidance, Navigation (core chapter)
  ch6_implementation.tex stack, engines, speech pipeline, deployment, tests
  ch7_evaluation.tex     door model, behavioral verification, live trials, limits
  ch8_conclusion.tex
```
