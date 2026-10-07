# ruleofthumb: ToDo (plain-language version)

The list of planned work for the `ruleofthumb` pip package
(repo `https://github.com/KaiRawal/Rule-of-Thumb`, PyPI `ruleofthumb-rot`,
docs `https://ruleofthumb.readthedocs.io/`), sorted by priority.

## How to read this file

- **P0** = do first, **P1** = next, **P2** = later, **P3** = maybe, low priority.
- **Item numbers never change.** Gaps in the numbering are finished or removed items.
- **Research** items are investigations or experiments. They may not end in code.
- Each item says **what** it is, **why** it matters, **when it counts as done**, and its **status**.

## Words used below

- **Black box**: the model being explained (we only see its answers, never its insides).
- **Surrogate / RoT**: the small, simple model this package trains to copy the black box. Its weights are the explanation.
- **Agreement / fidelity**: how often the surrogate gives the same answer as the black box.
- **Reveal curve**: show the input to the surrogate most-important-part first and track how fast it gets the right answer. Faster means a better explanation.
- **Floor**: the minimum score a test accepts.
- **Seed**: the number that fixes the randomness of a run, so it can be repeated.
- **Monorepo**: the original research code this package was built from.
- **Backbone / embedding**: a pretrained model that turns raw text or images into numbers first (e.g. ModernBERT for text, MobileNet for images).
- **Dropout**: during training, parts of the input are hidden at random so the surrogate learns from partial views.

---

## P0: do first

### 22. Make the test suite pass on any machine (P0)

- **What:** long training runs can land in slightly different places on different computers, because each machine's math libraries round numbers a little differently. So a test that says "score must be at least X" can pass on one machine and fail on another. The fix has two parts:
  - **Tests that check the output of a trained model** (explanations, plots, save/load, recovering a known answer) load saved weights instead of training. The result is then exact everywhere, so they can use tight checks. The weights are raw `.pt` files, made by `tests/mint_unit_weights.py`.
  - **Tests that must train live** (repeatability with a seed, tuning, raw-input fits, comparisons between fits) keep training. Their floors are measured by running many times (several seeds, thread counts and devices): floor = worst result seen minus a small margin, written down in `TEST_SUITE.md`.
  - Also: make training deterministic where possible, and use bigger test datasets (digits 500 → 1797, COMPAS 800 → 2000, reviews 31 → ~60, image digits 500 → 1000+).
- **Why:** a test suite that fails on some machines for no real reason can't be trusted.
- **Done when:** CI is green, saved-weight tests use tight checks, and live tests use measured floors.
- **Status:** in progress. Our work is on branch `todo22` (not merged yet).

### 36. Reproduce the research experiments with the package (research, P0)

- **What:** the monorepo has 9 hand-tuned experiments: adversarial, scientific, ai_auditing, explanation_example (local and remote), openxai, resume, movie, judicial and runtimes. A plain install of the package (using `fit()` and `autotune`) must reach the same headline numbers.
- **How**, one experiment at a time, cheapest first (Local → Scientific → OpenXAI → Adversarial → Auditing → Resume → Movie → Judicial → Pets-5000):
  1. Try `fit()` with defaults and record the gap.
  2. Try `autotune` and record the gap.
  3. Any setting the research code hand-picked that autotune can't find becomes a ticket to widen autotune's search.
  4. Translate old habits: the `-1` padding marker becomes `mask=`, unit vs element granularity, and "always 2 classes" becomes the real class count.
  5. Every bug or missing feature found becomes package work, done until the experiment passes.
- **Pass:** headline fidelity / wAUROC within ±0.02, or within the paper's own error bars if those are wider.
- **Rules:**
  - Never call paid APIs again; reuse the saved GPT CSVs.
  - Never rerun SHAP/LIME; compare against their saved outputs.
  - Run the heavy experiments last.
  - Never change the monorepo; only read its data and saved outputs.
  - Keep working notes in `scratch/replication/` (not committed).
  - HateXplain and SALICON are out of scope (already done the other way round).
- **Gaps already known:** hand-tuned settings everywhere; padding marker vs mask; `RoT_additive`/RBF only in the OpenXAI patch; the saliency power/trim view is missing; extra (non-input) features must be added by hand.
- **Why:** proves the package really does what the research code did, without hand-tuning.
- **Done when:** all 9 pass with no open gaps, and every package fix they needed has shipped.
- **Status:** not started.

### 37. Make tuning the default (P0)

- **What:** `fit`, `fit_tabular`, `fit_text` and `fit_image` get a `tune=` option, on by default:
  - `tune=True` runs `autotune()`: it tries several training settings automatically and keeps the best.
  - A dict customises the search (`space`, `n_candidates`, `scoring`).
  - `tune=None` / `False` runs today's single fixed training (for tests and when speed matters).
  - `autotune` must never call itself in a loop (recursion guard).
  - The result is still an `Explainer`, now with `best_params_` and `best_score_`. `autotune()` itself keeps returning `AutotuneResult`.
- **Takes over** item 29 (stop training early and drop bad candidates, which keeps the extra cost down) and item 30 (`evaluate()`).
- **Side work:** tests that check fixed behaviour pass `tune=None` (otherwise the suite gets about 8× slower); examples and docs get re-run; version bump, changelog and a README note ("`tune=None` gives the old behaviour").
- **Why:** most users never tune, so they get worse fits than they could.
- **Done when:** the test suite is green with tuning as the default (with its run time reported), the docs recommend `tune=True`, and items 29/30 are either done here or explicitly moved out.
- **Status:** not started upstream. Our work is on branch `todo37` (not merged yet).

### 38. Dropout audit: write down why, don't change it (P0)

- **What:** the package hides whole units during training (a full token or a full pixel). The research code mostly hid single values. The picture differs per data type:
  - **Tabular:** no difference.
  - **Text:** the research code disagreed with itself. The package matches the Judicial version (whole tokens).
  - **Images:** a real difference. The research code hid each colour channel of each pixel separately; the package hides whole pixels.
- **Decision already made: keep the package's way.**
  - At scoring time whole units are revealed, so training should hide whole units too.
  - Only whole-unit hiding at a 0.5 rate makes every partial view equally likely (see item 28).
  - It works cleanly with padding and needs 576× fewer random draws for images.
  - A test showed switching would gain only +0.002 (noise) and would break every saved weight and test floor.
- **Work:**
  1. Write this reasoning into code comments and the docs, where it exists nowhere today.
  2. Describe what dropout does to the result, without changing behaviour. Having dropout at all matters a lot: training without dropout closes the image gap, and zeroing the `b` weights ruins fidelity.
- **Why:** so nobody "fixes" this later by mistake.
- **Status:** not started.

---

## P1: next

### 20. Better documentation (P1)

- **What:** the docs are complete, but they undersell the package. Work, in order:
  - a) overview essays: when RoT fits, how it differs from other explainers, how to judge trust;
  - b) a gallery of real-task examples per data type, built from the test data;
  - c) a visual gallery of every plot type;
  - d) benchmarks;
  - e) release-notes and contributing pages, from the existing runbooks.
- **Presentation, keeping the current theme:** logo and favicon, landing image, a "pre-alpha" banner, gallery thumbnails, tidier navigation, a little custom CSS.
- **Not doing:** switching the docs theme.
- **Status:** not started.

### 21. Publication-quality plots with `shap-editorial` (P1)

- **What:** `shap-editorial` draws nicer beeswarm, waterfall, bar and scatter plots from `shap.Explanation` objects. It's young and has a single maintainer, so pin the exact version and treat it as replaceable. It's already a dependency, and our tabular plots already build `shap.Explanation` objects.
- **Steps:**
  1. Check that our objects work with it as-is.
  2. If they do, add editorial versions (`plot.editorial_*` or a `style=` option), starting with tabular waterfall, bar and beeswarm.
- **Decisions needed:** colours (editorial uses grey → red, while ours is red = pushes toward the class, blue = pushes against); multiclass (not supported upstream, so plot one class at a time).
- **Ship with:** tests, a gallery demo (item 20c) and docs updates.
- **Status:** not started.

### 24. Measure explanation quality beyond fidelity (research, P1)

- **What:** "copies the model well" (fidelity) and "matches what humans think is important" (plausibility) are different things and must be measured separately:
  - deletion/insertion curves, always compared against a random baseline;
  - rank-agreement measures;
  - always include simple controls, such as "look at the centre of the image", which often beats learned maps at matching humans.
- **Notes:** short or noisy inputs make it hard for the surrogate to pick up rare words (see item 16). Models that almost always give one answer need special handling before explaining them.
- **Status:** no commitment to build anything yet.

### 29. Stop training when it's done (P1, handled under item 37)

- **What:** instead of a fixed number of epochs, stop when agreement stops improving (with a maximum). Let `autotune` give up early on settings that are clearly losing, instead of training every candidate to the end.
- **Why:** saves time; it's the main way to make tuning-by-default affordable.
- **Related:** item 16 (capacity limits) and item 23 (results differ across machines).

### 30. An `evaluate()` method (P1, handled under item 37)

- **What:** `Explainer.evaluate(...)` reports:
  - a) **fit agreement** on the training data and on a held-back part, warning below 0.75;
  - b) **reveal gap**: how much better the explanation's order is than a random order (only when asked);
  - c) **match with human markings**, only if the user provides them (text first), compared against random importances.
- **Current defaults it records:** `autotune` scores by plain agreement unless `scoring="reveal"` is given, and fitting never runs the reveal machinery.
- **Not in scope:** repeat fits for stability, per-slice breakdowns, centre-guess controls, new dependencies, and scoring by changing inputs to the black box (the package never has the model, only its answers).

### 31. Warn when the surrogate gets a row wrong (P1)

- **What:** if a user passes the model's answers to `get_explanation(..., y_outputs=...)` and the surrogate disagrees on some rows, warn once per call with the number of rows and their indices, because explanations for those rows are meaningless.
  - No error and nothing blocks; nothing changes if `y_outputs` isn't passed.
  - Plot helpers stay silent.
  - Wrong length raises a `ValueError`.
- **Tests:** wrong rows warn with the right indices, correct rows stay quiet, plus multiclass and masked-text cases.
- **Status:** not started.

### 39. Fixes found by the sandbox replication (P1)

- **What:** the RoT-Sandbox runs (3 of 4 experiments passed; the Pima gap of 0.026 stays open as its own ticket) found:
  1. A crash on Apple GPUs (MPS) with float64 input in `predict`, `get_order`, `score` and tuning validation. Fix: convert inputs the way `fit_*` already does, plus tests.
  2. The old defaults differ from the package's (old: 1500 epochs, learning rate 0.005; package: 500 and 0.05). Add a docs table of the old defaults per experiment. A `legacy=` preset, if wanted, belongs to item 37.
  3. The "batch too big" warning repeats for every autotune candidate. Warn once per fit.
  4. Document which steps are random or seeded, and how much results are expected to vary (feeds item 23). No API change.
  5. `plot.saliency` defaults are `power=1.0, trim=2.0`, but the paper used `3.0, 3.0`. Document the paper values; changing the default would break things, so that's deferred.
- **Status:** not started.

### 42. Check a gender-bias study on text-to-image models (research, P1)

- **Status:** blocked until the paper releases its data.
- **What:** the paper "Bias before Pixels" claims that activities shape gender bias in text-to-image models more than objects do (over 9,000 object/activity prompts). Once released:
  - a) fit RoT-text on prompt words → perceived gender, and compare how important activity words are against object words;
  - b) explain their own surrogate with RoT and compare all three rankings.
- **Pass:** the rankings agree within a tolerance fixed in advance.
- **Limits:** don't regenerate all ~9,396 images (explaining their 3 surrogates is enough); the way gender is labelled is itself a variable; log costs as in item 36.

---

## P2: later

### 23. Results that differ across machines (research, P2)

- **What:** long fits on small datasets reach different results on different machines (the reason for item 22). Measure how big the differences are, and try fixes: early stopping, stronger regularisation, weight averaging, ensembles.
- **Done when:** explanations and fidelity come out the same on every machine. Any API change needs a migration note.
- **Status:** not started upstream. A measurement script exists locally (`tests/integration/measure_stability.py`), not committed.

### 25. Compare image explanations against a "centre" guess (research, P2)

- **What:** when checking image explanations against where humans look, a plain "look at the centre" guess beats learned maps. So any claim that saliency finds what people look at must beat the centre guess by a clear margin, or the docs must say plainly that saliency is not gaze prediction.
- **Status:** no commitment to build anything yet.

### 32. Logo and graphics refresh (P2)

- **What:** redraw the logo (light and dark), favicons, README banner, and the hero/card figures (made by `docs-src/_figures/generate_figures.py`, dummy data only).
- **Limits:** static files plus the README header and the docs front page only. No change to `plot.py`, no theme switch, and keep red = toward / blue = against.
- **Done when:** every asset exists in a light/dark pair, every page using them looks right in both themes, and the ReadTheDocs build is green.
- **Status:** not started.

### 33. Publish on conda-forge as well as PyPI (P2)

- **What:** the package isn't on conda-forge yet.
  1. Create the conda package recipe through a staged-recipes PR, after confirming the name (`ruleofthumb-rot` on PyPI vs `ruleofthumb` when imported).
  2. Map the dependencies to conda packages (check that CPU torch, transformers, torchvision, captum and shap are available).
  3. Let conda-forge's bot pick up new versions automatically, and add this step to the release checklist (item 18) and `PUBLISH.md`.
- **Done when:** `mamba install -c conda-forge ruleofthumb-rot` works in a clean environment, and the bot picks up the next release by itself.
- **Cost:** ongoing upkeep of the conda package.
- **Status:** not started.

### 35. Interactive explanation viewers (exploratory, P2)

- **What:** there are no interactive plots today. Prototype two approaches (`anywidget` views and `ipywidgets` sliders around matplotlib), with a reveal-step slider, saliency power/trim sliders and word hover, then pick the better one.
- **Rules:** a new optional extra only; `plot.*` keeps returning normal figures (widgets wrap them); a static version always available; no new required dependencies.
- **Ship:** the decision plus one pilot widget with tests, a gallery demo (item 20c) and docs.
- **Done when:** the pilot works in JupyterLab and VSCode (and Colab/ReadTheDocs if possible).
- **Status:** not started.

### 41. Example: explaining with extra features the model never saw (P2)

- **What:** a new example notebook where RoT is fitted on the inputs plus extra information the black box never used, as in the monorepo's resume experiment, which adds demographic info to each resume before embedding.
- **Limits:** dummy data only, quick on a CPU, like the existing examples `01` to `06`.
- **Status:** not started.

---

## P3: maybe, low priority

### 16. Optional per-position weights (P3, deferred: rich backbones already cover real use)

- **What:** the text and image surrogates use one shared set of weights for every position. In effect they only see a summary: the average word embedding for text, and the total of each channel over the whole image for images.
  - **Good:** explanations don't depend on where things are, and there are few weights to learn.
  - **Limit:** the surrogate can only be as good as that summary is at telling classes apart. Raw single-channel digit images reduce to "total ink", so 10-class fidelity stays near "always guess the most common class". Adding coordinates gets ~0.35, the TinyCNN features ~0.49, and MobileNet features ~0.99. For text, word order is invisible.
- **Advice for users:** use rich backbones (transformer embeddings, MobileNet features) and check the surrogate's accuracy.
- **Only if raw-input multiclass is really needed:** add an opt-in mode with separate weights per position, behind a flag, keeping today's behaviour as the default.
- **Status:** deferred upstream. Our work (`share_weights=`) is on branch `todo-16` (not merged).

### 27. Validate on full datasets at scale (research, P3)

- **What:** the tests use slices (3000 text posts, 150 images) because the development machine can't hold the full embeddings (about 4.5 GB) or finish long fits quickly. Two tracks:
  1. Run the human-annotation benchmarks in full on a big machine (GPU), and record how slice and full results differ.
  2. Train from disk in chunks, so data that doesn't fit in memory still works.
- **Status:** no commitment to build anything yet.

### 34. "Launch in Binder" buttons for all six example notebooks (P3)

- **What:** add the Binder setup (`environment.yml` plus `postBuild`) and launch buttons (README and the docs examples page) for all six notebooks, including the heavy `04_hatexplain` and `05_salicon`.
  - The environment is big (torch, transformers, etc.), so the first launch is slow; document that.
  - Keep the download cells as they are.
  - Check the ~2 GB memory limit by actually launching each notebook.
- **Not doing:** rewriting notebooks to fit Binder. The docs on ReadTheDocs stay the main source.
- **Done when:** all six run start to finish from their button in a fresh Binder session.
- **Status:** not started.

### 40. Explain a fact-checking LLM (research, P3)

- **What:** get predictions from `bespokelabs/Bespoke-MiniCheck-7B` on a benchmark, fit RoT on text embeddings (the model's own hidden states, or the default embedder), and compare the word importances with saliency taken from the model's attention. If that works, repeat it on `https://www.hallucination-probes.com/`.
- **Status:** ambitious, low priority, not started.

---

## Standing task (no priority, repeated every release)

### 18. Release checklist for PyPI

1. Bump the version in 3 places: `pyproject.toml`, `src/ruleofthumb/__init__.py`, and the assertion in `tests/test_explain.py`.
2. Add a changelog entry.
3. Commit, push and tag. CI then builds and uploads (TestPyPI on `main`, real PyPI on `v*` tags), and ReadTheDocs rebuilds.
4. Check each step as described in `PUBLISH.md`.

Last done for version 0.0.2 (commit `b4c32d4`).

## Done / parked

### 28. Dropout rates other than 0.5 (marked done: `8a9b902`)

- Dropout is fixed at 0.5 on purpose: only 0.5 makes every partial view equally likely. No constructor, factory or `autotune` search accepts a `dropout_rate`, and `test_dropout_rate_is_not_tunable` keeps it that way.
- Reopen only if the method is ever extended to uneven reveal patterns. Then add the option behind a flag, put it back into the search, and reweight reveal scoring to match.

## Not doing (non-goals)

- No bitstring / partial-information sampling scripts: RoT already works per token or pixel, so they aren't needed.

---

## Changelog (newest first)

- **v0.0.2**: padding clean-up and safety checks.
  - Text/image functions take only `mask=` (`lengths=` and `attention_mask=` were removed; use `lengths_to_mask`).
  - New `ruleofthumb.vision` module provides the default image backbone, and `embed_texts` gets a pinned `revision=`.
  - `autotune` passes `n_classes` and other settings through.
  - Clear errors for out-of-range labels, warnings for oversized batches and low agreement, and read-only arrays are accepted.
  - Save files load only with the same package version that wrote them.
- **v0.0.1**: first public pre-alpha; the version was reset from the internal 0.2.19, under the PyPI name `ruleofthumb-rot`. New build backend (`uv_build`), a full docs website on ReadTheDocs, GitHub Actions (tests, TestPyPI, PyPI on tags), and type hints on 5 public functions. No behaviour change.
- **v0.2.19**: optional non-linear explanations (`nonlinear="rbf"` or `"hinge"`); the default stays linear.
- **v0.2.18**: new `plot` module: SHAP-style plots for tabular data, word highlighting and word clouds for text, saliency overlays for images.
- **v0.2.17**: `autotune`: automatic search for good training settings.
- **v0.2.16**: save and load explainers (`Explainer.save`, `load_explainer`).
- **v0.2.15**: `fit_image` accepts image file paths directly.
- **v0.2.14**: `fit_text` accepts raw strings directly (ModernBERT embeddings by default).
- **v0.2.13**: `embed_texts` turns raw strings into embeddings.
- **v0.2.12**: more and stronger integration tests (COMPAS, wine, GPT cat/dog and more), each checking the surrogate's accuracy and known important features.
- **v0.2.11**: integration tests on real datasets and models, saved in the repo, so the tests train nothing and download nothing.
- **v0.2.10**: one `Explainer` class plus `fit` / `fit_tabular` / `fit_text` / `fit_image` replace the old wrapper classes (breaking).
- **v0.2.9**: `device=` option (auto picks cuda, then mps, then cpu).
- **v0.2.8**: text explanations and reveal scoring work with more than 2 classes.
- **v0.2.7**: tabular explanations are signed and comparable to SHAP; unused helpers removed.
- **v0.2.6**: `n_classes=` option (it was fixed at 2).
- **v0.2.5**: `seed=` option for repeatable fits.
- **v0.2.4**: training settings exposed as options (`pretrain_epochs`, `weight_decay`, `swa_burn_in`, `l1_penalty`).
- **v0.2.3**: `get_order` breaks ties the same way every time.
- **v0.2.2**: `mins` / `maxs` belong to each model instead of being shared.
- **v0.2.1**: reveal curves step one token or pixel at a time by default (`granularity="element"` gives the old behaviour).
- **v0.2.0**: explicit masks replace the `-1` padding marker (breaking).
- **v0.1.x**: first port of the research code, with dead code removed.
