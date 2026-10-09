# ToDo 36: replicating the research experiments with the `ruleofthumb` package

Package: `ruleofthumb` 0.0.2, source unchanged

## 1. What we did

ToDo 36 asks whether a plain install of the package can reproduce the headline numbers of the research monorepo's
experiments.

1. **Analysis.** We read every experiment folder in the monorepo (read-only) and wrote one note per
   experiment: the paper's headline numbers and their source, the data, the black box, the old code's hand-picked
   settings, and a plan for the package.
2. **Runs.** Every experiment was rebuilt with the package on the Azure VM (NVIDIA H100), **one run per variant**
   (seed 0). Each experiment ran up to five variants:

   | Variant | What it is |
   |---|---|
   | **Package defaults** | plain `fit_*()` (ToDo step 1) |
   | **autotune** | `autotune()` with its default search (ToDo step 2) |
   | **Package, legacy settings** | the package with the old code's hand-picked settings passed in |
   | **Package, b = 0** | diagnosis only: the package with `b` starting at 0, like the old code |
   | **Old code** | the monorepo's own RoT, imported read-only (torch 2 shim), as a comparison |


## 2. Results: package vs the monorepo

"Paper" is the number published in the paper / committed in the monorepo. "Old code" is our rerun of the monorepo's
code. Pass = within ±0.02 of the paper, or same sign and rank where the paper only has a plot.

| # | Experiment | Paper (monorepo) | Old code rerun | Package defaults | autotune | Package, legacy settings | Why autotune failed |
|---|---|---|---|---|---|---|---|
| 1 | ExplanationExampleLocal (Pima, plot only) | Glucose 1st, Age 2nd, both negative |  | **PASS** (Glucose, Age) | FAIL (Glucose, BloodPressure, Age) | PASS | Picked lr 0.1 for best held-out agreement, which reorders the features |
| 2 | ScientificDiscovery | acc 0.714, Age importance ~0 (last of 8) | PASS: 0.714, Age last, max \|Age\| 0.009 | FAIL: acc 0.760, Age 5th, max \|Age\| 1.12 | FAIL: 0.786, Age 6th | FAIL: 0.773, Age 5th | Old settings (lr 0.005, 1500 epochs) not in its search range, and `b` = -mean still differs |
| 3 | AIAuditing | acc 0.696; is_amazon 0.621, sold_by 0.279 |  | **PASS**: 0.689; 0.603 / 0.280 | FAIL: 0.700; 0.554 / 0.238 | (= defaults) | Agreement went up (0.700) but importances drifted; the winning settings (= defaults) are not in its range |
| 4 | AdversarialAttack (Table S7) | German/COMPAS 100%, CC 89.5-91% | PASS (identical) | **PASS 10/10** (= old code) | German/COMPAS PASS; CC 97.5% (FAIL by being higher) | (= defaults) | CC: picked lr 0.03, which fits the black box better than the paper, so it scores too high |
| 5 | OpenXAI (COMPAS LR bars) | e.g. baseline g[1] 1.0847, los slope -0.003775 | PASS: 1.0871, -0.003703 | FAIL: bars ~1.6x (1.183, -0.00618) | FAIL (1.152, -0.00578) | FAIL (0.903, -0.00178) | Old lr 0.01 with 500 epochs not reachable, and `b` = -mean still differs |
| 6 | Resume (plot only) | 6 job-category signs and ranks | PASS | **PASS** (sizes ~0.4x) | PASS (sizes ~50x smaller) | PASS (= old code to 3-4 decimals) | Passed; candidates tied, so its pick was arbitrary (importances ~50x smaller) |
| 7 | Movie | wAUROC 0.72 | PASS: 0.7238 | FAIL: 0.6788 | FAIL: 0.6764 | **PASS: 0.7267** | Old lr 1e-6, batch 16 and `l1_penalty=0` are outside its range (l1 not searchable) |
| 8 | Judicial | wAUROC 0.7669, PR-AUC 0.7742, fidelity 0.9505 | PASS: 0.7659 / 0.7735 / 0.9505 | FAIL: 0.7414 / 0.7517 / 0.891 | out of memory; 0.749 with batch 64 | **PASS: 0.7507-0.7572** | Runs out of memory; old lr 1e-5 and `l1_penalty=0` outside its range |
| 9 | Pets-5000 (cat/dog, GPT labels) | test acc 97.60%, train 98.53% | PASS: 97.60% test | **PASS: 97.04%** | PASS: 97.04% | PASS: 97.36% | Passed |
| 10 | Runtimes | | | | | skipped | skipped |

**Score:** package defaults pass **5 of 9** (Local, AIAuditing, Adversarial, Resume, Pets). With the old code's
settings passed in, the package passes **7 of 9** (adds Movie and Judicial). ScientificDiscovery and OpenXAI only
come close with the "b = 0" diagnosis, which reproduces the old code. The old code itself reproduces the paper
everywhere we could check it.

## 3. Why the package differs from the monorepo

1. **Where `b` starts (tabular).** The old code meant to start `b` at minus the data mean, but its line
   `self.b[None,:].data = -mean` writes to a temporary copy and does nothing, so `b` starts at 0. The package really
   starts at minus the mean. On unscaled or MinMax data this changes the result:
   - ScientificDiscovery: the package with `b` = 0 reproduces the old code to 4 decimals (on CPU). With minus the
     mean, the surrogate copies the black box better (agreement 0.96 vs 0.82) but Age is no longer ~0. **The paper's
     "Age ~ 0" result depends on the old code's no-op.**
   - OpenXAI: `b` = 0 plus the old learning rate 0.01 matches the baseline and slope; the package's own start does not.
   - No effect on standardised data (Local, AIAuditing, Adversarial).
2. **Default settings for text.** The package defaults `l1_penalty=0.01` and learning rate 0.05; the old text code
   used `l1_penalty=0` and much smaller learning rates (1e-5, 1e-6). With the old settings the package matches the
   old code (Resume to 3-4 decimals, Movie 0.7267 vs 0.7238).
3. **Default learning rate for tabular.** 0.05 vs the old code's 0.01 (OpenXAI) or 0.005 (Local, Scientific).
4. **Memory on long text.** Package text training builds one huge tensor and copies the whole dataset to the GPU, so
   Judicial and Movie cannot run at the paper's batch sizes without a workaround (section 7).
5. **Small model-form differences (text, image).** Judicial is ~0.01 below the old code even with identical
   settings (different `b` shape, warm-up trains different parameters). Pets: the package clamps `b` every step and
   drops whole pixels; small effect, all variants still pass.

## 4. autotune

autotune never beat the package defaults on a paper target, and in places it made things worse (Local ranking,
Resume importances ~50x smaller, OpenXAI, AIAuditing importances). Reasons:

- It picks the candidate that agrees best with the black box, but the paper's headlines are about the explanations.
  Agreement often went up while the explanation targets did not.
- Its search range excludes every setting that reproduced the paper (old learning rates, 1500 epochs, full batch,
  `l1_penalty=0`) and even the package's own defaults (lr 0.05, 500 epochs).
- On easy problems many candidates tie (7 of 8 at agreement 1.0 on Adversarial) and the first one tried wins.
- It costs 10-50x a single fit, runs out of memory on long text, and one failing candidate stops the whole search.

So making tuning the default (ToDo 37) would not bring the paper's results back with today's autotune.

## 5. Package bugs and gaps (24 items, none fixed)

### Bugs / behaviour differences

| # | Finding | Where | Evidence | Experiments hit | Decision |
|---|---|---|---|---|---|
| 1 | **`b` starts at -mean in the package, at 0 in the old code** (the old code's `self.b[None,:].data = -mean` writes to a temporary view, a no-op; `b` stays 0, then clamped). | package `core.py` `RoT.fit` (`self.b.copy_(-points.mean(0))`); legacy `rot_class.py:100` | ScientificDiscovery: package with `b`=0 reproduces the old code to 4 decimals (acc 0.7143, Age max\|imp\| 0.0092). With -mean: higher fidelity to the black box (0.95 vs 0.79) but Age is no longer ~0 (rank 4-6). No effect on standardised data. | ScientificDiscovery, OpenXAI (even on MinMax data), Pets | **Keep package as is** (user). Record that the paper's "Age ~ 0" comes from the old code's no-op. |
| 2 | **Text trains on the SUM of token scores, predicts on the MEAN.** | `text.py` `RoTText.loss` (`.sum(-1)` + g) vs `RoTText.score` (mean over length + g) | Judicial: \|g\| <= 0.028, fidelity identical under both rules. Movie: changes 2.4% of predictions (legacy_hp), 0.6% (defaults). Resume: 0%. Affects `train_agreement_` and autotune scoring, not wAUROC. | Judicial, Movie, Resume | **Record only, do not fix** (user). |
| 3 | **Text training runs out of GPU memory on long documents** (dense batch x classes x tokens x embedding tensor). | `text.py:108-158` | Judicial: 11.27 GiB allocation at batch 128, 5.63 GiB at 64; batch 32 is the maximum. Movie defaults OOM at batch 5000 and 500. An identical token-level (einsum) loss matches to ~1e-7 and is ~10x faster. | Judicial, Movie | open |
| 4 | **Text fit copies the whole dataset to the GPU and keeps two batches alive.** | `text.py:179` (`RoTText.fit`), `core.py` `training_loop` | Judicial: training set 17.65 GiB on GPU; full batch OOMs even with the low-memory loss. Padding wastes 3.8x (mean 3,056 vs max 11,538 tokens). | Judicial, Movie | open |
| 5 | **`autotune` cannot run on long-document text.** | `tune.py` | Judicial: stock OOMs on the first candidate; works only with batch limited to 64 (wAUROC 0.749). | Judicial, Movie | open |
| 6 | **~0.01 gap at identical settings (text).** | `text.py` model form and pretrain | Judicial, legacy settings: package 0.757 / 0.762 vs old code 0.766 / 0.772. Package a*(x+b) with b of shape (K, E); old `RoT_text` x.a + one scalar b per class; old pretrain trains only g. | Judicial | open |

### Gaps (missing features / limits)

| # | Gap | Where | Evidence | Experiments hit |
|---|---|---|---|---|
| 7 | **`autotune` cannot pick the package's own defaults or any legacy setting** (lr 0.05 / 0.01 / 0.005 / 5e-4 / 1e-5 / 1e-6, 500 / 1500 / 35 / 128 / 1 epochs, full batch, l1 0). `l1_penalty`, `pretrain_epochs` not searchable. | `tune.py:22-27` | Local: autotune picks lr 0.1 and fails the headline (defaults pass). | all |
| 8 | **`autotune` has no tie-break;** ties are not reported. | `tune.py` (`trials.sort`) | Adversarial: 7 of 8 candidates tie at 1.0; first sampled wins. | Adversarial, Resume |
| 9 | **Package defaults trigger the oversized-batch warning** on small data (batch 5000 > N). | `explain.py:61-72` | Adversarial German (900 rows), CC (1794); COMPAS (4937), Resume; inside autotune for batch 2000. | most |
| 10 | **No public unsigned importance and no public binary baseline (`g[1]`).** | `explain.py` `get_explanation` | Adversarial used `e.model.importance(x)`; OpenXAI used `exp.model.g[1]`. | Adversarial, OpenXAI |
| 11 | **No long-document embedding** (`embed_texts` truncates, no window/stride); no span-merge or rationale-AUROC helper. | `embed.py` | Judicial: 510-token / stride-410 windowing ported into the harness. | Judicial |
| 12 | **`embed_images` cannot match the legacy transform** (resizes to 224, legacy 256 -> crop 224; refuses `transform=` with a backbone). | `vision.py` | Pets: package features differ from legacy (mean cosine 0.85). | Pets |
| 13 | **`plot.saliency` has no shared normalisation bounds; `plot.force` has no `link="logit"` or raw-value display.** | `plot.py:260`, `plot.force` | Pets saliency and Local force plot cannot be redrawn as in the paper. | Pets, Local |
| 14 | **`text` extra allows `transformers>=4.30`; ModernBERT needs >=4.48.** | `pyproject.toml` | From analysis. | text |
| 15 | **README says autotune scores by "reveal fidelity"; the code default is agreement.** | `README.md:226` vs `tune.py:149` | From analysis. | docs |
| 16 | **`autotune` has no per-candidate error handling:** one OOM candidate kills the search. | `tune.py` candidate loop | Movie: batch 500/2000 candidates OOM; only `batch_size=[64]` works. | Movie, Judicial |
| 17 | **Package defaults are poor for text** (`l1_penalty=0.01`, lr 0.05; old code l1 0). | `explain.py` `fit_text` | Movie: defaults wAUROC 0.679 / fidelity 0.714 vs legacy settings 0.727; Resume: importance sizes unstable. | Movie, Resume, Judicial |
| 18 | **Warning rounds before comparing** ("low train agreement (0.750 < 0.75)"). | `explain.py` `_attach_fit_quality` | Movie defaults. | cosmetic |
| 19 | **Text dropout granularity** (package whole tokens, Movie old code single embedding values); not expressible, no measurable effect (0.7267 vs 0.7238). | `text.py` `stochastic_importance` | Movie. | Movie (ToDo 38) |
| 20 | **Default learning rate 0.05 vs the old code's 0.01** (with the `b` start, explains the OpenXAI ~1.6x bars). | `explain.py:396` | OpenXAI: lr 0.01 + b=0 gives g[1] 1.087 (paper 1.0847); defaults 1.183. | OpenXAI |
| 21 | **`get_explanation` returns only the class-1 map for binary tasks; no class option.** | `explain.py` | Pets figure needs the cat (class-0) map; read from private `e.model._unit_importance(X)[:, 0]`. | Pets |
| 22 | **`embed_images` is CPU-only, one image per call, slow at default threads.** | `vision.py` (`trunk(pixels[None].cpu())`) | Pets: 3.6 s/image at 40 threads on a busy VM, 110 ms/image at 4 threads; legacy-style GPU path 3 s for all 5000. | Pets |
| 23 | **Fitting on file paths vs the same precomputed arrays gives different results.** | `explain.py` `fit_image` paths branch | Pets: train 99.79% vs 99.33%, test 97.28% vs 96.24%. Not investigated. | Pets |
| 24 | **`b` is clamped every step in the package; the old image code never clamps.** | `core.py` `project` | Pets (with `b` start and dropout granularity, not settable). | Pets |
