# Third-party licenses

Except where noted below, the repository is MIT-licensed (see [../LICENSE](../LICENSE)). It contains these
third-party components:

| Component | Files | License |
|---|---|---|
| [adam-atan2](https://github.com/imoneoi/adam_atan2) 0.0.3, Copyright (c) its authors; ported to PyTorch (modified) | `models/adam_atan2_torch.py`, reference in `tests/test_adam_atan2_torch.py` | Apache License 2.0, [Apache-2.0.txt](Apache-2.0.txt) |
| [d3](https://d3js.org) v7.9.0, Copyright 2010-2023 Mike Bostock; inlined unmodified | `results/sudoku-contractive/report/TRM_contraction_benchmark.html` | ISC, [ISC-d3.txt](ISC-d3.txt) |

Code and data inherited from upstream TRM also include material from earlier projects, for which upstream TRM gives
no separate license notice. That material keeps the terms of its original sources:

* Upstream TRM's code is, per its README, based on the [HRM code](https://github.com/sapientinc/HRM) and the
  [HRM analysis code](https://github.com/arcprize/hierarchical-reasoning-model-analysis), both published under the
  Apache License 2.0 ([Apache-2.0.txt](Apache-2.0.txt)).
* `kaggle/combined/` holds ARC task data used by upstream's ARC-AGI recipes (not by this benchmark), unchanged from
  upstream. The ARC-AGI-1 and ARC-AGI-2 task sets are published by their authors under the Apache License 2.0.
  `arc-agi_concept_challenges.json` and `arc-agi_concept_solutions.json` hold the 160 ConceptARC tasks (used by
  upstream's `arc1concept` / `arc2concept` recipes), from [ConceptARC](https://github.com/victorvikram/ConceptARC)
  (Moskvichev, Odouard and Mitchell, 2023), whose repository is published under the MIT license. Each of these task
  sets keeps the terms of its original source.
