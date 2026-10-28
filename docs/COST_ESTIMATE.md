# Dollar cost estimate for the CVPR benchmark plan

*27 Oct 2026 snapshot. This prices [CVPR_BENCHMARK_PLAN.md](CVPR_BENCHMARK_PLAN.md) (plan v2): every row of tables
A.1, A.2, B, C.1–C.3 and D, and §2, §8 and §9 of the plan. It is an estimate made before any of the planned runs:
cloud prices move daily, and the throughput of every rented GPU except the RTX 5080 is unmeasured.*

*Prices: 27 Oct 2026 snapshot from the Vast.ai offer search API (re-fetched at 22:30–22:40 UTC), the RunPod API and
pricing page, provider pricing pages and the getdeploying.com aggregator. Each offer used is listed with its
provider, tier and public source URL in [cost/prices_2026-10-27.csv](cost/prices_2026-10-27.csv); the sources are
also listed at the end of this document. Vast figures are snapshot medians: they move by about ±15% within a day.
Throughput factors: §1.2. All numbers below are printed by the scripts in [cost/](cost/) (`dollar_cost_model.py`,
`dollar_cost_extra.py`, `roofline.py`; see [cost/README.md](cost/README.md)); a few are sums or ratios of printed
values, and one, flagged in §9, is a rough estimate outside the scripts. Rerun the scripts after the G0 probe (§6.5),
replacing the assumed S and U with measured values. USD; €1 = $1.1150 (ECB reference rate, 27 Oct 2026).*

*Hardware already owned: one RTX 4090 Laptop GPU and two RTX 5080s, available part-time. One of the two 5080s runs
in the plan's slow regime (0.61 of a fast 5080, plan §2.1).*

---

## 0. Headline

- **The $2,000 cap does not bind if Sudoku training runs on RTX 5080s** (the owned pair plus rented 1-GPU 5080s on
  Vast.ai, verified hosts, about $0.345/h including 40 GB of disk). Every tier, with the confirmatory 2 × 2 factorial at
  the paper budget L, costs **$561 central and $1,312 pessimistic** if everything is rented, overheads included. Vast does
  not charge VAT, so the central case has none; the pessimistic case adds 20%. Counting the owned cards' share it costs
  **$428 and $1,195**. The real constraints are engineering person-days and the 11 Nov freeze.
- **Per unit of work, a rented 4× RTX PRO 6000 node costs 2.2× (Vast marketplace, unverified hosts) to 4.3× (managed
  on-demand, with 20% VAT; 3.6× without) as much as rented 5080s** once utilisation, failures and VAT are included ($0.90
  and $1.75 vs $0.41 per fast-5080 G-h). It is cost-competitive only at or below **$2.33 per node-hour** ($2.89 if its
  speed-up reaches the high end, 2.6). For an RTX 6000 Ada node the threshold is $1.33. Market prices are $5.13–9.96. The
  $5.34 Vast node is on unverified hosts; the only verified 4× PRO 6000 SE on Vast costs $9.39.
- **The trap is keeping the node up.** Tier 0 is engineering-bound for 11–15 days. Renting the node continuously
  through the CVPR window costs $1,922 (Vast, unverified) to $3,380 (Vast, verified) or $3,586 (RunPod Secure) before
  VAT, at 13–28% utilisation. Rent in bursts billed per second, or use 1-GPU instances.
- **Recommended package (§6):** all tiers; the core factorial (A0, A2, A3, B★, B3, X1 × 6 seeds) at L, with
  X18a/b/c subsumed; T4/T7 behind their gates. Optional +P adds seeds 6–8 at L on X1, B3, A3 and B★ for +$71 (+$162
  pessimistic).
  - The pessimistic case with +P, all rented, is $1,474. That leaves the $400 reserve (20% of $2,000) intact with $126
    to spare.
  - Stress case: the unmeasured new-task rows at the plan's hi G-h with pessimistic rates. It costs $1,750, or $1,913
    with +P. That is still under $2,000 but uses most of the reserve (§6.3).

---

## 1. Inputs and formulas

### 1.1 Units

- **Work** is counted in the plan's **lo** G-h: hours on a *fast* RTX 5080 (5.14 TRM steps/s at B = 256, about
  2,000 puzzle-ACT-steps/s in evaluation).
  - Rented 5080s are not all fast. Vast's measured `gpu_mem_bw` is about 782 GB/s on most 5080 offers, but 3–4 of the
    30 1-GPU offers show 328–424 GB/s (half bandwidth), and one of those was on a verified host. Five are power-capped
    at 250 W, against the 360 W TGP.
  - lo is therefore the right base **only after the probe and the 4.6 steps/s rejection rule** (§6.2), with the search
    filtered on `gpu_mem_bw > 700`. S low = 0.75 stays in the pessimistic case.
  - The plan's midpoint (and hi) embeds the slower owned 5080. It is shown as a sensitivity column ("cen@mid") and
    is used for owned-hardware electricity.
- **S** is the throughput of one rented GPU in fast-5080 units (one run per GPU, B = 256). The same S is used for
  training and evaluation; evaluation is also mostly bandwidth-bound.

### 1.2 Throughput factors used (unmeasured except the 5080)

The S ranges are inputs to the model, not measurements. For every card except the RTX PRO 6000 they sit at or below
the lo/mid/hi roofline speed-ups printed by `docs/cost/roofline.py` (S low ≤ lo, central ≤ mid, high ≤ hi; a
per-kernel roofline of one TRM training step, calibrated to the fast 5080), with the low end discounted for power caps
and host overhead (the RTX 4090 Laptop GPU measured at about half its roofline value, §9). The RTX PRO 6000 range was
re-derived from Vast-measured memory bandwidth.

| GPU | S low | S central | S high | Note |
|---|---|---|---|---|
| RTX 5080 (rented) | 0.75 | 1.0 | 1.0 | Low end means an unprobed slow host (the slower owned 5080 is 0.61; half-bandwidth and 250 W hosts exist on Vast). A 5-minute probe and host rejection bring it to ≥ 0.9 |
| RTX PRO 6000 (Server Ed., Workstation, Max-Q) | 1.8 | 2.1 | 2.6 | One range for all editions: Vast-measured bandwidth is 1,391–1,406 GB/s for all three (5080: 782), and the workload is 58% memory-bound on the 5080. That gives ≈ 1.9–2.6, depending on the 5080's GEMM efficiency and on whether the PRO 6000 runs FP32 accumulate at full rate (not confirmed). 300 W hosts (Max-Q, capped WS) lean to the low end |
| RTX 6000 Ada | 0.95 | 1.2 | 1.4 | Vast-measured bandwidth 763 GB/s, about the 5080's |
| RTX 4090 / 5090 | 1.0 / 1.5 | 1.15 / 1.75 | 1.25 / 1.9 | Vast-measured bandwidth 879 / 1,455 GB/s |
| L40S | 0.9 | 1.1 | 1.3 | |
| A100 80 GB PCIe / SXM | 1.4 / 1.5 | 1.8 / 1.9 | 2.2 / 2.3 | The high end needs a ≥ 1,600 GB/s host: Vast-measured A100 bandwidth is 1,315–1,667 GB/s depending on the listing (1.68–2.13× the 5080) |
| H100 PCIe / SXM | 1.9 / 2.8 | 2.4 / 3.6 | 2.9 / 4.4 | |
| H200 / B200 | 3.2 / 4.0 | 4.2 / 6.0 | 5.2 / 8.5 | Host sync caps B200 |

The GeForce FP32-accumulate halving is applied: BF16 with FP32 accumulate is 112.6 TFLOPS dense on the 5080, 209.5 on the
5090 and 165.2 on the 4090 (NVIDIA RTX Blackwell whitepaper). MPS packing of 2–3 runs per PRO 6000 is expected to add
0–10%. It is left out of the central case.

### 1.3 Formula

For scenario s and scope Q (a set of plan rows):

```
raw rate        r      = p / S                                   ($ per fast-5080 G-h; table §2)
work            W      = Σ_rows (train_lo + eval_lo) · (1 + f_class)   [+ pairing rerun, §4.8]  · (1 + spot overhead)
billed GPU-h    H      = W / (S · U)                             (U = share of billed GPU-time doing useful work)
compute         C      = H · p                                   (p = GPU-only price; Vast: dph_base)
storage         D      = H · disk_GB · ($/GB-month) / 730        (at the host's own storage rate)
setup           Su     = n_launch · t_setup · p · k              (k = GPUs per billing unit: 4 for a node, 1 for an instance)
warm-up         Wu     = n_jobs · 0.05 h · p                     (process start, data load, torch.compile per job)
fixed           Fx     = pilots + egress/ingress                 (pool egress scales with the number of launches)
TOTAL                  = (C + D + Su + Wu + Fx) · (1 + VAT_s)
owned hardware         = G-h_mid · (1 + f) · e ,  e = $0.11 per mid-G-h (0.09–0.13); $0.13 per lo-G-h (0.11–0.16)
```

### 1.4 Parameters (optimistic / central / pessimistic)

| Parameter | Opt | Central | Pess | Basis |
|---|---|---|---|---|
| f, failed runs, reruns, debugging: Sudoku rows | 6% | 12% | 25% | New flags (RI/NI, B4, B5, C6, fp32 path); checkpoints every 5k steps bound each loss to ≤ 18 min on a fast 5080 |
| f, new tasks (T3, T3+, T4–T8) | 12% | 25% | 50% | Unmeasured rows (plan notes [n13]–[n16]), new data pipelines |
| U, 1-GPU instances (5080 pool) | 0.97 | 0.95 | 0.90 | Queue-driven, destroyed when idle, per-second billing |
| U, 4-GPU node in bursts | 0.90 | 0.80 | 0.65 | Dependency tails (§4.7) |
| U, interruptible node | 0.85 | 0.75 | 0.60 | Plus spot overhead 5 / 10 / 25% on W |
| Setup per launch or burst | 0.4 h | 0.5 h | 0.75 h | Image pull (10–15 GB), dataset build (1.3 GB, CPU minutes), smoke test |
| Warm-up per job | 0.05 h | 0.05 h | 0.05 h | × 195 (Tier 0 R) … 535 (all tiers, L) jobs |
| Launches (pool) / bursts (node) | | 15–55 / 8–27 | | Per scope |
| Disk per GPU | 30 GB | 40 GB | 60–80 GB | Results copied off the instance after each job |
| Disk $/GB-month, 5080 pool | 0.20 | 0.27 | 0.27 (on 80 GB) | Verified 5080 hosts charge $0.13–0.87 (median ≈ $0.27). Vast's `dph_total` already includes ≈ 8 GB of default disk, so the model prices the GPU at `dph_base` and adds the disk itself |
| Disk $/GB-month, nodes | 0.07–0.20 | 0.20 Vast; **0 Massed** | 0.20–0.25 | Massed includes 350–2,900 GB of local storage in the hourly price |
| Pilots (S probe, init-hash check, MPS test) | $5 | $10 | $20 | |
| Egress + ingress, 5080 pool (all tiers; scales with launches) | $2 | $12 | $30 | Verified 1-GPU 5080 hosts charge $0–41/TB each way (median ≈ $10–16/TB); about 0.84 TB in total (§4.5) |
| Egress + ingress, node | $0 | $2 | $5 | Massed and RunPod do not charge; the 4× PRO 6000 hosts charge $0–2.7/TB |
| VAT, Vast.ai (S2, S2s, S4) | 0% | **0%** | 20% | Vast is based in California, does not collect or remit VAT and puts none on invoices ([docs.vast.ai/host/guide-to-taxes](https://docs.vast.ai/host/guide-to-taxes)). 20% is kept in the pessimistic case: the RunPod fallback, or an institutional purchase under reverse charge without VAT recovery |
| VAT, managed providers (S1, S3) | 0% | 20% | 20% | Massed, RunPod and UpCloud pages say nothing about tax; buyer in an EU country with a 20% standard VAT rate, without a VAT ID |
| Prices | see §2 and §5 | | | |

---

## 2. Dollars per fast-5080 G-h (r = price ÷ S; before overheads)

Sorted by central r. "Spot" means interruptible. Vast values are live API medians at 27 Oct 22:30–22:40 UTC.
`dollar_cost_extra.py` prints every offer in the CSV; this table shows a selection, plus one excluded offer (Hetzner).

| GPU | Offer (27 Oct 2026) | $/GPU-h | S (low–central–high) | r central | r range |
|---|---|---|---|---|---|
| A100 80GB PCIe | Vast.ai spot (min bid) | 0.278 | 1.4–1.8–2.2 | 0.154 | 0.126–0.199 |
| H100 SXM | Nebius spot (floor only; floats up to $4.49) | 0.790 | 2.8–3.6–4.4 | 0.219 | 0.180–0.282 |
| A100 80GB SXM | Vast.ai spot | 0.460 | 1.5–1.9–2.3 | 0.242 | 0.200–0.307 |
| **RTX 5080** | **Vast.ai spot, verified hosts (min-bid median; range $0.227–0.267)** | 0.247 | 0.75–1.0–1.0 | **0.247** | 0.247–0.329 |
| RTX 5090 | Vast.ai spot | 0.474 | 1.5–1.75–1.9 | 0.271 | 0.249–0.316 |
| RTX 4090 | Vast.ai spot | 0.341 | 1.0–1.15–1.25 | 0.297 | 0.273–0.341 |
| **RTX 5080** | **Vast.ai on-demand, verified 1-GPU, GPU-only median (+$0.015/h for 40 GB disk)** | 0.330 | 0.75–1.0–1.0 | **0.330** | 0.330–0.440 |
| A100 80GB PCIe | Vast.ai on-demand (median; p25 $0.403) | 0.667 | 1.4–1.8–2.2 | 0.371 | 0.303–0.476 |
| RTX PRO 6000 SE | Vast.ai 4× spot, verified host (min bid; all-host median $0.62) | 0.780 | 1.8–2.1–2.6 | 0.371 | 0.300–0.433 |
| RTX PRO 6000 | HyperAI 1× on-demand (getdeploying; in stock) | 0.800 | 1.8–2.1–2.6 | 0.381 | 0.308–0.444 |
| RTX 5090 | Vast.ai on-demand | 0.674 | 1.5–1.75–1.9 | 0.385 | 0.355–0.449 |
| A100 80GB SXM | Vast.ai on-demand | 0.735 | 1.5–1.9–2.3 | 0.387 | 0.320–0.490 |
| RTX 5080 | RunPod Community (stock "Low") | 0.390 | 0.75–1.0–1.0 | 0.390 | 0.390–0.520 |
| RTX 4090 | Vast.ai on-demand | 0.478 | 1.0–1.15–1.25 | 0.416 | 0.382–0.478 |
| RTX 6000 Ada | Vast.ai 4× node, verified host (Türkiye, $2.40/node-h) | 0.601 | 0.95–1.2–1.4 | 0.501 | 0.429–0.633 |
| RTX 6000 Ada | Vast.ai on-demand | 0.654 | 0.95–1.2–1.4 | 0.545 | 0.467–0.688 |
| RTX PRO 6000 SE | Verda 4× spot ($5.13/node-h) | 1.284 | 1.8–2.1–2.6 | 0.611 | 0.494–0.713 |
| RTX PRO 6000 | Lyceum 4× spot (EU, $5.16/node-h) | 1.290 | 1.8–2.1–2.6 | 0.614 | 0.496–0.717 |
| **RTX PRO 6000 SE** | **Vast.ai 4× node on-demand, unverified hosts (Japan, Romania; $5.34/node-h)** | 1.334 | 1.8–2.1–2.6 | **0.635** | 0.513–0.741 |
| RTX 6000 Ada | Massed Compute 4× ($3.16/node-h; **out of stock** on 27 Oct) | 0.790 | 0.95–1.2–1.4 | 0.658 | 0.564–0.832 |
| L40S | Vast.ai on-demand | 0.735 | 0.9–1.1–1.3 | 0.668 | 0.565–0.817 |
| RTX PRO 6000 WS | Vast.ai 4× node, verified host (Poland, 500 W; $6.23/node-h; single offer) | 1.558 | 1.8–2.1–2.6 | 0.742 | 0.599–0.866 |
| H100 SXM | Vast.ai on-demand | 2.702 | 2.8–3.6–4.4 | 0.751 | 0.614–0.965 |
| RTX PRO 6000 SE | Vast.ai 1× on-demand median (verified $1.736) | 1.601 | 1.8–2.1–2.6 | 0.762 | 0.616–0.889 |
| RTX PRO 6000 WS | RunPod Community | 1.690 | 1.8–2.1–2.6 | 0.805 | 0.650–0.939 |
| RTX 6000 Ada | RunPod Secure 4× ($3.96/node-h; in stock) | 0.990 | 0.95–1.2–1.4 | 0.825 | 0.707–1.042 |
| RTX PRO 6000 SE | Nebius (1× or 8× only) | 1.800 | 1.8–2.1–2.6 | 0.857 | 0.692–1.000 |
| RTX PRO 6000 | UpCloud 4× ($7.52/node-h) | 1.880 | 1.8–2.1–2.6 | 0.895 | 0.723–1.044 |
| H200 | Hyperstack | 3.990 | 3.2–4.2–5.2 | 0.950 | 0.767–1.247 |
| B200 | Hyperstack | 6.000 | 4.0–6.0–8.5 | 1.000 | 0.706–1.500 |
| **RTX PRO 6000** | **Massed Compute 4× ($8.76/node-h) / Jarvislabs** | 2.190 | 1.8–2.1–2.6 | **1.043** | 0.842–1.217 |
| **RTX PRO 6000 SE** | **Vast.ai 4× node on-demand, the only verified host (Czechia, $9.39/node-h)** | 2.347 | 1.8–2.1–2.6 | **1.118** | 0.903–1.304 |
| **RTX PRO 6000 SE** | **RunPod Secure 4× ($9.96/node-h)** | 2.490 | 1.8–2.1–2.6 | **1.186** | 0.958–1.383 |
| RTX PRO 6000 SE | Verda on-demand | 2.567 | 1.8–2.1–2.6 | 1.222 | 0.987–1.426 |
| L40S | Scaleway (EU) | 1.639 | 0.9–1.1–1.3 | 1.490 | 1.261–1.821 |
| RTX PRO 6000 Max-Q | Hetzner GEX430 (€3,599/month + €3,599 setup) | – | – | – | First month $8,026: **excluded** |

Price drift since the first collection: 1× PRO 6000 SE $1.468 → $1.601; 5090 $0.803 → $0.674; 4090 $0.536 → $0.478;
A100 PCIe spot $0.333 → $0.278; 4× SE spot $0.77 → $0.62 median ($0.78 verified). None of these changes the ranking for
on-demand training.

Reading the table:

- **The rented RTX 5080 is the cheapest on-demand option that preserves pairing, and its S is the only one known.**
  - It is not the cheapest per G-h on price alone at high S. HyperAI's 1× PRO 6000 at $0.80 is $0.38 per G-h at
    S = 2.1 and $0.31 at S = 2.6. A Vast A100 PCIe on-demand at the p25 price of $0.403 is $0.22 per G-h at S = 1.8.
  - Packet·ai's $0.66 PRO 6000 is its "Dynamic" tier (shared GPUs, per getdeploying) and is excluded.
  - All of these break the pairing rule, so the 5080 recommendation rests on pairing, not on price alone.
  - By definition S = 1 for the 5080; host variance is checked by the probe.
  - A rented 5080 is the only rented card that keeps the plan's pairing rule (plan §7.2) with the owned 5080s and the
    published seeds. This needs a check: the seed-0 initial-weight hash on a rented 5080 should match the owned card's,
    with the same container and torch version (gate G0).
- Spot A100s are about half the price per G-h, but S is unmeasured and a different GPU model breaks pairing. At most
  they suit evaluation, where the saving is about $50–70 on the whole plan.
- **Break-even for a rented 4× node, against the 5080 pool** (S = 2.1, U = 0.80 vs 0.95, same VAT treatment as Vast):
  - RTX PRO 6000 ≤ $0.584/GPU-h = **$2.33 per node-hour**. At S = 2.6 the threshold is $2.89. With 20% VAT on the node
    it is $1.95.
  - RTX 6000 Ada ≤ $0.333/GPU-h = **$1.33 per node-hour**.
  - The cheapest 4× PRO 6000 that could be rented on 27 Oct was $5.13 (Verda spot) or $5.34 (Vast, unverified).

---

## 3. Workload scopes (G-h)

### 3.1 The L-core upgrade (core factorial at the paper budget L = 195k steps × 256)

Rules applied:

- Runs with `last_state.pt` (new runs and the A0_s0 rerun) continue from R or 45k to L.
- The published A0 s1–s2 and A2/A3 s0–s2 kept no optimizer state, so they are rerun from scratch at L.
- B1 and B2 stay at R, except the arm chosen as B★, which continues to L after the 2 Nov choice.
- DP0, DP1 and DP2 are still decided on the 35k prefixes. The R-prefix evaluations they need are costed as duplicates.

| Arm | Training added (lo–hi) | How | Evaluation added (lo–hi) | Evaluation content |
|---|---|---|---|---|
| A0 × 6 | 58.0–95.0 | 4 continued (45k → L, 8.8–14.4 each) + s1–s2 from scratch (11.4–18.7 each) | 8.0–15.3 | exact@16 at 35k and 70k prefixes; EV-F × 3 at 45k for DP0 |
| A2 × 6 | 62.7–103.8 | s0–s2 from scratch (11.5–19.0) + s3–s5 continued (9.4–15.6) | 8.0–15.3 | as A0 |
| A3 × 6 | 78.6–129.0 | s0–s2 from scratch (14.4–23.6) + s3–s5 continued (11.8–19.4) | 11.7–22.5 | prefixes; EV-S at 150k on s0–s2 (L compute match vs A0@195k); EV-F × 3 and C6 at 35k for DP1 |
| B★ × 6 | 56.4–91.8 | 6 continued (9.4–15.3); 0 if B★ = B3 | 8.0–15.3 | prefixes; EV-F × 3 on the L checkpoints of s0–s2 |
| B3 × 6 | 56.4–93.6 | 6 continued (9.4–15.6) | 10.0–19.1 | prefixes; EV-F, EV-V and C6 at 35k for DP1 and the B★ choice |
| X1 × 6 | 70.8–116.4 | 6 continued (11.8–19.4) | 11.7–22.5 | as A3 |
| R-prefix selector duplicates | – | – | 3.7–7.0 | EV-V × 3 at R on A0@45k and on A3 s0–s2 (DP0 compares against A3's best selector at R); C6 on B★ s0 at R (DP1) |
| **Total** | **382.9–629.6** | | **61.0–117.2** | **L-core = 443.9–746.8 G-h (mid 595)** |

- **Conditional duplicates not costed.** X1 and B★ also need EV-V at R if their selectors are chosen at L rather than
  reused from R. Each costs 1.7–3.2 G-h (about $1).
- **Deviation from plan note [n8].** At 70k the L-core uses exact@16 on 100k instead of EV-S0. Restoring EV-S0 there would add
  15.8–28.1 G-h over 36 runs, about $6 central.

Once the core runs to L, **X18a, X18b and X18c are subsumed** (their checkpoints become prefixes or duplicates): that is
72.8 + 194.1 + 28.6 = 295.5 lo G-h removed. All tiers at L therefore cost only +148 lo G-h more than all tiers at R.

### 3.2 Scopes

| Scope | lo | mid | hi | Train lo | Eval lo | Work incl. failures (central / pess) |
|---|---|---|---|---|---|---|
| T0R: Tier 0 at R (plan v2) | 316.5 | 446.6 | 576.7 | 140.6 | 175.9 | 359 / 404 |
| **T0L: Tier 0, core at L** | 760.4 | 1,041.9 | 1,323.5 | 523.5 | 236.9 | 856 / 959 |
| T01R: Tiers 0+1 at R | 630.4 | 922.1 | 1,213.8 | 351.7 | 278.7 | 721 / 816 |
| T01L: Tiers 0+1, core at L (−X18a) | 1,001.5 | 1,418.7 | 1,835.9 | 683.0 | 318.5 | 1,136 / 1,280 |
| AllR: all tiers at R | 1,097.2 | 1,596.2 | 2,095.2 | 699.1 | 398.1 | 1,270 / 1,450 |
| **AllL: all tiers, core at L (−X18a/b/c)** | 1,245.6 | 1,789.9 | 2,334.2 | 874.5 | 371.1 | 1,436 / 1,636 |

The R-scope row sums reproduce plan §8.1 exactly: Tier 0 is 140.6–244.6 training + 175.9–332.1 evaluation.

---

## 4. Overheads (explicit)

1. **Setup and idle per instance.** Each launch or burst costs 0.5 h (0.4–0.75) of instance time:
   - container pull: 10–15 GB PyTorch image, 3–10 min;
   - Sudoku-Extreme build (1.3 GB, CPU minutes) or rsync;
   - smoke test.
   Each job adds a 3-minute warm-up for process start and torch.compile. In total this is $9 + $9 for the 5080 pool on
   all tiers, and $72 + $36 for the Vast node with 27 bursts.
2. **Failures, reruns, debugging.** 12% of work on Sudoku rows and 25% on new tasks (pessimistic 25% / 50%).
   Checkpointing every 5k steps with `last_state.pt` caps each crash at ≤ 18 min on a fast 5080 (about 29 min on the slow owned card). Most of the allowance is
   debugging of new code (RI/NI, B4/B5, C6, T3) and the unmeasured T4–T8 rows.
3. **Pilots and calibration.** $10 ($20):
   - 30-min probe of a rented 5080 (speed and init hash);
   - optional 30-min probe of a 1× PRO 6000 (S, and MPS packing of 1/2/3 runs).
   The T4 and T7 pilots are already rows in the plan.
4. **Storage for the rental period.**
   - Volumes: Tier 0 ≈ 27 GB at R (plan §2.6); at L ≈ 50 GB (each L run is 0.9–1.1 GB vs 0.24–0.45 GB). All tiers ≈ 130–140 GB,
     including about 40 GB of optical-flow data.
   - Results are copied to local storage after each job, so cloud disk only lives with the instance.
   - Verified 5080 hosts charge $0.13–0.87/GB-month (median ≈ $0.27). One host in China that sells 1/8 slices charges
     $0.87, which adds $0.048/h for 40 GB.
   - At the median, 40 GB costs ≈ $0.015 per instance-hour, 4.5% of a 5080's GPU-only price. Hosts must therefore be
     ranked by `dph_base` + disk_GB × storage_cost / 730 + pull cost, not by `dph_total`. `dph_total` also already
     contains ≈ 8 GB of default disk.
   - Totals: $13 (Tier 0 at L) to $22 (all tiers) for the pool; $6–9 for the Vast node; $0 at Massed, where local
     storage is included in the hourly price.
   - A persistent network volume between node bursts adds $7–20/month (RunPod $0.07, Nebius $0.08/GB-month).
5. **Egress and ingress.**
   - Pool, all tiers: about 0.7 TB of image and data pulls over 55 launches (≈ 13 GB each), plus about 0.14 TB of
     results coming home.
   - Verified 1-GPU 5080 hosts at ≤ $0.40 charge $2.7–41/TB for download, with similar upload rates. The median is
     ≈ $10/TB at 22:40 UTC and $15–16/TB at 22:30. Examples: the slice host in China $16–17/TB, Taiwan $26.7/TB,
     Ontario $40/TB.
   - That is **$12 central ($30 pessimistic) for all tiers**, scaling with the number of launches (≈ $3 for Tier 0 at R).
   - Two levers: prefer hosts at ≤ $10/TB when prices tie, and slim the 10–15 GB image.
   - Nodes: free at Massed, RunPod, Nebius and Hyperstack; $0–2.7/TB on the Vast 4× PRO 6000 hosts → ≤ $2.
6. **Billing granularity.**
   - Vast and RunPod bill per second: nothing beyond setup. RunPod only requires holding one hour of credit, which is not
     a cost.
   - Hyperstack and Jarvislabs bill per minute; Verda in prepaid 10-minute blocks (≤ $0.85 per 4× burst).
   - Massed, Nebius and CoreWeave do not state their granularity: assume hourly rounding, about +0.25–0.5 node-h per
     burst (+$60–120 on all tiers with Massed).
   - Hetzner bills monthly plus a setup fee equal to one month: excluded.
   - Prepaid credit can be stranded: top up in ≤ $100 steps.
7. **Utilisation of a 4-GPU node** (dependency structure, plan §8.4/§8.5).
   - Bursts reach about 80% because each phase has independent filler work:
     - Phase A: 17 no-code runs plus D5.
     - Phase B: the two B1 σ pilots block B1 s1–s2, B2, B3 and X1 for 1.5–2 h on a PRO 6000; X10 × 3 and EV-F on
       A0@45k fill the other two GPUs.
     - Phase D: 9 seed-3–5 runs plus 6 T3 runs.
   - Utilisation is lost in tails. An L run lasts 5.4–6.9 h on a PRO 6000 (11.4–14.4 h on a 5080), so a burst ending on
     one L run idles 3 GPUs for up to 7 h. Mitigation: end bursts on short evaluation jobs, or pack two runs per GPU
     with MPS.
   - **Continuous rental is the failure mode.** Phases A–E are engineering-bound: RI flags land on day 1, C6/C7/C9 on
     days 5–6, T3 on days 3–8. Over the 15-day window, Tier 0 keeps a 4× PRO 6000 node only 13% busy (28% at L). Fifteen
     days cost (ex VAT):

     | Node | 15 days, continuous |
     |---|---|
     | Vast 4× PRO 6000 SE, unverified hosts | $1,922 |
     | Vast 4× PRO 6000 SE, verified host | $3,380 |
     | UpCloud | $2,707 |
     | Massed | $3,154 |
     | RunPod Secure | $3,586 |
     | Massed 4× Ada | $1,138 |

     A 1-GPU pool has no idle GPUs: an instance exists only while it has a queue.
8. **Pairing (plan §7.2).**
   - Training a family on any card other than a 5080 breaks pairing with the published 5080 seeds and with the owned
     cards.
   - The 4× node scenarios (S1–S3) at R therefore add a rerun of A0 s1–s2, A2 s0–s2 and A3 s0–s2: +19.3–31.4 G-h.
     In those scenarios the owned 5080s can only evaluate.
   - At L this penalty is zero, because those seeds are rerun anyway.
9. **VAT and FX.**
   - **Vast.ai:** Vast is based in California and does not currently collect or remit VAT; VAT is not shown on its
     invoices ([docs.vast.ai/host/guide-to-taxes](https://docs.vast.ai/host/guide-to-taxes)).
     - The central case for S2, S2s and S4 therefore has 0% VAT.
     - The pessimistic case keeps 20%. That covers the RunPod fallback, and a purchase through an institution that
       self-assesses VAT under reverse charge but cannot recover it.
   - **Managed providers (Massed, RunPod, UpCloud):** their pricing pages say nothing about tax, so 20% stays in the
     central and pessimistic cases for S1 and S3.
   - $2,000 = €1,794. Card FX fees of 0–2% (≤ $25) come out of the reserve.
10. **Local electricity on owned hardware.**
    - Desktop with a 5080: 0.40–0.55 kW; laptop: 0.17–0.25 kW; at $0.2231/kWh (regulated residential tariff, €0.2001/kWh).
    - That is $0.11 (0.09–0.13) per mid-G-h, or $0.13 (0.11–0.16) per lo-G-h of work. The slow card and the laptop need
      more wall-hours per unit of work.

---

## 5. Scenarios (total $ incl. VAT and all overheads)

Central uses lo G-h; "cen@mid" applies central parameters to the plan's midpoint G-h; "cen ex-VAT" drops VAT (equal to
central for the Vast scenarios, which have no central VAT).

| Scenario | Scope | Opt | **Central** | Pess | cen@mid | cen ex-VAT | Node-h or instance-h (central) |
|---|---|---|---|---|---|---|---|
| **S1** 4× PRO 6000, managed on-demand ($2.19/GPU-h, Massed $8.76/node-h, storage included; opt UpCloud $1.88; pess RunPod Secure $2.49); 20% VAT central | Tier 0 (R) | 335 | **677** | 1,231 | 919 | 564 | 57 node-h |
| | Tier 0, core at L | 713 | **1,445** | 2,628 | 1,941 | 1,204 | 127 |
| | Tiers 0+1 (R) | 648 | **1,309** | 2,390 | 1,849 | 1,091 | 110 |
| | All tiers (R) | 1,094 | **2,230** | 4,117 | 3,155 | 1,859 | 192 |
| | All tiers, core at L | 1,214 | 2,473 | 4,557 | 3,466 | 2,061 | 214 |
| **S2** 4× PRO 6000 SE, Vast.ai marketplace ($1.334/GPU-h = $5.34/node-h on unverified hosts; pess: the only verified 4× SE, $2.347/GPU-h = $9.39/node-h) | Tier 0 (R) | 240 | **351** | 1,164 | 475 | 351 | 57 |
| | Tier 0, core at L | 509 | **744** | 2,484 | 998 | 744 | 127 |
| | Tiers 0+1 (R) | 462 | **674** | 2,259 | 950 | 674 | 110 |
| | All tiers (R) | 780 | **1,145** | 3,890 | 1,618 | 1,145 | 192 |
| | All tiers, core at L | 865 | 1,269 | 4,305 | 1,778 | 1,269 | 214 |
| S2s, same node interruptible (Vast 4× SE verified-host bid $0.78; opt all-host median $0.62; pess WS bid $1.07) | Tier 0 (R) / at L | 125 / 264 | **242 / 512** | 720 / 1,535 | 328 / 687 | 242 / 512 | 66 / 149 |
| | Tiers 0+1 (R) / all tiers (R) | 239 / 403 | **462 / 783** | 1,385 / 2,384 | 652 / 1,110 | 462 / 783 | 130 / 225 |
| **S3** 4× RTX 6000 Ada 48 GB ($0.79/GPU-h, Massed $3.16/node-h, **out of stock on 27 Oct**, storage included; opt Vast verified 4× $0.601; pess RunPod Secure $0.99, in stock); 20% VAT central | Tier 0 (R) | 192 | **414** | 907 | 567 | 345 | 99 |
| | Tier 0, core at L | 414 | **893** | 1,958 | 1,206 | 744 | 223 |
| | Tiers 0+1 (R) | 368 | **796** | 1,753 | 1,136 | 663 | 193 |
| | All tiers (R) | 625 | **1,360** | 3,032 | 1,944 | 1,134 | 336 |
| | All tiers, core at L | 695 | 1,509 | 3,357 | 2,136 | 1,257 | 374 |
| **S4** RTX 5080 pool, 1-GPU on-demand (Vast verified: GPU-only median $0.330 + 40 GB disk at $0.27/GB-month = $0.345/h; opt p25 $0.28; pess $0.39 = RunPod Community / Vast p75, S 0.75) | Tier 0 (R) | 110 | **149** | 345 | 203 | 149 | 378 instance-h |
| | Tier 0, core at L | 253 | **335** | 771 | 450 | 335 | 901 |
| | Tiers 0+1 (R) | 215 | **289** | 672 | 412 | 289 | 759 |
| | All tiers (R) | 370 | **498** | 1,167 | 710 | 498 | 1,337 |
| | All tiers, core at L | 419 | **561** | 1,312 | 792 | 561 | 1,512 |
| **S5** owned hardware only (electricity on mid G-h incl. failure allowance; days incl. failures) | Tier 0 (R) | 43 | **56** | 74 | – | – | 13.9 d at 60% 5080 availability (18.9 d at 40%) |
| | Tier 0, core at L | 100 | **129** | 171 | – | – | 32.2 d (43.8 d): **misses the 11 Nov freeze** |
| | Tiers 0+1 (R) | 89 | **116** | 156 | – | – | 29.1 d (39.5 d) |
| | All tiers (R) | 155 | **204** | 276 | – | – | 50.9 d (69.2 d) |
| | All tiers, core at L | 174 | 228 | 307 | – | – | 56.9 d (77.3 d) |

All-in effective $ per lo-G-h of planned work, variable part (central; opt–pess):

| Scenario | $ per lo-G-h |
|---|---|
| S1 | 1.75 (0.85–3.21); 1.46 ex VAT |
| S2 | 0.90 (0.61–3.04) |
| S2s | 0.62 (0.32–1.89) |
| S3 | 1.11 (0.51–2.44); 0.92 ex VAT |
| **S4** | **0.41 (0.31–0.93)** |
| S5 | 0.13 (electricity) |

The pessimistic column stacks every adverse value at once (low S, high price, low U, high f, VAT): it is a roughly
5th-percentile case, not a forecast. **One exception:** it prices the unmeasured new-task rows (T3–T8: 314.5 lo-G-h vs
706.4 hi-G-h; T5 alone spans 4×) at their lo G-h, so for those rows it is not a worst case. The stress line in §6.3
covers them.

**Wall-clock.**

- **Tier 0 is engineering-bound in every rented scenario** (plan §8.4: DP0/DP1 on 4 Nov, freeze on 11 Nov).
- Pure compute time, if run back to back, central:

  | Scope | 4× PRO 6000 node (S1/S2) | 4× Ada node (S3) | 5080 pool (S4), 8 instances |
  |---|---|---|---|
  | Tier 0 (R) | 2.4 days | 4.1 days | 2.0 days |
  | Tier 0, core at L | 5.3 days | 9.3 days | 4.7 days |
  | All tiers, core at L | 8.9 days | 15.6 days | 7.9 days |

- The pool scales to the queue, limited by supply.
  - The live API lists 44–46 machines with rentable on-demand 5080s, but only 29–30 1-GPU offers. Of these, 14 are
    on verified hosts and 12 of those cost ≤ $0.40.
  - Some of those 12 are 1/8 slices of one host, or half-bandwidth or 250 W cards. That leaves **8–9 distinct
    verified, full-bandwidth hosts at ≤ $0.40**. RunPod Community stock is "Low".
  - Peak-day fallback order: (1) RunPod Community 5080 at $0.39; (2) verified 1-GPU hosts at ≤ $0.60 (Florida, Italy);
    (3) a second wave of L runs (+0.5–1 day, no change in $). Each step costs a few dollars at most; the main effect
    is on wall-clock.
- The L-core lengthens the confirmatory critical path by one B★ L run after the 2 Nov choice (the TRM path):
  - 9.4 h as a continuation from R on a fast 5080 (11.4 h from scratch, 18.7 h on the slow card);
  - about 4.5 h on a PRO 6000 (5.4 h from scratch).
  - DP1 is unaffected because it reads the 35k prefixes.
- Owned-only (S5) cannot fit Tier 0 at L in the window.

---

## 6. Recommended package (≤ $2,000 total, ≥ 20% reserve)

### 6.1 Content

- **All of plan v2 (Tiers 0, 1, 2), with the confirmatory core factorial at the paper budget L:**
  - A0, A2, A3, B★, B3 and X1 at seeds 0–5, 195k steps × 256.
  - B1/B2 stay at R except the B★ arm (§3.1).
  - X18a, X18b and X18c are dropped as subsumed.
  - Everything else keeps its plan seeds: must arms 3, sweeps 1 then 3 at the winner.
- **Gated rows.** T4 runs past the DP4 maze gate only. T7, T8, B8 and B10 run only if DP1–DP3 leave the paper alive.
  The Tier-1 mechanism rows (X12, X15, X16, X13) follow DP2.
- **Option +P (decide at G0; must go into the 29 Oct pre-registration):** seeds 6–8 at L for X1, B3, A3 and B★.
  H5 and H1 rise from n = 6 to n = 9, and the t-test power for the 10 pp / sd 7 effect goes from 0.80 to about 0.96.
  Cost: +$71 central, +$162 pessimistic.
- **Why L is worth it.**
  - It answers the main reviewer risk (our 62–69% vs published 84–87%) before submission.
  - It moves DP4's full-budget kill test into the CVPR window: A0 at L against A3's M2, reading on about 5–6 Nov.
  - All tiers at L cost only 148 lo G-h more than at R, because X18a/b/c disappear.

### 6.2 Hardware and providers

- **All Sudoku training runs on RTX 5080s:**
  - the owned pair, when available (part-time);
  - rented 1-GPU **Vast.ai on-demand 5080s on verified hosts at ≤ $0.40/h all-in**.
    - Rank hosts by `dph_base` + 40 GB × storage rate / 730 + bandwidth, not by `dph_total`.
    - Filter the search on `gpu_mem_bw > 700`. Prefer `gpu_max_power ≥ 300`; a 250 W host must pass the probe.
    - When prices tie, prefer bandwidth at ≤ $10/TB.
    - Fallbacks in order: RunPod Community ($0.39), then verified hosts at ≤ $0.60. Same GPU model, so the family stays
      paired; no spot for family training.
  - Container pinned to the published torch/CUDA versions.
  - Reject any host below 4.6 steps/s (A0, B = 256).
- **Evaluation runs on any idle card**: the laptop (0.31 of a 5080; evaluation only), an owned 5080, or a rented 5080.
  Vast spot (verified hosts, about $0.25) is fine for evaluation jobs, which restart cleanly.
- **High-memory rows** (X19, C6, B8, T4, T7) use the plan's micro-batch or checkpointing workaround on 16 GB; that
  overhead is already in their G-h. Use a 1× PRO 6000 (Vast median $1.60, verified $1.74; HyperAI $0.80 when in stock)
  only if its probe shows it cheaper per G-h for those configurations.
- **A 4× RTX PRO 6000 or RTX 6000 Ada node is not needed.** Use it only if it is offered at ≤ $2.33/node-h (PRO 6000; $2.89 if S
  measures 2.6) or ≤ $1.33/node-h (Ada), VAT-free, or free (§6.8).

### 6.3 Expected cost

| | Central | Pessimistic |
|---|---|---|
| All rented, VAT and overheads included | $561 | $1,312 |
| Owned share: 573 lo-G-h central (5080 pair at 40% in the window + laptop evaluation ≈ 330; half of Tiers 1–2 afterwards ≈ 243); 200 pessimistic | −$208 rented, +$74 electricity | −$149, +$32 |
| **Package total** | **$428** | **$1,195** |
| + Option P (seeds 6–8 at L, priced as rented) | +$71 | +$162 |
| **Package + P** | **$498** | **$1,357** (all-rented $1,474) |
| Reserve within $2,000 | $1,502 | $643 ($526 all-rented) |
| *Stress: new-task rows (T3–T8) at the plan's hi G-h, pessimistic rates; Sudoku rows at lo* | – | *$1,750 all-rented; $1,913 with +P* |
| *Central rates on the plan's hi G-h for every row ("cen on hi")* | *$1,022; $1,140 with +P at hi* | – |

Allocation cap: $1,600, plus a $400 reserve (20% of $2,000).

- The all-rented pessimistic case with +P ($1,474) stays under the cap.
- The stress case with +P ($1,913) is still within $2,000 but $313 into the reserve, leaving $87.
- The T4 and T7 pilots and T5's first run show which case applies. If any of them lands in the upper half of its
  G-h range, apply §6.6 cuts 1–2 (T4 beyond its gate, Tier 2 group 1) before Tranche 4.

### 6.4 Wall-clock

- **Tier 0 (with the L-core): plan §8.4 dates unchanged.**
  - The A0/A2/A3 L runs start on day 1 and finish by about 30 Oct.
  - B3/X1 s0–s2 continue to L in Phases B–C.
  - B★ is chosen on 2 Nov (R prefixes); B★ s0–s2 then continue, and B★/B3/X1 s3–s5 run at L.
  - L checkpoints arrive about 3–4 Nov, L evaluation about 5–6 Nov; DP0/DP1 on 4 Nov, DP3 8–9 Nov, freeze 11 Nov.
  - Peak rented demand: about 10–12 instances (16 with +P) for about one day in Phases A and D; average about 2. The
    peak exceeds the 8–9 distinct verified full-bandwidth hosts of 27 Oct, so it relies on the fallback order in §5.
- **Tier 1:** about 280 G-h more work including failures (241 lo-G-h), roughly 2.5–3 days of compute on 4 rented
  cards plus the owned ones. Engineering-bound at 15–22 person-days.
- **Tier 2:** about 300 G-h of work including failures (244 lo-G-h). T4's nine runs take about 1–1.5 days on nine
  instances (15–19 h per run on a fast 5080, plus micro-batching). Engineering-bound at 13–22 person-days. ICCV 2027
  (early March, unverified) leaves room.

### 6.5 Order of spending, with stop/go checkpoints

Cumulative $ are all-rented (central / pessimistic); the owned share lowers them.

| Step | When | What is paid for | $ this step | Cumulative | Stop/go |
|---|---|---|---|---|---|
| **G0** | 28 Oct, before the pre-registration | Probes: one rented 5080 (30 min); optional 1× PRO 6000 (30 min, S and MPS) | 10 / 24 | 10 / 24 | **Go** to L if: (i) rented 5080 ≥ 4.6 steps/s; (ii) seed-0 init hash = owned 5080; (iii) ≥ 8 verified 1-GPU 5080 offers at ≤ $0.40 on ≥ 6 distinct full-bandwidth hosts (27 Oct: 8–9 hosts, a narrow pass). Decide +P. **No-go:** keep plan v2 at R with the pool as slack (T0R: $149 / $345) and L later as X18a/b |
| **Tranche 1** | 28 Oct–4 Nov (Phases A–D) | Phase A–D rows at R; all A0/A2/A3 L runs; B3/X1 s0–s2 to L; B★ continuation; B★/B3/X1 s3–s5 at L; R-prefix duplicates (+P seeds) | 241 / 553 (+71 / 162) | ≈ 251 / 577 | **DP0 + DP1 (4 Nov)**: (c) → stop: cancel in-flight runs and Phases E–F (sunk ≤ $251 / $577; ≤ $322 / $740 with +P). (a)/(b) → continue |
| **Tranche 2** | 5–11 Nov (Phases E–F) | Evaluation of L checkpoints, EV-Full, EV-R/EV-T3, T3 B★ arm | 70 / 162 | **335 / 771** (Tier 0 at L, incl. fixed) | **DP2 (7 Nov)** selects the Tier-1 mechanism rows. **DP3 (8–9 Nov)**: fail → ICCV route, T5/T6 first in Tranche 3. **DP4 (full budget)** now reads here: A0@L reaching A3's M2 → kill the method claim before submission |
| **Tranche 3** | after 16 Nov | Tier 1 minus X18a (ablation seeds, X6–X17, X19, T3+, T5, T6, D6, extras) | 102 / 240 | ≈ 445 / 1,030 | Cut list §6.6 if the measured $/lo-G-h exceeds $0.61 |
| **Tranche 4** | Dec–Jan | Tier 2 minus T4 (X13, X16, X20, B8, B10, T7, T8, D7) | 36 / 87 | ≈ 489 / 1,134 | Gated by DP1–DP3 outcomes; apply cuts 1–2 first if the T4/T7/T5 pilots come in at the hi end (§6.3) |
| **Tranche 5** | Dec–Jan | T4: 2k-step pilot + first A0 run + EV-Maze (DP4 maze gate), then 8 runs | 8 / 19, then 65 / 160 | **561 / 1,312** | **DP4 maze gate:** continue only if A0 ≥ 43% with the CI lower bound ≥ 40% |

Spend control: after each tranche, recompute the realised $ per lo-G-h (cloud invoice ÷ plan lo G-h completed).

| Realised $ per lo-G-h | Action |
|---|---|
| > $0.61 (1.5× central) | Apply cuts 1–3 below |
| > $0.93 (the pessimistic value) | Apply cuts 1–6 |

### 6.6 What to cut first if prices or throughput come in worse

The pool columns show what each cut saves in the recommended scenario; the node columns show what it saves if the work
runs on the Vast 4× PRO 6000 node (pessimistic = verified-host price).

| Order | Cut | lo G-h saved | Pool $ cen | Pool $ pess | Node (S2) $ cen | Node $ pess |
|---|---|---|---|---|---|---|
| 1 | T4 beyond its gate (full T4 if the gate itself is skipped) | 142.7–159.4 | 65–72 | 160–178 | 143–160 | 520–581 |
| 2 | Tier 2 group 1: T7, B8, X13, X16, X20, T8, B10, D7 (42.1 new-task + 42.6 Sudoku G-h) | 84.7 | 36 | 87 | 80 | 283 |
| 3 | Option +P (seeds 6–8), if taken | 174.1 | 71 | 162 | 156 | 528 |
| 4 | Tier 1: X6, X7, X9 | 40.3 | 16 | 38 | 36 | 122 |
| 5 | Tier 1: X12, X14, X15 (unless DP2 makes them central) | 22.6 | 9 | 21 | 20 | 69 |
| 6 | T6, then T5 (if DP3 failed, keep them and cut T3+ instead: 34.5 G-h, $16 / $39) | 45.6 | 21 | 51 | 46 | 166 |
| 7 | L-core → L for seeds 0–2 only (confirmatory family back at R, L descriptive) | 189.0 | 77 | 176 | 169 | 574 |
| 8 | Plan §8.6 Tier-0 cuts 1–6 | 48.3 | 20 | 45 | 43 | 147 |
| – | Revert to plan v2 as written (L-core removed; X18a/b/c restored; difference of the §5 scenario totals) | 148.4 | 63 | 144 | 124 | 415 |
| – | L-core removed, X18a back in Tier 1, X18b/c not restored | 371.1 | 151 | 346 | 333 | 1,126 |

Never cut (plan §8.6): B1–B3 and X1 with B★; six seeds for the family arms; E1, E5, E6, E8; C5 and M2; the validation
split; D1 on the family arms; X8, X10, B4a/B4b; T3.

### 6.7 Marginal $ per optional row

Pool = rented 5080 pool (S4); node = Vast 4× PRO 6000 (S2).

| Row | G-h lo–hi | Pool $ cen | Pool $ pess | Node $ cen |
|---|---|---|---|---|
| **+P: seeds 6–8 at L for X1, B3, A3, B★** (n = 9 on H5/H1; power 0.80 → ≈ 0.96) | 174–290 | 71 | 162 | 156 |
| Seeds 6–8 at L for all 6 family arms (n = 9; Holm becomes usable, plan §7.6: 0.82 at Holm's first step) | 252–421 | 103 | 235 | 226 |
| T3+ ungrounded visual Sudoku | 34.5–68.0 | 16 | 39 | 35 |
| Tier-1 evaluation extras on X1 and B★ (EV-G/X/TT) | 17.5–32.6 | 7 | 16 | 16 |
| T5 restoration (with B9 and JR on T5) | 27.6–108.0 | 13 | 31 | 28 |
| T6 RAFT fine-tune | 18.0–48.0 | 8 | 20 | 18 |
| X17 V1 caps with seeds | 16.4–28.3 | 7 | 15 | 15 |
| X19 batch fidelity | 13.4–22.6 | 5 | 12 | 12 |
| X15 on-trajectory pairs | 11.4–21.5 | 5 | 11 | 10 |
| X16 metric variants / X13 adversary strength | 14.9–27.8 / 11.7–20.2 | 6 / 5 | 14 / 11 | 13 / 10 |
| X12 / B6 / D6 / B8 | 4.2–8.3 / 4.1–7.8 / 3.9–7.2 / 5.2–12.4 | 2 each | 4 / 4 / 4 / 5 | 4 / 4 / 3 / 5 |
| T4 gate only (2k-step pilot + first TRM-Att A0 run + EV-Maze) | 16.7–33.2 | 8 | 19 | 17 |
| T4 full (9 runs) | 159–317 | 72 | 178 | 160 |
| T7 ARC-1 fine-tunes / T8 Mini-ARC | 32.2–71.2 / 9.9–29.7 | 15 / 4 | 36 / 11 | 32 / 10 |

### 6.8 Using a 4× node anyway

- **Free or institutional node:**
  - Put whole families on it: all Sudoku training, including the published seeds, which are rerun at L anyway.
  - The owned 5080s and the laptop evaluate.
  - $ ≈ electricity on the owned cards only. Tier 0 at L needs about 127 node-hours (5.3 days of node time).
- **Rented RTX PRO 6000 node:**
  - Use the Vast 4× Server Edition ($5.34/node-h), in bursts only, from a script that destroys the node when its queue
    is empty.
  - **Host verification caveat.** All $5.34 offers are on unverified hosts (Japan, Romania): no Vast reliability check,
    so probe as for 5080s and keep checkpoints off-host. The only verified 4× SE costs $9.39/node-h (Czechia), about
    RunPod Secure's $9.96. A verified 4× WS (Poland, 500 W) costs $6.23, but it is a single offer.
  - Tier 0 at L: $744 central (unverified host); $2,484 pessimistic (verified-host price, before measurement).
  - **Gate after the first 4–6 h burst:** continue only if measured S ≥ 1.9 and U ≥ 0.75.
    - At the $5.34/node-h rental price, that caps the pessimistic case at $1,195 for Tier 0 at L and $1,613
      for Tiers 0+1 at L (at the cap).
    - On the verified $9.39 host the same case is $2,067 and $2,798, both above $2,000.
  - So the unverified-host node route affords Tier 0 at L plus the DP-selected Tier-1 rows, not all tiers. The
    verified-host route affords only Tier 0 at R.
  - Managed on-demand (Massed, RunPod) affords only Tier 0 at R ($677 central, $1,231 pessimistic).
- **RTX 6000 Ada node (48 GB):** S ≈ 1.2.
  - At Massed's $3.16/node-h it is about 2.7× the pool's all-in cost per G-h (2.3× at equal VAT; 2.0× on raw price
    ÷ S).
  - Massed's Ada was out of stock on 27 Oct. In stock: Vast 4× at $2.40/node-h (verified, Türkiye) and RunPod Secure at
    $3.96.
  - Tier 0 at L: $893 central, $1,958 pessimistic.

---

## 7. Per-table $ breakdown (recommended scenario, all-rented basis)

Priced at the S4 pool's effective rates (no VAT in the central case, 20% in the pessimistic case):

| Rate per lo-G-h | Central | Pessimistic |
|---|---|---|
| Sudoku rows | $0.406 | $0.932 |
| New-task rows | $0.454 | $1.119 |

The "cen on hi" column prices the plan's hi G-h at central rates, to show sensitivity to the row estimates themselves.
Each figure is rounded from its exact value, so columns may not sum exactly.

| Table | Tier | G-h lo–hi | $ central | $ pess | $ cen on hi |
|---|---|---|---|---|---|
| **A.1** trained baselines | 0 (B1, B2, B★, B3, B4a/b, B5) | 93.4–169.8 | 38 | 87 | 69 |
| | L-core (B★, B3 to L) | 130.8–219.9 | 53 | 122 | 89 |
| | 1 (B6, B7) | 10.5–19.3 | 4 | 10 | 8 |
| | 2 (B8, B10) | 9.5–20.8 | 4 | 9 | 8 |
| | **A.1 total** | 244.2–429.8 | **99** | **228** | 175 |
| **A.2** test-time baselines (E8, E9) | 0 | 7.0–13.0 | **3** | **7** | 5 |
| **B** internal arms | 0 (A0, A2, A3, A1…, X1–X5, X8, X10, X11) | 183.1–329.1 | 74 | 171 | 134 |
| | L-core (A0, A2, A3, X1 to L; R-prefix duplicates) | 313.1–526.9 | 127 | 292 | 214 |
| | 1 (minus X18a) | 129.1–229.3 | 52 | 120 | 93 |
| | 2 (X13, X16, X20; minus X18b/c) | 32.2–58.2 | 13 | 30 | 24 |
| | **B total** | 657.5–1,143.5 | **267** | **613** | 465 |
| **C** protocols and diagnostics | 0 (C0) / 1 (extras on X1, B★; D6) / 2 (D7) | 22.4–41.5 | **9** | **21** | 17 |
| **D** tasks | 0 (T3) | 32.9–64.7 | 15 | 37 | 29 |
| | 1 (T3+, T5, T6) | 80.1–224.0 | 36 | 90 | 102 |
| | 2 (T4, T7, T8) | 201.5–417.7 | 91 | 225 | 190 |
| | **D total** | 314.5–706.4 | **143** | **352** | 320 |
| Fixed (setup, warm-up, pilots, egress) | | | 40 | 92 | 40 |
| **Total, all rented** | | 1,245.6–2,334.2 | **561** | **1,312** | 1,022 |
| Owned-hardware offset (rented work saved − electricity) | | | −133 | −117 | |
| **Package total** | | | **428** | **1,195** | |

---

## 8. The ten most expensive rows (recommended scenario, all-rented, central / pessimistic)

| # | Row | Tier | G-h lo–hi | $ cen | $ pess |
|---|---|---|---|---|---|
| 1 | T4 Maze-Hard (gated, 9 runs + pilot) | 2 | 159.4–316.8 | 72 | 178 |
| 2 | A3 → L (6 seeds; s0–s2 from scratch) | L-core | 90.3–151.5 | 37 | 84 |
| 3 | X1 → L (6 seeds) | L-core | 82.5–138.9 | 34 | 77 |
| 4 | A2 → L (6 seeds; s0–s2 from scratch) | L-core | 70.7–119.1 | 29 | 66 |
| 5 | B3 → L (6 seeds) | L-core | 66.3–112.7 | 27 | 62 |
| 6 | A0 → L (6 seeds; s1–s2 from scratch) | L-core | 66.0–110.3 | 27 | 62 |
| 7 | B★ → L (6 seeds; 0 if B★ = B3) | L-core | 64.4–107.1 | 26 | 60 |
| 8 | A0 (R row: rerun s0, s3–s5, A0@45k evaluation) | 0 | 42.4–77.4 | 17 | 40 |
| 9 | T3+ ungrounded visual Sudoku | 1 | 34.5–68.0 | 16 | 39 |
| 10 | T3 visual Sudoku | 0 | 32.9–64.7 | 15 | 37 |

T7 follows at $15 / $36. Summed per arm (R row + L upgrade, central; the 3.7 G-h of R-prefix duplicates, about $1.5, are
not assigned to arms):

| Arm | $ central |
|---|---|
| A3 | 50 |
| X1 | 46 |
| A0 | 44 |
| B3 | 38 |
| A2 | 36 |
| B★ | 32 |

On the Vast 4× node (unverified host) multiply by about 2.2; on managed on-demand by about 4.3 (3.6 ex VAT).

---

## 9. Caveats

- **S is unmeasured for every card except the 5080.** The node scenarios carry the largest uncertainty; the G0 probe
  removes it for under $5.
  - The 4090 Laptop came in at half the roofline value, so power-capped parts (Max-Q, Ada, 300 W servers) can land at
    the low end.
  - The PRO 6000 range (1.8–2.1–2.6) rests on Vast-measured bandwidth. The high end needs full-rate FP32 accumulate,
    which is not confirmed. `docs/cost/roofline.py` still uses spec-derived bandwidth, and feeding it the measured `gpu_mem_bw`
    ratios is left for the probe.
  - Even at S = 2.6 the Vast node at $5.34 costs $0.51 per fast-5080 G-h on raw price ÷ S, against $0.33–0.35 for the
    pool. After utilisation the figures are $0.64 and $0.36. The node's break-even ($2.89) stays below every 4× price
    that could be rented on 27 Oct ($5.13).
- **The plan's own G-h for T3–T8, B8 and B10 are unmeasured** (plan notes [n13]–[n16]); T5 spans 4× (26–105 G-h).
  - The 25–50% failure allowance, the "cen on hi" column and the §6.3 stress line ($1,750; $1,913 with +P) cover this.
  - Their pilots replace the numbers.
- **5080 rental supply** is the pool's main risk.
  - The live API lists 44–46 machines but only 14 verified 1-GPU offers, and 8–9 distinct verified full-bandwidth hosts
    at ≤ $0.40. RunPod Community stock is "Low".
  - The peak needs 10–16 instances for about a day. If supply is short, follow the §5 fallback order: RunPod $0.39,
    then verified hosts at ≤ $0.60, then two waves (+0.5–1 day). The $ change is a few dollars.
- **Host quality varies.** On Vast, 3–4 of 30 1-GPU 5080 offers measure half bandwidth and 5 are capped at 250 W. The
  probe, the 4.6 steps/s rule and the `gpu_mem_bw > 700` filter are what make lo the right base.
- **The pairing check (init hash) is an assumption, not a fact.** If a rented 5080 initialises differently from the
  owned one, run whole families on rented 5080s only, with the owned cards evaluating: no change in $, less owned
  offset (about +$90 central; a rough estimate, not printed by the scripts).
- **VAT.**
  - Vast states that it does not collect or remit VAT, so the central case has none for Vast spend. If Vast starts
    collecting, or the purchase goes through an institution that self-assesses VAT without recovering it, add 20%:
    the central all-tiers-at-L figure becomes $673.
  - Massed's and RunPod's tax treatment was not verified; they keep 20%.
- **Spot prices are medians or floors** (Nebius "from $0.79" floats up to on-demand minus $0.01; Vast min bids are
  floors). They are used only in S2s and for evaluation.
- **Market snapshot.** Vast medians moved by 5–20% between the two fetches on 27 Oct, and host counts by ±2. Re-fetch at
  G0.
- **The owned-hardware share** assumes the 5080 pair at 40% availability (it is available part-time) and the laptop
  at 90%. At 0% the package is the all-rented column.

---

## Sources

Prices (27 Oct 2026 snapshot; per-offer rows in [cost/prices_2026-10-27.csv](cost/prices_2026-10-27.csv)):

- Vast.ai: offer search API (aggregated over the live listings; medians or min-bid medians), [vast.ai/pricing](https://vast.ai/pricing).
  Tax treatment: [docs.vast.ai/host/guide-to-taxes](https://docs.vast.ai/host/guide-to-taxes).
- RunPod: [runpod.io/pricing](https://www.runpod.io/pricing).
- Massed Compute: [vm.massedcompute.com/pricing](https://vm.massedcompute.com/pricing).
- UpCloud: [upcloud.com/pricing](https://upcloud.com/pricing/).
- Verda: [verda.com/pricing](https://verda.com/pricing).
- Lyceum: [lyceum.technology/pricing](https://lyceum.technology/pricing).
- Nebius: [nebius.com/prices](https://nebius.com/prices).
- Hyperstack: [hyperstack.cloud/gpu-pricing](https://www.hyperstack.cloud/gpu-pricing).
- Jarvislabs: [jarvislabs.ai/pricing](https://jarvislabs.ai/pricing).
- Scaleway: [scaleway.com/en/pricing/gpu](https://www.scaleway.com/en/pricing/gpu/).
- HyperAI, EcoHash and Packet·ai (via the getdeploying.com aggregator):
  [getdeploying.com/gpus/nvidia-rtx-pro-6000](https://getdeploying.com/gpus/nvidia-rtx-pro-6000),
  [getdeploying.com/packet-ai](https://getdeploying.com/packet-ai).
- Hetzner GEX430 (excluded): [hetzner.com/dedicated-rootserver/gex430](https://www.hetzner.com/dedicated-rootserver/gex430/).

Other inputs:

- GeForce BF16 tensor throughput with FP32 accumulate: NVIDIA RTX Blackwell GPU architecture whitepaper,
  [images.nvidia.com/…/nvidia-rtx-blackwell-gpu-architecture.pdf](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- Exchange rate: ECB euro reference rates,
  [ecb.europa.eu](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-usd.en.html).
- Electricity: regulated residential tariff, €0.2001/kWh ($0.2231/kWh).
- Measured throughput of the RTX 5080s and the RTX 4090 Laptop GPU, and every G-h figure: the plan (§2, §8, §10) and
  the runs in `results/sudoku-contractive/`.
