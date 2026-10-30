# Tractable polar decoding: research and implementation log

Updated 30 October 2026. This agent-maintained log records exploratory reasoning, controls and decisions; [Report 11](../reports/11_tractable_polar_decoding.tex) contains the concise algebra. The author's paper-outline scratchpad is separate.

## Aim and working boundary

Keep the Report-8 encoder as a fixed reference, known K/noise, IID source and multiplicity-aware PUPE. The 29 October sum-geometry continuation also studies explicitly labelled encoder modifications. Use Reports 7 and 9 to remove full-payload computations and identify useful structural approximations. A small representation is not enough: charge discovery, normalization, repeated residual contexts, mixing and count readout. New constructions need comparison with established methods before any novelty claim. Job038 subsequently adds a focused encoder comparison using existing receivers and matched D2 calibration; it does not introduce a learned inference family.

## 30 October: returned-result audit, pruning decisions and common-D2 encoder test

This supersedes the priorities and the two-receiver Job038 plan below; older entries retain their
original scope. The user requested analysis of returned Jobs036/037, closure of weak avenues, common-D2
comparison against balanced and learned dense encoders, and a concrete next decoder direction.

`jobs/037_polar_tractable/audit_results.py` independently checked **1,130,406** saved frame records:
manifest/config, reviewed source hashes and decoder budgets for completed runs, archived physical
matrices and input arrays against D2, observation hashes, legal counts, recomputed count-PUPE and
candidate recall. Only full 256-frame cells from certified complete runs enter matched contrasts;
partial cells are separate diagnostics. Seed sets are explicit, with paired frame and seed intervals.
No run was submitted/cancelled and no active-campaign runtime file changed.

| Local snapshot | Complete | Partial, uncertified | Source-guard failure | Absent |
|---|---:|---:|---:|---:|
| Job036, original 81-shard manifest | 36 | 27 | 1 | 17 |
| Job037, 60 rows | 42 | 18 | 0 | 0 |

All 42 selected balanced Job035 control runs are complete, including the three DEC-05 retries.
Job036 B14/DEC-19/seed3201/K22–26 contains 4096 frames but its downloaded log reports source changes
before final certification. It is excluded. Missing local files do not establish live scheduler state;
some absent original036 rows were previously deprioritized. Do not resubmit them automatically.
A refreshed copy is needed if HPC has finished more work than this local snapshot contains.

Three seeds, K26, 256 IID frames/seed/cell, balanced reference; PUPE in percent:

| Decoder | B12,4dB | B12,10dB | B14,4dB | B14,10dB |
|---|---:|---:|---:|---:|
| D2 | 2.519 | .456 | 40.400 | 8.569 |
| DEC-14 explicit multiset beam | .310 | .045 | 6.030 | .341 |
| DEC-25 SCL-only beam | 14.108 | 1.808 | 69.381 | 35.732 |
| DEC-26 basis beam | .671 | .050 | 31.510 | 4.778 |
| DEC-29 frozen-cache correction | .621 | .035 | 31.425 | 4.612 |
| DEC-31 XOR pair correction | .366 | .020 | 30.464 | 3.400 |
| DEC-32 SCL single-slot correction | .270 | .010 | 30.198 | 3.135 |

DEC-14 minus D2 at B14/K26/4dB is −.34370, paired three-seed t95 [−.37177,−.31563].
DEC-32 minus DEC-26 is −.01312 [−.01939,−.00685] there and −.01643 [−.01947,−.01339]
at 10 dB. These are descriptive exploratory intervals, not multiplicity-corrected held-out inference.
Recorded B14/4dB seconds/frame: D2 .049, DEC14 .587, DEC26 2.236, DEC32 3.855. Different HPC
executions and excluded training/setup prevent a controlled same-node timing claim; procedural
execution has not yet bought practical speed at these six-bit payload sizes.

**What can close at current budgets:** SCL-only beam25 as the main search, the fixed four-center
cache29, the unconditional common-product/beam IS33, and the union-beam34 are not priorities for
further broad sweeps. In completed B12/K26 4/10dB cells, DEC-33 returns exactly DEC-26's hard counts at about
5.6–5.8 times its recorded cost; ESS≈511–512 while the expected-risk bound is 1. High ESS is not
independent discovery or posterior accuracy. DEC-34 adds about 5.5 times cost without useful gain there;
its B14 runs remain uncertified, so the claim is B12-specific. DEC-31 loses to simpler DEC-32 at the highlighted
B12/B14 cells; keep it as an ablation, not the default joint correction. This closes these finite
implementations/budgets, not the underlying mathematical methods.

**Job036:** B14/seed3201/K26/4dB, DEC-15/16/17/18 yield 24.084/5.965/6.115/6.070% versus DEC-14's 6.130%.
At 10dB their errors are 1.277/.270/.165/.120% versus DEC-14's .346%. Product-free beam DEC-16 is already strong;
17/18's high-SNR gains are worth understanding but cost roughly 33–48s/frame versus beam .44s.
Current unconditional product mixtures and SMC composites DEC-20–24 do not justify a new broad bank.
Preserve a cheap, targeted correction question; do not infer family-wide failure from one seed or
finite budget. The failed DEC-19 hard-load row and incomplete B14 cells cannot settle all pair-refinement
comparisons. B14 parity chains DEC-27/28 and four-symbol DEC-30 remain open pending complete results.

**Failure localization:** at B14/K26/4dB, DEC-26's visited-word missing-copy floor is 4.41%, its positive
numerical-tail capacity floor 24.44%, and actual loss 31.51%. At 10dB these are .336%,2.639%,4.778%.
For a saved tail table define capacity c_w=max{j:tail(w,j)>0}; any action using that capacity misses
at least 1−sum_w min(a_w,c_w)/K. This is an optimistic information-support diagnostic, not a posterior
certificate: capacities need not coexist in one state, and zeros can reflect pruning, sampling or
floating underflow. Thus global candidate recall alone cannot identify the failing stage.
At B14/10dB, DEC-32 accepts 114.37 of 128 moves on average but changes state only .582 times. Its low empirical
support explains most remaining error; large nominal acceptance cannot be used as a mixing argument.

**Next decoder experiment:** keep DEC-14 as the performance reference and DEC-26→32 as the procedural reference.
Instrument where true prefixes leave the beam (truth for analysis only), comparing screen breadth and
beam width on the same predeclared B14/K26 4/10dB frames. Separate word discovery, joint-state retention,
positive-tail support and allocation loss; include full setup/repeated-context work. Then test a smaller
correction budget from the retained configurations at matched elapsed work. A mixed-pattern affine
provider is justified only if these errors require coordinated moves and its conditional draw pays
for setup/normalization. No new DEC family or broad HPC submission is needed before this diagnostic.

**Encoder:** Job038 now has nine encoder labels and only common D2 in the primary comparison. The four
reference/repair3 × uniform/balanced arms are joined by frozen/learned dense, frozen/learned sparse and
learned polar energy from Jobs032/034. All 30 archives are available. Their physical matrices and D2 are
reevaluated on the current paired IID count frames, with recorded float64 unit normalization of stored
float32 columns; old IID membership metrics are not imported. Only 12 repaired calibrations need
training. 54 rows, 27,648 D2-frame evaluations, 3,072 separate pattern/outside-oracle diagnostics. Dense
uses 256 occupied coordinates rather than 32; joint historical training spends encoder optimization too.
The original 2x2 effect is kept separately from comparisons against dense. No custom decoder is added.

All 42 reusable control rows completed locally, with 21,504 frames independently checked for physical
signals, unit power, count-PUPE and cross-encoder truth/noise pairing. All 12 archived reference replays
preserve their decisions. Maximum column rescaling is 4.915e-7. The nine-arm execution/merge smoke and 20
focused regression tests pass; Reports 8/10/11 compile. Repair production training remains pending:
submit Job038 rows 1–24 from a stable checkout (twelve reference replays plus twelve repair calibrations),
or copy the complete controls and let the full array skip them. No job has been submitted.

The fresh common-D2 controls also show why learned dense must not be the only comparator.
At B14/K26, three seeds and 128 IID frames/seed/cell, count-PUPE in percent is:

| Encoder | 4 dB | 10 dB |
|---|---:|---:|
| Balanced polar | 40.775 | 9.315 |
| Fixed dense | 40.525 | 9.535 |
| Learned polar energy | 41.496 | 9.445 |
| Learned dense | 43.219 | 14.253 |

These are 128-frame encoder-control results, separate from the 256-frame decoder table above.
The paired three-seed 95% t interval for balanced minus fixed dense at 10 dB is
[−2.48, 2.04] percentage points; against learned dense it is [−14.67, 4.80].
The point estimates do not establish a general ranking. Keep all these controls, report training costs,
and wait for the repair arms before drawing conclusions about the proposed signature.

Report 8 is retitled **Encoder geometry and multiuser sum design** and explains deterministic generation,
known frame/KL/adder-code foundations and the bounded next programme. The finite three-row search is
not a general construction theorem; the combinatorial encoder space is not exhausted. A clear stopping
rule is to evaluate this candidate, validate a useful win on new seeds/full grid, or close it if the
conditional-distance gain fails to yield common-receiver benefit. More encoder families are not needed
merely because the algebra permits them. Generated evidence and commands are in the experiment bank.

## 30 October: implement the discriminating encoder comparison

The user requested a new job extending earlier encoder work, comparison with balancing and historical
results, and a concrete decoder continuation. Job034 owns learned energy/D2, Job035 owns deterministic
balance/full-message controls, and Job037 already owns the tractable decoder bank. Added Job038 locally
under `jobs/038_polar_signatures/`; shared runtime files remain unchanged so existing source audits retain
their meaning. No new Markdown file or DEC family was needed. Commands and complete budgets are in the
existing experiment index.

The selected test is the four-arm reference/repair3 by uniform/balanced comparison, with fixed Nc32,
n256, B12/B14, placement and per-message power. Only repeated output-character rows are replaced;
the original native CRC/SCL receiver is incompatible. Matched D2 and existing DEC-26 basis proposals
score the actual modified waveforms. DEC-26's implementation constructs an unused polar object even
in basis-only mode: the wrapper carries the original CRC metadata for that constructor, the job rejects
SCL/union modes, and an independent test fails if SCL is called. This is a compatibility detail, not
an assertion that the new signature is still decoded by the original CRC/SCL map.

The four historical cells K7/26 at 4/10 dB, 128 IID frames/cell and three old seeds give 24 array rows,
24,576 receiver-frame evaluations. D2 retains the historical distinct-source training contract on the
full load/SNR range. Twelve reference checkpoints are reused; twelve repaired calibrations are new.
Job034's learned-energy optimum is historical context only, so this experiment must not claim superiority
to learned energies from a win over deterministic balancing. The 39-symbol extension is deferred because
it changes occupied resources and overlap, requiring its own fair control.

A separate first-two-slot oracle preserves source weighting without looking for favorable witnesses.
Given the outside users and the two patterns, its ordered 64² pair posterior exactly accounts for same-
pattern repeated messages and permutation masses. Count tails determine the two-copy action. It records
near-competitor distances, pair evidence/risk, overlap and cost, stratified by same/mixed patterns. This
does not compare alternative pattern allocations, solve message discovery, or establish global Bayes
risk. It provides a limited geometric control beside the full receivers' candidate recall and PUPE.

One provenance check changed the implementation: recomputing balanced amplitudes locally differs from
the archived HPC values by up to 5.55e-17. Exact checkpoint hashes correctly rejected that regeneration.
The default now validates and freezes archived amplitudes for both signature arms, reconstructing the
original matrix exactly. A separately labelled fresh-reference mode regenerates and retrains instead.
All12 historical one-frame D2 replays then preserved input hashes and count decisions. Eight B12/B14
four-arm smoke runs pass, including short matched training and the merger/plot path. No production
PUPE ranking follows from these checks. Final evidence is generated under
`jobs/038_polar_signatures/results_validation_final_2026-10-30/`.

Validation passes 46 focused tests covering physical operators, oracle pair tails against direct
enumeration, legal count actions, procedural compatibility, paired comparisons and job/reuse contracts.
Full-budget DEC-26 K26/4dB probes take approximately 0.55–0.70s/frame locally; previous reference D2
training took 19–51 min/calibration. A rough 9–10 aggregate job-hour estimate combines these proxies;
training on the repair and HPC hardware may differ. The job requests 2 CPUs/16 GB/24 h per row. The user
explicitly requested **no concurrency caps, now or in future**: removed caps from scripts, current
submission instructions and tests, and added the rule to AGENTS.md. Historical timing assumptions in
older entries describe their original planning context, not current submission policy. No jobs submitted.

For now the encoder's next useful step is this evidence, not more row searches. For decoding, the affine
orbit result gives a local invariant move but recorded exact witnesses are rare and it cannot discover
other waveform modes. Sparse-Walsh packet identities need measurements the reference does not transmit.
Neither warrants a new broad HPC family. Continue with the existing Job037 bank, then a bounded mixed-
pattern conditional-provider experiment: measure sector construction, envelope acceptance, actual moves
and conditional tail error in imperfect outside contexts, against existing pair proposals at matched
total work. If useful, integrate it as a proposal variant inside an existing corrected chain and test
global mixing/PUPE. If sector setup or envelope slack dominates, record that failure before adding a DEC.

## Earlier tractability passes: retained starting point

- Exact single-slot residual likelihood supplies polar LLRs. A finite SCL list is a candidate rule, not a partition function. Exact physical rescoring and an explicit prior mixture make its proposal density usable.
- Grouped characters admit a normalized independent-parity reference and pointwise envelope. A dependent triangle proves that exact rejection correction can still be exponentially inefficient.
- Replacing envelope weights on explored payloads improves the bound while a product-coordinate exclusion trie samples the complement directly. This repairs one triangle but a fixed global flip radius fails on repeated conflicts; small exact components repair that control, not every actual code.
- Score oscillation transfers bounds across nearby residuals and permits corrected reuse. Small count-tail representations remove the output alphabet axis, but obtaining representative configurations remains the difficult part.
- Same-pattern pair discovery has four-symbol polar and XOR-first variants. The XOR guide omits a code-consistency partition factor; physical rescoring corrects retained pair scores, not undiscovered pairs.

These observations motivated the implementations below and the mixed-pattern investigation that follows. The old three logs remain retired; this is the new tractability log requested by the user.

## Structural pass: overlap parity conditioning

## Question and scope

The same-pattern four-symbol channel does not handle arbitrary mixed-pattern overlaps. Report 7 already groups each pair by overlap parities, but its implemented contraction still groups/copies all V payload weights and stores up to 2^(r_left+r_right) table entries. Can its structural observation be combined with the affine character operations and pointwise envelopes in Report 11 to perform mixed-pattern block updates without materializing V?

The proposed answer is conditional and finite: fix a hypothesized ordered tuple of patterns and an outside residual. Condition only selected overlap parities, chosen so each retained interaction has at least one fixed endpoint. The remaining payload laws become independent affine one-user problems. Their exact or bounded normalizers and available full-space proposals give an executable block operation. The exponential variable is the total conditioned rank; it is not guaranteed small.

## Exact algebra (checkpoint before checks)

For m slots with fixed patterns p_k, expand the physical block score after removing the common residual norm and m unit self-energies:

    S(v_1,...,v_m) = sum_k s_k(v_k) - sum_{k<l} sum_{g,h} J_kl(g,h) chi_g(v_k) chi_h(v_l),
    J_kl(g,h) = sigma^-2 sum_{shared resources with local rows g,h} sqrt(e_ki e_lj).

The unaries s_k are r dot phi(p_k,v_k)/sigma^2. Constant row keys are exact: J(0,0) contributes a common constant; J(g,0) and J(0,h) shift one unary. Remove these before declaring interaction rank. Every remaining J is nonnegative for the fixed positive amplitude encoder. The equations can also use signed grouped J with absolute values in bounds.

Choose a subspace W_k of payload characters for each slot, with row basis H_k and rank r_k. A nonconstant cross-factor (g,h) is covered if g belongs to W_k or h belongs to W_l. Condition alpha_k=H_k v_k. Any covered factor is now either a known constant (both endpoints covered) or a unary character on the other endpoint. The retained score has the exact form

    S_W = c(alpha) + sum_k t_{k,alpha}(v_k), with H_k v_k=alpha_k.

Thus

    Z_W = sum_alpha exp(c(alpha)) product_k Z_{k,alpha},
    Z_{k,alpha} = sum_{v_k: H_k v_k=alpha_k} exp(t_{k,alpha}(v_k)).

No extra sector prior or 2^-r factor belongs in this unnormalized payload sum. Every payload tuple belongs to one sector, and each affine law contains 2^(b-r_k) payloads. Using restricted character maps retains all polar/CRC constraints. A standard unconstrained SCL call is not automatically a sampler for these arbitrary parity restrictions.

For a pair, choose W_p to span all nonconstant left overlap rows and W_q=0. Then c may be absorbed into the restricted left unary, while right fields shift by the known left signs. Only 2^r_left sectors are needed. Reverse orientation can use r_right instead; compare actual inner costs as well as rank. This replaces a two-sided table with one-sided affine queries, not with free known partitions.

## Rank budget, exact correction, and limits

A limited rank cover can omit uncovered factors. Write S=S_W+D with

    |D| <= T(W) = sum_uncovered |J_kl(g,h)|.

Then e^-T Z_W <= Z <= e^T Z_W, and TV(pi,pi_W) <= tanh(T/2) from log-density oscillation 2T. This is a fixed-pattern block comparison, not a guarantee on the stationary law of approximate Gibbs updates in changing contexts. If instead pi_W can be drawn exactly, rejection with exp(D-T) restores pi and has acceptance at least exp(-2T). Low noise can destroy this guarantee even for small raw energy overlaps.

More generally use Report 11's pointwise one-user envelopes exp(t_k)<=U_k q_k on each affine support. Define E_alpha=exp(c(alpha)) product U_k and U_W=sum E_alpha. Draw alpha proportional to E_alpha, draw the independent q_k, then correct by the physical score using the envelope exp(T) U_W. This uses normalized available proposals and finite sector sums, not unknown Z_k. Available one-user lower bounds similarly give L_W=sum exp(c) product L_k and acceptance lower bound exp(-2T) L_W/U_W. If no factors are omitted, the exp(-2T) penalty disappears; the inner envelope looseness remains.

The count prior remains uniform on ordered IID payloads conditioned on the patterns; no extra factorial factor is introduced. Projection of complete ordered K-slot samples yields the count prior. An exact corrected block move preserves its conditional posterior, while mixing and final count tails remain separate. Distinct-message exclusions introduce additional factors and require separate treatment.

## What is and is not structurally new

Conditioning to simplify a factor graph is established. Targeted primary comparison: Bidyuk and Dechter, *Cutset Sampling for Bayesian Networks*, JAIR 28 (2007), https://www.ics.uci.edu/~csp/r137.pdf and https://arxiv.org/abs/1110.2740. Their cutset/Rao-Blackwell mechanism already combines assigned variables with exact residual inference. Report 10 DEC-09 also already allows conditioned feature cutsets. No general novelty claim is warranted.

The concrete extension to review is the combination of physical cross-character coefficients, one-sided affine polar sectors, shared rank accounting across multiple overlaps, and Report 11's normalized pointwise envelopes. It avoids the existing full-V grouping stage for this conditional operation and provides an explicit rank/omitted-energy tradeoff. Testing whether current banks offer useful low ranks or cheap tail bounds remains necessary.

## Finite algorithm, cost and selection obstruction

Given a fixed pattern tuple and outside residual:

1. Intersect supports, group physical cross-coefficients by endpoint row keys, and absorb constant-row factors into unaries. Build candidate row spans by elimination. For the simple mixed pair operation, choose one endpoint's complete nonconstant overlap span; no combinatorial cover search is needed.
2. For each alpha in the 2^R joint sectors, solve H_k v_k=alpha_k using cached right-inverse/nullspace maps. Shift unary fields using covered factors; if both endpoints are fixed place the factor in c(alpha). Construct each affine character envelope, its lower bound and normalized full-support sampler. Accumulate sector upper/lower weights.
3. Draw sectors proportional to their upper weights; draw full payloads from their affine proposals, including uniform free bits when score rank is deficient. Correct with the actual physical score and exp(T) U_W. This is an executable exact conditional draw with random, potentially enormous runtime. Capped hold-on-failure or MH give invariant kernels if the outside context/pattern tuple and chosen proposal agree forward/reverse.
4. Reconstruct complete ordered K-slot configurations, then use the existing sparse count-tail/readout procedure. These conditional operations do not independently solve joint discovery or convergence.

Geometry, coefficient grouping, binary row maps and a chosen cover are independent of the residual and can be cached for a fixed pattern tuple. Unary field coefficients, inner envelope weights and sector weights depend on the residual. Per-sector envelope setup is polynomial using the independent-character reference (coarsely O(d b^2+d log d) scalar/bit operations per slot, after handling maps); the expected rejection count may be exponential. A conservative cover setup costs polynomial preprocessing plus 2^R(E+sum C_inner), where E is the grouped interaction count. For a rank-budget greedy heuristic, a naive implementation that tests all endpoint rows and rescans factors at each addition costs O(R E^2 b^2) binary/scalar work; this selection cost must not disappear from the account. An all-row one-sided cover avoids this optimization overhead.

Store 2^R sector weights plus one sector's working inner data and rebuild the chosen sector, or cache all inner proposals at proportional storage cost. Affine maps and restricted character keys cost polynomial memory in m,b,d; the bank remains Theta(Pd). Each rejection attempt pays sector selection, affine payload generation, code evaluation and physical scoring. A usable certified upper bound on mean attempts is exp(2T) U_W/L_W. Extra conditioned parities can reduce inner envelope slack, so the least-rank cover is not automatically the least-work cover.

For J outside contexts and H candidate pattern tuples, multiply setup/draw costs by JH unless a specific valid cache reuse is supplied. Exhausting pattern tuples costs P^m; exhaustive all-slot pair selection costs K choose 2 choices. Exact fixed-pattern payload updates preserve only that conditional block law and need other pattern-changing moves for irreducibility across pattern allocations. Distinct-message source laws add exclusion constraints and are not derived here.

An elementary obstruction prevents borrowing Report 11's greedy matroid optimality: in one-sided selection, weights 9 on e1,e2,e1+e2 and 10 on e3,e4 make greedy choose e3,e4, retaining weight20. The rank-two span of e1,e2 retains27. In the full rank-cover problem with one-bit slots, each selected slot covers incident factors, already giving ordinary vertex-cover choices. The constructive route does not assume cheap global optimization of the best cover.

For uniform independent d-subset placements, expected overlap is d^2/n. With d proportional to b and n proportional to d, this overlap grows with b; nothing forces the relevant parity rank to stay bounded. Maintaining constant expected overlap under that random placement law requires n on the order of d^2, which changes the channel-resource scaling. Rank b gives 2^b sectors in this procedure, not a lower bound against every other algorithm. Weak raw overlap coefficients also need not be negligible after division by sigma^2.

The fixed-block approximation has an additional conditional decision bound: with the outside K-m true copies and patterns fixed, any fixed action's PUPE varies by at most m/K across the remaining tuples. Therefore a Bayes action for pi_W has regret at most 2(m/K) tanh(T/2) under the exact block law. Every count-tail error is at most tanh(T/2). Neither statement proves global accuracy for repeated approximate updates with changing contexts.

## Completed independent checks

Script: `tests/polar_tractability_structural_checks.py`; run with `uv run python -m tests.polar_tractability_structural_checks`.

- The physical fixture uses the actual b3/d16/CRC4 generator, three unit-energy positive-amplitude patterns in a star, and deliberately includes constant-row interactions that must become unaries/constants. It is a conditional algebra control, not a full balanced bank or performance experiment.
- All512 payload tuples are evaluated for12 seeded residuals and four covers (rank0,1,2,4). The script independently compares the direct Gaussian waveform score with the factorized law, verifies affine constraints, exact full-cover partition, omitted-factor bounds, proposal normalization, pointwise acceptance and correction, TV and the certified acceptance lower bound. Maximum discrepancy: 5.329070518200751e-15.
- Both star edges use the same rank-two central span, so jointly conditioning it costs four sectors, not the16 obtained by redundantly treating edge signatures independently. On the first fixture: rank0 has T=.291777, TV=.060676 and corrected acceptance .189223; rank1 has T=.128012, TV=.039064 and acceptance .267283; rank2 retains every factor with acceptance .349873; additionally conditioning leaf parities (rank4,total16 sectors) raises envelope acceptance to .692925. These values illustrate the algebraic cost/acceptance tradeoff only.
- A separate same-pattern pair control includes all ordered duplicate payloads and full-rank affine restrictions. A check verifies the rank-greedy20 versus optimum27 counterexample.
- Final consistency review corrected that same-pattern control to reuse exactly the same energy vector as well as support for both slots; equality of payloads now gives identical physical waveforms, as required by the fixed encoder. All checks still pass. This did not alter the main star fixture or census.
- A full placement/generator census uses the repository's B12/n256/seed3201 bank, b6/d32/CRC8, all2016 unordered distinct pattern pairs, and removes constant-row factors first. Histograms of min(left nonconstant rank,right nonconstant rank): rank0:19, rank1:123, rank2:331, rank3:544, rank4:541, rank5:374, rank6:84. Sum of one-sided sector counts is31,941; sum of comparable two-sided nonconstant table entries is1,093,347. Neither number is a timing estimate; each sector requires inner envelope/partition work. This also differs slightly from Report7's original table census because it first absorbs factors whose opposite endpoint row is constant.

The proposed derivation is now integrated into Report 11. This structural subtask changed only its independent check script; runtime implementation and verification are recorded separately below.

## Defensible stopping point

The pass establishes an explicit structural bridge for mixed-pattern and small joint blocks, with all-sector normalization and exact correction available without payload tables. It does not establish small useful ranks at growing b, favorable inner acceptance, inexpensive pattern discovery, or fast outer mixing. The closest established methodology is conditioning/cutset inference; the concrete encoder-specific combination deserves evaluation but not an unsupported novelty claim. A targeted next experiment can measure cover setup, conditioned rank, inner acceptance and complete conditional draw cost on actual residuals alongside the routes being implemented. No broader architecture choice is implied.


## Decoder implementation pass, 29 October

The current operations now have complete finite-budget implementations in `framework/polar_tractable.py`, with pair discovery in `framework/polar_pair_proposals.py` and registration in the existing decoder bank.

| ID | Variant |
|---|---|
| DEC-25 | SCL candidate lists + multiset beam |
| DEC-26 | Independent reliable-basis flips + multiset beam |
| DEC-27 | Independent-parity proposal + single-slot physical MH |
| DEC-28 | Finite-set repaired parity proposal + single-slot MH |
| DEC-29 | Fixed cache of repaired proposals + residual selection and physical MH |
| DEC-30 | Same-pattern four-symbol pair MH alternating with single-slot SCL-list MH |
| DEC-31 | Same-pattern XOR-first pair MH alternating with single-slot SCL-list MH |
| DEC-32 | SCL-list/prior mixture + single-slot MH |

Every method is nonlearned and IID-only. `run(name, scenario, **params)` accepts a procedural `PolarScenario`; it returns a sparse dictionary of Python integer message IDs and count-tail metadata. Existing dense `Scenario` inputs use the same implementation through the harness's geometry provider and a dense hard-count adapter. Earlier decoders remain unchanged. The small-B harness still constructs Phi and therefore does not become a large-B harness merely by choosing one of the new methods.

Explicit implementation choices:

- A beam's final weights include the multiset factorial prior. A chain updates ordered slots under the uniform IID prior and must not add factorials again. Both allocate K copies from tails, not posterior means or MAP counts.
- The continuation guide is isotropic. Final retained-state weights and MH ratios use the physical Gaussian likelihood.
- Basis candidates traverse increasing flip cardinality in the declared greedy basis order, capped by `max_refine`. This is not a globally best-first weighted-flip search. The explored set consists of actual full payloads; its complement remains available.
- Pattern screening scores all P patterns, then sends at most `patterns` to SCL/basis discovery. Fresh parity samplers construct all P envelopes. This remaining pattern cost is explicit.
- Chains start from an SCL multiset beam of width `init_width`. Initialization is fixed across proposal families but can depend on each decoder's recorded RNG seed if a parent has no CRC-valid extension.
- CRC-empty lists are not interpreted as absent patterns. List proposals use the prior when empty; empty beam parents receive one recorded uniform prior draw. This gives legal output, not successful discovery.
- The repaired sampler uses the actual finite-set envelope and directly samples its complement. The cached variant freezes centers after initialization; nearest-center choice depends only on the outside residual and the exact proposal ratio is retained.
- Pair lists contain same-pattern proposals, but their defensive ordered-pair prior and alternating single-slot moves retain access to other pattern allocations. Pair physical scores include unoccupied coordinates when compared across patterns. XOR-first discovery still omits its code-consistency mass.
- `candidate_ids` means all physically scored/cached words, including rejected proposals and initialization. Its truth-only recall is a discovery diagnostic, not posterior support coverage. Tail metadata separately identifies the empirical/restricted inference support.
- Prototype waveforms are dense n-vectors cached within a call, so memory is O(n times generated words); polar recursion currently copies arrays. No optimized-runtime claim follows from avoiding M.
- Finite-chain output has no mixing guarantee; exact-invariant kernels are not exact posterior samples. Envelope guarantees refer to real-arithmetic identities checked numerically, not interval certificates.

The mixed-pattern parity-cover result remains theory plus independent finite checks. It needs affine restricted proposal providers, sector preparation and pattern-changing outer updates before becoming another decoder.

### Validation

The focused suite passes **66 tests**, comprising existing bank/family/combination regressions plus **25 new tests** (10 pair, 9 procedural algebra, 3 harness, 3 bounded execution). Tests independently check physical waveform equality, bit ordering, proposal normalization and branch laws, repaired bounds, cached/single/pair detailed balance, count collisions, exhaustive-beam REF endpoints and sparse/dense-interface equivalence. Python compilation also passes.

All eight new variants execute directly at B100 with P4/b98/N128/CRC16/n256, at K1 and K2 with tiny budgets, without a Phi attribute or an alphabet array. IDs above 64 bits survive the Python interface. The noiseless single-user observation is recovered at positive declared variance 1e-4. K2 checks legal counts and reproducibility, not recovery. This is an execution control, not a B100 PUPE result or a scaling theorem.

```bash
uv run python -m unittest tests.polar_decoder_bank_test tests.polar_decoder_families_test tests.polar_decoder_combinations_test tests.polar_tractable_test tests.polar_tractable_execution_test tests.polar_tractable_harness_test tests.polar_pair_proposals_test
uv run python -m compileall -q framework baselines benchmarks src tests
```

The five tractability algebra modules also pass; the structural control's maximum discrepancy is 5.33e-15. Reproduction commands are indexed in the experiment bank.

### Paired smoke runs

Generated outputs remain ignored under `results/`:

- `tractable_smoke_2026-10-29`: B4/n12/P4/N8/CRC4, balanced bank3217, IID, K1/2/4, Eb/N0 -4/4/10 dB, 32 frames/cell. REF, DEC14, DEC25–32, trained D2 and adapted ODMA: **3,456 decoder-frame evaluations**. The two common bound overlays remain in the plots. New beams use width8, patterns4, list8, max_refine8; chains use samples32/burn_in32/init_width4. All new methods improve with SNR in this smoke, but some retain high-SNR discovery/mixing errors. Finite observed PUPE may fall below REF by chance; conditional posterior regret remains nonnegative.
- `tractable_mini_B12_2026-10-29`: B12/n256/P64/b6/N32/CRC8, balanced bank3201, IID, K2/7, Eb/N0 4/10 dB, four frames/cell; DEC14, DEC25–32 and ODMA: 160 evaluations. All new variants recover every transmitted copy in these easy cells; this sample is too small to infer a low error rate.
- `tractable_stress_B12_2026-10-29`: same contract, K22, four frames/cell: 80 evaluations. `tractable_combined_B12_2026-10-29` merges both B12 runs with source/config/pairing checks. New beams use width8/patterns8/list8/max_refine8. Chains use samples16/burn_in16/init_width2. DEC14 width8 is the explicit control. No new B12 D2 training or formal performance comparison is claimed. B12 bounds were omitted to keep this diagnostic small; the B4 comparison retains them.

B12/K22 diagnostic, **four frames per cell**:

| Decoder | PUPE at 4 dB | PUPE at 10 dB |
|---|---:|---:|
| DEC-14 | 0.0114 | 0.0000 |
| DEC-25 | 0.5227 | 0.0114 |
| DEC-26 | 0.0114 | 0.0000 |
| DEC-27 | 0.3977 | 0.0341 |
| DEC-28 | 0.4545 | 0.1477 |
| DEC-29 | 0.5455 | 0.2159 |
| DEC-30 | 0.5455 | 0.1705 |
| DEC-31 | 0.4432 | 0.1136 |
| DEC-32 | 0.5114 | 0.1705 |
| ODMA | 0.4205 | 0.1250 |

The important signal is a discovery failure, not a justified winner across methods. At K22/4 dB, DEC25's physically scored candidate recall is 0.614 versus 1.0 for DEC26. About 86% of its screened pattern SCL calls return no CRC-valid payload, and 186 beam-parent extensions across four frames require the declared prior draw. Many screened patterns can be wrong, so this percentage alone is not a causal CRC ablation. Nevertheless, the basis route constructs valid CRC/polar payloads directly and avoids late CRC filtering; this is a concrete structural difference worth investigating. Both beam variants otherwise use the same continuation/scoring/readout machinery.

The chains have only 32 total moves across 22 slots and a narrower SCL initializer. Their results therefore confound initialization, candidate discovery, proposal quality and mixing. Cache acceptance can also be poor after moving away from a fixed center. Do not close corrected sampling or pair moves based on this smoke, and do not call accepted-move fraction a mixing statistic (self-proposals count as accepted).

### Next discriminating work

1. Use the same strong basis/union initializer across correction variants and record actual changes, initialization PUPE and final PUPE. Compare matched total work as well as fixed iteration counts; 32 updates is only about 1.5 nominal single-slot sweeps at K22.
2. Examine SCL list width and early CRC/parity consistency against basis/union discovery, then increase payload length. The present B12/B14 payload remains six bits; seven radius-one candidates are a substantial fraction of that small payload alphabet.
3. For the new parity-cover construction, measure conditioned rank, setup and corrected acceptance on actual residual contexts before building a full outer decoder. Exploit shared overlap rows, and include pattern discovery and repeated contexts in the cost.

These are follow-up controls, not implemented combinations or HPC jobs. The current pass adds no HPC scripts, commits or pushes.

## Whole-decoder error and paired HPC preparation, 29 October

The next question is whether the conditional guarantees can say something about a complete decoder in expectation. Existing bounds cover omitted conditional mass, envelope slack/acceptance, residual transfer and dropped interactions. They do not turn finite MH output into independent posterior samples. An empirical ESS or accepted-move fraction does not close that gap.

### Common tilted product plus an exchangeable beam

For a fixed observation and any fixed vector t, the repaired parity construction gives a normalized full-support word law q_t and a pointwise envelope exp(t^T phi(w)) <= U_t q_t(w). For an ordered K-tuple W, write S(W)=sum phi(w), Q_t(W)=product q_t(w), and f_y(W)=M^{-K} exp(-||y-S||^2/(2 sigma^2)). Completing the square gives

```
f_y(W) <= C_t Q_t(W),
C_t = (U_t/M)^K exp(-t^T y + sigma^2 ||t||^2/2).
```

The discarded factor is exp(-||S-(y-sigma^2 t)||^2/(2 sigma^2)) <= 1. This accounts for the whole physical interaction; it does not assert independence of the target. The current guide sets t=y/(sigma^2+(K-1)/n). Validity holds for any fixed t; usefulness need not.

Let A be the retained multiset beam, with exact count weights f_#(a)=K!/prod a_w! times the ordered weight. Then Z_A=sum_A f_# is a lower evidence bound. Draw from a mixture of Q_t and the beam's normalized count law followed by a uniform shuffle. Each tuple's beam density is its count probability divided by its orbit size. Both mixture components are exchangeable, so density evaluation uses a product and an orbit factor, not K! permutations or a permanent. A slot-dependent product would lose this simplification.

For beam weight lambda<1, importance weights obey w=f_y/q <= C_t/(1-lambda)=C, and the normalized ratio r=w/Z obeys r<=C/Z_A=bar_kappa. No full message alphabet is needed. The beam and all proposal choices are frozen before H independent draws. Setup comprises beam discovery and one all-pattern repaired reference; draws reuse it. A common product can be a poor multiuser proposal, so this removes repeated preparation rather than promising good variance.

### Expected decision loss without a denominator penalty

For every count tail I_{w,j}, r I is in [0,bar_kappa] with variance at most bar_kappa. Bernstein's exponential-moment bound, applied to both signs of all MK tail errors, yields

```
E[max_tail |H^-1 sum r_i I_i - T|] <= sqrt(2 bar_kappa log(2MK)/H) + bar_kappa log(2MK)/(3H).
E[excess conditional PUPE] <= min(1, twice the preceding expression).
```

The readout maximizes self-normalized empirical tails. Their denominator is common to every action and positive, so the same action maximizes unnormalized tails. Comparing the two actions directly avoids adding a self-normalization denominator error. The expectation is over independent proposal draws, conditional on y and the fixed proposal; this is not a pointwise error certificate. Averaging over Y gives an ensemble excess-risk bound. Absolute PUPE additionally includes the unknown Bayes risk. The bound can be 1, hence uninformative, when the global envelope/evidence ratio is large. A tighter worst-case upper bound is not itself a performance ranking.

This combines earlier Report-7 count tails and orbit accounting, Report-9 independent weighted sampling, and Report-11 parity repair into **DEC-33**. Mixture importance sampling and exponential tilting are established techniques; no general novelty claim is made. Independent tiny checks enumerate all ordered tuples at K1/2/3, verifying normalization, exchangeability, the complete envelope, the beam evidence lower bound and weighted count tails. A separate exactly summable sampling experiment checks the expected-regret inequality.

**DEC-34** combines SCL and basis candidates before the existing beam pruning. At a fixed residual its candidates contain both lists. The resulting complete beams need not be nested because their later contexts and pruning differ; no monotone PUPE claim follows.

For the planned Job037, chain proposals share a stronger basis-beam initializer. The runtime default remains SCL for historical reproducibility. The harness records initial PUPE only after decoding, and actual state changes separately from accepted self-proposals. These controls address the initialization confound in the earlier smoke.

Work in progress: complete matched smoke checks, measure full-budget frame costs, pin the historical-control compatibility audit and generate the CPU array. The flat report/shared style, test directory, decoder bank and ignored generated-result locations are preserved. The author-owned paper outline is untouched.

### Completed verification and observations

The final focused suite passes **81 tests**, including the old decoder bank, families, combinations and Job036 guards, all procedural/pair tests, the new whole-tuple tests and Job037 manifest/source/reuse checks. Python compilation, PBS shell syntax and whitespace checks pass. New DEC33/34 also execute directly at B100 for K1/2 with sparse Python-integer counts; the K1 noiseless control recovers its transmitted word, while K2 asserts legality and reproducibility only. No large-payload PUPE claim follows.

The paired B4 run `results/tractable_matched_smoke_2026-10-29/` contains **1,872 evaluations**: REF, DEC14, ten tractable variants and adapted ODMA; K1/2/4, -4/4/10dB, 16 frames/cell. Its saved config records the smaller local budgets and matched basis initialization. All new variants improve from low to high SNR. Exact conditional regrets are nonnegative even when finite observed PUPE happens to beat REF. Both coding bounds and plots are retained; the plot layout was inspected.

The global envelope was vacuous throughout this first smoke. A small exact tightening is available at K1: the selected t=y/sigma^2 makes the discarded square equal ||phi(w)||^2=1, so multiply C_t by exp(-1/(2 sigma^2)). This changes only the certificate, not draws or scores. The final runtime/source audit includes it. `results/tractable_importance_bound_smoke_2026-10-29/` repeats REF/DEC33 (288 evaluations) and verifies all 144 DEC33 outputs and observation hashes are identical. Average K1 expected-regret bounds become 0.496/0.483/0.483 at -4/4/10dB; observed conditional regret is zero there. At K2/4 the bounds remain 1 despite average measured conditional regret below .008. The bounds are mathematically valid, but not presently sharp predictors. Earlier smoke directories preserve their original source hashes and are not rewritten as final-source runs.

`results/tractable_matched_B12_2026-10-29/` uses actual Job037 budgets, B12/b6/N32/n256/CRC8, seed3201, K22, 4/10dB and four frames/cell (96 evaluations across DEC14, DEC25–34 and ODMA). At 4dB DEC25 has PUPE .01136; DEC26–34 and DEC14 have zero; ODMA has .42045. At 10dB all tested research decoders have zero; ODMA has .125. All six chains' initial PUPE is already zero. Across the four 4dB frames, DEC27/28/29/32 make no actual changes; DEC30 makes three and DEC31 one. The improvement over the earlier smoke therefore chiefly removes the initialization/search-budget confound. It does not demonstrate useful mixing or a correction benefit. The single B14/K26/4dB timing fixture still has .5 PUPE for most basis-initialized receivers: harder discovery remains a real boundary.

### HPC preparation and historical compatibility

Job037 contains 60 rows (DEC25–34 x B12/B14 x three seeds), each with four loads, eight SNRs and 256 frames/cell. The total is 491,520 decoder-frame evaluations. Requests are CPU-only, two CPUs/16GB/72h each, with four concurrent rows at most. All physical parameters, source model, seeds and observation streams match the balanced Job035/036 contract. No new D2 training or ODMA rerun is queued. All beams use width32; chains use basis init_width32, burn64, samples64; DEC33 uses 1024 independent tuples. The job README owns the full parameter matrix and queue/merge commands.

The compatibility contract was reconstructed from three actual Git snapshots, not inferred from result labels. Unchanged dependency modules match byte-for-byte. Existing encoder, fixture, metrics, bounds, training and decoder bodies remain unchanged; registry/sparse adapter and new harness diagnostic/setup branches were reviewed. Historical research overrides are pinned (notably DEC07 guide_log_floor=-1000 versus default -80). The Job036 source checker recognizes only this explicitly pinned forward transition, retaining rejection tests for altered historical dependencies/definitions/inventory. The current merger checks all selected complete cells/seeds, parameter values, source hashes, matrix hashes and per-frame input hashes. Missing controls cause an error, not silent cohort changes. `results/tractable_historical_reuse_2026-10-29/` successfully merged real B12/seed3201 D2, ODMA, DEC14 and DEC24 originals; provenance remains attached to each run.

Single-frame full-budget K26/4dB probes took .58–3.59s/receiver at B12 and .54–5.40s at B14. Straight multiplication gives about 1.2–12.3 row-hours, 271 job-hours overall, or 68h with four active jobs before queueing. These are deliberately labeled rough laptop-based extrapolations, not measured HPC throughput or a PUPE study. Saved probes are `results/tractable_job037_probe_B12.json` and `_B14.json`.

Report 11 was edited in place and compiled successfully using the existing desktop editor. Its flat location, shared report style and aim-led opening remain intact. Per the current editor instruction, no separate PDF was exported: the adjacent saved PDF predates this addition, while the editor preview uses the new source. The author-owned paper outline and preexisting log deletions were left untouched. Generated results remain ignored. No commit, push or job submission was performed.

### Next research gates

1. Improve full-configuration envelope tightness using known waveform geometry, or several shared tilts, without restoring exponential setup. The present multiuser bound identifies a gap, not a useful performance prediction.
2. Resolve discovery failures as payload length grows: parity/CRC-aware lists, basis search and their union need matched work/coverage measurements. B12/B14 still have six-bit payloads.
3. Turn the mixed-pattern parity-cover algebra into an affine conditional provider only if measured overlap ranks and preparation/acceptance costs support it. It remains checked theory, not an implemented DEC entry.
4. Test corrections on cases where the matched initializer is imperfect, reporting actual state changes and paired loss against that initializer. Zero errors from an unchanged correct initializer cannot rank these kernels.

These are concrete continuing paths; neither the available algebra nor this smoke establishes a feasible multiuser B100 receiver. Learning remains outside this pass.

## Sum geometry continuation, 29 October: question and discovery record

The current user reopens encoder design alongside tractable inference, keeping the present encoder as the reference. Read CURRENT_STATE, the experiment index, Report 7's identifiability/count-class derivations, Report 8's energy moments and distance caveats, Report 9's class-conditional correction, Report 10's count readout and Report 11's proposals/overlap operations. The four pre-existing Job036/037 shell/README modifications are outside this investigation.

**Precise question.** Does the first four-user alias reflect missing affine sets of characters, and can that description produce a finite repair criterion and an alphabet-free operation that preserves count uncertainty? For one known pattern with positive energies, put E_g=sum_{i:ell_i=g} e_i. For z=a-a', the squared distance is sum_g E_g (H z)_g^2. Thus row multiplicity and energy affect distances through E_g; a positive reweighting cannot change the integer kernel. CRC and the polar information set act through the effective generator, not through a generic label 'polar'.

**Algebra before measurement.** For an r-dimensional payload subspace U and two different cosets a+U,a+d+U, put F(U,d)={g:g annihilates U, g dot d=1}. Their Fourier difference has magnitude 2^(r+1) exactly on F and zero elsewhere. Consequently d_signal^2=4^(r+1) sum_{g in F} E_g. Equivalently, define W_U=span(G intersect U-perp)-perp. All cosets inside a+W_U have the same signal, with class size 2^(dim W_U-r). Computing W_U from a supplied basis of U is polynomial in Nc and b; searching every U is not. This gives a complete criterion for coset competitors only.

**Actively rejected explanations.** Pair injectivity does not imply triple injectivity for a general binary linear sign code. The b=4 row set {0,1,2,5,6,7,9,10,11,15} has full rank and passes every pair-rank condition, yet {0,1,2} and {4,8,15} have equal sums. This also rejects a universal power-of-two first-alias rule. A low-degree character/trade certificate was another possible explanation of the selected threshold: a complete degree-two set after a binary change of basis would exclude trades of volume below four. An exhaustive independent-basis clique search finds no such basis in the selected G. Do not use that sufficient theorem as an explanation of this fixture.

**Initial finite findings, awaiting the retained check below.** Enumerating the 651 binary payload planes gives precisely 392 two-member and 12 four-member ambiguous coset classes, matching the previous full K4 census. Their 464 pairwise equalities occupy 58 different missing Fourier supports, each occurring eight times. A set-cover calculation suggests seven missing characters are necessary and sufficient to separate all these equalities while retaining the old characters; a seven-row witness is {8,13,19,52,60,61,63}. An independent integer exhaustive check, rather than only the floating MILP solver status, will certify the lower bound. The best three-row addition hits only 40/58 supports. Replacing three duplicates therefore cannot eliminate the first aliases.

**Prior-work comparison.** The noiseless shared-codebook multiset target is the classical B_h-code problem; Gu, *Zero-error communication over adder MAC* (2018), https://arxiv.org/abs/1809.07364, distinguishes it from user-specific uniquely decodable code tuples. The low-degree sufficient condition is established trade theory: Ghorbani et al., *On the volumes and affine types of trades*, https://arxiv.org/abs/1810.02296, Lemma 4. These are not new general coding theorems. Sparse Walsh methods, especially Li et al.'s SPRIGHT, https://arxiv.org/abs/1508.06336, already use affine character queries and sparse recovery. Their query access, source/noise scaling and designed measurement sets must be reconciled with the fixed rows, unit message energy and unknown overlapping patterns here. The concrete contributions under examination are a verified fixture classification/repair obstruction, physical distance accounting and a conditional count-preserving operation.

### Validated outcome and direction change

The retained check is `tests/polar_sum_geometry_checks.py`; its command and generated JSON are in EXPERIMENT_BANK. It independently compares actual class members from the full integer K4 census against the affine census. They agree exactly. Distance identities across all plane/coset pairs agree within 7.11e-15. The integer cover certificate exhausts all 1,623,160 six-subsets: maximum coverage is 55/58; the seven-row witness covers all. The best three-row cover 8/20/61 leaves 144 two-member classes (288 multisets) after replacing only repeated rows, with minimum Hamming distance 12. This improves the earlier exploratory three-row choice's 152 classes but does not establish risk optimality.

A targeted noisy-geometry check uses exactly the B12/seed3201 reference placements, known pattern 0, and all 2,080 K2 multisets. Under uniform/balanced energies the reference minimum squared distances are 1/.8014282121; the three-row replacement gives 1.5/1.2716568767 with the same energies and coordinates. The reference's 2,162,160 competing-pair distances include 982,704 increases and 1,179,456 decreases under balancing. Thus balancing is not monotone for sum geometry, while the three-row candidate has a concrete conditional separation gain worth retaining. Neither fact compares full-system PUPE, and no receiver was run. The closest balanced reference competition is {0,43} versus {4,47}.

The seven-parity extension is a constructive B4 signature when all old characters remain. Allocating energy eta/7 to each new row and scaling the old profile by 1-eta separates every old four-user alias by squared distance at least 64 eta/7; old nonzero distances retain at least their 1-eta fraction. It costs 39 occupied symbols instead of 32, changes overlap and payload bits per occupied symbol, and does not automatically preserve resource balance. A fixed-length arbitrary redesign is not ruled out; deleting old distinct characters can create new aliases. Neither this extension nor the three-row replacement is silently substituted into the production polar/SCL encoder.

The affine criterion also yields an exact conditional operation. For four selected ordered slots on a common pattern whose distinct payloads form a plane, compute W_U and shift all four by a uniform d in W_U. Forward/reverse laws and likelihoods agree under IID uniform messages, so acceptance is one. The orbit's count-tail formula keeps outside multiplicities and gives each represented payload block-inclusion probability 1/J. With outside counts c12=2,c1=1, the two induced count vectors have factorial weights 1/6 and 1/2; the slot-selection multiplicities compensate exactly. A uniform count-space replacement would be wrong. Distinct-source shifts intersecting outside messages can be rejected symmetrically. The routine only moves within a waveform class, so it supplies no global discovery or mixing result. Basis construction is polynomial for a supplied plane, and a 100-bit nullspace control contains no payload alphabet. Enumerating eligible quadruples or affine groups is explicitly charged in Report 11.

The conditional near-alias bound preserves source-weighted identity and accounts for log-likelihood oscillation, with a separate group-evidence error. Twenty finite groups verify the stated TV inequality. A short-Walsh packet identity independently checks repeated-message singleton recovery and the unit-power bin-noise variance sigma² Nc/2^q. This is the known sparse-Walsh interface, not permission to query untransmitted characters. Existing orbital MCMC (Niepert, https://arxiv.org/abs/1206.5396) also precedes the general symmetry move; the contribution here is the explicit row-nullspace and count handling.

The strongest relevance limit is source frequency. The probability of containing a recorded four-user witness is at most 6.79e-5 at B12/K26 and 1.06e-6 at B14/K26 under IID, with nearly equal distinct-source bounds. Those are prior witness-event bounds, not all-alias or noisy-risk bounds. They redirect practical attention toward two/three-user near competitors, mixed patterns and proposal coverage rather than a broad campaign aimed only at eliminating K4 equalities. The next discriminating control is a matched conditional exact-posterior or candidate-coverage comparison of the reference and three-row signature, followed by mixed-pattern checks; a performance benefit remains open.

Validated results are integrated into the existing Reports 7/8/9/11, with the local-operation link in Report 10. No new Markdown files, DEC families, runtime sources or HPC jobs were created. A first packet-noise check used zero absolute tolerance for theoretically zero off-diagonals; changing it to 1e-14 accounts for observed 2.1e-17 floating roundoff, without changing the identity or the integer certificates.

### Final verification

The retained sum-geometry script passes with the final three-row distance control included (5.69 s on this local run). `uv run python -m unittest tests.polar_exact_reductions_test tests.polar_energy_geometry_test` passes all 12 tests; `compileall` on the new script and `git diff --check` also pass. All five edited LaTeX sources, Reports 7–11, compile successfully using the desktop editor. These compilations validate the saved sources/live previews; no adjacent PDF exports or generated results were added to git. The four pre-existing Job036/037 edits remain untouched.

---

## Preserved earlier tractability record, 26–27 October

The following is the original earlier log, retained in full. Its paths and commands describe the layout at that time; consult the experiment bank for current commands and report locations.

# Polar tractability investigation

## 26 October 2026 — first algebra-first installment

Scope: fixed analytical polar/CRC, transmission patterns and KL-balanced energies; known K and physical noise;
IID full messages and multiplicity-aware PUPE. Accuracy concessions are allowed when declared. This work adds
Report 11 at `docs/reports/11_tractable_polar_decoding/main.tex`, without changing existing reports, decoders,
experiments, the author-owned archive or HPC jobs. The project was created with the app's LaTeX project tool.

### Evidence and initial direction

Read Reports 7–10, the current state/experiment bank, relevant research logs, the 26 October scaling audit,
the polar implementation, and the parked revisable-sampling prototype. Earlier memory was used only to locate
the exact-versus-approximate boundary, then checked against current repository evidence.

The full-M Job035/036 results are useful small-B accuracy references, not the only existing algorithmic work:
`tests/polar_revisable_sampling.py` already contains SCL-list/prior full-support proposals, correct forward/reverse
pair densities and sparse count-tail allocation. Its exact-reference audit uses enumeration, whereas its proposal
operation does not enumerate payloads. Do not call those ingredients new or infer useful mixing from invariance.
Report 9 Entry Q already groups restricted character rows and gives exact independent-character partitions.

The working question is not how to hide M behind a sparse final list. Charge pattern preparation, all residual
contexts, candidate generation, proposal normalization, global inference, sparse readout and memory. With an arbitrary
frozen P-by-d placement/energy bank, P=2^Bp remains in input size. Payload scaling needs bounded Bp and growing b/d;
the B12/B14 experiment keeps b=6,d=32 and increases only P. Fixed search budgets are not fixed-PUPE guarantees.

### Derivation: a normalized independent-parity envelope

For an exact one-slot conditional, the outside residual r gives score
`s_p(v)=r^T phi(p,v)/sigma^2=sum_i h_pi (-1)^(ell_i^T v)` with `h_pi=sqrt(e_pi)r_position/sigma^2`.
Physical rescoring is necessary across patterns: SCL path metrics include pattern-dependent channel constants.

After fixing an affine payload prefix `v=v0+Rz`, group identical restricted character rows:
`s(z)=H0+sum_g H_g chi_g(z)`. Retain independent rows F and let
`D_F(z)=sum_{g notin F} H_g chi_g(z)`, `R_F=sum_{g notin F}|H_g|`.
Then `Z_F=2^(q-rank F) exp(H0) product_{g in F} 2cosh(H_g)` normalizes a full-support proposal.
Sample its independent parity bits with logistic probabilities and solve the binary system, choosing free bits
uniformly. This is polynomial arithmetic without a payload scan. Binary factorization and changing bases are charged.

The exact partition satisfies `exp(-R_F) Z_F <= Z <= exp(R_F) Z_F`. Greedy independent-row selection in decreasing
`|H_g|` minimizes both R_F and this particular upper bound: the latter's inclusion benefit is
`|H_g|-log cosh(H_g)`, also increasing with magnitude. This optimizes only this bound family.

Across patterns, choose pattern with probability `U_p/sum U_p`, draw its basis proposal, and accept with
`exp(D_F-R_F)`. The accepted law is the exact one-slot conditional, with acceptance `Z_all/U_all`.
No exact polar partition is needed to execute it; expected runtime is the unknown reciprocal acceptance.
A finite rejection cap must return failure or use a valid invariant alternative, not quietly substitute a raw proposal.
The bound also certifies a discovered list's conditional missed mass by `1-Z_list/U_all`; it is not a full multiuser
certificate when the outside configuration is uncertain.

### Reversal: exact correction can be exponentially slow

Do not promote the product-parity proposal to a chosen scalable decoder. The actual b=6,d=32,CRC8 generator
has 28 distinct nonzero character rows, rank six, and contains keys `(2,5,7)` with `2 xor 5=7`.
For coefficients `(t,t,-t)`, its score on the two independent signs is `t(x+u-xu)`. Any two-character basis
gives acceptance

`A(t)=(3 exp(t)+exp(-3t))/(exp(t)*(2cosh t)^2) ~ 3 exp(-2t)`.

Exact values: t=2 gives .0529940; t=4 gives .001005713; t=8 gives 3.37605e-7 (about 2.96 million attempts).
These scores are realizable on one known-pattern residual because amplitudes are positive and occupied resources
are distinct. This is a conditional algebra obstruction, not a measurement of typical residuals or PUPE.

A targeted alternative is to retain a few dependent parity factors jointly in a normalized reference. The triangle
requires only four parity assignments and is then exact. More generally the joint state size/elimination width and
the cost of choosing useful factors must be charged. Whether a small set resolves relevant conflicts remains open.

### Additional deductions retained for Report 11

* A Gaussian continuation guide with scalar covariance yields ordinary polar LLRs; a diagonal covariance adds a
  pattern-dependent quadratic constant. A dense covariance creates characters `ell_i xor ell_j`, so it is still a
  character score but not generally an ordinary d-channel SCL input. Dropping those terms is a guide approximation.
* H weighted legal configurations need only O(HK) nonzero tail levels: `sum_w max_h a_hw <= HK`. Heap allocation costs
  O(C_s+K log C_s), C_s<=HK, and message labels cost B bits. Sparse readout is existing machinery, with this accounting
  making clear that an M-by-K table is not mathematically required.
* Restricting actions to C using the original posterior loses at most the posterior expected outside-copy fraction.
  Conditioning the entire posterior on all copies lying in C instead has an omitted-event-mass bound. They differ:
  K=4, mass .25 on aaaa and .75 on bbbc, C={a,b}, gives outside-copy fraction .1875 but conditioning regret .5.
* If H independent exact posterior draws were available, a union bound over MK count-tail events gives
  `H >= [B log 2 + log(2K/delta)]/(2 eta^2)` for uniform tail error eta and excess Bayes risk at most 2eta.
  Sparse empirical readout still suffices. This isolates posterior sampling/discovery as the unresolved barrier;
  correlated chain states and self-normalized IS draws do not satisfy that premise merely by having exact targets.
* Bounded SCL/list proposals, restricted multiset beam, IS, local/pair MH and small joint-parity references remain
  alternatives. The number of distinct residual contexts J can be KW or HK and must not disappear in cost claims.

### Tiny independent checks

`uv run python docs/reports/11_tractable_polar_decoding/checks.py` passes. Forty random q=4 character fixtures
(16 payloads each) verify normalization, the envelope, optimal independent-basis weight by exhaustive subsets,
list-mass bounds, rejection identity and detailed balance of a list/parity mixture. Maximum discrepancies:
normalization 1.11e-15; correction identity 8.88e-16; detailed balance 2.78e-17. Envelope/partition ratios ranged
1.309–55.964 on arbitrary scores; this is not an operating-point study.

The actual CRC8 generator verifies the triangle and acceptance formula. Twelve random SCL calls return five
nonempty CRC-valid lists; their path metrics match physical channel metrics to 3.55e-15. Empty lists are explicit
outcomes, not silently repaired. All 15 count states for three words/K4 verify sparse tails and exhaustive minimum
risk, with tail discrepancy 1.11e-16. These are double-precision identity checks, not rigorous interval arithmetic.

### Current research boundary

The defensible next questions are list coverage on competing residuals, value per additional residual context,
envelope tightness and small joint-factor repair, and finite-budget inference error at matched complete work.
No evidence yet establishes favorable acceptance, adequate nonexhaustive discovery, fast mixing or fixed-PUPE
polynomial scaling. Do not turn this first installment into a selected architecture or a broad new survey.

## 27 October 2026 — continuation: spend finite search to tighten a positive envelope

The user requested completion of the first installment followed by further exploration. Parent review/compilation
owns `main.tex` and `checks.py`; this continuation uses `continuation.tex` and `continuation_checks.py`. The parent
is independently studying residual-context reuse. This pass instead asks whether a bounded deterministic search
can repair the independent-parity envelope without giving up its available normalization and sampler.

The relevant classical connection is ordered-statistics decoding: choose a reliable independent information set,
then reprocess low-weight flips of its hard decisions. The primary source is Fossorier and Lin, IEEE TIT 1995,
DOI 10.1109/18.412683, [publisher record](https://ieeexplore.ieee.org/document/412683/).
That attribution motivates candidate construction, not a transfer of its single-user performance claims to URA.
The additional object here is a positive partition envelope and exact conditional sampler for the full remaining
payload space, with multiplicity semantics retained in the outer receiver.

For the grouped character score, let F span all nonzero-coefficient keys and have rank r. Independent parity
coordinates under q_F disagree with their favored signs with probabilities `p_j=1/(1+exp(2|H_j|))`. Every assignment
has `2^(q-r)` equal-score payloads; rank-deficient cases must retain that cell multiplicity and sample free bits
uniformly rather than discard full message identities.

Let C_t contain parity-coordinate assignments with at most t flips. Its size is
`N_t=sum_{j=0}^t binom(r,j)`. Enumerate their exact **cell** masses, giving Z_t, while a capped Bernoulli-count DP
computes `tau_t=Q_F(number of flips>t)` in `O(r(t+1))` work. Since the initial pointwise envelope is `E=U q_F`,
replace its value by the exact physical weight on C_t and leave it unchanged outside. The total is
`U_t=Z_t+U*tau_t`, with `Z<=U_t<=U`, and omitted target mass at most `U*tau_t/(Z_t+U*tau_t)`.

An exact proposal samples the explored cells proportional to their exact mass with mixture weight `Z_t/U_t`,
or q_F conditioned on `number of flips>t` with mixture weight `U*tau_t/U_t`. Suffix Bernoulli-count tables supply
that conditioned draw directly; using rejection from q_F would hide a potentially large `1/tau_t` cost.
Accept explored cells automatically and outside cells with the original `exp(D_F-R_F)` correction. The accepted
law is exact, with acceptance `Z/U_t`. Unknown patterns still need their own envelope masses and a sum over P.

For the frustrated triangle `(a,a,-a)`, t=1 explores three parity cells and yields
`U_1=3 exp(a)+exp(-a)`, compared with `Z=3 exp(a)+exp(-3a)`. Its acceptance tends to one although the original
product-envelope acceptance tends to zero. This repairs that disclosed mechanism using O(r) explored cells,
without proving that bounded t repairs general residuals. The next check is an independent fixture that forces
many conflicts, so the remaining combinatorial boundary is not hidden by this favorable example.

### Correction and generalization after review

The parent noticed that **radius zero already repairs the triangle**: replacing the favored cell's overlarge
`exp(3a)` envelope mass by its exact `exp(a)` gives `U_0=3exp(a)+exp(-a)`. The single-flip cells already saturate
the envelope, so radius one has the same U. This correction is retained explicitly rather than presenting the
initial radius-one calculation as the minimal construction.

The more general identity needs no Hamming ball. For any finite explored set A and positive envelope `E=UQ>=f`,
set `E_A=f` on A and `E_A=E` outside. Then `U_A=Z_A+U Q(A^c)`, monotonically decreasing with additional states,
and each explored state's gain is `E(w)-f(w)`. Exact rejection uses the normalized repaired envelope and accepts
outside with `f/E`; its expected attempts are `U_A/Z`. A known lower envelope `LQ<=f` gives the interval
`[Z_A+L Q(A^c), Z_A+U Q(A^c)]`. The same proposal has normalized importance second moment at most `U_A/Z`.

For an arbitrary explored list, transform payloads into independent parity plus uniform free coordinates and
build an exclusion trie. A positive recursion computes the probability of avoiding listed leaves beneath each
prefix; those values give exact sequential complement draws in O(q) traversal after O(|A|q) trie preparation.
This avoids both cancellation in `1-Q(A)` and a hidden expected `1/Q(A^c)` cost from rejection-sampling the
complement. Label conversion and inverse binary maps are separately charged. Thus SCL lists can repair the same
envelope; reliable-basis flips are an alternative discovery rule, not a required architecture.

The triangle also separates discovery from sampling: one explored cell has target mass approaching 1/3, and its
omitted-mass bound approaches 2/3, although repaired full-space acceptance tends to one. The remaining sampler
actually finds the other modes. A list-only approximation must not inherit the repaired sampler's conclusion.

### A second boundary and its exact control

For m independent frustrated triangles, a fixed flip radius can still fail exponentially. At large strength a,
target flip count approaches Binomial(m,2/3). With fixed m and t<m-1, the remaining-envelope term has ratio to
the target of order `exp(2a(m-t-1))`. At a=4, one-cell repair acceptance is .999888 for one triangle, .000754348
for two, and 3.81784e-10 for four. Radius one only raises the four-triangle case to 3.25330e-7.

This is not a general hardness construction: the independent triangles factor into exact four-state components,
so an exact reference handles that family in O(m) local tables. Given any binary character basis, express all
other keys in that basis and connect the coordinates co-occurring in each character. Connected components yield
an exact partition factorization; charge their ranks and table sizes. Structural checks without payload
enumeration give one component of rank6 for b6/d32/CRC8, rank32 for b32/d128/CRC16 and rank88 for b88/d512/CRC16,
when every nonzero generator character is present. Hence no automatic favorable large-payload decomposition is
established for this construction. These are structure controls, not measured inference regimes.

### Continuation checkpoint and verification

`continuation.tex` contains section-only material for parent integration. `continuation_checks.py` verifies 30
five-bit fixtures (32 payloads), arbitrary-list repaired normalization, positive exclusion-trie probabilities,
monotone envelopes, omitted-mass bounds, count-DP tails and exact accepted densities. It also verifies the triangle,
uniform free-cell multiplicities, independent-triangle factorized partitions and the component structure controls.
Command: `uv run python docs/reports/11_tractable_polar_decoding/continuation_checks.py`.
Maximum identity discrepancy: 1.33227e-15. No production decoder or performance benchmark was added.

The new defensible boundary is finite explored-set repair plus an accessible complement sampler. Whether
preparation pays for itself, how many useful residual contexts/draws can share it, and whether full multiuser
inference moves between competing explanations remain unresolved. The parent's independent residual-reuse
derivation can transport the repaired pointwise envelope, so the two results combine without assuming a winner.

## 27 October 2026 — independent review and residual-context reuse

The first installment passed the desktop compiler and latexmk; all eight pages were rendered and inspected.
The expanded checks include constant/rank-deficient cases, affine-prefix grouping, a multi-pattern envelope
identity and exhaustive minimum-log-envelope selection. Their largest reported discrepancy was 7.11e-15.
The report now uses the updated deterministic-check numbers. No runtime or HPC source was changed.

A parent-side continuation asks whether the expensive residual contexts can be shared. For a fixed encoder,
noise and allowed domain, the score difference is Delta r^T phi(w)/sigma^2. A certified interval [l,u],
obtained from unit norms or grouped character sums without enumerating payloads, has width Omega=u-l.
It gives TV(pi_r,pi_r0)<=tanh(Omega/4) and transports a cached list's missed-mass bound epsilon0 to
exp(Omega)*epsilon0/(1-epsilon0+exp(Omega)*epsilon0). This is a conditional comparison, not a joint decoder
certificate. The derivation is saved in residual_reuse.tex.

More usefully for exact correction, if exp(s0)<=U0*q0, the same cached proposal satisfies
exp(sr)<=exp(u)*U0*q0. Its new rejection acceptance is at least exp(-Omega)*Z0_lower/U0; this also gives
a fixed-context independence-MH minorization. It saves reconstruction, not physical rescoring or mixing.
Nearest-center selection, grouped pattern scans, cache storage and the possibly poor covering number are
charged. High SNR shrinks useful residual neighborhoods. A frozen bank and identical forward/reverse
selection are essential; arbitrary state-history adaptation is not covered.

### Integrated review and deliverable, 27 October

Both investigations are now integrated in the 14-page Report11. The parent reviewed the continuation's
envelope/complement identities and counterexamples; the independent agent reviewed the residual-reuse
proofs. The latter review added the cached-law error term: eta0+tanh(Omega/4), capped at one, when the
stored conditional is itself approximate.

All three check scripts pass: 40 initial character fixtures, 30 continuation fixtures and 40 balanced
polar residual-context pairs, plus the stated edge cases and structural controls. Reuse's largest
reported correction/invariance discrepancy is 1.11e-16. No stochastic operating-point performance or
multiuser mixing claim follows. The desktop compiler and latexmk both succeed; all 14 final pages were
rendered and visually checked, with no overfull boxes or unresolved references. Source, checks and PDF
are in the self-contained project, and README/current-state/experiment-bank links are updated.

Further theoretical work remains: combine small joint-factor references with selective repair, choose
exploration by envelope slack versus target coverage, and understand global user-coupling and useful
residual reuse regions. This is a completed pair of investigations, not the closure of tractability work.
