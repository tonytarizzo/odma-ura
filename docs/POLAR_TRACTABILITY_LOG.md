# Tractable polar decoding: research and implementation log

Updated 29 October 2026. This agent-maintained log records exploratory reasoning, controls and decisions; [Report 11](reports/11_tractable_polar_decoding.tex) contains the concise algebra. The author's paper-outline scratchpad is separate.

## Aim and working boundary

Keep the fixed Report-8 encoder, known K/noise, IID source and multiplicity-aware PUPE. Use Reports 7 and 9 to remove full-payload computations and identify useful structural approximations. A small representation is not enough: charge discovery, normalization, repeated residual contexts, mixing and count readout. New constructions need comparison with established methods before any novelty claim. No learned components or HPC campaign are added in this pass.

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
