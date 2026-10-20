# Polar decoding research log: exact foundations

Consolidated **15 October 2026** from the 11–14 October derivation record, synthesis atlas and scaling audit.
**16 October:** exact-only Report 7 and separate Bayesian approximation Report 9 now have distinct ownership.
This log records deductions, corrections, evidence and stopping decisions; [Report 7](reports/07_polar_decoding_from_posterior.pdf)
retains the full derivations. The [current state](CURRENT_STATE.md) owns priorities and the
[experiment bank](EXPERIMENT_BANK.md#exact-foundations) maps each finding to its executable check.
The [approximation log](POLAR_APPROXIMATION_LOG.md) continues the investigation separately.

**Stage decision:** the broad exact investigation is complete for the specified known-K model. There is a
complete finite Bayes reference and enough structure to derive approximations. No available exact construction
has established end-to-end tractability. This closes a research phase, not all possible exact mathematics.

## 11 October: fix the model, prior and decision

Freeze the real unit-gain AWGN model, polar/CRC code, unknown pattern bank and deterministic KL-balanced
symbol energies of Report 8. Keep possible encoder improvement deferred. D2 is measured context, not the
starting equation or the required architecture. Receiver knowledge is the encoder, K and sigma²; no true
messages, patterns or interference cancellation are supplied.

Counts a and exchangeable ordered slots describe the same posterior:

\[
P(a\mid y)\propto\pi(a)\exp[-\|y-\Phi a\|^2/(2\sigma^2)],\quad \mathbf1^Ta=K,
\qquad \pi_{\rm IID}(a)=\frac{K!}{M^K\prod_w a_w!}.
\]

Distinct sampling restricts counts to 0/1 with a uniform legal prior. It is IID conditioned on no full-message
duplicates, not an excuse to remove same-pattern collisions. Uniform ordered slots induce factorial count
weights; a uniform count prior would change the experiment.

**Correction:** the first report followed the implementation's membership score. The user confirmed that
multiplicity-aware PUPE was always intended. Report 7 and the scorers were corrected; old results remain
historical. [PUPE_METRIC.md](PUPE_METRIC.md) owns the migration and rescoring details. The correct action uses
T_wj=Pr(a_w>=j|y), allocating each next copy by its marginal matching gain. IID posterior means alone are
insufficient; count MAP, ordered-slot MAP and minimum-PUPE decisions differ.

Conditioning on all other slot hypotheses gives residual r and exact one-word score
b(v)^T A_p^T r/sigma². **Maximum score, evidence summation and posterior sampling are different interfaces.**
A bounded polar SCL list supplies none of their exact guarantees automatically. Unknown-pattern probabilities
need evidence and normalization across patterns, not just each pattern's best payload score.

The initial report also contained candidate, bit-message and optimization formulations. On 16 October those
approximation routes were removed from the exact narrative. The exact polar/CRC/pattern factorization remains,
with one shared pattern variable per slot. The active investigation keeps the Bayes count-tail objective.

## 12 October: finish an exact decoder before simplifying it

For alphabet S, let H_m^S(r) sum physical Gaussian weights over ordered m-tuples. Then

\[
H_0(r)=e^{-\|r\|^2/(2\sigma^2)},\qquad
H_m^S(r)=\sum_{w\in S}H_{m-1}^S(r-\phi_w),
\]
\[
\Pr(a_w=j\mid y)=\binom Kj
\frac{H_{K-j}^{\mathcal W\setminus\{w\}}(y-j\phi_w)}{H_K^{\mathcal W}(y)}.
\]

The excluded word is essential: without exclusion, the expression is a factorial moment, not an exact count
probability. Distinct recursion removes each chosen word as well. These identities finish the decoder from
observation to count tails to a proven minimum-posterior-PUPE multiset action, albeit at prohibitive cost.

Positive inside measures on disjoint message-prefix alphabets remove slot-order redundancy. A leaf with j
copies has weight 1/j! for IID, or permits only 0/1 for distinct. Convolve measures, apply the Gaussian once,
and use outside/backward queries for each count law. Equal signals may be merged for evidence, but message
identities/count provenance must survive. An ambiguous fixture checks that a scalar root sum law alone is insufficient.

Polar linearity gives characters (-1)^(g·v). Grouping identical generator rows is lossless; the selected
32-by-6 CRC8/polar generator has 29 distinct characters, including a constant. Walsh transforms accelerate
all-payload one-user scores. They do not diagonalize the exponentiated multiuser likelihood. Marginal bit
counts cannot pass exactly through an XOR butterfly because they discard cross-user pairing.

Three alternative complete evaluations were derived with the same marked count queries:

| Representation | Exact operation and limitation |
|---|---|
| Finite Fourier convolution | Integer signatures admit an alias-free grid of side 2K+1. IID uses transform powers; distinct uses cardinality coefficients. A d-dimensional grid has (2K+1)^d points. |
| Joint variable elimination | Sum finite factors while retaining complete separator assignments and count markers. Cost follows intermediate bags, not initial circuit size. |
| Connected-user expansion | Expand pair factors as 1+f, group connected components and assemble all orders through K. Pairwise energy does not make higher connected groups vanish. |

An IID recursion through H_0,...,H_K is marginalization, not a series of successively smaller corrections.
Regrouping two/three-user sums changes parenthesization without discarding users. Fourier-mode decay or a
central-limit argument needs relative evidence/decision control before it justifies approximation.

## 12 October: safe forgetting, completion bounds and identifiability

At a legal alphabet/cardinality cut, future sums lie in t0+V. Two partial signals have proportional
continuation likelihoods exactly when their difference is orthogonal to V, provided legal futures agree.
Project onto V, absorb the perpendicular Gaussian contribution and retain count markers. Distinct-source
feasibility is additional state; geometry cannot erase exclusions.

The full six selected B12/B14 banks span all 256 observation directions. This uses character axes **and pattern-mean
differences**, with exact combinatorial certificates rather than a floating rank threshold. A known pattern
with fixed occupancy has 28 variation directions. Fixing 0,...,6 leading payload bits gives dimensions
28,23,15,7,3,1,0. From two fixed bits, the remaining character set identifies every local count vector:
a smaller geometric dimension can still leave H(V',h) different states.

A scalar geometric separator also need not have small likelihood rank: exp(-ij/sigma²) can have arbitrarily
large full rank. Matrix rank limits a specified exact linear factorization, not every symbolic algorithm.

For a residual r and legal completion sums T, any dual vector u gives

\[
\min_{t\in T}\|r-t\|^2\geq 2u^Tr-\|u\|^2-2h_T(u),\qquad h_T(u)=\max_{t\in T}u^Tt.
\]

IID support is h times the best allowed-word score; distinct support sums the h best distinct scores.
Optimizing the dual reaches distance to the convex completion hull. Every valid trial can strengthen a
positive subtree-mass bound. An incomplete candidate maximum underestimates support and can invalidate
certification: the required bound direction is an **upper** support bound.

Count-tail intervals, and later direct selected/unselected-copy contrasts, certify a Bayes action without
finishing the posterior. These are proved inequalities with finite floating checks, not outward-rounded
machine certificates or guaranteed fast stopping.

For Bc=6/Nc=32/CRC8, all 2,080 pair sums and 45,760 triple sums are distinct. Four users give 766,480 multisets
but 766,052 sums: 392 two-way and 12 four-way ambiguous classes, all involving distinct payloads.
One exact alias is {12,15,57,58} versus {20,23,33,34}. Character cancellation on binary cosets proves that
placement and fixed positive symbol energies preserve the equality. Conditional noiseless known-pattern
four-user Bayes risks are .0006122589 IID and .0006736169 distinct; these are not full-system error floors.
The 428 saved sum entries are about .056% of the count states, too few to support massive collision compression.

## 13 October: compatible schedules and the cost of certification

Nested future spaces give a sequential projected-state recurrence whose orthogonal likelihood factors
telescope to the original Gaussian. The resource-frontier specialization finalizes each resource once and
uses recorded forward/backward transitions for full count laws. Reapplying the whole likelihood independently
at each cut would reuse the data incorrectly.

All 24 pattern orders on the tiny P4/V4/K3 fixture agree on evidence/counts, while peak states range
286–455 IID and 196–299 distinct. Some best/worst orders have the same frontier width eight. On the research
banks, tested natural/greedy resource frontiers peak at 231–256 resources; the greedy order is not optimal.

A sharper obstruction holds at the first-eight-pattern cut of every selected bank: one user per processed
pattern yields at least 64^8=2^48 distinct, nonproportional continuation rows, with positive future occupancy
and interior occupancy for the distinct case. Private-coordinate binary ranks prove it without enumeration.
This rules out a small exact row-merging table **at that cut**, not every exact representation or every load.

Direct decision contrasts charge uncertainty only when the adverse event could reverse an allocation.
The exact conditional **prior** mass of that event can multiply a common likelihood envelope; this is not a
posterior-independence assumption. Cached support directions remain valid for descendant suffixes/occupancies.
The experiment bank records the paired bounds/search audit: fewer evaluated hypotheses are offset by support,
frontier and event-query work. No runtime speedup follows from the hypothesis fraction or score counter alone.

Observation-weighted connected majorants use positive spanning-tree bounds and a partition recurrence to
bound the entire marked polynomial. A constructed high-noise K4 fixture certifies its action using only
two-user blocks; the tested two-block bounds certify 0/24 polar observations. This is a failure of these
sufficient bounds, not a general theorem that connected approximations must fail.

## 13 October: pair structure, rooted trees and the first cycle

Cross-pattern interactions depend only on generator parities on overlapping resources. For endpoint ranks
r_pq,r_qp, class tables have 2^(r_pq+r_qp) entries, plus payload-to-class maps. Same-pattern interactions are XOR
convolutions and admit Walsh application. Distinct exclusions alter the repeated-full-word entry; all unknown
patterns and same-pattern different-payload collisions remain represented.

The six-bank census saves 6.83–7.24x intersecting-pair table entries against symmetry-reduced dense tables.
Maps, kernels and setup are additional. These give a complete K2 count posterior without a stored pair posterior.

For F=|B-1| and unary law nu, the rooted series

\[
T_w(z)=z\exp\left(\sum_v\nu_vF_{wv}T_v(z)\right),\qquad
Q_m=(m-1)!\sum_w\nu_w[z^m]T_w
\]

computes exact weighted tree sums using K-1 operator applications and O(K²M) polynomial work. It improves
majorants without tree enumeration, but still certifies 0/24 in the polar audit. A cheap bound ingredient is
not an exact posterior solver.

For K3 let A=diag(sqrt(nu)) B diag(sqrt(nu)). The triangle normalizer is tr(A³). For a marked word w, put
x_w(u)=nu_u B_wu for u!=w and zero otherwise. Positive count numerators for one, two and three copies are

\[
3\nu_w x_w^TBx_w,\qquad 3\nu_w^2B_{ww}\sum_{u\ne w}\nu_uB_{wu}^2,\qquad
\nu_w^3B_{ww}^3.
\]

Their count-weighted sum over w is 3Z. One B application per word supplies all count laws with O(M) scratch
beyond the operator/output, avoiding a triple tensor. At low noise, computing B=1+(B-1) can cancel positive
entries; the positive helper pays extra same-pattern work to avoid that subtraction.

Joint parity signatures must include **both** neighbours. In the restricted first-eight-pattern B12 census,
median triple-class counts are 131,072/131,072/65,536 versus 262,144 ungrouped; 14/15/12 of 56 triples expose
all six bits at every endpoint. Pair compression can disappear when composed. tr(A^K) for K>=4 gives a cycle,
where the physical model needs a clique; it also misses distinct exclusions between nonadjacent vertices.

Deleting a Gaussian edge gives P=Q_open R/E_open R, so any count contrast H changes by
Cov_open(H,R)/E_open R. Keep all source exclusions. The identity is an exact comparison resource. Its previous exploratory edge-deletion PUPE table has been
removed from the active record; no edge deletion is adopted.

## 13 October: connect geometry, invariants and decision error

After scalar reweighting, a partial-signal displacement delta has worst continuation log error bounded by
w_T(delta)/(2sigma²), where w_T is the directional width of legal future sums. Width zero recovers exact merging.
Count markers carry this bound to posterior tails and output decisions. Candidate restriction must separately
pay omitted posterior mass. Several cuts need one compatible likelihood schedule. The quantitative
approximation/error-transfer development now belongs to Report 9; Report 7 retains lossless continuation
states, exact reference changes, ambiguity identities and reweighting contrasts.

Exact unary absorption/recentering changes a connected reference without changing the posterior or four-point
interaction ratios. Positive pair applications survive through diagonal scaling; |B_centered-1| need not retain
cheap original tables. A better-looking error bound may therefore cost more to construct.

Within an exact equal-signal class, identity probabilities are prior determined, independent of y. Restoring
those ratios at fixed class masses contracts both KL directions and TV, without a per-observation guarantee
of a better Bayes action. Slot symmetrization instead leaves the count distribution unchanged.

For multiple positive corrections, the covariance identity uses each updated reference. Their effects do not
add independently: independent uniform signs U,V, contrast UV and corrections 1+U/2,1+V/2 have zero individual
effect but combined effect 1/4. Support deletion requires separate omitted-mass accounting.

## 14 October: consolidate the computational accounting

Let E count intersecting unordered pattern pairs and J_pair=sum 2^(r_pq+r_qp) their table entries.
The following are available schedules, not universal lower bounds or stable-arithmetic guarantees.

| Construction | Complete cost consequence |
|---|---|
| Streaming count reference | H_IID(M,K)=binom(M+K-1,K), H_dist(M,K)=binom(M,K). Attainable work O(H(M,K)(K Nc+n)+MK), sparse updates per hypothesis; the dense verifier stores more. |
| One-user Walsh scoring | O(P Nc+M log V) work, O(V) scratch if streamed by pattern. No generally sub-exhaustive exact evidence/max oracle has been supplied. |
| Structured pair application | O(PV log V+EV+J_pair) work, O(PV+EV+J_pair) representation. Charge maps and parity/table construction separately. |
| Positive pair helper | O(PV²+P²V+J_pair+P²) per application; direct positive same-pattern sums/disjoint maps avoid particular cancellations, at extra cost. |
| Complete K2 / K3 | One pair application for K2; K3 uses M applications plus row extraction/O(M²) scalar work. O(M) extra streaming scratch does not remove explicit M. |
| Inside/frontier circuit | Cost follows generated transitions T, retained states S and keys/projections. Scalar reverse differentiation gets all leaf-count queries in O(T) work plus readout. The forward circuit can remain exponential. |
| Fourier | Grid (2K+1)^d remains. IID binomial markers/distinct polynomial division allow O(grid × M K) marked work once phases/transforms exist. The generic verifier is intentionally more expensive. |
| Connected coefficients | Positive partition sums plus inverse recurrence avoid graph enumeration. Sparse shared enumeration costs O(sum_h h²H(M,h)+MK²), then marked polynomial inversion O(MK⁴); the expensive partition sums remain. |
| Rooted-tree majorant | O(K C_pair+K²M) after constructing the absolute operator; computes a bound, with unproved useful tightness. |
| Decision-certified search | Add construction, every node bound, frontier scans, readout and adverse-event queries. Neither a small visited fraction nor 128 cached scores measures total work. |

Concrete implementation overheads remain in the finite helpers: repeated suffix scans/sorts, recomputed coordinate
boxes, dense count markers, frontier rescans and (MK)-square adverse-event matrices. Exact reorganizations can
remove some overhead without changing the exponential search/state issue. Shared local-option adjoints and some
scalar-reverse/Fourier-marker reorganizations were one-off finite checks, not permanent optimized implementations.

Growing B12 to B14 at fixed Bc=6 quadruples P and M; measured parity entries grow about 16–17x. Increasing Bc
instead grows V and potentially the overlap ranks. Increasing K changes count-state/connected/separator growth.
Any hypothetical bounded-rank/degree regime must be established for the actual growing family.

Full IID tail output has MK entries. A final K-message list contains only O(KB) bits: no M-sized lower bound
for every possible exact list decoder has been proved. Likewise, finite compression factors do not rescue an
uncontrolled exponential construction. Count static setup, observation work, complete inference, marked
readout, peak memory, numerical precision and error-control costs together.

## 15 October: close the broad exact stage and clean the record

The completed exact derivations remain in Report 7. This consolidated log replaces repeated report-build
histories, superseded next-step instructions and the separate atlas/79-entry tracking apparatus. Their useful
identities, cost corrections and obstructions are retained above or in the report. Verification modules and
historical research are preserved; the experiment bank supplies the reproducible mapping.

Approximation work should now start from a named expensive exact operation and a derived cost/error claim.
Further exact investigation is warranted when it resolves that claim, rather than as another open-ended survey.

## 16 October: separate exact resources from approximation assumptions

The user confirmed the Bayes decoder route. Exact posterior evaluations remain alternative calculations of
the same target; each exposes expensive objects where assumptions can be derived. Report 7 is now exact-only:
the bit/pattern factorization was preserved beside polar algebra, while candidate/BP/Gaussian/mean-field/MAP
route sections and the approximation mechanism menu were removed. Exact known-cardinality count-factor
reparameterization from the former working notes was retained as an algebraic resource.

Report 9 owns algebraically motivated assumptions and associated accuracy/cost arguments. Its initial
foundation contains the bottleneck map and shared error calculus, with no concrete family selected. The
previous approximation recipes and numerical rankings were cleared from the active notes; scripts remain
parked. No receiver implementation, new tests or PUPE experiments are part of this stage.
