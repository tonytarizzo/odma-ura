# Required energy versus active users

At each load the plotted receiver ordinate is the physical $E_b/N_0$ where the measured per-user missed-copy probability crosses 0.05. The encoder spends one unit of energy per 100-bit message and the real chip noise has variance $N_0/2$. The per-load bootstrap resamples whole frames, preserving the shared geometry, messages and standardized noise across energies and both decoders. A downward or upward triangle gives the corresponding one-sided search limit. A recrossing leaves a gap. The machine-readable summary retains every crossing status and unbounded interval endpoint.

## The two mathematical references

The dotted sufficient-energy curve evaluates Polyanskiy's finite-blocklength Theorem 1 under IID source messages at the same 5% target. The audited implementation uses all Gallager $p_t$ terms and the common-noise $q_1$ integral, following the numerical recipe in Section III. It includes the clipping and collision penalties. The collision-as-error loss upper-bounds the count loss for the constructed code. A reported decoder can therefore attain the requested error at a lower energy than this sufficient-energy curve. The numerical roots use the original exponent and auxiliary-power grids, quadrature order 96 and 0.002 dB root tolerance. The illustrative Fig. 1 in the original paper uses 10%. [Polyanskiy, 2017](https://people.lids.mit.edu/yp/homepage/data/isit17_mac.pdf) · [Audited numerical bound](../../benchmarks/ura_bounds.py)

The dashed necessary-energy line uses the repository's single-user genie/list-Fano converse. Choose a uniformly tagged transmission. Pad the decoder's set of distinct returned messages to a list of size $K$. Its membership miss probability $e$ is at most the original missed-copy probability, including repeated messages and partial count outputs. A genie reveals all other transmitted words. Independence of the IID source leaves the tagged message uniform on $M=2^{100}$ possibilities, and subtraction gives a single-user real Gaussian channel. Its mutual information in nats is at most $C=n\log(1+2B(E_b/N_0)/n)/2$. List-Fano requires

$$
\log M-C \le h$e$+(1-e)\log K+e\log(M-K).
$$

The right side increases up to $e=1-K/M$, so substitution of $e=0.05$ gives a necessary energy for the original count decoder. The curve is loose because the genie removes all interference and the list may contain $K$ words. Its slight decrease with load comes from the larger allowed list. A direct inversion of this equation agrees with the implemented converse at all six loads.

## Published scalar comparison

The open squares are the six magenta points of Özates, Kazemi and Duman, Fig. 3. The released figure form on publisher page 4 has vertical origin 367.5 and spacing 425 drawing units per dB. The authors' proposed-receiver ordinates give 0.25, 0.40, 0.98, 1.77, 2.70 and 3.65 dB for $K=50,100,150,200,250,300$. Their target uses message membership. Their BPSK waveform occupies one real quadrature in each of 30,000 samples; the paper's complex observation also supplies an orthogonal noise quadrature to a magnitude detector. They change placement bits, move to two power levels at $K=150$, three levels above $K=200$, and shorten the polar block to 256 above $K=200$. The measured curves use the Job 043 encoder declaration. [Özates et al., Sections III–IV and Fig. 3](https://spiral.imperial.ac.uk/server/api/core/bitstreams/284d4dc9-4e39-4086-a7eb-c50813dfe01a/content#page=4)

The same published figure also draws a black dashed random-coding reference citing Polyanskiy. Its digitized six ordinates are 0.3342, 0.4532, 0.5115, 0.8458, 1.3033 and 1.8041 dB. The paper does not specify a separate numerical recipe for that trace. The figure here displays the recomputed 5% theorem curve, with the digitized trace preserved separately in the JSON for comparison.

Run `uv run python jobs/043_polar_balanced_load/plot_load.py --references-only` to regenerate the reference-only PDF, PNG and JSON in `results/dec43_balanced_load/`. The normal merge renders `required_ebn0_vs_load.pdf` and its PNG from the complete holdout summary. The two-load eight-geometry local preview has separate filenames and identifies its pilot sample in the figure.
