# Job 029: Joint Encoder/Decoder Learning at B=14

Job `027` fixed each codebook and trained only its decoder. This batch tests whether co-adapting the nonzero amplitudes
with D0 or D1 closes part of the remaining sparse-to-dense gap or removes the late sparse-support deterioration.

The 20 rows use one operating point (`B=14,n=256`), the same two seeds as job `028`, D0/D1, and five encoder settings: learned dense;
arbitrary iid support at `T=16,32`; and selected linear-hash support at `T=16,32`. Sparse support locations remain fixed.
After every optimiser step, zero entries are restored exactly and every codeword is projected to unit energy. Dense and
sparse models all receive 120 joint-training epochs; this is six times the old D1 decoder budget and 1.5 times the old
D0 budget. Initial and final geometry plus all loss curves are saved so lack of convergence is visible.

Local checks:

```bash
bash jobs/029_joint_encoder_decoder_B14/local_smoke.sh
bash jobs/029_joint_encoder_decoder_B14/local_mini.sh
```

Submit all 20 tasks:

```bash
qsub jobs/029_joint_encoder_decoder_B14/029_joint_encoder_decoder_B14.sh
```

The sparse rows are indices `1-16`; dense references are `17-20`:

```bash
qsub -J 1-16 jobs/029_joint_encoder_decoder_B14/029_joint_encoder_decoder_B14.sh
qsub -J 17-20 jobs/029_joint_encoder_decoder_B14/029_joint_encoder_decoder_B14.sh
```

After jobs `028` and `029` return, merge them into a direct fixed-versus-joint comparison:

```bash
uv run python -m tests.framework_joint_learning_merge \
  --joint-root jobs/029_joint_encoder_decoder_B14/results \
  --joint-manifest jobs/029_joint_encoder_decoder_B14/manifest.tsv \
  --fixed-root jobs/028_hash_skeleton_B14/results \
  --fixed-manifest jobs/028_hash_skeleton_B14/manifest.tsv \
  --out-dir jobs/029_joint_encoder_decoder_B14/results/merged
```

The central comparison is iid versus selected-hash at equal `T`, decoder, and seed, before and after joint learning.
Dense is a learned reference. A favourable result would show that fixed generated support is compatible with useful
co-adaptation; it would not make the current global D0/D1 receiver scalable in `B`.
