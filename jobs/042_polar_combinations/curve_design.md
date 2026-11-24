# Job 042 — DEC-38 on the paper reference and paired energy maps

The default public DEC-38 fixes the tag-conditioned serial native receiver, then a deficit scan and a rate-triggered occupied-word challenge. Select the construction on the balanced Study 042 development and confirmation panels. Freeze the curve release before running the flat energy grid. This job evaluates the crossover of the full decoder relative to the native scalar literature implementation and the original default DEC-35. At K100 also run DEC-39, the independently confirmed serial receiver with a dimension-203 constituent challenge. The registered DEC-39 geometry requires 87 payload bits. The smaller K50 case uses 88. The job also evaluates the effect of balancing for all four K100 receivers. The soft-output paper has a different pilot/MIMO channel; its local fixed-pattern control and the published operating point stay in the separate paper comparison.

Reuse Job 039's flat scalar B100 geometry, 256 independent seeds per load, and all 12 energies for K100 and 14 for K50. Each frame has 30,000 real observations and 512 active coded chips per user, CRC16, known load and noise. Sources are IID complete 100-bit messages, including multiplicities. Warm/cold scheduling shares the physical frame, charges one full geometry per comparison and rotates the standalone receiver calls by seed. Native uses ordinary SCL128. Record complete count actions, ordinary set membership, false copies, receiver and geometry-inclusive time, and paid native lists. The first 416 shards each contain 16 flat cases, 6,656 in all.

A second set of 12 shards uses the first 64 K100 seeds at 0.25, 0.40 and 0.50 dB from the same grid with the established balanced amplitudes. Compare them with their 192 flat counterparts from the primary grid. The source bits, positions, thermal noise draw and information set must match exactly. For each receiver retain the three paired missed/false-copy totals, number of improved/regressed frames, gain after deleting the best frame and a paired frame-bootstrap interval. These three paired energies support a fixed-energy balancing comparison.

The merge verifies release, physical hashes, sparse count actions and complete arm inventories. The flat energy endpoint uses the original Job 039 first downward 5% crossing rule with explicit censoring and ambiguity when the curve recrosses. Two thousand paired bootstrap replicates preserve the geometry across all energies and arms, with crossing-status counts, individual bands and contrast bounds. A threshold or an energy benefit requires the returned data. Existing Job 039 files are read for definitions and integrity; outputs go to a separate job directory.

From the repository root after updating main and running the existing environment setup, submit both jobs with one shell line.

```sh
array_id=$(qsub jobs/042_polar_combinations/042_curve.sh) && qsub -W "depend=afterok:${array_id}" jobs/042_polar_combinations/042_merge.sh
```

Use `uv run python jobs/042_polar_combinations/paper_curve_extension.py --inventory` before submission. The merger writes `results/dec42_paper_curve/summary.json`. Re-submission verifies existing frame files and resumes missing work. There is no array concurrency cap.
