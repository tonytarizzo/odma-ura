
- implement whole armada of new decoder approaches
- test all thoroughly
- draw conclusions on next steps

- separately, look into quantifying bounds/theory in terms of quantities such as average block overlap per resource, active symbol load, or more graph-native measures such as local treewidth, maximum resource degree, overlap spectrum, or chordality of the block-intersection graph. These are exactly the quantities that should control whether block-state BP, junction-tree elimination, or coordinate descent remain tractable. This gap is especially attractive because it connects coding design, graph structure, and decoder complexity in one place.

- separately, has grokking been seen in learned decoders, and if not, can we first see if grokking type behaviour can be reached, and if yes, could this then be used to reverse engiener what structures/process the learning found exploited the setup, which might then inform us on how to modify the classical unlearnt decoders directly?