

- figure out cause of instability
- figure out how to gaurantee convergence even if less optimal
- figure out how to decrease complexity to approach OMP glob method


- have a setup that systematically runs the approaches for combos of inputs e.g
10 devices, 8 patterns, SNRs 10 and 0, antenna count etc
20 devices, 8 patterns, SNRs 10 and 0, antenna count etc
30 devices, 8 patterns, SNRs 10 and 0, antenna count etc
40 devices, 8 patterns, SNRs 10 and 0, antenna count etc
50 devices, 8 patterns, SNRs 10 and 0, antenna count etc
etc




- implement an alternative of v1 that doesnt throw away cross variance terms in step 1, this is to evaluate how much of an effect those have, even if it increases computation a lot (curiosity)
- figure out how best to now add designs for pilot coding that allows current v2 decoder to be used effectively with pilot codes - can either do conventional pilots, or send pilots using odma method and have an initial decode before the main message decode (future stage)