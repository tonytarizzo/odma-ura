- Understand current v1 decoder implemenation and line up with algebra
- Note/remove any enhancements added to make it run & clean up code with Opus
- Check defined inputs to decoder reflect what would be known to the decoder in actuality

- Algebraicly determine best initialisations for variables
- Understand which latent variables, e.g K and SNR, are being defined/used in decoding
- Understand how they are being updated, when and with what approach (EM?)
- Make sure any updates to these variables are algebraicly correct

- Evaluate limits, thresholds, damping, stopping conditions and other implementation additions
- Draw up list of these additions/enhancements used in 3-4 key competing methods/research papers
- Compare both, see what is justifiable, what can be improved and implement

- Delve into adding more structures eg. baysien compatible rounding approaches and priors

- Test runs at 10db, 5db and 0db. 
- Add clear plotting functions
- Add simple robust slugging method to save folder of plots and outputs
