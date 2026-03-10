# v1:
Works really well

# v2: 
Works really well

# v2 full: 
Compares to other baseline decoders, beats but is more complex

# v3: 
Attempts to solve the prohibitively complex problem of estimating H directly with x. 
Since both are unknown without the other, it creates a bootstrapping issue causing collapse to mean 0.

# v3a: 
Implements a simpler method based on MD-AirComp's approach that directly uses the knowledge of the first antenna being a count vector whilst the other antennas share same support, very useful prior but requires the reciever to know the transmitters are doing perfect channel pre-equalisation. Device truncation added too. Results show this approach just isnt useful. Fading has to be approached with a more design first idea, eg. pilot codes.

# v4:
Pilot assisted full fading setup. TBD

