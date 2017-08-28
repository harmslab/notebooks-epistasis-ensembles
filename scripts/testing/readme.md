# Does population size matter?

## Experimental design


1. Build a set of trajectories with infinite population size
    1. Randomly sample 3 possible moves at each step, weighted by fixation probabilities
    2. Get the sequences and fixation probabilites
2. Sample those trajectories with different popsize
3. Compare true trajectories (probabilities from actual lattice proteins) to predicted trajectories (using 2nd-order epistasis model)