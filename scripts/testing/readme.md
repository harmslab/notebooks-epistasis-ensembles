# Does population size matter?

## Experimental design


1. Build a set of trajectories with infinite population size
    1. Randomly sample 3 possible moves at each step, weighted by fixation probabilities
    2. Get the sequences and fixation probabilites
2. Sample those trajectories with different popsize
3. Compare true trajectories (probabilities from actual lattice proteins) to predicted trajectories (using 2nd-order epistasis model)

We set out to determine whether population size for our predictions. In smaller populations,
selection has a smaller effect on the fixation probability, and therefore genetic drift plays
a major role in determining trajectories. Conversely, in larger population sizes, selection
dominates. In our simulations, we probe the extreme case of infinite population size. In this
regime, we would expect our predictive power is strongest. 
