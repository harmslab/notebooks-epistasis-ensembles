import numpy as np
import pickle


n_states = 3

def overlap(p,q):
    """Calculates the difference in overlap."""
    return np.array(abs(p - q)).sum() / (p.sum() + q.sum())

def shannon(probabilities):
    """Shannon entropy."""
    return -np.sum(probabilities * np.log(probabilities))    

thetas = []
omegas = []
for j, popsize in enumerate([1,10,100,1000,10000,100000, 1000000]):
    
    theta, omega = [], []
    for i in range(1000):
        try:
            true_fname = "results/true-trajectories-{}-{}-{}.pickle".format(
                i,
                popsize, 
                n_states)

            pred_fname = "results/pred-trajectories-{}-{}-{}.pickle".format(
                i,
                popsize, 
                n_states)

            with open(true_fname, "rb") as f:
                true_trajectories = pickle.load(f)

            with open(pred_fname, "rb") as f:
                pred_trajectories = pickle.load(f)   

            true_spect = np.array(list(true_trajectories[7].values()))   
            pred_spect = np.array(list(pred_trajectories[7].values()))   

            # Quality check
            trajs = np.array(list(true_trajectories[7].values()))
            x = sum(np.nan_to_num(trajs))
            if x < 1.0:
                continue
            else:
                theta.append(overlap(true_spect, pred_spect)) 
                omega.append(shannon(trajs))
        except FileNotFoundError:
            pass

    omegas.append(omega)
    thetas.append(theta)
    
# Write to file
with open("summary/theta-omega-{}.pickle".format(n_states), "wb") as f:
    data = {"omegas" : omegas, "thetas": thetas}
    pickle.dump(data, f)
    