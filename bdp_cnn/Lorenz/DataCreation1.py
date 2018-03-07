from LorenzDataCreation import Lorenz
from joblib import Parallel, delayed


def run_ensemble(model,i):
    model.run_model(boundaries=(i,i+0.0005),label=i)
    return (i,model.results[i])

def main(members,init_time=1000,duration_time=365*10,stepsize=6):
    model = Lorenz(init_time,stepsize,duration_time)
    results = Parallel(n_jobs=-1,verbose=5)(delayed(run_ensemble)(model,i) for i in range(members))

    result = {}
    for tup in results:
        result[tup[0]] = tup[1]

    model.results = result.copy()
    del results,result
    return model

if __name__ == "__main__":
    model  = main(4)