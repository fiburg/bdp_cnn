from .LorenzDataCreation import Lorenz
from joblib import Parallel, delayed


def _run_ensemble(model,i):
    """
    Used inside the parallelisation-loop to call the method run_model of the lorenz-class.

    Args:
        model: obj: instance of the Lorenz-class.
        i: int: loop variable.

    Returns:
         Returns a tuple with two values: first value is the loop variable second value is an xarray with the results.
    """

    model.run_model(boundaries=(i,i+0.0005),label=i)
    return (i,model.results[i])

def main(members,init_time=1000,duration_time=365*10,stepsize=6,cpus=-1):
    """
    Main function to easily create lots of data for CNN input.

    Args:
        members:  int: Number of ensemble members.
        init_time: float: number of days for initialization.
        duration_time: float: number of days to run the model.
        stepsize: float: stepsize in hours.
        cpus: int: number of cpus used for parallelization. -1 = all available.

    Returns:
        obj: instance of the Lorenz-class containing the results.

    Example:
        >>> model = main(4)

        This would create an object called "model" with the default settings and 4 model members.

    """

    model = Lorenz(init_time,stepsize,duration_time)
    results = Parallel(n_jobs=-1,verbose=5)(delayed(_run_ensemble)(model,i) for i in range(members))

    result = {}
    for tup in results:
        result[tup[0]] = tup[1]

    model.results = result.copy()
    del results,result
    return model

if __name__ == "__main__":
    model  = main(4)