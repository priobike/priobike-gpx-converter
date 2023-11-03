# GPX import

For the priobike mobile app we want to add a GPX import feature.
GPX files can contain thousands of points even for short routes through Hamburg.
We don't want to save shortcuts in the app with thousands of waypoints,
therefore we want to reduce the waypoint number as much as possible whilst maintaining the route shape.

## first approach - optimization

To decide whether a waypoint should be included these checks are used:
- is the waypoint further away from the last one than the `d1` distance threshold

or

- is the angle between the last segment and the current one larger than the `a` angle threshold and the distance larger than the `d2` distance threshold

These parameters `a, d1, d2` can be optimized based on random routes through Hamburg.

### sampled exhaustive search

A sensible sample of parameters is chosen and tested on several random routes and the mean. 
This does not guarantee optimality, the search space is largely unexplored and even when a sensible ranges for each parameter is chosen a lot of combinations are nonsensical 
Therefore, a lot of time is wasted on uninteresting solutions

solution for route1: ~10s
: `cost: 4184.410528, a_threshold: 0.3141592653589793, d_threshold_1: 8000.0, d_threshold_2: 6000.0`

### differential evolution 

Is a popular genetic algorithm that tries to find the global minimum of a multivariate function.
It doesn't rely on gradient methods and therefore is applicable to non-smooth, non-differentiable, non-continuous, noisy, changing over time functions with [plateaus](https://stackoverflow.com/questions/52742336/scipy-optimize-is-only-returning-x0-only-completing-one-iteration).
But it does not guarantee optimality and requires more function evaluations and can take longer to finish than other optimization methods.

solution for route1: ~30min
: `cost: 3906.1969384357176, a_threshold: 0.300940870, d_threshold_1: 5608.33038, d_threshold_2: 4672.55096` 

Because our cost function takes a long time to execute finding a solution is very costly and doing so for hundreds of random routes is not advised.

## second approach - iterative improvement

Instead of finding a combination of the three parameters that is good enough for all routes,
the distance to the original route can be checked several times during the approximation and
waypoints can be added at segments with a large distance. We use the first approach with the optimized parameters
to get a first approximation and then approve upon it iteratively until a cost threshold is undercut.

solution for route1: ~1min
: `cost: 77.75880700319742, a_threshold: 0.3141592653589793, d_threshold_1: 10000.0, d_threshold_2: 5000.0`