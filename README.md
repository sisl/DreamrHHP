# DreamrHHP

Code repository for the paper on Dynamic Real-time Multimodal Routing (DREAMR) with Hierarchical Hybrid Planning (HHP). The paper is currently in submission to IEEE Intelligent Vehicles Symposium 2019 and is available on arXiv at https://arxiv.org/abs/1902.01560. The supplementary video for visualizing behavior on two grid problems is available on YouTube [here](https://youtu.be/e5IcB79TEXY) and the one for a real-world-scale street scenario is [here] (https://youtu.be/c3nfTa8BA-E)

**N.B - The code is stable but is a continuous work in-progress. The experiments for the paper were done in Julia 0.6 and is available as the v0.2 release (the module is called HitchhikingDrones rather than DreamrHHP).**

## Setup
The DreamrHHP repository is set up as a package with its own environment in [Julia 1.0](https://julialang.org/downloads/). Look at **Using someone else's project** at the Julia [package manager documentation](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1) for the basic idea. To get the code up and running (after having installed Julia):
```shell
$ git clone https://github.com/sisl/DreamrHHP.git
$ cd DreamrHHP
```
Then start the Julia REPL and go into [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/) mode by pressing `]`, followed by:
```shell
(v1.0) pkg> activate .
(DreamrHHP) pkg> instantiate
```
This will install the necessary depedendencies and essentially reproduce the Julia environment required to make the package work.

## Usage

**This code is primarily provided for illustrative purposes. There are several moving parts that work with each other, rather than stand-alone. The description that follows assumes some familiarity with the paper.**

The `DreamrHHP` module (defined in the `src`) folder implements the global open-loop layer, the local closed-loop layer, and any other utilities required to bridge them. The top-level HHP logic that invokes the two layers and interacts with the problem simulator, however, is in `scripts/` as an executable Julia file but is not imported as such by the module.


### Global Open-loop Layer

The logic for the global layer is implemented in `src/graph_plan/`. Following are the files of interest:

- `src/graph_plan/astar_visitor_light.jl` implements an **implicit** A* search, following the framework and convention of [Graphs.jl](https://github.com/JuliaAttic/Graphs.jl). A re-implementation is necessary so that successors can be generated with a neighbour function, rather than using the edges of the graph.
- `src/graph_plan/graph_solution.jl` implements the actual open-loop layer logic. It interfaces with the online route information that is available from the environment, updates the implicit graph, and calls the implicit A* search on it.
- `src/graph_plan/graph_helpers.jl` implements a collection of auxiliary methods that are used by `astar_visitor_light`, defines the vertex types and edge weight functions.

### Local Closed-loop layer

The logic required the local layer is implemented in `src/macro_action_policy` and the code that actually generates the closed-loop policies for macro-actions is written in `scripts/`. This layer's implementation relies heavily on the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

- `src/macro_action_policy/uavdynamics.jl` defines an interface for the agent dynamics (it can be any general dynamics model that an MDP can represent). It also implements a simple 2D multirotor model.
- `src/macro_action_policy/partial_control_mdp.jl` defines the MDPs for constrained flight, unconstrained flight, and riding. All the components necessary to use [LocalApproximationValueIteration](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl) to obtain policies for each of them, are also defined.
- `scripts/pcmdp_hopon_*.jl` is a pair of scripts that generates constrained flight policies with varying values of the beta risk-threshold parameter for abort. From the paper, note that the abort logic can only be introduced in the CF policy _after_ the partial control value iteration has been done once (to compute the worst values from each horizon). Therefore, `_preabort` computes the value function for the CF MDP without abort logic, while `_vary_abort_thresh` uses the in-horizon value function from `_preabort` and then regenerates the in-horizon policy with a specific beta.
- `scripts/ucmdp_` and `scripts/unconstr_` execute logic similar to `scripts/pcmdp_hopon_*` for the unconstrained flight and riding policies as `scripts/pcmdp_hopon_` does. There is only one script for each because there is no varying parameter like beta for CF.


### Simulators

There are two components to the DREAMR simulation - the car routes which are updated online and the MDP simulator which implements the transition function and computes the trajectory cost. We separate the two in order to be able to generate car route data separately and more easily work with standalone route data if it should be available.
**For anyone wishing to benchmark, the code for simulators can be exported and used without any heavy dependencies.**

- ` data/grid_data_generator.jl` implements a generator for car routes for a full episode of 360 epochs, each of 5 seconds (total 30 minutes). It follows the format expected by the global open-loop layer logic in `src/graph_plan/graph_solution.jl`. The script generates a dictionary of information, with a nested dictionary for each epoch, and saves it as a JSON file.
- ` src/simulators/sdmc_simulator.jl` implements the transition and cost functions for the DREAMR MDP (sdmc is short-hand for Single Drone Multiple Cars). The car route information generated from `data/grid_data_generator.jl` is provided to it (basically the car route info is generated standalone in advance), but the decision-making algorithm (HHP or RHC) only sees the updates to the currents information, and not the future. The general structure of the `SDMCSimulator` resembles that of an [OpenAI Gym](https://gym.openai.com/) environment.



### HHP Framework

The top-level code for HHP that invokes both layers and interacts with the DREAMR simulator is in `scripts/sdmc_solver.jl`. It broadly matches the `ONLINE` procedure in Algorithm 1 of the paper. It takes as arguments the offline policies, the parameter files, and the car routes for the full episode. The parameter and car route files are used to instantiate the `SDMCSimulator` object, which the script then interacts with. The solver script can also log the drone state for each epoch during execution (in order to postprocess or visualize the solution).


### Parameters

The set of spatio-temporal and cost parameters define the specific policies and behaviour of the framework. There are two relevant components:

 - `data/paramsets/*.toml` are the files that contain sets of values for the parameters in [TOML](https://github.com/toml-lang/toml) format. We split the parameters into `scale-*.toml` for spatial parameters, `simtime-*.toml` for temporal parameters, and `cost-*.toml` for cost function parameters. This allows easy combinations of parameters for different settings. The `data-allinone-*.toml` file is used by `data/grid_data_generator.jl` and contains parameters necessary for specifying the car routes.
 - `src/parameters.jl` implements a `Parameters` struct that reads in the parameter files and has nested structures called `ScaleParameters`, `SimTimeParameters` and `CostParameters` respectively. The struct instantitions are passed around across the various code modules.
