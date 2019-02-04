# DreamrHHP

Code repository for the paper on Dynamic Real-time Multimodal Routing (DREAMR) with Hierarchical Hybrid Planning (HHP). The paper is currently in submission to IEEE Intelligent Vehicles Symposium 2019 and is available on arXiv at [LINK]. 

**N.B - The code is stable but is a continuous work in-progress. The experiments for the paper were done in Julia 0.6 and is available as the v0.2 release (the module is called HitchhikingDrones rather than DreamrHHP).**

## Usage
The DreamrHHP repository is set up as a package with its own environment in [Julia 1.0](https://julialang.org/downloads/). Look at **Using someone else's project** at the Julia [package manager documentation](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1) for the basic idea. To get the code up and running (after :
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