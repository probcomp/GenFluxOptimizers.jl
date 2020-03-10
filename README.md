# GenFluxOptimizers.jl

A plugin for [Gen](https://github.com/probcomp/Gen), enabling the use of any of [Flux](https://github.com/FluxML/Flux.jl)'s [optimizers](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Optimiser-Reference-1) for parameter learning in generative functions from Gen's static or dynamic modeling languages.

```julia
using Gen
using Flux.Optimise
using GenFluxOptimizers

@gen function f()
  @param p :: Float64
  ...
end

data = choicemap(...)
tr, = generate(f, (), data)

# Use Flux ADAM optimizer
adam_update = ParamUpdate(FluxOptimConf(Optimise.ADAM, (0.1, (0.9, 0.999))), f)
for i=1:1000
  accumulate_param_gradients!(tr)
  apply!(adam_update)
end
```
