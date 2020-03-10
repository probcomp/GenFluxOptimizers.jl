using Gen
using GenFluxOptimizers
using Flux.Optimise

@gen function f(xs)
    @param slope :: Float64
    @param intercept :: Float64
    @param noise :: Float64
    for (i, x) in enumerate(xs)
        {:y => i} ~ normal(slope*x + intercept, exp(noise))
    end
end
init_param!(f, :slope, 0.0)
init_param!(f, :intercept, 0.0)
init_param!(f, :noise, 0.0)

xs = -5.0:1:5.0
ys = choicemap([(:y => i) => normal(3*x + 2, 0.3) for (i, x) in enumerate(xs)]...)
tr, = generate(f, (xs,), ys)

up = ParamUpdate(FluxOptimConf(Optimise.ADAM, (0.1, (0.9, 0.999))), f)
for i=1:1000
    accumulate_param_gradients!(tr)
    apply!(up)
end

println(Gen.get_param(f, :slope))
println(Gen.get_param(f, :intercept))
println(exp(Gen.get_param(f, :noise)))
