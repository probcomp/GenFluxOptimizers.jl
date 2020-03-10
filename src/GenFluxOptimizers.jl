module GenFluxOptimizers

using Gen
using Flux

"""
    conf = FluxOptimConf(func, args)
"""
struct FluxOptimConf{F}
    func :: F
    args :: Tuple
end

struct FluxParam{T}
    name :: Symbol
    gen_fn :: T
end

struct FluxOptimState{T}
    state  :: T
    gen_fn :: Union{DynamicDSLFunction, Gen.StaticIRGenerativeFunction}
    params :: Vector{FluxParam}
end


Base.zero(p::FluxParam) = flux_zero(get_param(p.gen_fn, p.name))
flux_zero(::Float64) = [0.0]
flux_zero(x) = zero(x)
flux_delta(delta::Float64) = [delta]
flux_delta(delta) = delta

function Gen.init_update_state(conf :: FluxOptimConf,
                               gen_fn :: Union{DynamicDSLFunction, Gen.StaticIRGenerativeFunction},
                               param_list::Array{T, 1} where T)
    FluxOptimState(conf.func(conf.args...), gen_fn, FluxParam[FluxParam(name, gen_fn) for name in param_list])
end

function Gen.apply_update!(state::FluxOptimState{T}) where T
    for param in state.params
        grad = get_param_grad(param.gen_fn, param.name)
        delta = Flux.Optimise.apply!(state.state, param, flux_delta(grad))
        value = Gen.get_param(param.gen_fn, param.name)
        if grad isa Float64
            set_param!(state.gen_fn, param.name, value + delta[1])
        else
            set_param!(state.gen_fn, param.name, value + delta)
        end
        zero_param_grad!(state.gen_fn, param.name)
    end
end

export FluxOptimConf

end # module
