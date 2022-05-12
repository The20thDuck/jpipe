using CUDA
using Flux
using Zygote

submodels = []
for d in devices()
    device!(d)
    push!(submodels, Dense(10 => 10))
end
model = Chain(submodels)

x = randn(10)
model(x)

function loss(x, y)
    ŷ = model(x)
    return sum((ŷ .- y)^2)
end

gs = gradient(Flux.params(model)) do 
    loss(x, x)
end
