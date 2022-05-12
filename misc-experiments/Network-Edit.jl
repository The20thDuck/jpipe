###########
# IMPORTS #
###########


using CUDA
using Flux
using MPI

##################
# NETWORK STRUCT #
##################


using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

struct Network
    layers
    devs
end

function Network(layers)
    # remember initial device
    # init_dev = device()

    # initialize things
    devs = [dev for dev in devices()]
    net_layers = []
    net_devs = []
    layers_per_dev = length(layers) ÷ length(devs)

    for (idx, layer) in enumerate(layers)
        # find the device that the layer should be copied to
        dev_idx = 1 + ((idx - 1) ÷ layers_per_dev)
        dev = devs[dev_idx]

        # copy the layer to the corresponding device
        # device!(dev)
        # layer = gpu(layer)

        # update things
        push!(net_layers, layer)
        push!(net_devs, dev)
    end

    # restore initial device
    # device!(init_dev)

    # return `Network` struct
    return Network(net_layers, net_devs)
end


# naive forward pass implementation
function (network::Network)(x)
    # for dev in devices()
        # copy input to given device,
        # passing though the CPU because this sucks
    println("rank:", rank, "network")
    device!(rank)
    if rank > 0
        x, status = MPI.recv(rank-1, rank, comm)
    end

    println("rank:", rank, "recv")

    dev_layers = [
        layer for (d, layer)
        in zip(network.devs, network.layers)
        if d == CuDevice(rank)
    ]
    println("rank:", rank, "dev calculated")

    for layer in dev_layers
        layer = gpu(layer)
        x = layer(x)
    end
    # end
    println("rank:", rank, "layer calculated")

    # copy input to CPU and return it

    if rank < size-1
        MPI.send(x, rank+1, rank+1,comm)
    end

    return cpu(x)
end


function mse(network, x, ŷ)
    # evaluate network output
    y = network(x)

    # copy output to CPU
    y = cpu(y)
    
    # return mean squared loss
    return sum((ŷ .- y)^2)
end


#########
# TESTS #
#########

if rank == 0
    layers = Chain(
        Dense(1, 1), Dense(1, 1), Dense(1, 1), Dense(1, 1)
    )
    N = length(layers)
    layer = layers[1:N/size]
    for r in 2:size
        MPI.send(layers[(r-1)*N/size + 1: r*N/size], r-1, 0, comm)
    end
else
    layer, status = MPI.recv(0, 0, comm)
end
println(rank, "got layer")

device!(rank)
layer = gpu(layer)
println(rank, "Got gpu")

if rank == 0
    x = cu([1])
else
    x, status = MPI.recv(rank-1, rank, comm)
end
println(rank, " Got x ", x, " ", typeof(x))

x = layer(x)

println(rank, "calc x")
if rank < size-1
    MPI.send(x, rank+1, rank+1,comm)
end

println(x)

# return cpu(x)

# out = network(x)
# print(layers(cpu(x)))
