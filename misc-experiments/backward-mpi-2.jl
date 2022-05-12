
# # Simple multi-layer perceptron


# In this example, we create a simple [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP) that classifies handwritten digits
# using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). A MLP consists of at least *three layers* of stacked perceptrons: Input, hidden, and output. Each neuron of an MLP has parameters 
# (weights and bias) and uses an [activation function](https://en.wikipedia.org/wiki/Activation_function) to compute its output. 


# ![mlp](../mlp_mnist/docs/mlp.svg)

# Source: http://d2l.ai/chapter_multilayer-perceptrons/mlp.html



# To run this example, we need the following packages:


using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy, mse
using Base: @kwdef
using CUDA
using MLDatasets
using Zygote
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
num_devs = MPI.Comm_size(comm)


# We set default values for learning rate, batch size, epochs, and the usage of a GPU (if available) for the model:

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
    split::Bool = true
    identity_dataset = true
end

# If a GPU is available on our local system, then Flux uses it for computing the loss and updating the weights and biases when training our model.


# ## Data

# We create the function `getdata` to load the MNIST train and test data from [MLDatasets](https://github.com/JuliaML/MLDatasets.jl) and reshape them so that they are in the shape that Flux expects. 

function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    ## Load dataset	
    if args.identity_dataset
        num_examples = 10
        xtrain = randn((1, num_examples))
        xtest = randn((1, num_examples))
        ytrain, ytest = xtrain, xtest
    else
        xtrain, ytrain = MLDatasets.MNIST.traindata()
        xtest, ytest = MLDatasets.MNIST.testdata()
        
        ## Reshape input data to flatten each image into a linear array
        xtrain = Flux.flatten(xtrain)
        xtest = Flux.flatten(xtest)

        ## One-hot-encode the labels
        ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
    end
    ## Create two DataLoader objects (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

# The function `getdata` performs the following tasks:

# * **Loads MNIST dataset:** Loads the train and test set tensors. The shape of train data is `28x28x60000` and test data is `28X28X10000`. 
# * **Reshapes the train and test data:**  Uses the [flatten](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.flatten) function to reshape the train data set into a `784x60000` array and test data set into a `784x10000`. Notice that we reshape the data so that we can pass these as arguments for the input layer of our model (a simple MLP expects a vector as an input).
# * **One-hot encodes the train and test labels:** Creates a batch of one-hot vectors so we can pass the labels of the data as arguments for the loss function. For this example, we use the [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) function and it expects data to be one-hot encoded. 
# * **Creates mini-batches of data:** Creates two DataLoader objects (train and test) that handle data mini-batches of size `1024 ` (as defined above). We create these two objects so that we can pass the entire data set through the loss function at once when training our model. Also, it shuffles the data points during each iteration (`shuffle=true`).

# ## Model

# As we mentioned above, a MLP consist of *three* layers that are fully connected. For this example, we define our model with the following layers and dimensions: 

# * **Input:** It has `784` perceptrons (the MNIST image size is `28x28`). We flatten the train and test data so that we can pass them as arguments to this layer.
# * **Hidden:** It has `32` perceptrons that use the [relu](https://fluxml.ai/Flux.jl/stable/models/nnlib/#NNlib.relu) activation function.
# * **Output:** It has `10` perceptrons that output the model's prediction or probability that a digit is 0 to 9. 


# We define the model with the `build_model` function: 


function build_model(args; imgsize=(28,28,1), nclasses=10)
    if args.identity_dataset
        return Chain(
            Dense(1, 1, relu),
            Dense(1, 1, relu),
            Dense(1, 1, relu),
            Dense(1, 1)
        )
    else
        return Chain( 
            Dense(prod(imgsize), 64, relu),
            Dense(64, 48, relu),
            Dense(48, 32, relu),
            Dense(32, nclasses))
    end
end

# Note that we use the functions [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) so that our model is *densely* (or fully) connected and [Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain) to chain the computation of the three layers.

# ## Loss function

# Now, we define the loss function `loss_and_accuracy`. It expects the following arguments:
# * ADataLoader object.
# * The `build_model` function we defined above.
# * A device object (in case we have a GPU available).

function loss_and_accuracy(data_loader, model, device, args)
    # return loss, accuracy for rank == 0, else (0, 0)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        if (rank == 0 || args.split == false)
            if args.identity_dataset
                ls += mse(ŷ, y, agg=sum)
            else
                ls += logitcrossentropy(ŷ, y, agg=sum)
                acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
            end
        end
        num +=  size(x)[end]
    end
    return ls / num, acc / num
end

# This function iterates through the `dataloader` object in mini-batches and uses the function 
# [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) to compute the difference between 
# the predicted and actual values (loss) and the accuracy. 


# ## Train function

# Now, we define the `train` function that calls the functions defined above and trains the model.

mutable struct SplitModel
    layers::Chain
    params::Params
    last_input
end

function SplitModel(build_model, args)
    device!(rank)
    if rank == 0
        model = build_model(args)
        N = length(model)
        layers = model[1:N/num_devs]
        for r in 2:num_devs
            MPI.send(model[(r-1)*N/num_devs + 1: r*N/num_devs], r-1, 0, comm)
        end
    else
        layers, status = MPI.recv(0, 0, comm)
    end
    layers = gpu(layers)
    return SplitModel(layers, Flux.params(layers), nothing)
end

function (split_model::SplitModel)(x)
    # if rank == 0, return (layer_in, layer_out). Else, return (layer_in, nothing)
    if rank > 0
        x, status = MPI.recv(rank-1, rank, comm)
    else
        x = gpu(x)
    end
    split_model.last_input = x
    
    # run layers
    act = split_model.layers(x)
    
    # send output
    if rank < num_devs-1
        MPI.send(act, rank+1 , rank+1,comm)
    else
        MPI.send(act, 0, 0, comm)
    end
    if rank == 0
        out, status = MPI.recv(num_devs-1, 0, comm)
        return out
    end
end

function get_gradient(split_model::SplitModel, loss_func, y, input)
    
    # forward pass
    out, back_in = pullback(split_model.layers, input)

    if rank < num_devs-1
        d_out, status = MPI.recv(rank+1, 0, comm)
    else
        # Do something with final activation?
        y = gpu(y)
        _, back = pullback(ŷ -> loss_func(ŷ, y), out)
        d_out = back(1)
    end

    d_in = back_in(d_out[1])

    if rank > 0
        MPI.send(d_in, rank-1, 0, comm)
    end

    _, back_param = pullback(() -> split_model.layers(input), params(split_model))
    gs = back_param(d_out[1])
    return (gs)
end

params(split_model::SplitModel) = split_model.params
last_input(split_model::SplitModel) = split_model.last_input

function train(; kws...)
    args = Args(; kws...) ## Collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    ## Create test and train dataloaders
    train_loader, test_loader = getdata(args)

    ## Construct model
    if args.split
        model = SplitModel(build_model, args)
    else
        model = build_model(args) |> device
    end
    # println("output: ", size(model(randn(28*28))))
    # ps = Flux.params(model) ## model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.η)
    
    if args.identity_dataset
        loss_func = mse
    else
        loss_func = logitcrossentropy
    end
    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            if args.split
                "               
                act = model(x)
                input = last_input(model)
                if rank == 0
                    @assert input == gpu(x)
                end
                # STORE DIFFERENT INPUTS
                gs = get_gradient(model, logitcrossentropy, y, input)
                Flux.Optimise.update!(opt, params(model), gs) ## update parameters
                "
                
                if rank > 0
                    x, status = MPI.recv(rank-1, rank, comm)
                else
                    x = gpu(x)
                end
                println("rank ", rank, " x ", size(x), " ", typeof(x))
                
                # run layers
                out, back_in = pullback(model.layers, x)

                println("rank ", rank, "sending ", size(out), " ", typeof(out))
                # send output
                if rank < num_devs-1
                    MPI.send(out, rank+1 , rank+1,comm)
                end
                

                # get derivative
                if rank < num_devs-1
                    d_out, status = MPI.recv(rank+1, 0, comm)
                else
                    # Do something with final activation?
                    y = gpu(y)
                    _, back = pullback(ŷ -> loss_func(ŷ, y), out)
                    d_out = back(1)
                end
                println("rank ", rank, "received d_out", size(d_out[1]))
            
                d_in = back_in(d_out[1])
            
                if rank > 0
                    MPI.send(d_in, rank-1, 0, comm)
                    println("rank ", rank, "sending d_in ", size(d_in[1]))
                end
                ps = Flux.params(model.layers)
                _, back_param = pullback(() -> model.layers(x), ps)
                gs = back_param(d_out[1])

                println("rank ", rank, "optimizing")
                Flux.Optimise.update!(opt, ps, gs) ## update parameters
                break

            else
                x, y = device(x), device(y)
                ps = Flux.params(model)
                gs = gradient(() -> logitcrossentropy(model(x), y), ps)
                Flux.Optimise.update!(opt, ps, gs) ## update parameters
            end
        end
        
        ## Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model, device, args)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, device, args)
        if (rank == 0 || args.split == false)
            println("Epoch=$epoch")
            println("  train_loss = $train_loss, train_accuracy = $train_acc")
            println("  test_loss = $test_loss, test_accuracy = $test_acc")
        end
        break
    end
end

# ## Run the example 

# We call the `train` function:

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

@time train()
