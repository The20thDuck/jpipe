# Imports


using CUDA
using Flux
using MLDatasets
using MPI
using Serialization
using Zygote


# MPI setup


MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)


# Body of code


println("Rank $(rank) of $(comm_size)")


# Dataset parameters


microbatch_size = 256
num_microbatches = 16 # number of microbatches per batch
batch_size = num_microbatches * microbatch_size
num_classes = 10 # number of classes in the dataset
train_size = 500_000 # number of training examples
test_size = 100_000  # number of testing examples
image_size = (32, 32, 2) # size of images


function getdata()
    device!(rank)

    if rank == 0 || rank == comm_size - 1
        x_train = CuArray{Float32}(undef, (*image_size, train_size))
        x_test = CuArray{Float32}(undef, (*image_size, test_size))
        y_train = CuArray{Float32}(undef, train_size)
        y_test = CuArray{Float32}(undef, test_size)
    end

    # master process loads dataset
    if rank == 0
        ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

        x_train, y_train = CIFAR10.traindata()
        x_test, y_test = CIFAR10.testdata()

        y_train = Flux.onehotbatch(y_train, 0:(num_classes - 1))
        y_test = Flux.onehotbatch(y_test, 0:(num_classes - 1))
    end
end

getdata()
