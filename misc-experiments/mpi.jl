using CUDA
# using MPIPreferences

ENV["JULIA_MPI_BINARY"]="system"
ENV["JULIA_MPI_LIBRARY"]="/opt/openmpi/lib/libmpi.so"

using Pkg
Pkg.add("MPI")
Pkg.build("MPI", verbose=true)
using MPI
# Pkg.build("CUDA")
using CUDA
CUDA.versioninfo()
MPI.Init()

print(MPI.has_cuda())

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dst = mod(rank+1, size)
src = mod(rank-1, size)

N = 4

send_mesg = CuArray{Float64}(undef, N)
recv_mesg = CuArray{Float64}(undef, N)

fill!(send_mesg, Float64(rank))

rreq = MPI.Irecv!(recv_mesg, src,  src+32, comm)

print("$rank: Sending   $rank -> $dst = $send_mesg\n")
sreq = MPI.Isend(send_mesg, dst, rank+32, comm)

stats = MPI.Waitall!([rreq, sreq])
device!(rank)
cu_mesg = CuArray{Float64}(undef, N)
copyto!(cu_mesg, recv_mesg)

dev = string(device())
print("$rank: Received $src -> $rank = $cu_mesg, $dev\n")


MPI.Barrier(comm)

delete!(ENV, "JULIA_MPI_LIBRARY")
delete!(ENV, "JULIA_MPI_PATH")

delete!(ENV, "JULIA_MPI_BINARY")

using Libdl
dlopen("/home/ubuntu/opt/openmpi/lib/libmpi.so")
