
using MPI
using CUDA
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dst = mod(rank+1, size)
src = mod(rank-1, size)

MSG_LENGTH = 100
device!(rank)

send_mesg = CuArray{Float64}(undef, MSG_LENGTH)
recv_mesg = CuArray{Float64}(undef, MSG_LENGTH + 100)


function communicate(n, alloc)

    if rank > 0
        if alloc
            alloc_recv, status = MPI.recv(src, 0, comm)
        else
            status = MPI.Recv!(recv_mesg, src, 0, comm)
            println(recv_mesg)
        end
    end
    if rank < size - 1
        fill!(send_mesg, Float64(n))
        # if alloc
        if alloc
            status = MPI.send(send_mesg, dst, 0, comm)
        else
            status = MPI.Send(send_mesg, dst, 0, comm)
        end
    end
end


function multi_communicate(N, alloc)
    for n in 1:N
        communicate(n, alloc)
    end
end

# const N = (60000 ÷ 256) *10*2*4
const N = 1

@time multi_communicate(N, false)
@time multi_communicate(N, false)

@time multi_communicate(N, true)
@time multi_communicate(N, true)
