
using ArgParse
using Statistics
using Flux
using DataStructures
using Zygote
include("./utils.jl")

shared_embed = 1000
shared_hid   = 1000
BATCH_SIZE   = 64
NUM_BLOCKS   = 12
mutable struct Node
   id::Integer
   sigma::Function
end

Node(id::Integer, sigma=tanh) = Node(id,sigma)

struct Leaf
end


dag = Dict(0=> [Node(1, tanh)], 1=> [Node(2,tanh)], 2=> [Node(3,tanh),Node(4,relu)],3=>[Leaf],4=>[Node(5,tanh),Node(6,tanh),Node(7,relu)], 5=>[Node(8,relu)], 6=>[Leaf], 7=>[Leaf], 8=>[Node(9,relu)],9=> [Node(10,relu),Node(11,relu), Node(12,relu)],10=>[Leaf],11=>[Leaf],12=>[Leaf])


struct Linear
  W
end

Linear(in::Integer, out::Integer) =
  Linear((randn(out, in)))

(m::Linear)(x) = (m.W * x')'

function init_wdict(wdicts,num_blocks, shared_hid)
   for idx in 1:num_blocks 
      for jdx in (idx+1):num_blocks
         for wd in wdicts
            wd[idx][jdx] = Linear(shared_hid,shared_hid)
         end
      end
   end
end

mutable struct ChildRNN
   w_xc #weights x->c
   w_xh #weights x->h
   w_hh
   w_hc
   w_c::DefaultDict  #default dict
   w_h::DefaultDict  #default dict
   h
end

function ChildRNN(shared_embed, shared_hid, batch_size, num_blocks=NUM_BLOCKS,init=randn) 
   w_c = DefaultDict(Dict)
   w_h = DefaultDict(Dict)
   init_wdict([w_c,w_h],num_blocks,shared_hid)
   ChildRNN(Linear(shared_embed,shared_hid),
            Linear(shared_embed,shared_hid),
            (randn(shared_embed, shared_hid)),
            (randn(shared_embed, shared_hid)),
            w_c,
            w_h,
            zeros(batch_size,shared_hid))
end

function loss(xs, ys)
  l = sum(sqrt.((xs .- ys).^2))
  return l
end

function cell(rnn::ChildRNN, x, h_prev, dag)
   c = Dict()
   h = Dict()
   f = Dict()

   f[1] = dag[0][1].sigma
   println("h_prev size: $(size(h_prev))")
   println("rnn.w_hc size: $(size(rnn.w_hc))")
   c[1] = sigmoid.(rnn.w_xc(x) .+ (h_prev * rnn.w_hc))
   h[1] = (c[1] .* f[1].(x + h_prev * rnn.w_hh) .+ (1 .- c[1]) .* h_prev)

   leaf_node_ids = []
   q = DataStructures.Deque{Int}()
   pushfirst!(q,1)

   while true
      if(length(q) == 0)
         break
      end
      node_id = popfirst!(q)
      @show node_id
      nodes = dag[node_id]
      for next_node in nodes
         @show next_node
         if next_node == Leaf
            push!(leaf_node_ids,node_id)
            continue
         end
         @show next_node
         next_id = next_node.id
         @show next_id
#         println("w_c[$node_id][$next_id]")

         @show keys(rnn.w_h)
         @show keys(rnn.w_c)
         w_h = rnn.w_h[node_id][next_id]
         w_c = rnn.w_c[node_id][next_id]
         f[next_id] = next_node.sigma
         c[next_id] = sigmoid.(w_c(h[node_id]))
         h[next_id] = (c[next_id] .* f[next_id].(w_h(h[node_id])) +
                                     (1 .- c[next_id]) .* h[node_id])
         push!(q, next_id)
      end #for
   end #while
   leaf_nodes = [h[node_id] for node_id in leaf_node_ids]
   @show size(leaf_nodes)
   output = mean(leaf_nodes)

   return(output, h[NUM_BLOCKS])
end

function forward(inputs, dag, hidden=nothing)
   time_steps = size(inputs,1)
   batch_size = size(inputs,2)
   if(hidden == nothing)
      #TODO: hidden = 
   end

end

function (m::ChildRNN)(xi,hi)
   w_xc, w_hc, W_h, W_c = m.w_xc, m.w_hc, m.w_h, m.w_c 
   w_hh, w_hc = m.w_hh, m.w_hc
   left = w_xc(xi)
   right = hi * w_hc
   println("size(hi): $(size(hi))")
   println("size(w_hc): $(size(w_hc))")
   println("size(right): $(size(right))")
   c1 = sigmoid.(left .+ right)
   h1 = (c1 .*  tanh.(xi  + hi * w_hh)  .+ (1 .- c1) .* hi)

   w_h = W_h[1][2]
   w_c = W_c[1][2]
   c2  = sigmoid.(w_c(h1))   
   h2  = (c2 .*  tanh.(w_h(h1)) + 
                (1 .- c2) .* h1)

   w_h = W_h[2][3]
   w_c = W_c[2][3]
   c3  = sigmoid.(w_c(h2))   
   h3  = (c3 .*  tanh.(w_h(h2)) + 
                (1 .- c2) .* h2) #leaf


   w_h = W_h[2][4]
   w_c = W_c[2][4]
   c4  = sigmoid.(w_c(h2))   
   h4  = (c4 .*  relu.(w_h(h2)) + 
                (1 .- c2) .* h2) #leaf

   w_h = W_h[4][5]
   w_c = W_c[4][5]
   c5  = sigmoid.(w_c(h4))   
   h5  = (c5 .*  relu.(w_h(h4)) + 
                (1 .- c4).* h4) 


   w_h = W_h[4][6]
   w_c = W_c[4][6]
   c6  = sigmoid.(w_c(h4))   
   h6  = (c6 .*  tanh.(w_h(h4)) + 
                (1 .-c4) .* h4) #leaf


   w_h = W_h[4][7]
   w_c = W_c[4][7]
   c7  = sigmoid.(w_c(h4))   
   h7  = (c6 .*  relu.(w_h(h4)) + 
                (1 .- c4).*h4) #leaf

   w_h = W_h[5][8]
   w_c = W_c[5][8]
   c8  = sigmoid.(w_c(h5))   
   h8  = (c8 .*  relu.(w_h(h5)) + 
                (1 .- c5).*h5) 

   w_h = W_h[8][9]
   w_c = W_c[8][9]
   c9  = sigmoid.(w_c(h8))   
   h9  = (c9 .*  relu.(w_h(h8)) + 
                (1 .-c8).*h8) 

   w_h = W_h[9][10]
   w_c = W_c[9][10]
   c10  = sigmoid.(w_c(h9))   
   h10  = (c10 .*  relu.(w_h(h9)) + 
                  (1 .-c9).*h9) #leaf

   w_h = W_h[9][11]
   w_c = W_c[9][11]
   c11  = sigmoid.(w_c(h9))   
   h11  = (c11 .*  relu.(w_h(h9)) + 
                  (1 .-c9).*h9) #leaf

   w_h = W_h[9][12]
   w_c = W_c[9][12]
   c12  = sigmoid.(w_c(h9))   
   h12  = (c12 .*  relu.(w_h(h9)) + 
                  (1 .-c9).*h9) #leaf
   output = mean([h3,h6,h7,h10,h11,h12])
   return output, h12
end

hidden(m::ChildRNN) = m.h

Flux.@treelike ChildRNN
m_rnn = Chain(
              ChildRNN(shared_embed, shared_hid, BATCH_SIZE),
              Flux.softmax
              )

x = rand(64,1000)

rnn = ChildRNN(shared_embed, shared_hid, BATCH_SIZE)
#yhat = Flux.softmax(rnn(x,zeros(64,1000)))
yhat = (rnn(rand(64,1000),zeros(64,1000)))
ce_loss = loss(yhat[1], randn(64,1000))
print("ce_loss is: $ce_loss")

#OK, forget Chain, make our own model chain:
child = ChildRNN(shared_embed, shared_hid, BATCH_SIZE)

function model(x,y)
   yhat    = child(x,zeros(64,1000))
   ce_loss = loss(yhat[1], y)
   return ce_loss
end
#forward eval of child rnn:
println(cell(child, x, zeros(64,1000),dag))

nabla = Zygote.gradient(model, (rand(64,1000), rand(64,1000))...)

#@code_typed(gradient(loss, m_rnn, x, zeros(64,1000)))
@code_typed Zygote.gradient(model, (rand(64,1000), rand(64,1000))...)

Zygote.@code_adjoint model(rand(64,1000), rand(64,1000))

