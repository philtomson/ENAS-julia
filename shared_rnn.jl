using Random
using ArgParse
using Statistics
using Flux
using DataStructures
using Zygote

shared_embed = 1000
shared_hid   = 1000
BATCH_SIZE   = 64
NUM_BLOCKS   = 12
mutable struct Node
   id::Integer
   sigma::Function
end

Node(id::Integer, sigma=tanh) = Node(id,sigma)

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
   w_c::DefaultDict  #default dict
   w_h::DefaultDict  #default dict
end

function ChildRNN(shared_embed, shared_hid, batch_size, num_blocks=NUM_BLOCKS,init=randn) 
   w_c = DefaultDict(Dict)
   w_h = DefaultDict(Dict)
   init_wdict([w_c,w_h],num_blocks,shared_hid)
   ChildRNN(Linear(shared_embed,shared_hid),
            Linear(shared_embed,shared_hid),
            w_c,
            w_h)
end

rnn = ChildRNN(shared_embed, shared_hid, BATCH_SIZE)