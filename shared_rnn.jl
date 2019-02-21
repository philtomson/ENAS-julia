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
   w_c  #default dict
   w_h  #default dict
end

function ChildRNN(shared_embed, shared_hid, batch_size, init=randn) 
   w_c = DefaultDict(Dict)
   w_h = DefaultDict(Dict)
   init_wdict([w_c,w_h],numblocks,shared_hid)
   ChildRNN(Linear(shared_embed,shared_hid),
            Linear(shared_embed,shared_hid),
            w_c,
            w_h)
end
