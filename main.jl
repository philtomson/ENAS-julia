include("text.jl")
using Random
using ArgParse
using Statistics
using Flux
using DataStructures
using Zygote


s = ArgParseSettings()
@add_arg_table s begin
   "--epochs"
      help = "number of epochs to run"
      arg_type = Int
      default = 1
   "--num_blocks"
      help = "number of blocks in child RNNs"
      arg_type = Int
      default = 12
   "--random_seed"
      arg_type = Int
      default = 42
   "--data_path"
      help = "path to training data"
      arg_type = String
      default = "../ENAS-pytorch/data/ptb"
end
args = parse_args(ARGS, s)
for a in args
   println(" $(a[1]) => $(a[2])")
end

#args = let random_seed=42, 
#           data_path="../ENAS-pytorch/data/ptb"; () -> Any[random_seed, data_path]; end
function main(args)
   println("""Path to data is: $(args["data_path"])""")
   Random.seed!(args["random_seed"]) 
   dataset = Corpus(args["data_path"])
end

main(args)

#function linear(ws)
#   x -> ws * x #return a closure
#end

shared_embed = 1000
shared_hid   = 1000
#w_hc = param(randn(shared_hid, shared_hid))
#w_hh = param(randn(shared_hid, shared_hid))
w_hc = (randn(shared_embed, shared_hid))
w_hh = (randn(shared_embed, shared_hid))



struct Linear
  W
end

#Linear(in::Integer, out::Integer) =
#  Linear(param(randn(out, in)))

Linear(in::Integer, out::Integer) =
  Linear((randn(out, in)))

(m::Linear)(x) = (m.W * x')'

w_h = DefaultDict(Dict)
w_c = DefaultDict(Dict)
for idx in 1:args["num_blocks"]
   for jdx in (idx+1):args["num_blocks"]
        w_h[idx][jdx] = Linear(shared_hid, shared_hid)
        w_c[idx][jdx] = Linear(shared_hid, shared_hid)
   end
end

W_xc = Linear(shared_embed,shared_hid)
W_xh = Linear(shared_embed,shared_hid)

a = rand(64,1000)
left  = W_xc(a)
b = rand(64,1000)
right = (b * w_hc)

#best network from paper
function network(xi, hi, w_xc, w_xh, W_h, W_c)
   println("size(xi): $(size(xi))")
   println("size(hi): $(size(hi))")
   println("size(w_hc): $(size(w_hc))")
   left = w_xc(xi)
   right = hi * w_hc
   #c1 = sigmoid.(w_xc(xi) + hi * w_hc)
   println("size(left): $(size(left)) type: $(typeof(left))")
   println("size(right): $(size(right)) type: $(typeof(right))")
   c1 = sigmoid.(left .+ right)
   println("size(c1): $(size(c1)) type: $(typeof(c1))")

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
   #x4 = relu(x2)

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

   #x6 = tanh(x4) #leaf

   w_h = W_h[4][7]
   w_c = W_c[4][7]
   c7  = sigmoid.(w_c(h4))   
   h7  = (c6 .*  relu.(w_h(h4)) + 
                (1 .- c4).*h4) #leaf

   #x7 = relu(x4) #leaf
   w_h = W_h[5][8]
   w_c = W_c[5][8]
   c8  = sigmoid.(w_c(h5))   
   h8  = (c8 .*  relu.(w_h(h5)) + 
                (1 .- c5).*h5) 

   #x8 = relu(x5)
   w_h = W_h[8][9]
   w_c = W_c[8][9]
   c9  = sigmoid.(w_c(h8))   
   h9  = (c9 .*  relu.(w_h(h8)) + 
                (1 .-c8).*h8) 
   #x9 = relu(x8) 

   w_h = W_h[9][10]
   w_c = W_c[9][10]
   c10  = sigmoid.(w_c(h9))   
   h10  = (c10 .*  relu.(w_h(h9)) + 
                  (1 .-c9).*h9) #leaf
   #x10= relu(x9) #leaf

   w_h = W_h[9][11]
   w_c = W_c[9][11]
   c11  = sigmoid.(w_c(h9))   
   h11  = (c11 .*  relu.(w_h(h9)) + 
                  (1 .-c9).*h9) #leaf
   #x11= relu(x9) #leaf

   w_h = W_h[9][12]
   w_c = W_c[9][12]
   c12  = sigmoid.(w_c(h9))   
   h12  = (c12 .*  relu.(w_h(h9)) + 
                  (1 .-c9).*h9) #leaf
   Flux.softmax(mean([h3,h6,h7,h10,h11,h12]))
end

#function loss(xs, ys)
#  l = sum(Flux.crossentropy.(xs, ys))
#  return l
#end
function loss(xs, ys)
  l = sum(sqrt.((xs .- ys).^2))
  return l
end

ys = network(randn(64,1000),randn(64,1000),W_xc,W_xh,w_h,w_c)

ce_loss = loss(ys, randn(64,1000))
print("ce_loss is: $ce_loss")

