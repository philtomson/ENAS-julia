include("text.jl")
using Random
using ArgParse
using Flux

s = ArgParseSettings()
@add_arg_table s begin
   "--epochs"
      help = "number of epochs to run"
      arg_type = Int
      default = 1
   "--num_blocks"
      help = "number of blocks in child RNNs"
      arg_type = Int
      default = 6
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
