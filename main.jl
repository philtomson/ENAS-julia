include("text.jl")
using Random
using ArgParse
using Flux
args = let random_seed=42, 
           data_path="../ENAS-pytorch/data/ptb"; () -> Any[random_seed, data_path]; end
function main(args)
   Random.seed!(args.random_seed) 
   dataset = Corpus(args.data_path)
end
main(args)
