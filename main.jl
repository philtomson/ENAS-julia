include "text.jl"
using Flux

function main(args)
   Random.seed!(args.random_seed) 
   dataset = Corpus(args.data_path)
end
