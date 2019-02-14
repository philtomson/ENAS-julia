##################################################################
# duplicates test.py functionality in Julia
##################################################################
import Base.length

mutable struct Dictionary
    word2idx # Dict()
    idx2word # array
    counter  # Dict()
    total    # int 

    Dictionary() = new(Dict(),[],Dict(),0)
end

function length(dictionary::Dictionary)
   length(dictionary.idx2word)
end

function add_word(dictionary,word)
   if(!haskey(dictionary.word2idx,word))
       push!(dictionary.idx2word, word)
       dictionary.word2idx[word] = length(dictionary.idx2word)-1
   end
   token_id = dictionary.word2idx[word]
   if(!haskey(dictionary.counter,token_id))
       dictionary.counter[token_id] = 0
   end
   dictionary.counter[token_id] += 1
   dictionary.total += 1
   return token_id
end

function tokenize(corpus,path)
   tokens = 0
   open(path) do file
      for line in eachline(file)
         words = push!(split(line), "<eos>")
         tokens += length(words)
            for word in words
               add_word(corpus.dictionary,word)
            end
      end
   end   
   ids   = zeros(tokens)
   open(path) do file
      token = 1
      for line in eachline(file)
         words = push!(split(line), "<eos>")
         for word in words
            ids[token] = corpus.dictionary.word2idx[word]
            token += 1
         end
      end
   end
   return ids
end

mutable struct Corpus
   dictionary
   train
   valid
   test
   num_tokens
   Corpus(path) = ( self = new(); self.dictionary = Dictionary(); 
                    self.train = tokenize(self,joinpath(path, "train.txt"));
                    self.valid = tokenize(self,joinpath(path, "valid.txt"));
                    self.test  = tokenize(self,joinpath(path, "test.txt"));
                    self.num_tokens = length(self.dictionary);
                    self
                  )
end

#Test:
corpus = Corpus("/home/patomson/devel/ENAS-pytorch/data/ptb")   

