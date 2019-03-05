using LinearAlgebra

function clip_grad_norm(params, max_norm, norm_type=2)
   total_norm = 0
   for p in params
      param_norm = LinearAlgebra.norm(p, norm_type)
      total_norm += param_norm ^  norm_type
   end
   total_norm = total_norm ^ (1.0 / norm_type)
   clip_coef = max_norm / (total_norm + 1e-6)
   if clip_coef < 1
      for i in 1:size(params,1)
         params[i] = params[i] .* clip_coef
      end
   end
   return total_norm
end
