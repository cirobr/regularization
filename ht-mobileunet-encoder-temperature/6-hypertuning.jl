@info "hypertuning start..."

# loss function
lossFunction(yhat, y) = cosine_loss(yhat, y)   # fixed for this study
@info "loss function OK"

# optimizer parameters
optfn = Flux.Adam
η = 1e-4
λ = 0.0      # default 5e-4
@info "optimizer params OK"


# results DataFrame
results = DataFrame(
      T         = Float32[],
      validloss = Float32[],
)
# results = CSV.read("ht.csv", DataFrame)



# tuning function
function objective(trial)
      # search variables
      @unpack T = trial
      @info "hyper parameters: T=$T"

      # # Check if T is already in results
      # if any(row -> row.T == T, eachrow(results))
      #       @info "Combination already evaluated: T=$T"
      #       return Inf
      # end
      
      # cosine_loss fixed for this study only
      train_lossfn(yhat, y) = softloss(yhat, y, lossFunction; T=T, dims=3)
      valid_lossfn(yhat, y) = softloss(yhat, y, lossFunction; T=1.f0, dims=3)

      # model
      model = instantiate_model() |> gpu   # always with same initial conditions

      # optimizer
      modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), optfn(η)) : optfn(η)
      optimizerState = Flux.setup(modelOptimizer, model)

      # callbacks
      es = Flux.early_stopping(()->validloss, number_since_best;
            init_score = Inf, min_dist = 1.f-4)
      pl = Flux.plateau(()->validloss, patience;
            init_score = Inf, min_dist = 1.f-4)


      ### training
      metrics = []
      final_loss = Inf

      Random.seed!(1234)   # to enforce reproducibility
      for epoch in 1:epochs
            _ = trainEpoch!(train_lossfn, model, distillation_trainset, optimizerState)
            global validloss, _ = evaluateEpoch(valid_lossfn, model, distillation_validset, metrics)
            @info "Thread: $(Threads.threadid()), Epoch: $epoch, Validation loss: $validloss"
            if validloss < final_loss   final_loss = validloss   end

            # callbacks
            if isnan(validloss)
                  @info "nan loss detected"
                  break
            elseif es()
                  @info "early stopping"
                  break
            elseif pl()
                  @info "plateau"
                  break
            elseif epoch == epochs
                  @info "max epochs"
                  # no break here, leave the loop
            end
      end

      # save partial results
      push!(results, [T, final_loss])
      outputfile = script_name[1:end-3] * ".csv"
      CSV.write(outputfile, results)
      
      LibCUDA.cleangpu()
      return final_loss |> Float64
end


### hyperparameters tuning
ts = 1 : 35 .|> Float32

if debugflag
      ts      = ts[1:4]
end


scenario = Scenario(
      T       = ts,
      sampler = GridSampler(),
)

HyperTuning.optimize(objective, scenario)

# sort and save results
results = sort(results, :validloss)
outputfile = script_name[1:end-3] * ".csv"
CSV.write(outputfile, results)

# show results
display(scenario)
display(history(scenario))

outputfile = script_name[1:end-3] * ".txt"
h = history(scenario)
open(outputfile, "w") do io
      println(io, "scenario:")
      println(io, scenario)
      println(io, "history:")
      println(io, history(scenario))
end
@info "hypertuning OK"
