@info "hypertuning start..."

# loss function
lossFunction(yhat, y) = IoU_loss(yhat, y)
@info "loss function OK"

# optimizer parameters
optfn = Flux.Adam
η = 1e-4
λ = 0.0      # default 5e-4
@info "optimizer params OK"


# results DataFrame
# results = DataFrame(
#       d1        = Float32[],
#       d2        = Float32[],
#       d3        = Float32[],
#       d4        = Float32[],
#       d5        = Float32[],
#       validloss = Float32[],
# )
results = CSV.read("ht.csv", DataFrame)
@info "results DataFrame OK"

# tuning function
function objective(trial)
      # search variables
      @unpack d1,d2,d3,d4,d5 = trial
      @info "hyper parameters: $d1, $d2, $d3, $d4, $d5"
      drop_dec = (d1, d2, d3, d4, d5)

      # Check if drop_dec is already in results
      if any(row -> row.d1 == d1 && row.d2 == d2 && row.d3 == d3 && row.d4 == d4 && row.d5 == d5, eachrow(results))
            @info "Combination already evaluated: $d1, $d2, $d3, $d4, $d5"
            return Inf
      end

      train_lossfn = lossFunction
      valid_lossfn = lossFunction

      # model
      Random.seed!(1234)   # to enforce reproducibility
      drop_enc = (0.0, 0.2, 0.1, 0.1, 0.2)   # from previous experiment
      model = MobileUNet(3,C; drop_enc=drop_enc, drop_dec=drop_dec, verbose=false) |> dev

      # optimizer
      modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), optfn(η)) : optfn(η)
      optimizerState = Flux.setup(modelOptimizer, model)

      # callbacks
      es = Flux.early_stopping(()->validloss, number_since_best; init_score = Inf)
      pl = Flux.plateau(()->validloss, patience; init_score = Inf)


      ### training
      metrics = []
      final_loss = Inf

      Random.seed!(1234)   # to enforce reproducibility
      for epoch in 1:epochs
            _ = trainEpoch!(train_lossfn, model, trainset, optimizerState)
            global validloss, _ = evaluateEpoch(valid_lossfn, model, validset, metrics)
            @info "Epoch: $epoch, Validation loss: $validloss"
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
      push!(results, [d1,d2,d3,d4,d5,final_loss])
      outputfile = script_name[1:end-3] * ".csv"
      CSV.write(outputfile, results)
      
      LibCUDA.cleangpu()
      return final_loss |> Float64
end


### hyperparameters tuning
d1s = [0.0, 0.1, 0.2]
d2s = [0.0, 0.1, 0.2]
d3s = [0.0, 0.1, 0.2]
d4s = [0.0, 0.1, 0.2]
d5s = [0.0, 0.1, 0.2]

if debugflag
      d1s = d1s[1:2]
      d2s = d2s[1:1]
      d3s = d3s[1:1]
      d4s = d4s[1:1]
      d5s = d5s[1:1]
end

scenario = Scenario(
      d1 = d1s,
      d2 = d2s,
      d3 = d3s,
      d4 = d4s,
      d5 = d5s,
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
