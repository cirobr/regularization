@info "hypertuning start..."

# loss function
lossFunction(yhat, y) = cosine_loss(yhat, y)
@info "loss function OK"

# optimizer parameters
optfn = Flux.Adam
η = 1e-4
λ = 0.0      # default 5e-4
@info "optimizer params OK"



# tuning function
function objective(trial)
      # search variables
      @unpack d1,d2,d3,d4,d5 = trial
      @info "hyper parameters: $d1, $d2, $d3, $d4, $d5"
      drop_enc = (d1, d2, d3, d4, d5)

      train_lossfn(yhat, y) = softloss(yhat, y, lossFunction; T=1.f0, dims=3)
      valid_lossfn(yhat, y) = softloss(yhat, y, lossFunction; T=1.f0, dims=3)

      # model
      Random.seed!(1234)   # to enforce reproducibility
      model = Chain(MobileUNet(3,C; drop_enc=drop_enc, verbose=true),
                        x -> x[2][5],           # (16,16,1280,1)
                        tm.ConvK1(1280, 2048)   # (16,16,2048,1)
      ) |> dev

      # optimizer
      modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), optfn(η)) : optfn(η)
      optimizerState = Flux.setup(modelOptimizer, model)

      # callbacks
      number_since_best = 10
      patience = 5
      es = Flux.early_stopping(()->validloss, number_since_best, init_score = Inf)
      pl = Flux.plateau(()->validloss, patience, init_score = Inf)


      ### training
      # @info "start scenario..."
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
      d1s     = d1s[1:2]
      d2s     = d2s[2:2]
      d3s     = d3s[2:2]
      d4s     = d4s[2:2]
      d5s     = d5s[2:2]
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
@info "project finished"
