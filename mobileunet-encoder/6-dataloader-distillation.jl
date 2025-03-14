# data
@info "creating distillation data loaders..."

train_features     = []
valid_features     = []
ys  = [train_features, valid_features]
dls = [trainset, validset]

for (dl, y) in zip(dls, ys)
    for (X, _) in dl
        yhat = teachermodel(X) |> cpu
        push!(y, yhat)
    end
end
ytrain_features = cat(train_features...; dims=4);  train_features=nothing
yvalid_features = cat(valid_features...; dims=4);  valid_features=nothing
@info "distillation tensors OK"


# dataloaders
Random.seed!(1234)   # to enforce reproducibility
distillation_trainset = Flux.DataLoader((Xtrain, ytrain_features),
                                        batchsize=minibatchsize,
                                        shuffle=true
                                        ) |> gpu
distillation_validset = Flux.DataLoader((Xvalid, yvalid_features)) |> gpu

Random.seed!(1234)   # to enforce reproducibility
trainset = Flux.DataLoader((Xtrain, ytrain),
                            batchsize=minibatchsize,
                            shuffle=true
                            ) |> gpu
validset = Flux.DataLoader((Xvalid, yvalid)) |> gpu
@info "distillation dataloaders OK"
