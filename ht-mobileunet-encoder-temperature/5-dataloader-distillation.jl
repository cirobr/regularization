# data
@info "creating distillation data loaders..."

# prepare tensors
ytr_features = teachermodel(Xtr) |> cpu
dims = size(ytr_features)[1:3]

Ntrain = size(dftrain, 1)
# ytrain_features = Array{Float32, 4}(undef, (dims...,Ntrain))
ytrain_features_fpfn = tmp_path * "ytrain_features_" * random_string() * ".bin"
io_ytrain_features   = open(ytrain_features_fpfn, "w+", lock=true)
ytrain_features      = mmap(io_ytrain_features, Array{Float32, 4}, (dims...,Ntrain))

Nvalid = size(dfvalid, 1)
# yvalid_features = Array{Float32, 4}(undef, (dims...,Nvalid))
yvalid_features_fpfn = tmp_path * "yvalid_features_" * random_string() * ".bin"
io_yvalid_features   = open(yvalid_features_fpfn, "w+", lock=true)
yvalid_features      = mmap(io_yvalid_features, Array{Float32, 4}, (dims...,Nvalid))

# fill tensors
dls       = [trainset, validset]
yfeatures = [ytrain_features, yvalid_features]
for (dl, yfeature) in zip(dls, yfeatures)   # no @floop here
    for (i,X) in pb.ProgressBar(enumerate(dl))
        yhat_feature = teachermodel(X) |> cpu
        yfeature[:,:,:,i] = yhat_feature .|> Float32
    end
    Mmap.sync!(yfeature)
end
@info "distillation tensors OK"


# dataloaders
Random.seed!(1234)   # to enforce reproducibility
distillation_trainset = Flux.DataLoader((Xtrain, ytrain_features),
                                        batchsize=minibatchsize,
                                        shuffle=true
                                        ) |> gpu
distillation_validset = Flux.DataLoader((Xvalid, yvalid_features)) |> gpu

# Random.seed!(1234)   # to enforce reproducibility
# trainset = Flux.DataLoader((Xtrain, ytrain),
#                             batchsize=minibatchsize,
#                             shuffle=true
#                             ) |> gpu
# validset = Flux.DataLoader((Xvalid, yvalid)) |> gpu
@info "distillation dataloaders OK"
