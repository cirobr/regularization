### datasets
using PascalVocTools; const pv=PascalVocTools


@info "creating encoder datasets..."

classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [pv.voc_classname2classnumber[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
# dftrain = dftrain[dftrain.segmented .== 1,:]

fpfn = expanduser(datasetpath) * "dfvalid-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
# dfvalid = dfvalid[dfvalid.segmented .== 1,:]


########### debug ############
if debugflag
      dftrain = first(dftrain, 3)
      dfvalid = first(dfvalid, 2)
      minibatchsize = 1
      epochs  = 2
else
      minibatchsize = 4
      epochs  = nepochs
end
##############################


train_images = []
train_masks  = []

valid_images = []
valid_masks  = []

dfs = [dftrain, dfvalid]
Xs  = [train_images, valid_images]
ys  = [train_masks, valid_masks]

@floop for (df, X, y) in zip(dfs, Xs, ys)
      N = size(df, 1)
      for i in 1:N   # no @floop here
            local fpfn = expanduser(df.X[i])
            img = Images.load(fpfn)
            img = p.color2Float32(img)
            push!(X, img)

            # local fpfn = expanduser(df.y[i])
            # local mask = Images.load(fpfn)
            # local mask = pv.voc_rgb2classes(mask)
            # local mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            # local mask = permutedims(mask, (2,3,1))
            # push!(y, mask)
      end
end

Xtrain_encoder = cat(train_images...; dims=4); train_images=nothing
ytrain_encoder = Flux.onehotbatch(dftrain.label, [0,1], 0)
ytrain_encoder = Flux.label_smoothing(ytrain, 0.05);  train_masks=nothing

Xvalid_encoder = cat(valid_images...; dims=4); valid_images=nothing
yvalid_encoder = Flux.onehotbatch(dfvalid.label, [0,1], 0)
yvalid_encoder = Flux.label_smoothing(yvalid, 0.05);  valid_masks=nothing
@info "encoder tensors OK"

# dataloaders
trainset_encoder = Flux.DataLoader((Xtrain_encoder, ytrain_encoder)) |> gpu   # minibatch==1 to create features
validset_encoder = Flux.DataLoader((Xvalid_encoder, yvalid_encoder)) |> gpu
@info "encoder dataloaders OK"
