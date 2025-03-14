### datasets
@info "creating datasets..."

classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [pv.voc_classname2classnumber[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
# dftrain = dftrain[dftrain.segmented,:]

fpfn = expanduser(datasetpath) * "dfvalid-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
# dfvalid = dfvalid[dfvalid.segmented,:]


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

            local fpfn = expanduser(df.y[i])
            mask = Images.load(fpfn) .|> RGB
            mask = pv.voc_rgb2classes(mask)
            mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            mask = permutedims(mask, (2,3,1))
            push!(y, mask)
      end
end

Xtrain = cat(train_images...; dims=4); train_images=nothing
ytrain = cat(train_masks...; dims=4);  train_masks=nothing

Xvalid = cat(valid_images...; dims=4); valid_images=nothing
yvalid = cat(valid_masks...; dims=4);  valid_masks=nothing
@info "tensors OK"

# dataloaders
trainset = Flux.DataLoader((Xtrain, ytrain)) |> gpu   # minibatch==1 to create features
validset = Flux.DataLoader((Xvalid, yvalid)) |> gpu
@info "dataloaders OK"
