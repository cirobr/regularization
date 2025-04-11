"""
LibFluxML.Learn!()
Author: cirobr@GitHub
Date: 07-03-2025

Template for training semantic segmentation models.
      * Two training steps: train and tune. Tune learning rate is reduced by 1/10.
      * Depends on private libs: LibFluxML.jl, PreprocessingImages.jl, LibCUDA.jl.
"""

@info "Project start"
cd(@__DIR__)

### arguments
# envpath    = ARGS[1]
# cudadevice = parse(Int64, ARGS[2])
# nepochs    = parse(Int64, ARGS[3])
# debugflag  = parse(Bool,  ARGS[4])

envpath    = "~/envs/d11reg/"
cudadevice = 1
nepochs    = 400
debugflag  = true

@assert isdefined(Main, :envpath) "envpath not defined"
@assert isdefined(Main, :cudadevice) "cudadevice not defined"
@assert isdefined(Main, :nepochs) "nepochs not defined"
@assert isdefined(Main, :debugflag) "debugflag not defined"

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "envpath: $envpath"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"

minibatchsize = 4
epochs        = nepochs



### libs
using Pkg
Pkg.activate(expanduser(envpath))

using CUDA
CUDA.device!(cudadevice)
CUDA.versioninfo()

using TinyMachines
using Flux
import Flux: relu, leakyrelu, dice_coeff_loss, focal_loss
dev = CUDA.has_cuda_gpu() ? gpu : cpu
using Images
using DataFrames
using CSV
using JLD2
using FLoops
using Random
using Mmap
using Dates

# private libs
using CocoTools; const c=CocoTools
using PreprocessingImages; const p=PreprocessingImages
using LibFluxML
import LibFluxML: IoU_loss, ce1_loss, ce3_loss, cosine_loss, softloss, AccScore, F1Score, IoUScore
using LibCUDA

LibCUDA.cleangpu()
@info "environment OK"


### generate a random string based on the current date and time
random_string() = string(Dates.format(now(), "yyyymmdd_HHMMSSsss"))


### constants
const KB = 1024
const MB = KB * KB
const GB = KB * MB


### folders
outputfolder = script_name[1:end-3] * "/"

# pwd(), homedir()
datasetpath = "~/projects/kd-coco-dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

modelspath  = "./models/" * outputfolder
mkpath(expanduser(modelspath))

tblogspath  = "./tblogs/" * outputfolder
rm(tblogspath; force=true, recursive=true)
mkpath(expanduser(tblogspath))

tmp_path = "/scratch/cirobr/tmp/"
# tmp_path = "/tmp/"
# mkpath(tmp_path)     # it should already exist
@info "folders OK"


### datasets
@info "creating datasets..."
classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [c.coco_classnames[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-cow-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
dftrain = dftrain[dftrain.cow,:]

fpfn = expanduser(datasetpath) * "dfvalid-cow-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
dfvalid = dfvalid[dfvalid.cow,:]


########### debug ############
if debugflag
      dftrain = first(dftrain, 3)
      dfvalid = first(dfvalid, 2)
      minibatchsize = 1
      epochs  = 2
end
##############################


# check memory requirements
Xtr = Images.load(expanduser(dftrain.X[1]))
ytr = Images.load(expanduser(dftrain.y[1]))
Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)

dpsize   = sizeof(Xtr) + sizeof(ytr)
dpsizeGB = dpsize / GB
dbsizeGB = dpsizeGB * (Ntrain + Nvalid)

@info "dataset points = $(Ntrain + Nvalid)"
@info "dataset size = $(dbsizeGB) GB"

# create tensors
@info "creating tensors..."
dims = size(Xtr)
Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)

# Xtrain = Array{Float32, 4}(undef, (dims...,3,Ntrain))
Xtrain_fpfn = tmp_path * "xtrain_" * random_string() * ".bin"
io_xtrain   = open(Xtrain_fpfn, "w+", lock=true)
Xtrain      = mmap(io_xtrain, Array{Float32, 4}, (dims...,3,Ntrain))

# ytrain = Array{Bool, 4}(undef, (dims...,C,Ntrain))
ytrain_fpfn = tmp_path * "ytrain_" * random_string() * ".bin"
io_ytrain   = open(ytrain_fpfn, "w+", lock=true)
ytrain      = mmap(io_ytrain, Array{Bool, 4}, (dims...,C,Ntrain))

# Xvalid = Array{Float32, 4}(undef, (dims...,3,Nvalid))
Xvalid_fpfn = tmp_path * "xvalid_" * random_string() * ".bin"
io_xvalid   = open(Xvalid_fpfn, "w+", lock=true)
Xvalid      = mmap(io_xvalid, Array{Float32, 4}, (dims...,3,Nvalid))

# yvalid = Array{Bool, 4}(undef, (dims...,C,Nvalid))
yvalid_fpfn = tmp_path * "yvalid_" * random_string() * ".bin"
io_yvalid   = open(yvalid_fpfn, "w+", lock=true)
yvalid      = mmap(io_yvalid, Array{Bool, 4}, (dims...,C,Nvalid))

dfs = [dftrain, dfvalid]
Xouts = [Xtrain, Xvalid]
youts = [ytrain, yvalid]

for (df, Xout, yout) in zip(dfs, Xouts, youts)   # no @floop here
      N = size(df, 1)
      @floop for i in 1:N
            local fpfn = expanduser(df.X[i])
            img = Images.load(fpfn)
            img = p.color2Float32(img)
            Xout[:,:,:,i] = img

            local fpfn = expanduser(df.y[i])
            mask = Images.load(fpfn)
            mask = p.gray2Int32(mask)
            mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            mask = permutedims(mask, (2,3,1)) .|> Bool
            yout[:,:,:,i] = mask
      end
      Mmap.sync!(Xout)
      Mmap.sync!(yout)
end
dfs = nothing
Xouts = nothing
youts = nothing
@info "tensors OK"

# dataloaders
Random.seed!(1234)   # to enforce reproducibility
trainset = Flux.DataLoader((Xtrain, ytrain),
                            batchsize=minibatchsize,
                            shuffle=true) |> dev
validset = Flux.DataLoader((Xvalid, yvalid),
                            batchsize=1,
                            shuffle=false) |> dev
@info "dataloader OK"


### model
Random.seed!(1234)   # to enforce reproducibility
drop_enc = (0.0,0.2,0.1,0.1,0.2)
drop_dec = (0.1,0.0,0.2,0.0,0.2)
modelcpu = MobileUNet(3,C; drop_enc=drop_enc, drop_dec=drop_dec, verbose=false) |> dev
# fpfn = expanduser("")
# LibFluxML.loadModelState!(fpfn, modelcpu)
model = modelcpu |> dev
@info "model OK"


# check for matching between model and data
Xtr = Xtrain[:,:,:,1:1] |> dev
ytr = ytrain[:,:,:,1:1] |> dev
@assert size(model(Xtr)) == size(ytr) || error("model/data features do not match")
@info "model/data matching OK"


# loss functions
lossFunction(yhat, y) = IoU_loss(yhat, y)
# lossFunction(yhat, y) = dice_coeff_loss(yhat, y) + focal_loss(yhat, y; dims=3)
lossfns = [lossFunction]
@info "loss function OK"


# optimizer
optimizerFunction = Flux.Adam
η = 1e-4
λ = 0.0
modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), optimizerFunction(η)) : optimizerFunction(η)


optimizerState = Flux.setup(modelOptimizer, model)
# Flux.freeze!(optimizerState.encoder)
@info "optimizer OK"


### training
@info "start training ..."

number_since_best = 20
patience = 5
metrics = [
      AccScore,
      F1Score,
      IoUScore,
]

LibCUDA.cleangpu()
Random.seed!(1234)   # to enforce reproducibility
Learn!(
      lossfns,
      model,
      (trainset, validset),
      optimizerState,
      epochs,
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=modelspath * "train/",
      tblogspath=tblogspath * "train/"
)
fpfn = expanduser(modelspath) * "train/model.jld2"
mv(fpfn, expanduser(modelspath) * "train/bestmodel.jld2", force=true)
@info "training OK"


### tuning
@info "start tuning ..."
fpfn = expanduser(modelspath) * "train/bestmodel.jld2"
LibFluxML.loadModelState!(fpfn, modelcpu)
model = modelcpu |> dev

Flux.thaw!(optimizerState)
Flux.adjust!(optimizerState, η/10)
@info "optimizer adjusted"

LibCUDA.cleangpu()
Random.seed!(1234)   # to enforce reproducibility
Learn!(
      lossfns,
      model,
      (trainset, validset),
      optimizerState,
      epochs,
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=modelspath * "tune/",
      tblogspath=tblogspath * "tune/"
)

fpfn = expanduser(modelspath) * "tune/model.jld2"
mv(fpfn, expanduser(modelspath) * "tune/bestmodel.jld2", force=true)
@info "tuning OK"


### clean memory
close(io_xtrain)
close(io_ytrain)
close(io_xvalid)
close(io_yvalid)
rm(Xtrain_fpfn; force=true)
rm(ytrain_fpfn; force=true)
rm(Xvalid_fpfn; force=true)
rm(yvalid_fpfn; force=true)

LibCUDA.cleangpu()
@info "project finished"
