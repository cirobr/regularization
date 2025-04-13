"""
Knowledge distilation for semantic segmentation
Author: cirobr@GitHub
Date: 09-Apr-2025

Template for KD semantic segmentation models.
      * Three training steps: front-end, back-end, and tune. Tune learning rate is reduced by 1/10.
      * Depends on private libs: LibFluxML.jl, PreprocessingImages.jl, LibCUDA.jl.
"""

@info "Project start"
cd(@__DIR__)

### arguments
# envpath    = ARGS[1]
# cudadevice = parse(Int64, ARGS[2])
# nepochs    = parse(Int64, ARGS[3])
# debugflag  = parse(Bool,  ARGS[4])

envpath    = "~/envs/d12reg/"
cudadevice = 1
nepochs    = 400
debugflag  = false

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

using TinyMachines; tm=TinyMachines
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
using ProgressBars; const pb=ProgressBars

# private libs
using LibMetalhead
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


# back-end tensors
@info "creating back-end tensors..."
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
@info "back-end tensors OK"


###########################################
# teacher model
###########################################
@info "loading pretrained teacher model..."
modelcpu = ResNet50()
fpfn = expanduser("~/projects/kd-coco-new/2-teachers/models/resnet50/tune/bestmodel.jld2")
LibFluxML.loadModelState!(fpfn, modelcpu)
model = modelcpu |> dev

teachermodel = model[1]   # resnet encoder (20,20,2048,1)

modelcpu = nothing
model = nothing
@info "teacher model OK"
###########################################


# front-end tensors
@info "creating front-end tensors..."
trainset_x = Flux.DataLoader((Xtrain)) |> dev
validset_x = Flux.DataLoader((Xvalid)) |> dev
yhatf = teachermodel(first(trainset_x))
dims = size(yhatf)[1:3]

# ytrain_features = Array{Bool, 4}(undef, (dims...,C,Ntrain))
ytrainfeat_fpfn = tmp_path * "ytrainfeat_" * random_string() * ".bin"
io_ytrainfeat   = open(ytrainfeat_fpfn, "w+", lock=true)
ytrain_features = mmap(io_ytrainfeat, Array{Float32, 4}, (dims...,Ntrain))

# yvalid_features = Array{Bool, 4}(undef, (dims...,C,Nvalid))
yvalidfeat_fpfn = tmp_path * "yvalidfeat_" * random_string() * ".bin"
io_yvalidfeat   = open(yvalidfeat_fpfn, "w+", lock=true)
yvalid_features = mmap(io_yvalidfeat, Array{Float32, 4}, (dims...,Nvalid))


# fill distillation tensors
dls       = [trainset_x, validset_x]
yfeatures = [ytrain_features, yvalid_features]
for (dl, yfeature) in zip(dls, yfeatures)   # no @floop here
    for (i,(X)) in pb.ProgressBar(enumerate(dl))
        global yhat_feature = teachermodel(X) |> cpu#; @assert false
        yfeature[:,:,:,i:i] = yhat_feature .|> Float32
    end
    Mmap.sync!(yfeature)
end
trainset_x = nothing
validset_x = nothing
dls = nothing
yfeatures = nothing
teachermodel = nothing
@info "front-end tensors OK"


# dataloaders
Random.seed!(1234)   # to enforce reproducibility
trainset = Flux.DataLoader((Xtrain, ytrain),
                            batchsize=minibatchsize,
                            shuffle=true) |> dev
validset = Flux.DataLoader((Xvalid, yvalid),
                            batchsize=1,
                            shuffle=false) |> dev

distillation_trainset = Flux.DataLoader((Xtrain, ytrain_features),
                            batchsize=minibatchsize,
                            shuffle=true) |> dev
distillation_validset = Flux.DataLoader((Xvalid, yvalid_features),
                            batchsize=1,
                            shuffle=false) |> dev
@info "dataloaders OK"


###########################################
# student models
###########################################
drop_enc = (0.0,0.2,0.1,0.1,0.2)
drop_dec = (0.1,0.0,0.2,0.0,0.2)

# front-end
Random.seed!(1234)   # to enforce reproducibility
frontend_studentmodelcpu =
      Chain(
            MobileUNet(3,C; drop_enc=drop_enc, drop_dec=drop_dec, verbose=true),
            x -> x[2][5],           # (16,16,1280,1)
            tm.ConvK1(1280, 2048)   # (16,16,2048,1)
)
frontend_studentmodel = frontend_studentmodelcpu |> dev

# backend
Random.seed!(1234)   # to enforce reproducibility
backend_studentmodelcpu = 
      MobileUNet(3,C; drop_enc=drop_enc, drop_dec=drop_dec, verbose=false)

# backend_studentmodelcpu |> gpu after transfer learning only
@info "student models OK"
###########################################


# check front-end matching
Xtr = Xtrain[:,:,:,1:1] |> dev
yf = ytrain_features[:,:,:,1:1]
@assert size(frontend_studentmodel(Xtr)) == size(yf) || error("front-end features do not match")
@info "front-end features matching OK"


# loss functions
backendLossFunction(yhat, y) = IoU_loss(yhat, y)
backendlossfns = [backendLossFunction]

frontend_T = 1.f0   ### Distillation temperature ###

# frontendLossFunction(yhat, y) = cosine_loss(yhat, y)
# frontendLossFunction(yhat, y) = Flux.kldivergence(yhat, y)
frontendLossFunction(yhat, y) = Flux.poisson_loss(yhat, y)
frontendTrainLossFunction(yhat, y) = softloss(yhat, y, frontendLossFunction; T=frontend_T, dims=3)
frontendValidLossFunction(yhat, y) = softloss(yhat, y, frontendLossFunction; T=1.f0, dims=3)
frontendlossfns = [frontendTrainLossFunction, frontendValidLossFunction]
@info "loss functions OK"


# training parameters
number_since_best = 20
patience = 5
metrics = [
      AccScore,
      F1Score,
      IoUScore,
]


###########################################
### front-end distillation
###########################################
@info "start front-end distillation ..."

# optimizer
frontendOptimizerFunction = Flux.Adam
frontend_η = 1e-4
frontend_λ = 0.0
frontendModelOptimizer = frontend_λ > 0 ? Flux.Optimiser(WeightDecay(frontend_λ), frontendOptimizerFunction(frontend_η)) : frontendOptimizerFunction(frontend_η)
frontendOptimizerState = Flux.setup(frontendModelOptimizer, frontend_studentmodel)

# training
LibCUDA.cleangpu()
mpath = modelspath * "frontend/"
tpath = tblogspath * "frontend/"

Random.seed!(1234)   # to enforce reproducibility
Learn!(
      frontendlossfns,
      frontend_studentmodel,
      (distillation_trainset, distillation_validset),
      frontendOptimizerState,
      epochs,
      metrics=[],   # no front-end metrics
      earlystops=(number_since_best, patience),
      modelspath=mpath,
      tblogspath=tpath,
)

# save best frontend model
fpfn = expanduser(mpath) * "model.jld2"
mv(fpfn, expanduser(mpath) * "frontend_studentmodel.jld2", force=true)

# reload best frontend model
fpfn = expanduser(mpath) * "frontend_studentmodel.jld2"
LibFluxML.loadModelState!(fpfn, frontend_studentmodelcpu)
frontend_studentmodel = frontend_studentmodelcpu |> dev

# transfer learning to backend model
model_state = Flux.state(cpu(frontend_studentmodel[1]))
Flux.loadmodel!(backend_studentmodelcpu, model_state)
backend_studentmodel = backend_studentmodelcpu |> dev   # ready for backend 

# clean memory
frontend_studentmodelcpu = nothing
frontend_studentmodel = nothing
@info "front-end distillation OK"


###########################################
### back-end distillation
###########################################
@info "start back-end distillation ..."

# optimizer
backendOptimizerFunction = Flux.Adam
backend_η = frontend_η
backend_λ = frontend_λ
backendModelOptimizer = backend_λ > 0 ? Flux.Optimiser(WeightDecay(backend_λ), backendOptimizerFunction(backend_η)) : backendOptimizerFunction(backend_η)
backendOptimizerState = Flux.setup(backendModelOptimizer, backend_studentmodel)

# front-end freezing (not needed for kd logits)
Flux.freeze!(backendOptimizerState.d)


# training
LibCUDA.cleangpu()
mpath = modelspath * "backend/"
tpath = tblogspath * "backend/"

Random.seed!(1234)   # to enforce reproducibility
Learn!(
      backendlossfns,
      backend_studentmodel,
      (trainset, validset),
      backendOptimizerState,
      epochs,
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=mpath,
      tblogspath=tpath,
)

# save best backend model
fpfn = expanduser(mpath) * "model.jld2"
mv(fpfn, expanduser(mpath) * "backend_studentmodel.jld2", force=true)

# reload best backend model
fpfn = expanduser(mpath) * "backend_studentmodel.jld2"
LibFluxML.loadModelState!(fpfn, backend_studentmodelcpu)
backend_studentmodel = backend_studentmodelcpu |> dev   # ready for tuning
@info "back-end distillation OK"


###########################################
### fine-tuning
###########################################
@info "start fine-tuning ..."

# optimizer
tuning_η = backend_η / 10
Flux.thaw!(backendOptimizerState)               # unfreeze the model
Flux.adjust!(backendOptimizerState, tuning_η)   # adjust the learning rate

# training
LibCUDA.cleangpu()
mpath = modelspath * "tuning/"
tpath = tblogspath * "tuning/"

Random.seed!(1234)   # to enforce reproducibility
Learn!(
      backendlossfns,
      backend_studentmodel,
      (trainset, validset),
      backendOptimizerState,
      epochs,
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=mpath,
      tblogspath=tpath,
)

# save best tuned model
fpfn = expanduser(mpath) * "model.jld2"
mv(fpfn, expanduser(mpath) * "bestmodel.jld2", force=true)
@info "fine-tuning OK"
###########################################


# clean memory
close(io_xtrain)
close(io_ytrain)
close(io_ytrainfeat)

close(io_xvalid)
close(io_yvalid)
close(io_yvalidfeat)

rm(Xtrain_fpfn; force=true)
rm(ytrain_fpfn; force=true)
rm(ytrainfeat_fpfn; force=true)

rm(Xvalid_fpfn; force=true)
rm(yvalid_fpfn; force=true)
rm(yvalidfeat_fpfn; force=true)

LibCUDA.cleangpu()
@info "Congrats! KD process completed."
