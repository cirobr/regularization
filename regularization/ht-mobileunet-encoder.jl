"""
LibFluxML.Learn!()
Author: cirobr@GitHub
Date: 20-Oct-2024
"""

@info "Project start"
cd(@__DIR__)

### arguments
cudadevice = parse(Int64, ARGS[1])
nepochs    = parse(Int64, ARGS[2])
debugflag  = parse(Bool,  ARGS[3])

# cudadevice = 1
# nepochs    = 100
# debugflag  = true

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"

minibatchsize = 4
epochs        = nepochs


### libs
using Pkg
envpath = expanduser("~/envs/deeptraining/")
Pkg.activate(envpath)

using CUDA
CUDA.device!(cudadevice)
CUDA.versioninfo()

using HyperTuning
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
# using Statistics: mean, minimum, maximum, norm
# using StatsBase: sample
# using MLUtils: splitobs, kfolds, randobs
using Mmap
using Dates


# private libs
using PascalVocTools;      const pv=PascalVocTools
using PreprocessingImages; const p=PreprocessingImages
using LibFluxML
import LibFluxML: IoU_loss, ce1_loss, ce3_loss, cosine_loss, AccScore, F1Score, IoUScore
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
datasetpath = "../dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

# modelspath  = "./models/" * outputfolder
# mkpath(expanduser(modelspath))

# tblogspath  = "./tblogs/" * outputfolder
# rm(tblogspath; force=true, recursive=true)
# mkpath(expanduser(tblogspath))

tmp_path = "/scratch/cirobr/tmp/"
# tmp_path = "/tmp/"
# mkpath(tmp_path)     # it should already exist
@info "folders OK"


### datasets
@info "creating datasets..."
classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [pv.voc_classname2classnumber[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-coi-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
dftrain = dftrain[dftrain.segmented .== 1,:]

fpfn = expanduser(datasetpath) * "dfvalid-coi-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
dfvalid = dfvalid[dfvalid.segmented .== 1,:]


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
# @assert dbsizeGB < 2.0 || error("dataset is too large")
# @info "dataset size OK"


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
            mask = pv.voc_rgb2classes(mask)
            mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            mask = permutedims(mask, (2,3,1))
            yout[:,:,:,i] = mask
      end
      Mmap.sync!(Xout)
      Mmap.sync!(yout)
end
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


LibCUDA.cleangpu()


# loss function
lossFunction(yhat, y) = dice_coeff_loss(yhat, y) + focal_loss(yhat, y; dims=3)
@info "loss function OK"


Xtr = Xtrain[:,:,:,1:1] |> dev
ytr = ytrain[:,:,:,1:1] |> dev


# results DataFrame
results = DataFrame(
      dropd1 = Float32[],
      dropd2 = Float32[],
      dropd3 = Float32[],
      dropd4 = Float32[],
      dropd5 = Float32[],
      validloss = Float32[],
)

# tuning function
function objective(trial)
      @unpack dropd1, dropd2, dropd3, dropd4, dropd5 = trial
      @info "objective: dropd1=$dropd1, dropd2=$dropd2, dropd3=$dropd3, dropd4=$dropd4, dropd5=$dropd5"

      # model
      Random.seed!(1234)   # to enforce reproducibility
      model = MobileUnet(3,C; verbose=false,
                        dropd1=dropd1, dropd2=dropd2, dropd3=dropd3, dropd4=dropd4, dropd5=dropd5) |> dev
      # check for matching between model and data
      @assert size(model(Xtr)) == size(ytr) || error("model/data features do not match")

      # optimizer
      η = 1e-4
      λ = 0.0
      modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), Adam(η)) : Adam(η)
      optimizerState = Flux.setup(modelOptimizer, model)

      # callbacks
      number_since_best = 5
      patience = 5
      es = Flux.early_stopping(()->validloss, number_since_best; init_score = Inf)
      pl = Flux.plateau(()->validloss, patience; init_score = Inf)

      ### training
      @info "start training ..."
      metrics = []
      final_loss = Inf

      Random.seed!(1234)   # to enforce reproducibility
      for epoch in 1:epochs
            _ = trainEpoch!(lossFunction, model, trainset, optimizerState)
            global validloss, _ = evaluateEpoch(lossFunction, model, validset, metrics)
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
      # push!(results, [string(optfn), η, λ, final_loss])
      push!(results, [dropd1, dropd2, dropd3, dropd4, dropd5, final_loss])
      outputfile = script_name[1:end-3] * ".csv"
      CSV.write(outputfile, results)
      
      LibCUDA.cleangpu()
      return final_loss |> Float64
end


LibCUDA.cleangpu()


### hyperparameters tuning
# lossfns = [ce3_loss, cosine_loss, Flux.kldivergence, Flux.poisson_loss]
# optfns = [Flux.Adam, Flux.RMSProp]
# ηs = [1.e-4, 5.e-4]
# λs = [0.0, 5.e-7, 5.e-5]
dropd1s = [0.0, 0.1, 0.2]
dropd2s = [0.0, 0.1, 0.2]
dropd3s = [0.0, 0.1, 0.2]
dropd4s = [0.0, 0.1, 0.2]
dropd5s = [0.0, 0.1, 0.2]

if debugflag
      # lossfns = lossfns[1:2]
      # optfns  = optfns[1:2]
      # ηs      = ηs[1:2]
      # λs      = λs[1:2]
      dropd1s = dropd1s[1:2]
      dropd2s = dropd2s[1:2]
      dropd3s = dropd3s[1:1]
      dropd4s = dropd4s[1:1]
      dropd5s = dropd5s[1:1]
end


scenario = Scenario(
      # lossfn  = lossfns,
      # optfn   = optfns,
      # η       = ηs,
      # λ       = λs,
      dropd1  = dropd1s,
      dropd2  = dropd2s,
      dropd3  = dropd3s,
      dropd4  = dropd4s,
      dropd5  = dropd5s,
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
