### libs
using Pkg
envpath = expanduser("~/envs/dev/")
Pkg.activate(envpath)

using CUDA
CUDA.device!(cudadevice)
CUDA.versioninfo()

using Flux
import Flux: relu, leakyrelu, softmax, kaiming_normal
using Images
using DataFrames
using CSV
using JLD2
using FLoops
using Random
using Statistics: mean, minimum, maximum, norm
using StatsBase: sample
using MLUtils: splitobs, kfolds, obsview, ObsView
using HyperTuning

# private libs
using TinyMachines; tm=TinyMachines
using LibMetalhead
using PreprocessingImages; const p=PreprocessingImages
using PascalVocTools; const pv=PascalVocTools
using LibFluxML
import LibFluxML: IoU_loss, ce1_loss, ce3_loss, cosine_loss, softloss
import LibFluxML: trainEpoch!, evaluateEpoch
using LibCUDA

LibCUDA.cleangpu()
@info "environment OK"


### constants
const KB = 1024
const MB = KB * KB
const GB = KB * MB


### folders
outputfolder = script_name[1:end-3] * "/"

# pwd(), homedir()
workpath = pwd() * "/"
workpath = replace(workpath, homedir() => "~")
datasetpath = "../dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

# modelspath  = workpath * "models/" * outputfolder
# mkpath(expanduser(modelspath))

# tblogspath  = workpath * "tblogs/" * outputfolder
# rm(tblogspath; force=true, recursive=true); sleep(1)   # sleep to ensure removal
# mkpath(expanduser(tblogspath))
@info "folders OK"
