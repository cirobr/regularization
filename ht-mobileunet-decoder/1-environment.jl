### libs
using Pkg
envpath = expanduser("~/envs/d11reg/")
Pkg.activate(envpath)

using CUDA
CUDA.device!(cudadevice)
CUDA.versioninfo()

using Flux
import Flux: relu, leakyrelu
dev = CUDA.has_cuda_gpu() ? gpu : cpu
using Images
using DataFrames
using CSV
using JLD2
using FLoops
using Random
using Statistics: mean, minimum, maximum, norm
using StatsBase: sample
using MLUtils: splitobs, kfolds, obsview, ObsView, randobs
using Mmap
using HyperTuning
using Dates
using ProgressBars; const pb=ProgressBars

# private libs
using TinyMachines; const tm=TinyMachines
using LibMetalhead
using PreprocessingImages; const p=PreprocessingImages
using CocoTools; const c=CocoTools
using LibFluxML
import LibFluxML: IoU_loss, ce1_loss, ce3_loss, cosine_loss, softloss,
                  AccScore, F1Score, IoUScore
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
workpath = pwd() * "/"
workpath = replace(workpath, homedir() => "~")
datasetpath = "~/projects/kd-coco-dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

# modelspath  = workpath * "models/" * outputfolder
# mkpath(expanduser(modelspath))

# tblogspath  = workpath * "tblogs/" * outputfolder
# rm(tblogspath; force=true, recursive=true); sleep(1)   # sleep to ensure removal
# mkpath(expanduser(tblogspath))

tmp_path = "/scratch/cirobr/tmp/"
# tmp_path = "/tmp/"
# mkpath(tmp_path)     # it should already exist
@info "folders OK"
