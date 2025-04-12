@info "Project start"
cd(@__DIR__)
foldername = basename(pwd())

### arguments
cudadevice = parse(Int64, ARGS[1])
nepochs    = parse(Int64, ARGS[2])
debugflag  = parse(Bool,  ARGS[3])

# cudadevice = 1
# nepochs    = 200
# debugflag  = true

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"



include("1-environment.jl")



include("2-dataloader.jl")



include("3-teacher-model-resnet50.jl")



include("4-student-model-mobileunet.jl")

# check frontend student matching
Xtr = first(trainset) |> gpu

size_teachermodel = size(teachermodel(Xtr))
@info "teacher model feature size = $size_teachermodel"
size_frontend_studentmodel = size(frontend_studentmodel(Xtr))
@info "frontend student model feature size = $size_frontend_studentmodel"
@assert size_teachermodel == size_frontend_studentmodel || error("frontend model mismatch")
@info "frontend model matching OK"

# # check backend student matching
# size_backend_studentmodel = size(backend_studentmodelcpu(cpu(Xtr)))
# @info "backend student model feature size = $size_backend_studentmodel"
# size_ytr = size(ytr)
# @info "mask size = $size_ytr"
# @assert size_backend_studentmodel == size_ytr || error("backend model mismatch")
# @info "backend model matching OK"



include("5-dataloader-distillation.jl")
teachermodelcpu = nothing
teachermodel    = nothing



### hypertuning
LibCUDA.cleangpu()
number_since_best = 10
patience = 5
include("6-hypertuning.jl")



### clean memory
close(io_xtrain)
# close(io_ytrain)
close(io_xvalid)
# close(io_yvalid)
close(io_ytrain_features)
close(io_yvalid_features)
rm(Xtrain_fpfn; force=true)
# rm(ytrain_fpfn; force=true)
rm(Xvalid_fpfn; force=true)
# rm(yvalid_fpfn; force=true)
rm(ytrain_features_fpfn; force=true)
rm(yvalid_features_fpfn; force=true)

LibCUDA.cleangpu()
@info "congrats, you've just finished the hypertuning process!"
