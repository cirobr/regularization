@info "Project start"
cd(@__DIR__)
foldername = basename(pwd())

### arguments
# cudadevice = parse(Int64, ARGS[1])
# nepochs    = parse(Int64, ARGS[2])
# debugflag  = parse(Bool,  ARGS[3])

cudadevice = 0
nepochs    = 200
debugflag  = true

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"



include("1-environment.jl")



include("2-dataloader.jl")
Xtr  = Xtrain[:,:,:,1:1] |> gpu
ytr  = ytrain[:,:,:,1:1] |> gpu
include("2-dataloader-encoder.jl")
Xtr_encoder = Xtrain_encoder[:,:,:,1:1] |> gpu
ytr_encoder = ytrain_encoder[:,:,:,1:1] |> gpu


include("3-teacher-model-resnet50.jl")


# include("5-student-model-mobileunet.jl")

# check frontend student matching
size_teachermodel = size(teachermodel(Xtr))
@info "teacher model feature size = $size_teachermodel"
size_frontend_studentmodel = size(frontend_studentmodel(Xtr))
@info "frontend student model feature size = $size_frontend_studentmodel"
@assert size_teachermodel == size_frontend_studentmodel || error("frontend model mismatch")
@info "frontend model matching OK"

# check backend student matching
size_backend_studentmodel = size(backend_studentmodelcpu(cpu(Xtr)))
@info "backend student model feature size = $size_backend_studentmodel"
size_ytr = size(ytr)
@info "mask size = $size_ytr"
@assert size_backend_studentmodel == size_ytr || error("backend model mismatch")
@info "backend model matching OK"


include("6-dataloader-distillation.jl")
teachermodelcpu = nothing
teachermodel    = nothing


number_since_best = 10
patience = 5

### hypertuning
LibCUDA.cleangpu()
# include("7-hypertuning.jl")


LibCUDA.cleangpu()
@info "congrats, you've just finished the hypertuning process!"
