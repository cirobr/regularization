### UResNet50 model
@info "loading pretrained teacher model..."

modelcpu = ResNet50()
fpfn = expanduser("~/projects/knowledge-distillation/train-resnets/models/resnet50-ht/tune/bestmodel.jld2")
LibFluxML.loadModelState!(fpfn, modelcpu)
model = modelcpu |> gpu

# teachermodel = model[1]                  # (512,512,2,1)
# teachermodel = model[1].layers[1][1:2]   # (512,512,67,1)
teachermodel = model[1]   # resnet encoder (16,16,2048,1)

modelcpu = nothing
model = nothing
@info "teacher model OK"
