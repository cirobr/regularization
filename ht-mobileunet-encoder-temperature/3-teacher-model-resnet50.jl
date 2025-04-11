### UResNet50 model
@info "loading pretrained teacher model..."

modelcpu = ResNet50()
fpfn = expanduser("~/projects/kd-coco-new/2-teachers/models/resnet50/tune/bestmodel.jld2")
LibFluxML.loadModelState!(fpfn, modelcpu)
model = modelcpu |> gpu

teachermodel = model[1]   # resnet encoder (20,20,2048,1)

modelcpu = nothing
model = nothing
@info "teacher model OK"
