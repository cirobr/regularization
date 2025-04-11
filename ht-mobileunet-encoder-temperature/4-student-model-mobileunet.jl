# frontend student model
function instantiate_model()
      Random.seed!(1234)   # to enforce reproducibility
      drop_enc = (0.0,0.2,0.1,0.1,0.2)
      drop_dec = (0.1,0.0,0.2,0.0,0.2)
      return Chain(
            MobileUNet(3,C; drop_enc=drop_enc, drop_dec=drop_dec, verbose=false),
            x -> x[2][5],           # (20,20,1280,1)
            tm.ConvK1(1280, 2048)   # (20,20,2048,1)
      )
end
