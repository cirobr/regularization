# using Pkg
# envpath = expanduser("~/envs/d8/")
# Pkg.activate(envpath)

using TensorBoardLogger; tb=TensorBoardLogger
using ValueHistories
using DataFrames


# learndatalogs_cb
function get_learndatalogs(path)
    tblog = tb.TBReader(path)
    hist = []
    
    tb.map_summaries(tblog) do tag, iter, val
        push!(hist, [tag, iter, val])
    end
    
    # create grouped data frame
    df = DataFrame(mapreduce(permutedims, vcat, hist), :auto)
    rename!(df, Dict(:x1 => :tag, :x2 => :iter, :x3 => :val))
    gdf = groupby(df, :tag)
    gdf = filter(g -> occursin("Epoch", g.tag[1]), gdf)   # filter groups with "Epoch" in tag
    # [first(g.tag) for g in gdf]   # show tag from each group

    # extract metrics
    ismetrics = length(gdf) > 2 ? true : false
    train_loss = gdf[1].val .|> Float32
    valid_loss = gdf[2].val .|> Float32
    metrics = []
    if ismetrics
        metrics = hcat([g.val for g in gdf[3:end]]...) .|> Float32
    end

    return train_loss, valid_loss, metrics
end

# path = "./2-teachers/tblogs/uresnet50/train/_1"
# t, v, m = get_learndatalogs(path)