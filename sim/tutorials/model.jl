using Flux

xs = Int.(zeros(10, 10, 10))
println(xs |> size)

layer = Conv((3,3),1 => 4, relu)

xs2 = layer(xs)
println(xs2 |> size)

layer2 = Conv((3,3), 4=> 4, relu)
xs3 = layer2(xs2)
println(xs3 |> size)
flatten(xs3) |> size

#= println(AdaptiveMaxPool((2,2))(xs2) |> size) =#

