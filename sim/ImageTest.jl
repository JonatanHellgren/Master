using Plots
using Images
using FileIO
gr()
h = 400
w = 600
#= a = Array(RGB{FixedPointNumbers.UFixed{UInt8,8}}, h, w) =#
img_path = "kisspng-arrow-scalable-vector-graphics-clip-art-black-arrow-5aa8e8acc61c23.9217803715210190528115.jpg"
img = load(img_path)
p=plot(img)
