using Plots

num_layers = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
gpipe_times = [29.4, 46.0, 65.7, 88.2, 109.9, 131.6, 152.8, 174.9, 203.1, 237.5]
naive_times = [36.2, 55.4, 81.4, 107, 133.9, 162.0, 190.3, 227.4, 270.4, 330.3]

slope = sum([l*t for (l, t) in zip(num_layers, gpipe_times)])/sum([l*t for (l, t) in zip(num_layers, num_layers)])
slope2 = sum([l*t for (l, t) in zip(num_layers, naive_times)])/sum([l*t for (l, t) in zip(num_layers, num_layers)])
slope2/slope

plot(num_layers, gpipe_times, xlims=(0, 220), ylims=(0,350), seriestype=:scatter, label = "GPipe", color=:blue, xlabel="Number of Layers", ylabel="Time (s)", )
plot!(num_layers, slope*num_layers, smooth=:true, color=:blue, label="")
plot!(num_layers, naive_times, seriestype=:scatter, legend=:topleft, label="Naive", color=:red)
plot!(num_layers, slope2*num_layers, smooth=:true, color=:red, label="")


scatter(num_layers, [n/g for (n, g) in zip(naive_times, gpipe_times)], ylims=(1.15, 1.45), xlims=(0, 220), label="", markersize=5, yticks=([1.15:0.1:1.45;], ["15%", "25%", "35%"]), xlabel="Number of Layers", ylabel="Relative Performance Gain")