using Documenter, Rayuela

makedocs(
    format = :html,
    sitename = "Rayuela.jl",
    modules = [Rayuela],
    pages = [
        "Home"    => "index.md",
        # "Manual"  => [
        #     "man/usage.md"
        # ],
        "Library" => [
            "PQ.md",
            "OPQ.md",
            "RVQ.md",
            "ERVQ.md",
            "ChainQ.md",
            "LSQ.md"
        ]
    ]
    # doctest = test
)

deploydocs(
	repo = "github.com/una-dinosauria/Rayuela.jl.git",
  julia  = "",
  osname = "",
  # no need to build anything here, re-use output of `makedocs`
	target = "build",
	deps = nothing,
	make = nothing,
)
