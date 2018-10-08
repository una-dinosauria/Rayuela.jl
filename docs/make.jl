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
            "PQ.md"
        ]
    ]
    # doctest = test
)

deploydocs(
	repo = "github.com/una-dinosauria/Rayuela.jl.git",
	target = "build",
	osname = "linux",
	julia  = "1.0",
	deps = nothing,
	make = nothing,
)
