using Documenter, Rayuela

# makedocs()
makedocs(
    modules = [Rayuela],
    format = :html,
    sitename = "Rayuela.jl",
    pages = [
        "Home"    => "index.md",
        # "Manual"  => [
        #     "man/usage.md"
        # ],
        # "Library" => [
        #     "lib/api.md",
        #     "lib/array.md"
        # ]
    ]
    # doctest = test
)
