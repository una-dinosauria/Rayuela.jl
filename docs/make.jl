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
