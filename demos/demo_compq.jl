
using Rayuela

# === Main
Xt = fvecs_read(Int(1e4), "../data/sift/sift_learn.fvecs")
m  = 8   # Number of codebooks
h  = 256 # Number of entries in each codebook
lr = 0.5f0 # learning rate
H  = 32 # depth for search during encoding

n_its = 250

C, B = train_rvq(Xt, m, h, 25, true)
qerr = qerror(Xt, B, C)
@show B[:, 1]
print( "Error after initialization is $qerr\n" )

train_competitiveq(Xt, C, n_its, H, B, lr )
