
/*** Run one (batched) conditioning step of ICM ***/
void _condition(
  unsigned char* B, // In/out. n-by-m matrix of codes that we are working on
  float* ub, // In. n-by-h matrix with the unary terms of the code that we are exhaustively exploring
  float* binaries, // (m(m-1)/2)-by-h matrix with the binary terms of the encoding MRFS
  float* binaries_t, // Same as binaries, but each matrix has been transposed
  int* cbpair2binaryidx, // m-by-m matrix. cbpair2binaryidx[i,j] returns the linear index of the catted binaries going from codebook i to j
  int* to_condition, // m-1-vector of codes to look at (ie, which we are conditioning on)
  const int j, // Entry in to_condition that indicates which codebook we are looking at right now
  const int n, // Number of vectors we are processing in parallel
  const int m  // Number of codebooks
) {

  const int H = 256; // number of entries per codebook
  float* bb;         // pointer to ??
  int binariidx = 0; //
  int k = 0;
  unsigned char codek = 0;

  float minv = 0;
  float ubi  = 0;
  unsigned char mini = 0;

  // #pragma omp parallel for private(codek,mini,minv)
  for (int l=0; l<n; l++) {
    // Pointer to the binaries we'll use
    for (int kidx=0; kidx<m-1; kidx++) {

      k = to_condition[kidx];

      binariidx = cbpair2binaryidx[ j*m + k ];
      if( j < k ) {
        bb = binaries + H*H*binariidx;
      } else {
        bb = binaries_t + H*H*binariidx;
      }

      // Now condition
      codek = B[ l*m + k ];
      for (int ll=0; ll<H; ll++) {
        ub[ l*H + ll ] += bb[ codek*H + ll ];
      }

    }

    minv = ub[ l*H ];
    mini = 0;

    // Loop through the rest h-1 entries
    for (int k=1; k<H; k++) {
      ubi = ub[ l*H + k ];
      if (ubi < minv) {
        minv = ubi;
        mini = k;
      }
    }
    B[ l*m + j ] = mini;
  }

}

extern "C"
{
  void condition(
    unsigned char* B,
    float* ub,
    float* binaries,
    float* binaries_t,
    int* cbpair2binaryidx,
    int* to_condition,
    const int j,
    const int n,
    const int m) {
    _condition( B, ub, binaries, binaries_t, cbpair2binaryidx, to_condition, j, n, m );
  };
}
