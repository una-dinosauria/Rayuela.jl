
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

  #pragma omp parallel for private(codek,mini,minv)
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

} // end _condition

void _viterbi_encoding(
  unsigned char* B, // In/out. n-by-m matrix of codes that we are working on
  float* unaries, // In. n-by-h matrix with the unary terms of the code that we are exhaustively exploring
  float* binaries, // (m(m-1)/2)-by-h matrix with the binary terms of the encoding MRFS
  float* mincost,
  float* U,
  int* minidx,
  float* cost,
  int* backpath,
  const int n, // Number of vectors we are processing in parallel
  const int m  // Number of codebooks
) {

  const int H = 256;
  float * bb;

  float ucost, bcost, minv, costi;
  int mini;

  // #pragma omp parallel for private(U,minv,mini,ucost,bcost,bb,costi,cost,backpath,minidx,mincost)
  // #pragma omp parallel for private(U,ucost,bcost,minv,costi,mincost,minidx,cost,backpath,mini,bb)
  // #pragma omp parallel for private(U,ucost,bcost,minv,costi,mincost,minidx,cost,backpath,mini,bb) shared(B)
  for (int idx=0; idx<n; idx++) { // Loop over datapoints

    // Put all the unaries of this item together
    for (int i=0; i<m*H; i++) {
      U[i] = unaries[ idx*H*m + i];
    }

    // Forward pass
    for (int i=0; i<m-1; i++) {

      // If this is not the first iteration, add the precomputed costs
      if (i>0) {
        for (int j=0; j<H; j++) {
          U[i*H + j] += mincost[j];
        }
      }

      bb = binaries + H*H*i; // bb points to the ith codebook
      for (int j=0; j<H; j++) { // Loop over the cost of going to j
        for (int k=0; k<H; k++) { // Loop over the cost of coming from k
          ucost =  U[i*H + k]; // Pay the unary of coming from k
          bcost = bb[j*H + k]; // Pay the binary of going from j to k
          cost[k] = ucost + bcost;
        }

        // findmin because C++'s is too slow?
        minv = cost[0];
        mini = 0;
        for (int k=1; k<H; k++) {
          costi = cost[k];
          if (costi < minv) {
            minv = costi;
            mini = k;
          }
        }

        mincost[j] = minv;
         minidx[i*H + j] = mini;
      }
    }

    for (int j=0; j<H; j++) {
      U[(m-1)*H + j] += mincost[j];
    }

    minv = U[(m-1)*H + 0];
    mini = 0;
    for (int j=1; j<H; j++) {
      if (U[(m-1)*H + j] < minv) {
        minv = U[(m-1)*H + j];
        mini = j;
      }
    }

    // backward trace
    backpath[0] = mini;
    int backpathidx = 1;
    for (int i=m-2; i>=0; i--) {
      backpath[ backpathidx ] = minidx[i*H + backpath[backpathidx-1]];
      backpathidx++;
    }

    for (int i=0; i<m; i++) {
      B[idx*m + i] = backpath[(m-1)-i];
    }

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
    const int m ) {
    _condition( B, ub, binaries, binaries_t, cbpair2binaryidx, to_condition, j, n, m );
  };

  void viterbi_encoding(
    unsigned char* B, // In/out. n-by-m matrix of codes that we are working on
    float* unaries, // In. n-by-h matrix with the unary terms of the code that we are exhaustively exploring
    float* binaries, // (m(m-1)/2)-by-h matrix with the binary terms of the encoding MRFS
    float* mincost,
    float* U,
    int* minidx,
    float* cost,
    int* backpath,
    const int n, // Number of vectors we are processing in parallel
    const int m // Number of codebooks
  ) {
    _viterbi_encoding(B, unaries, binaries, mincost, U, minidx, cost, backpath, n, m);
  };
}
