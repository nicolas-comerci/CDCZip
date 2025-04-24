#ifndef DELTA_COMP_H
#define DELTA_COMP_H

namespace delta_comp_constants {
  // Coefficients (coprime pairs) for N-Transform feature extraction, only have 16 as more than 4SF-4F is unlikely to yield good results
  static constexpr std::pair<uint16_t, uint16_t> N_Transform_Coefs[16] = {
    {2, 3},
    {3, 5},
    {5, 7},
    {7, 11},
    {11, 13},
    {13, 17},
    {17, 19},
    {19, 23},
    {23, 29},
    {29, 31},
    {31, 37},
    {37, 41},
    {41, 43},
    {43, 47},
    {47, 53},
    {53, 59},
  };

  // Sampling mask for Content Defined Sampling with 1/64 frequency and somewhat spread 1bits as used by ODESS paper
  static constexpr uint32_t CDS_SAMPLING_MASK = 0x40030341;
}

#endif
