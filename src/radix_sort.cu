#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/span>
#include <cstdio>
#include <random>
#include <vector>
#include <iostream>

constexpr int N = 5000000;
constexpr int RADIX_BITS = 2;
constexpr int RADIX = (1 << RADIX_BITS);
constexpr int SHIFT_BITS = 0;
constexpr int MASK = RADIX - 1;
constexpr int SEGMENT_LENGTH = 256;
constexpr int SEGMENT_CNT = (N + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;
constexpr int SLOT_HISTOGRAM_MAX_LENGTH = RADIX * SEGMENT_CNT;

__device__ int get_curr_slot(
  cuda::std::span<int const> array_length,
  cuda::std::span<int const> input_tile_id,
  cuda::std::span<float const> input_depth,
  int eid
) {
  const int shift = SHIFT_BITS & 31;
  int key = 0xFFFFFFFF;
  if(eid < array_length[0]) {
    if (SHIFT_BITS < 32) {
      key = input_tile_id[eid];
    } else {
      key = __float_as_int(input_depth[eid]);
    }
  }
  return (key >> shift) & MASK;
}

__device__ int get_conflict_free_pos(int pos) {
  return pos + (pos >> 5);
}

__global__ void radix_sort_pass_1_histogram(
  cuda::std::span<int const> array_length,
  cuda::std::span<int const> input_tile_id,
  cuda::std::span<float const> input_depth,
  cuda::std::span<int> slot_histogram
) {
  int const SID = blockIdx.x;
  int const TID = threadIdx.x;
  int const EID_1 = SID * 256 + TID * 2 + 0;
  int const EID_2 = SID * 256 + TID * 2 + 1;

  int const curr_slot_1 = get_curr_slot(array_length, input_tile_id, input_depth, EID_1);
  int const curr_slot_2 = get_curr_slot(array_length, input_tile_id, input_depth, EID_2);

  __shared__ int local_slot_histogram[RADIX];

  if(TID < RADIX) {
    local_slot_histogram[TID] = 0;
  }

  __syncthreads();

  atomicAdd(&local_slot_histogram[curr_slot_1], 1);
  atomicAdd(&local_slot_histogram[curr_slot_2], 1);

  __syncthreads();

  if(TID <= MASK) {
    slot_histogram[TID * gridDim.x + SID] = atomicAdd(&local_slot_histogram[TID], 0);
  }
}

__global__ void radix_sort_pass_3_scatter(
  cuda::std::span<int const> array_length,
  cuda::std::span<int const> input_tile_id,
  cuda::std::span<float const> input_depth,
  cuda::std::span<int const> input_gauss_id,
  cuda::std::span<int const> slot_histogram_psum,
  cuda::std::span<int> sorted_tile_id,
  cuda::std::span<float> sorted_depth,
  cuda::std::span<int> sorted_gauss_id
) {
  int const SID = blockIdx.x;
  int const TID = threadIdx.x;
  int const EID_1 = SID * 256 + TID * 2 + 0;
  int const EID_2 = SID * 256 + TID * 2 + 1;

  int const arr_length = array_length[0];

  int const curr_slot_1 = get_curr_slot(array_length, input_tile_id, input_depth, EID_1);
  int const curr_slot_2 = get_curr_slot(array_length, input_tile_id, input_depth, EID_2);

  __shared__ int slot_histogram_preload[RADIX];
  __shared__ int psum[284];

  psum[get_conflict_free_pos(TID * 2 + 0)] = 1 << (curr_slot_1 << 3);
  psum[get_conflict_free_pos(TID * 2 + 1)] = 1 << (curr_slot_2 << 3);

  if(TID < RADIX) {
    slot_histogram_preload[TID] = slot_histogram_psum[TID * gridDim.x + SID];
  }

  __syncthreads();

  int d = 1;
  int offset = 1 << d;
  while(offset <= 256){
    int const idx = ((TID + 1) << d) - 1;

    if(idx < 256) {
      psum[get_conflict_free_pos(idx)] += psum[get_conflict_free_pos(idx - (offset >> 1))];
    }
    d++;
    offset <<= 1;
    __syncthreads();
  }

  if(TID == 0){
    psum[get_conflict_free_pos(255)] = 0;
  }

  __syncthreads();

  d = 256;
  while(d >= 2) {
    int const curr_idx = (TID + 1) * d - 1;
    if(curr_idx < 256) {
      int const idx_1 = get_conflict_free_pos(curr_idx -  (d>>1));
      int const idx_2 = get_conflict_free_pos(curr_idx);
      int t = psum[idx_1];
      psum[idx_1] = psum[idx_2];
      psum[idx_2] += t;
    }
    d>>=1;
    __syncthreads();
  }

  if(EID_1 < arr_length) {
    int const local_rank = (psum[get_conflict_free_pos(TID * 2 + 0)] >> (curr_slot_1 << 3)) & 0xff;
    int const scatter_pos = slot_histogram_preload[curr_slot_1] + local_rank;
    sorted_tile_id[scatter_pos] = input_tile_id[EID_1];
    sorted_depth[scatter_pos] = input_depth[EID_1];
    sorted_gauss_id[scatter_pos] = input_gauss_id[EID_1];
  }
  if(EID_2 < arr_length) {
    int const local_rank = (psum[get_conflict_free_pos(TID * 2 + 1)] >> (curr_slot_2 << 3)) & 0xff;
    int const scatter_pos = slot_histogram_preload[curr_slot_2] + local_rank;
    sorted_tile_id[scatter_pos] = input_tile_id[EID_2];
    sorted_depth[scatter_pos] = input_depth[EID_2];
    sorted_gauss_id[scatter_pos] = input_gauss_id[EID_2];
  }
}

void radix_sort(
  const thrust::device_vector<int> &array_length,
  const thrust::device_vector<int> &input_tile_id,
  const thrust::device_vector<float> &input_depth,
  const thrust::device_vector<int> &input_gauss_id,
  thrust::device_vector<int> &slot_histogram,
  thrust::device_vector<int> &sorted_tile_id,
  thrust::device_vector<float> &sorted_depth,
  thrust::device_vector<int> &sorted_gauss_id
) {
  const int pass_1_num_blocks = SEGMENT_CNT;
  const int pass_1_block_size = 128;
  radix_sort_pass_1_histogram<<<pass_1_num_blocks, pass_1_block_size>>>(
    cuda::std::span<int const>(thrust::raw_pointer_cast(array_length.data()), array_length.size()),
    cuda::std::span<int const>(thrust::raw_pointer_cast(input_tile_id.data()), input_tile_id.size()),
    cuda::std::span<float const>(thrust::raw_pointer_cast(input_depth.data()), input_depth.size()),
    cuda::std::span<int>(thrust::raw_pointer_cast(slot_histogram.data()), slot_histogram.size())
  );

  cudaDeviceSynchronize();

  thrust::exclusive_scan(
    thrust::device,
    slot_histogram.begin(), slot_histogram.end(),
    slot_histogram.begin()
  );

  cudaDeviceSynchronize();

  const int pass_3_num_blocks = SEGMENT_CNT;
  const int pass_3_block_size = 128;
  radix_sort_pass_3_scatter<<<pass_3_num_blocks, pass_3_block_size>>>(
    cuda::std::span<int const>(thrust::raw_pointer_cast(array_length.data()), array_length.size()),
    cuda::std::span<int const>(thrust::raw_pointer_cast(input_tile_id.data()), input_tile_id.size()),
    cuda::std::span<float const>(thrust::raw_pointer_cast(input_depth.data()), input_depth.size()),
    cuda::std::span<int const>(thrust::raw_pointer_cast(input_gauss_id.data()), input_gauss_id.size()),
    cuda::std::span<int const>(thrust::raw_pointer_cast(slot_histogram.data()), slot_histogram.size()),
    cuda::std::span<int>(thrust::raw_pointer_cast(sorted_tile_id.data()), sorted_tile_id.size()),
    cuda::std::span<float>(thrust::raw_pointer_cast(sorted_depth.data()), sorted_depth.size()),
    cuda::std::span<int>(thrust::raw_pointer_cast(sorted_gauss_id.data()), sorted_gauss_id.size())
  );
}

int main() {
  //std::cout << "Hello!\n";

  thrust::device_vector<int> array_length(1, N);

  std::mt19937_64 rng;
  std::random_device rd;
  rng.seed(rd());
  std::uniform_real_distribution<float> float_dist(-1e6f, 0.0f);
  std::uniform_int_distribution<uint32_t> int_dist(0, 3);

  thrust::host_vector<int> input_array_tile_id_host(N);
  thrust::host_vector<float> input_array_depth_host(N);
  thrust::host_vector<int> input_array_gauss_id_host(N);

  std::cout << "[Input Array]\n";
  for(int i = 0; i < N; i++) {
    input_array_tile_id_host[i] = int_dist(rng);
    input_array_depth_host[i] = float_dist(rng);
    input_array_gauss_id_host[i] = int_dist(rng);

    //std::cout << input_array_tile_id_host[i] << ' ' << input_array_depth_host[i] << ' ' << input_array_gauss_id_host[i] << '\n';
  }

  std::cout << '\n';

  thrust::device_vector<int> input_array_tile_id = input_array_tile_id_host;
  thrust::device_vector<float> input_array_depth = input_array_depth_host;
  thrust::device_vector<int> input_array_gauss_id = input_array_gauss_id_host;

  thrust::device_vector<int> slot_histogram(SLOT_HISTOGRAM_MAX_LENGTH);
  thrust::device_vector<int> sorted_array_tile_id(N);
  thrust::device_vector<float> sorted_array_depth(N);
  thrust::device_vector<int> sorted_array_gauss_id(N);

  radix_sort(array_length, input_array_tile_id, input_array_depth,input_array_gauss_id,  slot_histogram, sorted_array_tile_id, sorted_array_depth, sorted_array_gauss_id);

  auto const err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  // std::cout << "[Sorted Array]\n";
  // thrust::host_vector<int> sorted_array_tile_id_host = sorted_array_tile_id;
  // thrust::host_vector<float> sorted_array_depth_host = sorted_array_depth;
  // thrust::host_vector<int> sorted_array_gauss_id_host = sorted_array_gauss_id;
  // for(int i = 0; i < N; i++) {
  //     std::cout << sorted_array_tile_id_host[i] << ' ' << sorted_array_depth_host[i] << ' ' << sorted_array_gauss_id_host[i] << '\n';
  // }

  return 0;
}
