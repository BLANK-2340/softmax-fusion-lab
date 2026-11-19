// MIT, copyright 2026 BLANK-2340.
// EXPERIMENTAL: CUDA compilation, execution and performance are UNVERIFIED.
#include <cuda_runtime.h>
#include <math_constants.h>
#define FUSION_NO_MAIN
#include "softmax_cpu.cpp"
#include "bench_common.hpp"

constexpr int threads=256;
void cuda_check(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Device {
    T* ptr=nullptr;
    explicit Device(std::size_t n) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T))); }
    ~Device() { if (ptr) (void)cudaFree(ptr); }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
struct Event {
    cudaEvent_t event{};
    Event() { cuda_check(cudaEventCreate(&event)); }
    ~Event() { (void)cudaEventDestroy(event); }
    Event(const Event&)=delete;
    Event& operator=(const Event&)=delete;
};

__global__ void transform(const float* x, const float* bias, float* z, int count, int cols, float scale) {
    const int i=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<count) z[i]=(x[i]+bias[i%cols])*scale;
}

// One block/row. Every block has exactly 256 threads and all reach each barrier.
// The n==0 branch is uniform within the block and writes zeros before returning.
template<bool Fused> __global__ void row_softmax(const float* x, const float* bias,
        const int* lengths, float* out, int cols, float scale) {
    __shared__ float scratch[threads];
    const int r=blockIdx.x, lane=threadIdx.x, n=lengths[r];
    if (n==0) {
        for (int c=lane;c<cols;c+=threads) out[r*cols+c]=0;
        return;
    }
    float maximum=-CUDART_INF_F;
    for (int c=lane;c<n;c+=threads) {
        const float z=Fused ? (x[r*cols+c]+bias[c])*scale : x[r*cols+c];
        maximum=fmaxf(maximum,z);
    }
    scratch[lane]=maximum; __syncthreads();
    for (int stride=threads/2;stride>0;stride/=2) {
        if (lane<stride) scratch[lane]=fmaxf(scratch[lane],scratch[lane+stride]);
        __syncthreads();
    }
    maximum=scratch[0];
    __syncthreads(); // all threads read max before scratch is reused
    float sum=0;
    for (int c=lane;c<n;c+=threads) {
        const float z=Fused ? (x[r*cols+c]+bias[c])*scale : x[r*cols+c];
        sum+=expf(z-maximum);
    }
    scratch[lane]=sum; __syncthreads();
    for (int stride=threads/2;stride>0;stride/=2) {
        if (lane<stride) scratch[lane]+=scratch[lane+stride];
        __syncthreads();
    }
    const float denominator=scratch[0];
    for (int c=lane;c<cols;c+=threads) {
        if (c<n) {
            const float z=Fused ? (x[r*cols+c]+bias[c])*scale : x[r*cols+c];
            out[r*cols+c]=expf(z-maximum)/denominator;
        } else out[r*cols+c]=0;
    }
}

void launch(bool fused, const Device<float>& x, const Device<float>& bias,
        const Device<int>& lengths, Device<float>& intermediate, Device<float>& out,
        int rows, int cols, float scale) {
    if (fused) row_softmax<true><<<rows,threads>>>(x.ptr,bias.ptr,lengths.ptr,out.ptr,cols,scale);
    else {
        transform<<<(rows*cols+threads-1)/threads,threads>>>(x.ptr,bias.ptr,intermediate.ptr,rows*cols,cols,scale);
        cuda_check(cudaGetLastError());
        row_softmax<false><<<rows,threads>>>(intermediate.ptr,bias.ptr,lengths.ptr,out.ptr,cols,scale);
    }
    cuda_check(cudaGetLastError());
}

void run(const fusion::Batch& b, bool fused, int warmup, int repetitions, bool report) {
    fusion::validate(b);
    fusion::require(b.rows>0,"GPU harness requires nonempty batch");
    const int rows=static_cast<int>(b.rows),cols=static_cast<int>(b.cols);
    const auto bytes=b.x.size()*sizeof(float);
    Device<float> x(b.x.size()),bias(b.bias.size()),intermediate(b.x.size()),out(b.x.size());
    Device<int> lengths(b.rows);
    std::vector<int> host_lengths(b.lengths.begin(),b.lengths.end());
    cuda_check(cudaMemcpy(x.ptr,b.x.data(),bytes,cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(bias.ptr,b.bias.data(),b.bias.size()*sizeof(float),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(lengths.ptr,host_lengths.data(),b.rows*sizeof(int),cudaMemcpyHostToDevice));
    auto execute=[&] { launch(fused,x,bias,lengths,intermediate,out,rows,cols,b.scale); };
    auto validate_output=[&] {
        std::vector<float> actual(b.x.size());
        cuda_check(cudaMemcpy(actual.data(),out.ptr,bytes,cudaMemcpyDeviceToHost));
        return bench::verify(b,actual);
    };
    execute(); cuda_check(cudaDeviceSynchronize());
    (void)validate_output(); // correctness BEFORE warmup or timing
    for (int i=0;i<warmup;++i) execute();
    cuda_check(cudaDeviceSynchronize());
    Event start,stop;
    std::vector<double> samples;
    for (int i=0;i<repetitions;++i) {
        cuda_check(cudaEventRecord(start.event));
        execute();
        cuda_check(cudaEventRecord(stop.event));
        cuda_check(cudaEventSynchronize(stop.event));
        float ms=0; cuda_check(cudaEventElapsedTime(&ms,start.event,stop.event));
        samples.push_back(ms);
    }
    const auto error=validate_output(); // correctness AFTER timing
    if (report) {
        const auto stats=bench::summarize(samples);
        std::cout<<std::setprecision(12)<<"variant="<<(fused?"fused":"materialized")
                 <<" rows="<<rows<<" cols="<<cols<<" warmup="<<warmup<<" repetitions="<<repetitions
                 <<" min_ms="<<stats.minimum<<" median_ms="<<stats.median<<" mean_ms="<<stats.mean
                 <<" max_ms="<<stats.maximum<<" max_abs_error="<<static_cast<double>(error.max_abs)
                 <<" max_mass_error="<<static_cast<double>(error.max_mass)<<'\n';
    }
}

int main(int argc,char** argv) {
    try {
        const auto options=bench::parse(argc,argv);
        int count=0; cuda_check(cudaGetDeviceCount(&count));
        fusion::require(count>0,"No CUDA device available; GPU validation PENDING");
        cuda_check(cudaSetDevice(0));
        cudaDeviceProp device{}; cuda_check(cudaGetDeviceProperties(&device,0));
        int driver=0,runtime=0; cuda_check(cudaDriverGetVersion(&driver)); cuda_check(cudaRuntimeGetVersion(&runtime));
        std::cout<<"device="<<device.name<<" sm="<<device.major<<'.'<<device.minor
                 <<" global_bytes="<<device.totalGlobalMem<<" driver="<<driver<<" runtime="<<runtime
                 <<" nvcc="<<__CUDACC_VER_MAJOR__<<'.'<<__CUDACC_VER_MINOR__<<'.'<<__CUDACC_VER_BUILD__
                 <<" host_compiler="<<__VERSION__<<" block_threads="<<threads<<'\n';
        std::cout<<"scope=CUDA events on default stream; kernel pipeline only, includes device idle/launch gaps; excludes allocation, copies, CPU reference, validation and event creation; no end-to-end timing\n";
        // All original diagnostic widths/seeds/modes, both variants, before any reported timing.
        for (auto cols:{1,2,3,7,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096})
            for (auto seed:{7u,42u,2026u}) for (int mode=0;mode<4;++mode)
                for (bool fused:{false,true}) run(fusion::make_case(cols,seed,mode),fused,0,0,false);
        for (bool fused:{false,true}) {
            run(fusion::Batch{2,3,{80,80,80,-80,80,-80},{0,0,0},{3,3},4},fused,0,0,false);
            run(bench::workload(options),fused,0,0,false);
        }
        for (bool fused:{false,true}) {
            run(bench::workload(options),fused,options.warmup,options.repetitions,true);
        }
        std::cout<<"PASS device numerical suite and benchmark correctness; GPU results apply only to this execution\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
