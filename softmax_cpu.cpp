/*
Softmax Fusion Lab: original bounded bias/scale/prefix-softmax CPU foundation.
Repository: https://github.com/BLANK-2340/softmax-fusion-lab
Created PUBLIC 2026-11-19, owner BLANK-2340 (105676435), repo 1463846170.

Build/run (C++17, standard library only):
  g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror softmax_cpu.cpp -o /tmp/softmax_cpu
  /tmp/softmax_cpu test
  /tmp/softmax_cpu demo
  /tmp/softmax_cpu study
Sanitizers:
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer softmax_cpu.cpp -o /tmp/softmax_san
  ASAN_OPTIONS=detect_leaks=0 /tmp/softmax_san test
Leak detection is disabled here because LeakSanitizer reports a fatal ptrace
restriction in this runtime. No leak-detection success is claimed.

Question: can a bounded online normalizer implement bias+scale+prefix-masked
softmax without a materialized logits/exponential buffer while maintaining
numerical agreement with an independent high-precision reference? This CPU
study tests arithmetic and contracts, NOT GPU fusion or performance.

Contract: z[r,c]=(x[r,c]+bias[c])*scale for c<length[r]; output softmax(z)
on that prefix, exactly zero elsewhere. Fully masked rows return all zeros
by explicit API convention (not a probability distribution). Empty row batches
are valid. Columns 1..4096, rows 0..1024; finite |x|,|bias|<=80; 0<scale<=4.
Bounds keep FP32 transformation finite. NaN/Inf inputs are rejected even in
masked cells. No strides, arbitrary masks, aliasing, backward pass or FP16.

Reference: materialize long-double logits, max shift, exponentials and sum.
Candidate: float arithmetic, online max/normalizer recurrence, then recompute
transformed logits and write probabilities in a second pass. This is a serial
CPU algorithm; it is neither a CUDA simulator nor proof of parallel reduction.
Tests compare complete output vectors with abs<=1e-4 + rel1e-3*|reference|,
row mass within 1e-5, finite/nonnegative outputs and exact masked zeros.
Mixed-precision comparison includes FP32 transform and accumulation error.
Long double precision is platform-dependent; study prints mantissa bits.
No fast-math flags. No timing or speedup measurements are reported.

Original deterministic generated data: xorshift32 integer PRNG, seeds7/42/2026;
21 widths around powers-of-two/warp-like boundaries, 4 input modes, 5 prefix
lengths (0,1,n/2,n-1,n) =252 matrices,1260 rows. Narrow widths repeat some
lengths intentionally. Cases are diagnostics, not a workload distribution.
Modes: random dyadic values, zeros, alternating extreme +/-80, descending
dyadic values. Common row bias has original dyadic values. Fixed scale .5.
Separate tests exercise scale4, equal/extreme logits, singleton and known
probabilities, invalid input, shift/permutation invariance and no input mutation.

Attribution: online normalization is an established mathematical technique:
Milakov/Gimelshein (2018), Online normalizer calculation for softmax,
https://arxiv.org/abs/1805.02867 . This is an original implementation, not
copied code or a claimed novel algorithm/reproduction of paper speedups.
Future CUDA timing/synchronization reference (NVIDIA official docs):
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
No external code, assets, data or private Papers content incorporated.

Initial CPU study observed on GCC14.2: FP32 mantissa24 bits, long-double
mantissa64 bits; max absolute probability error 1.0926696584369847e-7,
max row-mass error 7.31821885510285011e-6 across the fixed252 matrices.
These are CPU arithmetic diagnostics, NOT GPU correctness/performance.
Validation: strict GCC14.2 build, all7 CPU test groups, demo, study JSON
subprocess byte repeatability and invalid CLI rejection passed. AddressSanitizer
and UndefinedBehaviorSanitizer passed with leak detection disabled as above.

Task continuity 2026-11-19: fifth NEW November project, CPU foundation only.
Four earlier bounded projects complete; first-three public demos verified.
No CUDA source, CUDA compilation, GPU correctness, GPU benchmark, README,
separate license file or CI yet. Current runtime has GCC14.2, no nvcc or
nvidia-smi. All GPU components remain PENDING. Next: add original coherent
fused CUDA kernel and benchmark harness with independent CPU correctness,
warmup/repetitions, hardware/compiler metadata and explicit timing scope;
add README/license/CI. Run CUDA only on actual compatible free hardware.
Do not mark the fifth project fully complete from these CPU checks.

MIT License
Copyright (c) 2026 BLANK-2340
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion {
struct Batch {
    std::size_t rows, cols;
    std::vector<float> x, bias;
    std::vector<std::size_t> lengths;
    float scale;
};

void validate(const Batch& b) {
    if (b.cols < 1 || b.cols > 4096 || b.rows > 1024 ||
        b.x.size() != b.rows*b.cols || b.bias.size() != b.cols ||
        b.lengths.size() != b.rows || !std::isfinite(b.scale) ||
        b.scale <= 0 || b.scale > 4) throw std::invalid_argument("shape/scale outside bounded contract");
    for (auto n : b.lengths) if (n > b.cols) throw std::invalid_argument("invalid prefix length");
    for (const auto* data : {&b.x, &b.bias}) {
        for (float x : *data) {
            if (!std::isfinite(x) || std::abs(x) > 80) throw std::invalid_argument("invalid input value");
        }
    }
}

std::vector<long double> reference(const Batch& b) {
    validate(b);
    std::vector<long double> out(b.x.size(), 0);
    for (std::size_t r=0; r<b.rows; ++r) {
        const auto n=b.lengths[r];
        if (n==0) continue;
        std::vector<long double> logits(n), exps(n);
        for (std::size_t c=0; c<n; ++c)
            logits[c]=(static_cast<long double>(b.x[r*b.cols+c])+b.bias[c])*b.scale;
        const auto maximum=*std::max_element(logits.begin(), logits.end());
        long double sum=0;
        for (std::size_t c=0; c<n; ++c) { exps[c]=std::exp(logits[c]-maximum); sum+=exps[c]; }
        for (std::size_t c=0; c<n; ++c) out[r*b.cols+c]=exps[c]/sum;
    }
    return out;
}

std::vector<float> online(const Batch& b) {
    validate(b);
    std::vector<float> out(b.x.size(), 0);
    for (std::size_t r=0; r<b.rows; ++r) {
        const auto n=b.lengths[r];
        if (n==0) continue;
        float maximum=0, sum=0;
        for (std::size_t c=0; c<n; ++c) {
            const float z=(b.x[r*b.cols+c]+b.bias[c])*b.scale;
            if (c==0) { maximum=z; sum=1; }
            else if (z<=maximum) sum+=std::exp(z-maximum);
            else { sum=sum*std::exp(maximum-z)+1; maximum=z; }
        }
        for (std::size_t c=0; c<n; ++c) {
            const float z=(b.x[r*b.cols+c]+b.bias[c])*b.scale;
            out[r*b.cols+c]=std::exp(z-maximum)/sum;
        }
    }
    return out;
}

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }

struct Error { long double max_abs=0, max_mass=0; };
Error check(const Batch& b) {
    const auto expected=reference(b);
    const auto actual=online(b);
    Error error;
    for (std::size_t r=0; r<b.rows; ++r) {
        long double mass=0;
        for (std::size_t c=0; c<b.cols; ++c) {
            const auto i=r*b.cols+c;
            require(std::isfinite(actual[i]) && actual[i]>=0, "invalid probability");
            const auto delta=std::abs(actual[i]-expected[i]);
            error.max_abs=std::max(error.max_abs,delta);
            require(delta<=1e-4L+1e-3L*std::abs(expected[i]), "reference mismatch");
            if (c>=b.lengths[r]) require(actual[i]==0, "masked output not exactly zero");
            mass+=actual[i];
        }
        const auto mass_error=std::abs(mass-(b.lengths[r]==0 ? 0.L : 1.L));
        error.max_mass=std::max(error.max_mass,mass_error);
        require(mass_error<=1e-5L,"row mass mismatch");
    }
    return error;
}

std::uint32_t next(std::uint32_t& state) {
    state^=state<<13; state^=state>>17; state^=state<<5; return state;
}
Batch make_case(std::size_t cols, std::uint32_t seed, int mode) {
    Batch b{5,cols,std::vector<float>(5*cols),std::vector<float>(cols),{0,1,cols/2,cols-1,cols},.5f};
    for (auto& bias : b.bias) bias=static_cast<float>(static_cast<int>(next(seed)%65)-32)/16;
    for (std::size_t i=0; i<b.x.size(); ++i) {
        if (mode==0) b.x[i]=static_cast<float>(static_cast<int>(next(seed)%2049)-1024)/128;
        else if (mode==1) b.x[i]=0;
        else if (mode==2) b.x[i]=(i%2==0 ? -80.f : 80.f);
        else b.x[i]=static_cast<float>(cols-(i%cols))/64;
    }
    return b;
}

Error study() {
    Error total;
    const std::vector<std::size_t> widths{1,2,3,7,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096};
    for (auto cols : widths) for (auto seed : {7u,42u,2026u}) for (int mode=0; mode<4; ++mode) {
        const auto e=check(make_case(cols,seed,mode));
        total.max_abs=std::max(total.max_abs,e.max_abs);
        total.max_mass=std::max(total.max_mass,e.max_mass);
    }
    return total;
}

void tests() {
    // 1. Analytic probabilities, not a second copy of the implementation.
    Batch b{1,3,{0,1,2},{0,0,0},{3},1};
    auto out=online(b);
    const long double d=1+std::exp(1.L)+std::exp(2.L);
    for (std::size_t i=0;i<3;++i) require(std::abs(out[i]-std::exp(static_cast<long double>(i))/d)<1e-7L,"analytic mismatch");
    // 2. Uniform, fully masked, singleton, large logits and extreme scale.
    Batch edge{4,3,{80,80,80, -80,80,80, 80,80,80, -80,80,-80},{0,0,0},{3,0,1,3},4};
    check(edge); out=online(edge);
    require(out[6]==1 && out[7]==0 && out[8]==0 && out[10]==1,"edge convention");
    for (std::size_t c=0;c<3;++c) require(std::abs(out[c]-1.f/3)<1e-7f,"uniform mismatch");
    check(Batch{1,3,{.1f,-.3f,.9f},{.7f,.2f,-.1f},{3},.7f});
    // 3. Shift invariance on exactly representable transforms.
    auto shifted=b; for (auto& x:shifted.x) x+=32;
    const auto base=online(b), translated=online(shifted);
    for (std::size_t i=0;i<3;++i) require(std::abs(base[i]-translated[i])<1e-7f,"shift mismatch");
    // 4. Permuting input AND bias permutes full-prefix probabilities.
    b.bias={.5f,1.f,-1.f}; const auto before=online(b);
    std::reverse(b.x.begin(),b.x.end()); std::reverse(b.bias.begin(),b.bias.end()); out=online(b);
    for (std::size_t i=0;i<3;++i) require(std::abs(out[i]-before[2-i])<1e-7f,"permutation mismatch");
    // 5. Invalid shapes/domain including nonfinite masked values rejected by BOTH paths.
    std::vector<Batch> invalid;
    auto bad=b; bad.cols=0; invalid.push_back(bad);
    bad=b; bad.rows=1025; invalid.push_back(bad);
    bad=b; bad.cols=4097; invalid.push_back(bad);
    bad=b; bad.x.pop_back(); invalid.push_back(bad);
    bad=b; bad.bias.pop_back(); invalid.push_back(bad);
    bad=b; bad.lengths={4}; invalid.push_back(bad);
    bad=b; bad.lengths.clear(); invalid.push_back(bad);
    for (float value : {0.f,-1.f,5.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        bad=b; bad.scale=value; invalid.push_back(bad);
    }
    for (float value : {81.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        bad=b; bad.lengths={0}; bad.x[0]=value; invalid.push_back(bad);
        bad=b; bad.bias[0]=value; invalid.push_back(bad);
    }
    for (const auto& item : invalid) {
        bool rejected=false; try { (void)online(item); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"online accepted invalid input");
        rejected=false; try { (void)reference(item); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"reference accepted invalid input");
    }
    // 6. Empty batch and immutable input contract; byte-repeatable computation.
    require(online(Batch{0,1,{}, {0},{},1}).empty(),"empty batch");
    const auto saved=b; const auto once=online(b), twice=online(b);
    require(once==twice && b.x==saved.x && b.bias==saved.bias && b.lengths==saved.lengths,"mutation/repeatability");
    // 7. Complete fixed boundary/seed/mode study against long-double reference.
    (void)study();
    std::cout << "PASS 7 CPU test groups; CUDA validation PENDING\n";
}
} // namespace fusion

int main(int argc, char** argv) {
    try {
        const std::string command=argc==2 ? argv[1] : "";
        if (command=="test") fusion::tests();
        else if (command=="study") {
            const auto e=fusion::study();
            std::cout << std::setprecision(18)
                      << "{\"scope\":\"CPU arithmetic only; no GPU or performance claim\",\"matrices\":252,\"rows\":1260,"
                      << "\"float_mantissa_bits\":" << std::numeric_limits<float>::digits
                      << ",\"reference_mantissa_bits\":" << std::numeric_limits<long double>::digits
                      << ",\"max_absolute_error\":" << e.max_abs
                      << ",\"max_row_mass_error\":" << e.max_mass << ",\"gpu_validation\":\"PENDING\"}\n";
        } else if (command=="demo") {
            const fusion::Batch b{2,4,{0,1,2,3,1,2,3,4},{0,.5f,0,-.5f},{3,0},1};
            const auto out=fusion::online(b); fusion::check(b);
            std::cout << std::setprecision(9) << "CPU prefix-softmax:\n";
            for (std::size_t r=0;r<b.rows;++r) {
                for (std::size_t c=0;c<b.cols;++c) std::cout << out[r*b.cols+c] << (c+1==b.cols ? '\n' : ' ');
            }
        } else throw std::invalid_argument("usage: softmax_cpu {test|study|demo}");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
