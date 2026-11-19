// MIT, copyright 2026 BLANK-2340. Host-only benchmark contracts.
// Include after softmax_cpu.cpp with FUSION_NO_MAIN defined (one translation unit).
#pragma once
#include <charconv>
#include <numeric>

namespace bench {
struct Options { int rows=128, cols=257, warmup=10, repetitions=50; };
int integer(const std::string& text, int lo, int hi) {
    int result=0;
    const auto p=std::from_chars(text.data(),text.data()+text.size(),result);
    if (p.ec!=std::errc{} || p.ptr!=text.data()+text.size() || result<lo || result>hi)
        throw std::invalid_argument("invalid bounded integer argument");
    return result;
}
Options parse(int argc, char** argv) {
    if (argc==1) return {};
    if (argc!=5) throw std::invalid_argument("usage: gpu_bench [rows cols warmup repetitions]");
    return {integer(argv[1],1,1024),integer(argv[2],1,4096),
            integer(argv[3],0,100),integer(argv[4],1,1000)};
}
fusion::Batch workload(const Options& o) {
    fusion::require(o.rows>=1 && o.rows<=1024 && o.cols>=1 && o.cols<=4096,"invalid workload shape");
    fusion::Batch b{static_cast<std::size_t>(o.rows),static_cast<std::size_t>(o.cols),
                    std::vector<float>(o.rows*o.cols),std::vector<float>(o.cols),{},.7f};
    std::uint32_t seed=2026;
    for (auto& x:b.x) x=static_cast<float>(static_cast<int>(fusion::next(seed)%2049)-1024)/128;
    for (auto& x:b.bias) x=static_cast<float>(static_cast<int>(fusion::next(seed)%65)-32)/16;
    for (int r=0;r<o.rows;++r) b.lengths.push_back(r%4==0 ? 0 : r%4==1 ? 1 : r%4==2 ? o.cols/2 : o.cols);
    fusion::validate(b);
    return b;
}
fusion::Error verify(const fusion::Batch& b, const std::vector<float>& actual) {
    const auto expected=fusion::reference(b);
    fusion::require(actual.size()==expected.size(),"output size mismatch");
    fusion::Error error;
    for (std::size_t r=0;r<b.rows;++r) {
        long double mass=0;
        for (std::size_t c=0;c<b.cols;++c) {
            const auto i=r*b.cols+c;
            fusion::require(std::isfinite(actual[i]) && actual[i]>=0,"nonfinite/negative output");
            const auto delta=std::abs(actual[i]-expected[i]);
            fusion::require(delta<=1e-4L+1e-3L*std::abs(expected[i]),"numerical correctness failure");
            fusion::require(c<b.lengths[r] || actual[i]==0,"masked output not zero");
            mass+=actual[i]; error.max_abs=std::max(error.max_abs,delta);
        }
        const auto delta=std::abs(mass-(b.lengths[r] ? 1.L : 0.L));
        fusion::require(delta<=1e-5L,"mass correctness failure");
        error.max_mass=std::max(error.max_mass,delta);
    }
    return error;
}
struct Summary { double minimum, median, mean, maximum; };
Summary summarize(std::vector<double> milliseconds) {
    fusion::require(!milliseconds.empty(),"empty timings");
    for (auto x:milliseconds) fusion::require(std::isfinite(x) && x>=0,"invalid timing");
    std::sort(milliseconds.begin(),milliseconds.end());
    const auto n=milliseconds.size();
    const auto median=n%2 ? milliseconds[n/2] : (milliseconds[n/2-1]+milliseconds[n/2])/2;
    return {milliseconds.front(),median,std::accumulate(milliseconds.begin(),milliseconds.end(),0.)/n,milliseconds.back()};
}
} // namespace bench
