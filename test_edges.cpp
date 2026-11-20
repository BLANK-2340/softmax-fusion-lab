// MIT, copyright 2026 BLANK-2340. CPU edge audit, NOT CUDA validation.
#define FUSION_NO_MAIN
#include "softmax_cpu.cpp"
#include "bench_common.hpp"

void rejects(const fusion::Batch& b, const std::vector<float>& output) {
    bool failed=false;
    try { (void)bench::verify(b,output); } catch (const std::runtime_error&) { failed=true; }
    fusion::require(failed,"fault injection escaped the numerical checker");
}

int main() {
    try {
        for (float gap:{.001f,.01f,.1f}) {
            fusion::Batch b{1,4096,std::vector<float>(4096,-gap),std::vector<float>(4096,0),{4096},1};
            b.x[0]=0;
            const auto out=fusion::online(b);
            long double mass=0; for (float x:out) mass+=x;
            std::cout << std::setprecision(18) << "gap="<<gap<<" mass_error="<<std::abs(mass-1)<<'\n';
            (void)bench::verify(b,out);
            // Closed-form reference independent of vectorized max/exp/sum code.
            const long double p0=1/(1+4095*std::exp(-static_cast<long double>(gap)));
            fusion::require(std::abs(out[0]-p0)<1e-9L,"near-tie analytic first probability");
            fusion::require(std::abs(out[1]-std::exp(-static_cast<long double>(gap))*p0)<1e-9L,
                            "near-tie analytic other probability");
        }
        for (float step:{.00001f,.001f,.01f}) {
            fusion::Batch b{1,4096,std::vector<float>(4096),std::vector<float>(4096,0),{4096},1};
            for (std::size_t c=0;c<4096;++c) b.x[c]=static_cast<float>(c)*step;
            (void)bench::verify(b,fusion::online(b));
            std::reverse(b.x.begin(),b.x.end());
            (void)bench::verify(b,fusion::online(b));
        }
        // Largest supported shape with non-dyadic near-ties and varied masks.
        fusion::Batch large{1024,4096,std::vector<float>(1024*4096),std::vector<float>(4096),{},.7f};
        for (std::size_t c=0;c<large.cols;++c) large.bias[c]=static_cast<float>(c%7)/10000;
        for (std::size_t r=0;r<large.rows;++r) {
            large.lengths.push_back(r%4==0 ? 0 : r%4==1 ? 1 : r%4==2 ? large.cols/2 : large.cols);
            for (std::size_t c=0;c<large.cols;++c)
                large.x[r*large.cols+c]=.1f+static_cast<float>((r+3*c)%11)/10000;
        }
        const auto large_error=bench::verify(large,fusion::online(large));
        std::cout<<"max_shape_rows=1024 cols=4096 max_abs_error="<<large_error.max_abs
                 <<" max_mass_error="<<large_error.max_mass<<'\n';
        // Shift and permutation invariance on non-dyadic inputs within domain.
        fusion::Batch b{1,4,{.1f,-.3f,.9f,.2f},{.7f,.2f,-.1f,.4f},{4},.7f};
        const auto base=fusion::online(b);
        auto shifted=b; for (auto& x:shifted.x) x+=32;
        const auto translated=fusion::online(shifted);
        for (std::size_t i=0;i<4;++i) fusion::require(std::abs(base[i]-translated[i])<2e-6f,"non-dyadic shift");
        auto permuted=b;
        std::reverse(permuted.x.begin(),permuted.x.end());
        std::reverse(permuted.bias.begin(),permuted.bias.end());
        const auto reversed=fusion::online(permuted);
        for (std::size_t i=0;i<4;++i) fusion::require(std::abs(base[i]-reversed[3-i])<1e-7f,"non-dyadic permutation");
        // Independent failure injection. Uniform accepted reference is exactly .25.
        fusion::Batch uniform{1,4,{0,0,0,0},{0,0,0,0},{4},1};
        std::vector<float> good(4,.25f); (void)bench::verify(uniform,good);
        rejects(uniform,{}); // wrong length
        auto bad=good; bad[0]=std::numeric_limits<float>::quiet_NaN(); rejects(uniform,bad);
        bad=good; bad[0]=std::numeric_limits<float>::infinity(); rejects(uniform,bad);
        bad=good; bad[0]=-.1f; rejects(uniform,bad);
        bad=good; bad[0]+=.01f; bad[1]-=.01f; rejects(uniform,bad); // mass preserved but wrong probabilities
        bad=good; for (auto& x:bad) x+=4e-6f; rejects(uniform,bad); // per-pixel close, mass fails
        auto masked=uniform; masked.lengths={2};
        rejects(masked,{.5f,.5f,1e-8f,0}); // tiny masked leak, even below abs/mass tolerances
        rejects(masked,{0,0,0,0}); // dropped valid prefix
        std::cout<<"PASS CPU near-tie/max-shape/invariance/checker fault audit; CUDA NOT executed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
