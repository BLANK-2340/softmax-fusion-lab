// MIT, copyright 2026 BLANK-2340. CPU tests DO NOT execute/validate CUDA.
#define FUSION_NO_MAIN
#include "softmax_cpu.cpp"
#include "bench_common.hpp"

int main() {
    try {
        const auto s=bench::summarize({4,1,3,2});
        fusion::require(s.minimum==1 && s.maximum==4 && s.median==2.5 && s.mean==2.5,"known statistics");
        fusion::require(bench::summarize({7}).median==7,"singleton statistics");
        for (const auto& v:std::vector<std::vector<double>>{{},{-1},{std::numeric_limits<double>::quiet_NaN()}}) {
            bool rejected=false; try { (void)bench::summarize(v); } catch (const std::runtime_error&) { rejected=true; }
            fusion::require(rejected,"invalid timing accepted");
        }
        for (const std::string x:{"", "0", "4097", "12x", "-1", "9999999999999999999", " 2"}) {
            bool rejected=false; try { (void)bench::integer(x,1,4096); } catch (const std::invalid_argument&) { rejected=true; }
            fusion::require(rejected,"invalid integer accepted");
        }
        char app[]="gpu_bench", r[]="5", c[]="33", w[]="0", n[]="2";
        char* args[]{app,r,c,w,n};
        const auto o=bench::parse(5,args);
        fusion::require(o.rows==5 && o.cols==33 && o.warmup==0 && o.repetitions==2,"parse contract");
        fusion::require(bench::parse(1,args).repetitions==50,"defaults");
        bool rejected=false; try { (void)bench::parse(2,args); } catch (const std::invalid_argument&) { rejected=true; }
        fusion::require(rejected,"argument count accepted");
        auto b=bench::workload(o), copy=bench::workload(o);
        fusion::require(b.x==copy.x && b.bias==copy.bias && b.lengths==copy.lengths,"seed repeatability");
        fusion::require(b.lengths==std::vector<std::size_t>{0,1,16,33,0},"prefix schedule");
        auto good=fusion::online(b); (void)bench::verify(b,good);
        auto bad=good; bad[0]=1; rejected=false;
        try { (void)bench::verify(b,bad); } catch (const std::runtime_error&) { rejected=true; }
        fusion::require(rejected,"corrupted output accepted");
        rejected=false;
        try { (void)bench::workload({0,33,0,1}); } catch (const std::runtime_error&) { rejected=true; }
        fusion::require(rejected,"invalid workload shape accepted");
        for (auto cols:{1,31,32,33,255,256,257,4095,4096}) {
            b=bench::workload({8,cols,0,1}); (void)bench::verify(b,fusion::online(b));
        }
        std::cout << "PASS host harness statistics/arguments/workload/correctness; CUDA NOT executed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
