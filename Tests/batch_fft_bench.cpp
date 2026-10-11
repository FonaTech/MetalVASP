#include "metal_backend.h"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
int main(){
 metal_backend *b=nullptr;metal_backend_create(&b,1);if(!metal_backend_gpu_enabled(b)) return 86;
 using Clock=std::chrono::steady_clock;
 for(size_t side:{12u,24u,32u}) for(size_t count:{4u,16u}) {
  size_t len=side*side*side,stride=len+7;std::vector<metal_complex64> input(count*stride),out(input);
  for(size_t i=0;i<input.size();++i) input[i]={std::sin(0.11*i),std::cos(0.07*i)};
  metal_backend_zfft3d_batch_gpu(b,side,side,side,count,stride,input.data(),out.data(),-1,0);
  auto scalar=[&](){for(size_t j=0;j<count;++j) if(metal_backend_zfft3d_batch_gpu(b,side,side,side,1,len,input.data()+j*stride,out.data()+j*stride,-1,0)) std::exit(1);};
  scalar();double times[2];uint64_t submits[2];const int repeats=5;
  for(int mode=0;mode<2;++mode){auto begin=Clock::now();auto n=metal_backend_command_count(b);
   for(int r=0;r<repeats;++r) {if(mode==0) scalar();else if(metal_backend_zfft3d_batch_gpu(b,side,side,side,count,stride,input.data(),out.data(),-1,0)) std::exit(2);}
   times[mode]=std::chrono::duration<double>(Clock::now()-begin).count()/repeats;submits[mode]=(metal_backend_command_count(b)-n)/repeats;
  }
  std::printf("{\"side\":%zu,\"batch\":%zu,\"scalar_seconds\":%.9f,\"batch_seconds\":%.9f,\"speedup\":%.5f,\"scalar_submissions\":%llu,\"batch_submissions\":%llu}\n",side,count,times[0],times[1],times[0]/times[1],(unsigned long long)submits[0],(unsigned long long)submits[1]);
 }
 metal_backend_destroy(b);
}
