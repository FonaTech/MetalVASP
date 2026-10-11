// Optional Instruments target: public Core ML only, no Python/exec launch.
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <os/signpost.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>

int main(int argc, char **argv) {
 std::printf("native CoreML Mach-O entered main pid=%ld\n",(long)getpid());std::fflush(stdout);
 if(argc==2 && std::strcmp(argv[1],"--launch-check")==0) return 0;
 @autoreleasepool {
  if(argc!=6) {
   std::fprintf(stderr,"usage: native-coreml-trace compiled.mlmodelc inputs.fp32 references.fp64 result.json repeats\n");
   return 2;
  }
  const size_t channels=512,positions=1024,count=channels*positions,variants=3;
  const int repeats=std::atoi(argv[5]); if(repeats<=0 || repeats>10000) return 2;
  NSString *output=[NSString stringWithUTF8String:argv[4]];
  if([[NSFileManager defaultManager] fileExistsAtPath:output]) return 2;
  NSString *identifier=[NSString stringWithFormat:@"MetalVASP.ANEProbe.Conv512.FP16.Native.PID%ld",(long)getpid()];
  NSMutableDictionary *report=[@{@"schema":@1,@"pid":@(getpid()),@"status":@"running",
    @"correlation_identifier":identifier,
    @"compiled_model":[NSString stringWithUTF8String:argv[1]],@"repeats":@(repeats),
    @"performance_claim_accepted":@NO,@"vasp_integration":@NO,
    @"observed_ane_execution_trace":[NSNull null]} mutableCopy];
  auto save=[&](){NSError *error=nil;NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    return json && [json writeToFile:output options:NSDataWritingAtomic error:&error];};
  auto fail=[&](NSString *message){report[@"status"]=@"failed";report[@"error"]=message ?: @"unknown failure";save();
    std::fprintf(stderr,"native CoreML target failed: %s\n",[message UTF8String]);return 1;};
  if(!save()) return 2;
  NSData *input=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
  NSData *reference=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[3]]];
  if(input.length!=variants*count*sizeof(float) || reference.length!=variants*count*sizeof(double))
    return fail(@"prepared raw tensors have unexpected sizes");
  MLModelConfiguration *configuration=[MLModelConfiguration new];configuration.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
  configuration.modelDisplayName=identifier;
  os_log_t predictionLog=os_log_create("org.metalvasp.aneprobe","prediction");
  NSError *error=nil;
  MLModel *model=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
    configuration:configuration error:&error];
  if(!model) return fail(error.localizedDescription);
  MLMultiArray *array=[[MLMultiArray alloc] initWithShape:@[@1,@512,@1,@1024] dataType:MLMultiArrayDataTypeFloat32 error:&error];
  if(!array || array.count!=count || [array.strides[1] unsignedLongLongValue]!=positions ||
     [array.strides[3] unsignedLongLongValue]!=1) return fail(error.localizedDescription ?: @"input allocation mismatch");
  MLDictionaryFeatureProvider *provider=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":array} error:&error];
  if(!provider) return fail(error.localizedDescription);
  report[@"prediction_start_unix_seconds"]=@([NSDate date].timeIntervalSince1970);
  NSMutableArray *intervals=[NSMutableArray arrayWithCapacity:repeats];double maxError=0;
  std::printf("native CoreML trace target pid=%ld prediction phase\n",(long)getpid());std::fflush(stdout);
  for(int iteration=0;iteration<repeats;++iteration) {
   @autoreleasepool {
    const size_t variant=size_t(iteration)%variants;
    std::memcpy(array.dataPointer,static_cast<const float *>(input.bytes)+variant*count,count*sizeof(float));
    double begin=[NSDate date].timeIntervalSince1970;
    os_signpost_id_t predictionID=os_signpost_id_generate(predictionLog);
    os_signpost_interval_begin(predictionLog,predictionID,"CoreMLPrediction","iteration=%d input=%zu",iteration,variant);
    id<MLFeatureProvider> prediction=[model predictionFromFeatures:provider error:&error];
    os_signpost_interval_end(predictionLog,predictionID,"CoreMLPrediction","iteration=%d",iteration);
    double end=[NSDate date].timeIntervalSince1970;
    if(!prediction) return fail(error.localizedDescription);
    MLMultiArray *actual=[prediction featureValueForName:@"result"].multiArrayValue;
    if(!actual || actual.dataType!=MLMultiArrayDataTypeFloat32 || ![actual.shape isEqualToArray:@[@1,@512,@1,@1024]])
      return fail(@"prediction has unexpected dtype/shape");
    auto values=static_cast<const float *>(actual.dataPointer);auto ref=static_cast<const double *>(reference.bytes)+variant*count;
    const size_t cStride=[actual.strides[1] unsignedLongLongValue],pStride=[actual.strides[3] unsignedLongLongValue];
    for(size_t c=0;c<channels;++c) for(size_t p=0;p<positions;++p) {
      double value=values[c*cStride+p*pStride];if(!std::isfinite(value)) return fail(@"nonfinite output");
      maxError=std::max(maxError,std::abs(value-ref[c*positions+p]));
    }
    [intervals addObject:@{@"index":@(iteration),@"input_index":@(variant),@"start_unix_seconds":@(begin),@"end_unix_seconds":@(end)}];
   }
  }
  report[@"prediction_end_unix_seconds"]=@([NSDate date].timeIntervalSince1970);
  report[@"prediction_intervals"]=intervals;report[@"status"]=@"prediction-completed";
  report[@"completed_prediction_count"]=@(repeats);report[@"max_absolute_error"]=@(maxError);
  report[@"fp16_probe_gate_passed"]=@(maxError<=0.005);report[@"ds_precision_gate_passed"]=@(maxError<=1e-10);
  if(!save()) return 2;
  std::printf("native prediction completed count=%d error=%.12g\n",repeats,maxError);return 0;
 }
}
