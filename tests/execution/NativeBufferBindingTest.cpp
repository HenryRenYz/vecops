// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <array>
#include "vecops/execution/SourceKernelPlan.h"

namespace {
using namespace vecops;
using namespace vecops::execution;
using namespace vecops::tensor;
struct Owner {
  unsigned refs=1;
  alignas(64) std::array<std::byte,512> storage{};
  static void retain(void* p) { ++static_cast<Owner*>(p)->refs; }
  static void release(void* p) { --static_cast<Owner*>(p)->refs; }
  VecopsBufferOwnerV1 token() { return {this,retain,release}; }
};
struct Deferred {
  std::array<float,4> input{1,2,3,4}, output{};
  int64_t size=4, stride=1;
  std::array<VecopsValue,2> values{};
  VecopsExecutionContext context{};
  VecopsCall call{};
  VecopsPreparedKernelV1 abi{};
  Deferred() {
    for(unsigned i=0;i<2;++i) {
      values[i].struct_size=sizeof(VecopsValue);values[i].kind=VECOPS_VALUE_TENSOR;
      values[i].value.tensor={sizeof(VecopsTensorView),i?output.data():input.data(),0,VECOPS_DEVICE_CPU,0,
        VECOPS_DTYPE_F32,1,&size,&stride,i?VECOPS_TENSOR_WRITE:VECOPS_TENSOR_READ};
    }
    context.struct_size=sizeof(context);context.defer_buffer_binding=1;
    call={sizeof(call),2,values.data(),nullptr,0,&context};
    abi.struct_size=sizeof(abi);
    auto build=[](auto& b,auto x,auto y) {
      auto scratch=b.template tensor<float>("scratch",make_shape(meta::cint<4>));
      b.template grid<1>("define",0,4,1,kernel_bind(x,overwrite(scratch)),
        [](auto,auto work,auto x,auto scratch) {scratch(work.begin())=x(work.begin())*2;});
      b.template grid<1>("read",0,4,1,kernel_bind(read(scratch),y),
        [](auto,auto work,auto scratch,auto y) {y(work.begin())=scratch(work.begin())+1;});
    };
    source_plan_details::launch<1>(&call,&abi,build,make_tensor(input.data(),make_shape(meta::cint<4>)),
      make_tensor(output.data(),make_shape(meta::cint<4>)));
  }
  ~Deferred() {abi.destroy(abi.context);}
  VecopsKernelBufferBindingV1 mapping(Owner& owner) {
    const auto& r=abi.buffers[0];
    return {r.name,owner.storage.data(),r.replica_stride*r.replicas,r.replica_stride};
  }
};
TEST(NativeBufferBinding, MetadataPrepareAllocatesNoWorkspaceAndSurvivesMappingSwitch) {
  Owner first,second;
  Deferred f;
  ASSERT_EQ(f.abi.buffer_count,1u);
  const auto& r=f.abi.buffers[0];
  EXPECT_STREQ(r.name,"slot2/scratch"); EXPECT_EQ(r.bytes,16u);
  EXPECT_GE(r.alignment,64u); EXPECT_EQ(r.replicas,1u); EXPECT_EQ(r.retained,0u);
  ASSERT_EQ(r.lifetime_count,1u); EXPECT_EQ(r.lifetimes[0].begin,0u);EXPECT_EQ(r.lifetimes[0].end,2u);
  EXPECT_NE(f.abi.replay(f.abi.context,nullptr),VECOPS_STATUS_OK);
  auto mapping=f.mapping(first);auto token=first.token();
  EXPECT_EQ(f.abi.activate_buffers(f.abi.context,&mapping,1,&token,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(first.refs,2u);
  EXPECT_EQ(f.abi.bind(f.abi.context,&f.call,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(f.abi.replay(f.abi.context,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(f.output,(std::array<float,4>{3,5,7,9}));
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,&mapping,1,&token,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(first.refs,2u);
  EXPECT_EQ(f.abi.suspend_buffers(f.abi.context,nullptr),VECOPS_STATUS_OK);EXPECT_EQ(first.refs,1u);
  EXPECT_NE(f.abi.replay(f.abi.context,nullptr),VECOPS_STATUS_OK);
  mapping=f.mapping(second);token=second.token();
  EXPECT_EQ(f.abi.activate_buffers(f.abi.context,&mapping,1,&token,nullptr),VECOPS_STATUS_OK);
  f.input.fill(17);
  EXPECT_EQ(f.abi.bind(f.abi.context,&f.call,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(f.abi.replay(f.abi.context,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(f.output,(std::array<float,4>{35,35,35,35}));
  EXPECT_EQ(f.abi.suspend_buffers(f.abi.context,nullptr),VECOPS_STATUS_OK);EXPECT_EQ(second.refs,1u);
}
TEST(NativeBufferBinding, ActivationValidatesNamesStrideAlignmentSizeAndOwnerBeforeFactories) {
  Deferred f;Owner owner;auto token=owner.token();
  const auto good=f.mapping(owner);
  auto bad=good;bad.name="unknown";
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,&bad,1,&token,nullptr),VECOPS_STATUS_OK);
  bad=good;bad.replica_stride++;
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,&bad,1,&token,nullptr),VECOPS_STATUS_OK);
  bad=good;bad.address=owner.storage.data()+1;
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,&bad,1,&token,nullptr),VECOPS_STATUS_OK);
  bad=good;bad.bytes=1;
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,&bad,1,&token,nullptr),VECOPS_STATUS_OK);
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,nullptr,0,&token,nullptr),VECOPS_STATUS_OK);
  EXPECT_NE(f.abi.activate_buffers(f.abi.context,&good,1,nullptr,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(owner.refs,1u);
}
TEST(NativeBufferBinding, SuspensionRejectsActiveResidentUse) {
  Deferred f;Owner owner;auto token=owner.token();auto mapping=f.mapping(owner);
  ASSERT_EQ(f.abi.activate_buffers(f.abi.context,&mapping,1,&token,nullptr),VECOPS_STATUS_OK);
  ASSERT_EQ(f.abi.bind(f.abi.context,&f.call,nullptr),VECOPS_STATUS_OK);
  ASSERT_EQ(f.abi.enter(f.abi.context,nullptr),VECOPS_STATUS_OK);
  EXPECT_NE(f.abi.suspend_buffers(f.abi.context,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(owner.refs,2u);
  f.abi.leave(f.abi.context,0);
  EXPECT_EQ(f.abi.suspend_buffers(f.abi.context,nullptr),VECOPS_STATUS_OK);
  EXPECT_EQ(owner.refs,1u);
}
TEST(NativeBufferBinding, LiveBindingsCannotOverlapButConsumedBuffersCanReuseStorage) {
  KernelBuilder live(1);
  auto a=live.tensor<int>("a",make_shape(meta::cint<4>));
  auto b=live.tensor<int>("b",make_shape(meta::cint<4>));
  live.single("overlap",kernel_bind(a,b),[](auto,auto) {});
  auto plan=live.finish();auto owner=std::make_shared<Owner>();
  auto address=owner->storage.data();
  auto bindings=std::vector<KernelBufferBinding>{{"slot0/a",address,64,64,owner},{"slot1/b",address,64,64,owner}};
  EXPECT_THROW((KernelInstance{plan,nullptr,bindings}),std::invalid_argument);
  KernelBuilder serial(1);
  a=serial.tensor<int>("a",make_shape(meta::cint<4>));
  b=serial.tensor<int>("b",make_shape(meta::cint<4>));
  const auto stage=serial.single("a",kernel_bind(overwrite(a)),[](auto a) {for(int i=0;i<4;++i)a(i)=1;});
  serial.consume(a,stage);
  serial.single("b",kernel_bind(overwrite(b)),[](auto b) {for(int i=0;i<4;++i)b(i)=2;});
  KernelInstance instance(serial.finish(),nullptr,std::move(bindings));
  instance.replay();
  EXPECT_EQ(*reinterpret_cast<int*>(address),2);
}
TEST(NativeBufferBinding, RepeatedDebugLabelsDescribeDistinctStableSlots) {
  KernelBuilder builder(1);
  auto large=builder.tensor<int>("normalized",make_shape(meta::cint<4>));
  auto small=builder.tensor<int>("normalized",make_shape(meta::cint<2>));
  int result=0;
  auto output=builder.input(make_tensor(&result,make_shape(meta::cint<1>)));
  builder.single("define",kernel_bind(overwrite(large),overwrite(small)),[](auto a,auto b) {
    for(int i=0;i<4;++i) a(i)=i+1;
    for(int i=0;i<2;++i) b(i)=10+i;
  });
  builder.single("read",kernel_bind(read(large),read(small),output),[](auto a,auto b,auto out) {
    out(0)=a(0)+a(3)+b(0)+b(1);
  });
  auto plan=builder.finish();
  const auto& requirements=plan.buffer_plan().allocations;
  ASSERT_EQ(requirements.size(),2u);
  EXPECT_EQ(requirements[0].canonical_path,"slot0/normalized");
  EXPECT_EQ(requirements[1].canonical_path,"slot1/normalized");
  EXPECT_EQ(requirements[0].request.bytes,16);
  EXPECT_EQ(requirements[1].request.bytes,8);
  auto owner=std::make_shared<Owner>();
  KernelInstance instance(plan,nullptr,std::vector<KernelBufferBinding>{
    {requirements[1].canonical_path,owner->storage.data()+64,64,64,owner},
    {requirements[0].canonical_path,owner->storage.data(),64,64,owner}});
  instance.set(output,&result);
  instance.replay();
  EXPECT_EQ(result,26);
}
} // namespace
