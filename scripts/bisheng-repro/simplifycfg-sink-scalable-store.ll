; ModuleID = '/tmp/vD.red.bc'

define void @tile2d_gemm_scalable_probe(ptr %0, ptr %1, i1 %2, i64 %.idx) {
  br i1 %2, label %4, label %5

common.ret:                                       ; preds = %5, %4
  ret void

4:                                                ; preds = %3
  store <vscale x 4 x float> zeroinitializer, ptr %0, align 1
  br label %common.ret

5:                                                ; preds = %3
  %6 = getelementptr i8, ptr %1, i64 %.idx
  store <vscale x 4 x float> zeroinitializer, ptr %6, align 1
  br label %common.ret
}
