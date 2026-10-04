import vecops.language as T
import torch

@T.jit(out_idx=[1])
def the_op(N):
    @T.template
    def kernel(a: T.Tensor((N,),dtype="float32"), o: T.Tensor((N,),dtype="float32")):
        x = T.load(a)
        y = T.exp(x)
        y /= (y + 1)
        T.store(o, y)
    return kernel

@T.jit(out_idx=[-1])
def mm(M, N, K, BM=64, BN=64):
    @T.template
    def kernel(
        a: T.Tensor((M, K), dtype="bfloat16"),
        b: T.Tensor((K, N), dtype="bfloat16"),
        c: T.Tensor((M, N), dtype="bfloat16")
    ):
        for i,j in T.grid(T.ceildiv(M, BM), T.ceildiv(N, BN)):
            ta = T.load(a[i:i+BM])
            tb = T.load(b[:, j:j+BN])
            tc = T.dot(ta, tb)
            T.store(c[i:i+BM,j:j+BN], tc)
    return kernel

op = the_op(1023)

a = torch.rand((1023,), dtype=torch.float32)
b = op(a)
print(f"b = {b}")

op = mm(1024, 1023, 256)
a = torch.randn((1024, 256), dtype=torch.bfloat16)
b = torch.randn((256, 1023), dtype=torch.bfloat16)
c = op(a, b)
print(f"c = {c}")