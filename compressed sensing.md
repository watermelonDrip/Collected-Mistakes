1. natural signal 都是可压缩的，包括语音信号，图像，steaming television。.

2. sparsity: 
+  x = /Psi * s  
  where x :true signal , /Psi：universal basis,e.g. DFT， s:sparse 
  也就是把原始的信号表示到在其他basis 上的
  比如，jpeg就是用fast fourier transform basis, 也就是选了一个我们想要压缩的basis

3.  compressed sensing
+ 如果我们只有5%的图片信息 ，我们可以恢复出原始的图片吗? 也就是说，我们现在有一个massively down sampled of image, 推断出稀疏向量s 来恢复出真实的图像吗？

![databook-pdf 11-23-2021 1-57-46 PM](https://user-images.githubusercontent.com/69283174/142965098-72057134-edaa-4e04-9c25-514825c645ac.png)
y 是 down undersampled picture， 其中图中的花体F 指的是 矩阵/Psi，也就是basis
我们得到一个under-determined system: y = C * x = C * /Phi * s = /Theta * s
where y is measurement, s  which is solving for, is consistence with y

4. code
