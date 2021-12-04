1. natural signal 都是可压缩的，包括语音信号，图像，steaming television。.

2. sparsity: 
+  x = /Psi * s  
  where x :true signal , /Psi：universal basis,e.g. DFT， s:sparse 
  也就是把原始的信号表示到在其他basis 上的
  比如，jpeg就是用fast fourier transform basis, 也就是选了一个我们想要压缩的basis
  这时x是像素的向量。x 可以表示一个图片比如一个猫咪
  

3.  compressed sensing
+ 如果我们只有5%的图片信息 ，我们可以恢复出原始的图片吗? 也就是说，我们现在有一个massively down sampled of image, 推断出稀疏向量s 来恢复出真实的图像吗？

![databook-pdf 11-23-2021 1-57-46 PM](https://user-images.githubusercontent.com/69283174/142965098-72057134-edaa-4e04-9c25-514825c645ac.png)
从上面的图可以看出，我们一开始就得到的第一张图片是有不完整信息的图片(y)。 这个不完整的图片是通过完整图片上采样得到的。
换句话说，比如真实的图片信息x假如说10000维度的，这时候我们的y可能是1000或者更小维度。

y 是 down undersampled picture， 其中图中的花体F 指的是 矩阵/Psi，也就是basis
我们得到一个under-determined system: y = C * x = C * /Phi * s = /Theta * s
where y is measurement, s  which is solving for, is consistence with y， x 和 s 表示相同的信息，因为他们只进行了一个变化。
+ 随机采样

 
<img src="https://user-images.githubusercontent.com/69283174/144697599-a690861e-ec1f-4111-b1e1-cf68d752bfeb.png" style="zoom:10%" />

  
C 相当于  x = /Psi * s  的矩阵中，抽出想要的列

5. When it works:
+ incoherence: C to be incoherence  wrt.\Psi. In other words, C cannot be too parallel to /Psi. ( searching features)
+ RIP（有限等距性质）实际是保证了观测矩阵不会把两个不同的稀疏信号映射到同一个集合中（保证原空间到稀疏空间的唯一对应映射关系）。


6. 06年的东西，到现在已经十年了，已经发展出了很多算法，原来的基于l1 minimization的BP算法很慢，现在都是快速算法，而且求解算法也从纯优化方面扩展到了estimation方面，有很多基于贝叶斯估计的方法了，目前最火的也是Donoho他们组搞得AMP算法，是用Graph model里面的message passing算法通过近似求解MMSE（MAP）解。在测量矩阵方面，也已经设计出了各种矩阵，除了i.i.d. Gaussian的矩阵还有很多正交的矩阵，比如partial random DFT/DCT 矩阵。对信号的要求也从稀疏变成了存在某种结构，比如low rank，group sparse等等。
