# Gaussian Mixture Model

## Background

​	下面介绍高斯混合模型，在进行概率密度估计的时候，我们可能会遇到以下情形。

![8-1 概率密度估计](https://raw.githubusercontent.com/IshiKura-a/Machine_Learing/master/img/9_1.png)

​	这是一组一维的样本点，纵轴表示其概率密度，概率密度曲线如橙色曲线所示。在此情况下，我们当然可以用一个高斯函数去估计概率密度函数，但是这样的方法显然是不够精确的。直观来看，把概率密度曲线看成是两个高斯函数的加权平均值（如蓝线所示），更能贴合真正的概率密度曲线，即
$$
\left\{
\begin{aligned}
	&p(x)=\sum_{k=1}^K\alpha_kN(x|\mu_k,\Sigma_k) \\
	&\sum_{k=1}^K\alpha_k=1
\end{aligned}
\right.
$$
​	这就是高斯混合模型。

​	从另外一个角度来看，假设我们有一组二维的数据$(x_1,x_2)$

![9-2 2D](https://raw.githubusercontent.com/IshiKura-a/Machine_Learing/master/img/9_2.png)

​	我们引入一个隐变量(latent variable)$z$来表示样本$x$属于哪一个高斯分布，显然$z$是离散型随机变量，其概率分布为

| $\mathbf{z}$ | $\mathbf{c_1}$ | $\mathbf{c_2}$ | $\mathbf{\cdots}$ | $\mathbf{c_K}$ |
| :----------: | -------------- | -------------- | ----------------- | -------------- |
|     $p$      | $p_1$          | $p_2$          | $\cdots$          | $p_K$          |

​	因为那些边缘的样本点的归属不是很明确，所以用随机变量来描述它的归属更为方便。由此
$$
\begin{aligned}
	p(x)&=\sum_zp(x,z) \\
	&=\sum_{k=1}^Kp(x,z=c_k) \\
	&=\sum_{k=1}^Kp(z=c_k)\cdot{p(x|z=c_k)} \\
	&=\sum_{k=1}^Kp_k\cdot{N(x|\mu_k,\Sigma_k)}
\end{aligned}
$$
​	和一维的公式是一致的。

## 参数估计

​	我们用$X$表示$N$组观测数据$(x^{(1)},x^{(2)},\cdots,x^{(N)})$，把$(X,Z)$称为完整数据(Complete data)， 用$\Theta$表示参数，$\Theta=\{p_1,p_2,\cdots,p_k,\mu_1,\mu_2,\cdots,\mu_K,\Sigma_1,\Sigma_2,\cdots,\Sigma_K\}$，因此$\Theta$的极大似然估计为：
$$
\hat{\Theta}_{MLE}=\text{argmax}_\Theta\log{p(X)}
$$
​	由于样本之间相互独立，所以
$$
\begin{aligned}
	\hat{\Theta}_{MLE}&=\text{argmax}_\Theta\log\prod_{i=1}^Np(x^{(i)}) \\
	&=\text{argmax}_\Theta\sum_{i=1}^N\log{p(x^{(i)})} \\
	&=\text{argmax}_\Theta\sum_{i=1}^N\log(\sum_{k=1}^Kp_k\cdot{N(x^{(i)}|\mu_k,\Sigma_k)}) \\
	
\end{aligned}
$$
​	由于$\log$中是连加项，所以以极大似然估计法求解GMM是无法得出解析解的。因此我们需要使用近似的方法求解。对于这样的含有隐变量的混合模型，可以使用我们之前提到的EM算法。
$$
\begin{aligned}
	Q(\theta,\theta^{(t)})&=\int_z\log{p(X,Z|\theta)\cdot{p}(Z|X,\theta^{(t)})}dZ \\
	&=\sum_{z^{(1)},z^{(2)},\cdots,z^{(N)}}{\log\prod_{i=1}^Np(x^{(i)},z^{(i)}|\theta)}\cdot\prod_{i=1}^Np(z^{(i)}|x^{(i)},\theta^{(t)}) \\
	&=\sum_{z^{(1)},z^{(2)},\cdots,z^{(N)}}{\sum_{i=1}^N\log{}p(x^{(i)},z^{(i)}|\theta)}\cdot\prod_{i=1}^Np(z^{(i)}|x^{(i)},\theta^{(t)})
\end{aligned}
$$
​	单独考虑一项，有
$$
\begin{aligned}
	&\sum_{z^{(1)},z^{(2)},\cdots,z^{(N)}}\log{}p(x^{(1)},z^{(1)}|\theta)\cdot\prod_{i=1}^Np(z^{(i)}|x^{(i)},\theta^{(t)}) \\
	=&\sum_{z^{(1)},z^{(2)},\cdots,z^{(N)}}\log{}p(x^{(1)},z^{(1)}|\theta)\cdot{p}(z^{(1)}|x^{(1)},\theta^{(t)})\prod_{i=2}^Np(z^{(i)}|x^{(i)},\theta^{(t)}) \\
	=&\sum_{z_1}\log{}p(x^{(1)},z_1|\theta)\cdot{p}(z^{(1)}|x^{(1)},\theta^{(t)})\cdot\sum_{z_2,\cdots,z_N}\prod_{i=2}^Np(z^{(i)}|x^{(i)},\theta^{(t)}) \\
	=&\sum_{z^{(1)}}\log{}p(x^{(1)},z^{(1)}|\theta)\cdot{p}(z^{(1)}|x^{(1)},\theta^{(t)})\cdot\prod_{i=2}^N\sum_{z_i}p(z^{(i)}|x^{(i)},\theta^{(t)}) \\
	=&\sum_{z^{(1)}}\log{}p(x^{(1)},z^{(1)}|\theta)\cdot{p}(z^{(1)}|x^{(1)},\theta^{(t)})\cdot\prod_{i=2}^N1 \\
	=&\sum_{z^{(1)}}\log{}p(x^{(1)},z^{(1)}|\theta)\cdot{p}(z^{(1)}|x^{(1)},\theta^{(t)}) \\
\end{aligned}
$$
​	因此，
$$
\begin{aligned}
	Q(\theta,\theta^{(t)})&=\sum_{i=1}^N\sum_{z^{(i)}}\log{p(x^{(i)},z^{(i)}|\theta)}\cdot{p(z^{(i)}|x^{(i)},\theta^{(t)})} \\
	&=\sum_{i=1}^N\sum_{z^{(i)}}\log({p_{z^{(i)}}}\cdot{N(x^{(i)}|\mu_{z^{(i)}},\Sigma_{z^{(i)}})})\cdot{\dfrac{{p_{z^{(i)}}^{(t)}}\cdot{N(x^{(i)}|\mu_{z^{(i)}}^{(t)},\Sigma_{z^{(i)}}^{(t)})}}{\sum_{k=1}^Kp_k^{(t)}N(x^{(i)}|\mu_k^{(t)},\Sigma_k^{(t)})}} \\
	&=\sum_{z_i}\sum_{i=1}^N\log({p_{z^{(i)}}}\cdot{N(x_i|\mu_{z^{(i)}},\Sigma_{z^{(i)}})})\cdot {p(z^{(i)}|x_i,\theta^{(t)})} \\
	&=\sum_{k=1}^K\sum_{i=1}^N\log({p_{k}}\cdot{N(x^{(i)}|\mu_{k},\Sigma_{k})})\cdot {p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})} \\
	&=\sum_{k=1}^K\sum_{i=1}^N(\log{p_{k}}+\log{N(x^{(i)}|\mu_{k},\Sigma_{k})})\cdot {p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}
\end{aligned}
$$
​	接下来求解$p^{(t+1)}=(p_1^{(t+1)},p_2^{(t+1)},\cdots,p_K^{(t+1)})$
$$
\begin{aligned}
	p_k^{(t+1)}&=\text{argmax}_{p_k}\sum_{k=1}^K\sum_{i=1}^N\log{p_{k}}\cdot {p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}
\end{aligned}
$$
​	因为存在约束
$$
\sum_{k=1}^Kp_k=1
$$
​	所以，我们使用拉格朗日乘子法来求解。定义
$$
\mathcal{L}(p,\lambda)=\sum_{k=1}^K\sum_{i=1}^N\log{p_{k}}\cdot {p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}+\lambda(\sum_{k=1}^Kp_k-1)
$$
​	对$p_k$求偏导
$$
\dfrac{\partial\mathcal{L}}{\partial{p_k}}=\sum_{i=1}^N\frac{1}{p_k}\cdot{p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}+\lambda
$$
​	令其为0，则
$$
\sum_{i=1}^N{p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}+p_k\lambda=0
$$
​	对$k$进行求和，有
$$
\sum_{i=1}^N\sum_{k=1}^K{p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}+\sum_{k=1}^Kp_k\lambda=0
$$
​	因为$p(z_i|x_i,\theta^{(t)})$是概率密度函数，所以
$$
\lambda=-\sum_{i=1}^N1=-N
$$
​	因此，
$$
p_k^{(t+1)}=\frac{1}{N}\sum_{i=1}^N{p(z^{(i)}=c_k|x^{(i)},\theta^{(t)})}
$$
​	