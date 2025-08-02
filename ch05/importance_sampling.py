import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# 기댓값의 참값 계산
e = np.sum(x * pi)
print("참값(E_pi[x]):", e)

# 몬테카를로법으로 계산
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)

# 몬테카를로법으로 계산한 기댓값
mean = np.mean(samples)
var = np.var(samples)
print("몬테카를로법(E_pi[x]):", mean)
print("몬테카를로법(Var_pi[x]):", var)


# b = np.array([1/3, 1/3, 1/3])
b = np.array([0.2, 0.2, 0.6])
n = 1000
samples = []
for _ in range(n):
    idx = np.arange(len(x))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(s * rho)

mean = np.mean(samples)
var = np.var(samples)
print("중요도 샘플링(E_pi[x]):", mean)
print("중요도 샘플링(Var_pi[x]):", var)
