import pandas as pd
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a[:, :-1]

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, np.nan],  # np.nan表示NA
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
# print("data.keys()", data.keys())
f = pd.DataFrame(data)
# k = f.iloc[[1, 3, 6], :]
# print("k:", k)
v = f.iloc[:, 1].value_counts().keys()[1]
vv = f[f.iloc[:, 1] == 2001]
# print("f", f)
k = vv.index
kk =f.index
s1 = set(k)
s2 = set(kk)
s = s1 & s2
n = len(s)
print("s:", s)
print("n:", n)

print("k:", k)
# print("vv:", vv)
# k = vv.loc[[1, 2, 3], :]
# print("k:", k)
a = f.loc[f["pop"] > 2]
b = f[f["pop"] > 2]
# print("a:\n", a, "\n")
# print("b:\n", b, "\n")
# print("a.type:\n", type(a), "\n")
# print("b.type:\n", type(b), "\n")


dt = {}
dt["a"] = (1, 2, 5)
dt["b"] = (4, 3, 9)
dt["c"] = (3, 4, 7)
res = sorted(dt.items(), key=lambda x: x[1][0], reverse=True)  # 按信息增益排名
a = res[0][0]
b = res[0][1][2]
c = res[0][1][2]

# print("b:", b, "\n")
# print("c:", c, "\n")
# for tmp in dt:
#         print("tmp:", tmp, "\n")
#         # print("tmp[0]:", tmp[0])
#         # print("tmp[1]:", tmp[1])


fc = f['pop']
fcv = fc.values
v1 = fcv[:-1]
v2 = fcv[1:]
v3 = (v1 + v2) / 2
# for v in v3:
#         print("v:", v, "\n")
fv = f.values
ll = len(fc)
item = fc[2]
item2 = f.iloc[0, 2]
# print("fv:", fv, "\n")
# print("fv.type:", type(fv), "\n")
# print("fc:", fc, "\n")
# print("fc.value:", fc.values, "\n")
# print("fc.value.type:", type(fc.values), "\n")
# print("fc.value.shape:", fc.values.shape, "\n")
# print("fc.value[:-1]", fcv[:-1], "\n")
# print("ll:", ll, "\n")

# print("f.type:", type(f), "\n")
# print("data_label.type:", type(data_label), "\n")
# print("ll.type:", type(ll), "\n")
# print("data_label:", data_label, "\n")
# print("item:", item, "\n")
# print("item2:", item2, "\n")
# print("item2.type:", type(item2), "\n")
