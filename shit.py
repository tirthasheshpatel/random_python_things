from math import *

n = int(input())

s = input()
t = input()

res = "c"
i = 1

n_a = n_b = n_c = n

l = ["a", "b", "c"]

cnt = {"a": n_a, "b": n_b, "c": n_c - 1}

j = 0
k = 0

while i != 3 * n:
    k += 1
    if cnt[l[j]] and res[i - 1] + l[j] not in [s, t]:
        res += l[j]
        cnt[l[j]] -= 1
        i += 1
        k = 0
    elif k == 4:
        break
    j = (j + 1) % 3

# print(res)

if len(res) == 3 * n:
    print("YES\n" + res)
else:
    print("NO")
