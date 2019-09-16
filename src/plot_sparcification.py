from matplotlib import pyplot as plt

c0 = 100
median = 0.5

ptd = {}
def pruning_treshold(t, sf):
    if (t,sf) in ptd.keys():
        return ptd[(t,sf)]
    else:
        pt = median * (sf ** sparcity(t, sf))
        ptd[(t,sf)] = pt
        return pt
    
sd = {}
def sparcity(t, sf):
    if (t,sf) in sd.keys():
        return sd[(t,sf)]
    else:
        s = weight_count(0, sf) / weight_count(t, sf)
        sd[(t,sf)] = s
        return s

wcd = {}
def weight_count(t, sf):
    if t == 0:
        return c0
    elif (t,sf) in wcd.keys():
        return wcd[(t,sf)]
    else:
        wc = weight_count(t-1, sf) - int(weight_count(t-1, sf) * pruning_treshold(t-1, sf))
        wcd[(t,sf)] = wc
        return wc

values = [[weight_count(t, sfp/100) for sfp in range(0,100,10)] for t in range(0,25) ]

fig, ax = plt.subplots()
ax.plot(values)

ax.set(xlabel='Pruning Step', ylabel='Weight Count',
       title='Convergence of Weight Count')
ax.grid()

#fig.savefig("test.png")
plt.show()