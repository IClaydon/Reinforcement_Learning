import numpy as np
from matplotlib import pyplot as plt

human = np.loadtxt('revenue_1000_human.txt')
human = human/1000000.

random = np.loadtxt('revenue_100_rand.txt')
random = random/1000000.

mc_100 = np.loadtxt('revenue_100_exp.txt')
mc_100 = mc_100/1000000.


N_100 = np.loadtxt('../Nstep_old/revenue_100_exp.txt')
N_100 = N_100/1000000.

"""mc_400 = np.loadtxt('revenue_400_exp.txt')
mc_400 = mc_400/1000000."""

mc_100_learn = np.loadtxt('revenue_100_learn.txt')
mc_100_learn = mc_100_learn/1000000.
mc_100_learn = mc_100_learn[-100:-1]

"""mc_1000_learn = np.loadtxt('revenue_1000_learn.txt')
mc_1000_learn = mc_1000_learn/1000000.
mc_1000_learn = mc_1000_learn[-100:-1]"""

mc_1000_lft_10000 = np.loadtxt('revenue_1000_lft_10000.txt')
mc_1000_lft_10000 = mc_1000_lft_10000/1000000.
mc_1000_lft_10000 = mc_1000_lft_10000[-100:-1]

mc_1k = np.loadtxt('revenue_1000_exp.txt')
mc_1k = mc_1k/1000000.


mc_10k = np.loadtxt('revenue_10000_exp.txt')
mc_10k = mc_10k/1000000.

mc_14k = np.loadtxt('revenue_14000_exp.txt')
mc_14k = mc_14k/1000000.

mc_9k = np.loadtxt('revenue_9000_exp.txt')
mc_9k = mc_9k/1000000.

mc_4k = np.loadtxt('revenue_4000_exp.txt')
mc_4k = mc_4k/1000000.

N_1k = np.loadtxt('../Nstep_old/revenue_1000_exp.txt')
N_1k = N_1k/1000000.

N_2k = np.loadtxt('../Nstep_old/revenue_2000_exp.txt')
N_2k = N_2k/1000000.

N_100_learn = np.loadtxt('../Nstep_old/revenue_100_learn.txt')
N_100_learn = N_100_learn/1000000.
N_100_learn = N_100_learn[-100:-1]

N_1000_learn = np.loadtxt('../Nstep_old/revenue_1000_learn.txt')
N_1000_learn = N_1000_learn/1000000.
N_1000_learn = N_1000_learn[-100:-1]


bins=np.linspace(0.95,1.3,71)
#bins2=np.linspace(1.0,1.1,21)
plt.hist(human,bins=bins,histtype='step',weights=[(1/len(human))]*len(human),label='Human')
#plt.hist(random,bins=bins,histtype='step',weights=[(1/len(random))]*len(random),label='Random')
#plt.hist(mc_100,bins=bins,histtype='step',weights=[(1/len(mc_100))]*len(mc_100),linestyle=('dashed'),label='MC 100')
#plt.hist(N_100,bins=bins,histtype='step',weights=[(1/len(N_100))]*len(N_100),color='green',label='Nstep 100')
#plt.hist(mc_400,bins=bins,histtype='step',weights=[(1/len(mc_400))]*len(mc_400),label='MC 400')
#plt.hist(mc_1k,bins=bins,histtype='step',weights=[(1/len(mc_1k))]*len(mc_1k),label='MC 1000')
#plt.hist(mc_4k,bins=bins,histtype='step',weights=[(1/len(mc_4k))]*len(mc_4k),label='MC 4000')
plt.hist(mc_10k,bins=bins,histtype='step',weights=[(1/len(mc_10k))]*len(mc_10k),label='MC 10000')
plt.hist(mc_14k,bins=bins,histtype='step',weights=[(1/len(mc_14k))]*len(mc_14k),label='MC 14000')
#plt.hist(mc_9k,bins=bins,histtype='step',weights=[(1/len(mc_9k))]*len(mc_9k),label='MC 9000')
#plt.hist(N_1k,bins=bins,histtype='step',weights=[(1/len(N_1k))]*len(N_1k),color='red',label='Nstep 1000')
#plt.hist(N_2k,bins=bins,histtype='step',weights=[(1/len(N_2k))]*len(N_2k),color='red',label='Nstep 2000')

#plt.hist(mc_100_learn,bins=bins,histtype='step',weights=[(1/len(mc_100_learn))]*len(mc_100_learn),linestyle=('dashed'),label='MC 100 Learn')
#plt.hist(mc_1000_learn,bins=bins,histtype='step',weights=[(1/len(mc_1000_learn))]*len(mc_1000_learn),linestyle=('dashed'),label='MC 1000 Learn')
#plt.hist(mc_1000_lft_10000,bins=bins,histtype='step',weights=[(1/len(mc_1000_lft_10000))]*len(mc_1000_lft_10000),linestyle=('dashed'),label='MC 1000 Lft 10000')
#plt.hist(N_100_learn,bins=bins,histtype='step',weights=[(1/len(N_100_learn))]*len(N_100_learn),label='N 100 learn')
#plt.hist(N_1000_learn,bins=bins,histtype='step',weights=[(1/len(N_1000_learn))]*len(N_1000_learn),linestyle=('dashed'),label='Nstep 1000 learn')

plt.xlabel('Yearly revenue (Million)')
plt.legend(loc=2)
plt.xlim([0.95,1.3])
plt.ylim([0,0.4])
plt.savefig('histogram.png')
plt.show()
