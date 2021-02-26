############################ IMPORTS #############################
import numpy as np
import pandas as pd
import pickle
import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from bltools import letter_subplots
# make figures better:
font = {'weight':'normal','size':11}
rc('font',**{'family':'serif','serif':['Palatino']})
rc('figure', figsize=(5.5,3.5))
rc('text', usetex=True)
#rc('xtick.major', pad=10) # xticks too close to border!
rc('xtick', labelsize=9)
rc('ytick', labelsize=8)
rc('legend',**{'fontsize':9})

import warnings
warnings.filterwarnings('ignore')
####################################################################

#    get average satisfation for every population
#        for each transparency level, for each voting system
V_systems = ['general','ranked','approval']
V_systems2label = {'general':"Plurality",'ranked':"Ranked-Choice",'approval':"Approval"}
happ_dist_files = {}
#uncomment once all voting systems are run
for V in V_systems:
    happ_dist_files[V] = glob.glob('ElectoralProcesses/{}/*'.format(V))

transparencies = [1,2,3,4,5,6,7]
happ_avgs = {V:{T:[] for T in transparencies} for V in happ_dist_files}
full_dists = {V:{T:[] for T in transparencies} for V in happ_dist_files}
allhappavgs = []
allhapps = []
for Vsys in happ_dist_files:
    for t in transparencies:
        popnum = 0
        print('{}_T{:02}_'.format(Vsys,t))
        for fname in [f for f in happ_dist_files[Vsys] if '_T{:02}_'.format(t) in f]:
            with open(fname,'rb') as file:
                diss_dist = np.array(pickle.load(file))
                # calculate happines from dissatisfaction:
                happ_dist = [(2-diss)/2 for diss in diss_dist]
                #happ_dist = [(2-diss) for diss in diss_dist]
                # store average happiness & std for every population:
                happ_mean = np.mean(happ_dist)
                happ_avgs[Vsys][t].append(happ_mean)
                allhappavgs.append(happ_mean)

                # store every happiness score for every population:
                full_dists[Vsys][t] += happ_dist
                allhapps += happ_dist
                popnum += 1


print(popnum)
minavg = np.min(allhappavgs)
maxavg = np.max(allhappavgs)
minhapp = np.min(allhapps)
maxhapp = np.max(allhapps)


cpairs = [('darkorange','gold'),('crimson','plum'),('darkslategrey','paleturquoise')]
# Plot candidate transparency vs avg happiness for each voting system:
if input('Would you like to plot transparency vs avg happiness? (y/Y) ').upper() == 'Y':
    plt.figure()
    for i,V in enumerate(happ_avgs):
        t_avgs = np.array([np.mean(happ_avgs[V][t]) for t in transparencies])
        # computing the standard error instead of std because we're concerned with the
        # range of values the mean of the means can take on
        t_std_err_ci = 1.96*(np.array([np.std(happ_avgs[V][t])for t in transparencies])/np.sqrt(popnum))
        plt.scatter(transparencies,t_avgs,
                    marker='x',color=cpairs[i][0],alpha=0.7,zorder=4)
        plt.plot(transparencies,t_avgs,
                 color=cpairs[i][0],alpha=0.25,
                 label='{}'.format(V_systems2label[V]))

        plt.fill_between(transparencies, t_avgs + t_std_err_ci, t_avgs - t_std_err_ci,
                         alpha=0.2,
                         color=cpairs[i][1])
        #                 [u+1.96*s for u,s in zip(t_avgs,t_std)],
        #                 alpha=0.4,color=cpairs[i][1])
        #plt.fill_between(transparencies,t_avgs,[u-1.96*s for u,s in zip(t_avgs,t_std)],
        #                 alpha=0.4,color=cpairs[i][1],
        #                 label='95\% CI for {}'.format(V_systems2label[V]))
    plt.ylim(np.min(t_avgs)-np.max(t_std_err_ci)-0.01,np.max(t_avgs)+np.max(t_std_err_ci)+0.005)
    plt.ylabel(r"Avg. Satisfaction")
    plt.xlabel('Opinion Transparency')
    plt.legend(loc='upper left',frameon=True)
    #plt.title('End-of-Election Happiness vs. Candidate Transparency \nfor each Voting System')
    plt.savefig('figs/new/avghapp_transparency.pdf',bbox_inches='tight')
    plt.clf()


# Plot full distributions of happ scores of all populations for each V sys
# One plot for every transparency :O
if input('Would you like to plot the full distributions for all happiness? (y/Y) ').upper() == 'Y':
    fig, axarr = plt.subplots(3,1, sharex=True)
<<<<<<< HEAD:final_project-showerte-dberenberg/src/analyze_elections.py
<<<<<<< HEAD:final_project-showerte-dberenberg/src/analyze_elections.py
    letter_subplots(axarr, xoffset=3*[-0.08])
=======
    letter_subplots(axarr, xoffset=3*[-0.05])
>>>>>>> 74483dd... fixedtransparency:final_project/src/analyze_elections.py
=======
    letter_subplots(axarr, xoffset=3*[-0.08])
>>>>>>> 9016c20... fixedtransparency:final_project/src/analyze_elections.py
    axarr[0].set_xlim(minavg - 0.01,maxavg + 0.01)
    axarr[2].set_xlabel("Avg. Satisfaction with elected candidate per population")
    for (ax,t) in zip(axarr, [1,4,7]):
        for i,V in enumerate(full_dists):
<<<<<<< HEAD:final_project-showerte-dberenberg/src/analyze_elections.py
<<<<<<< HEAD:final_project-showerte-dberenberg/src/analyze_elections.py
            ax.hist(happ_avgs[V][t],bins=10,
=======
            ax.hist(happ_avgs[V][t],bins='auto',
>>>>>>> 74483dd... fixedtransparency:final_project/src/analyze_elections.py
=======
            ax.hist(happ_avgs[V][t],bins=10,
>>>>>>> 9016c20... fixedtransparency:final_project/src/analyze_elections.py
                    color=cpairs[i][1], histtype='step',
                    label='{}'.format(V_systems2label[V])) #round(np.mean(full_dists[V][t]),2)))
            ax.set_ylabel('Count')
        #ax.set_title('$T = {}$'.format(t))
        ax.text(0.99,0.05, "$T={}$".format(t), transform=ax.transAxes, ha='right')
    axarr[2].legend(frameon=True, loc='upper left')
    plt.savefig('figs/new/satisfaction-hist_avgperpop.pdf'.format(t),bbox_inches='tight')
    plt.clf()


    fig, axarr = plt.subplots(3,1, sharex=True)
    letter_subplots(axarr, xoffset=3*[-0.05])
    axarr[0].set_xlim(0.45,0.95)
    axarr[2].set_xlabel("Satisfaction with elected candidate per person")
    for (ax,t) in zip(axarr, [1,4,7]):
        for i,V in enumerate(full_dists):
            ax.hist(full_dists[V][t],bins='auto',
                    color=cpairs[i][1], histtype='step',
                    label='{}'.format(V_systems2label[V])) #round(np.mean(full_dists[V][t]),2)))
            ax.set_ylabel('Count')
        #ax.set_title('$T = {}$'.format(t))
        ax.text(0.99,0.05, "$T={}$".format(t), transform=ax.transAxes, ha='right')
    axarr[2].legend(frameon=True, loc='upper left')
    plt.savefig('figs/new/satisfaction-hist_allpeep.pdf'.format(t),bbox_inches='tight')
<<<<<<< HEAD:final_project-showerte-dberenberg/src/analyze_elections.py
    plt.clf()

if input('Would you like to plot the outcome similarity map for all elections and transparencies? (y/Y) ').upper() == 'Y':
    import pandas as pd
    same_winners = pd.read_csv('ElectoralProcesses/same_winners.csv').transpose()
    labels = []
    for s in same_winners.index:
        v1 = s.split(' ')[0]
        v1 = v1.strip("(),'’")
        v2 = s.split(' ')[1]
        v2 = v2.strip("(),'’")
        labels.append(V_systems2label[v1]+', \n'+V_systems2label[v2])
    plt.figure()
    plt.imshow(same_winners)
    plt.yticks([0,1,2],labels)
    plt.xlabel('transparency level')
    #plt.colorbar(orientation='horizontal')
    for i in range(len(same_winners.index)):
        for j in range(len(same_winners.columns)):
            plt.text(j, i, str(same_winners.loc[same_winners.index[i], same_winners.columns[j]])+'\%',
                           ha="center", va="center", color="w")
    plt.title('Percent of Same Candidate Winning \nper Voting System Pair and Transparency Level')
    plt.savefig('../report_showerter-dberenberg/figs/same_winners.pdf',bbox_inches='tight')
=======
>>>>>>> 74483dd... fixedtransparency:final_project/src/analyze_elections.py
    plt.clf()

if input('Would you like to plot the outcome similarity map for all elections and transparencies? (y/Y) ').upper() == 'Y':
    import pandas as pd
    same_winners = pd.read_csv('ElectoralProcesses/same_winners.csv').transpose()
    labels = []
    for s in same_winners.index:
        v1 = s.split(' ')[0]
        v1 = v1.strip("(),'’")
        v2 = s.split(' ')[1]
        v2 = v2.strip("(),'’")
        labels.append(V_systems2label[v1]+', \n'+V_systems2label[v2])
    plt.figure()
    plt.imshow(same_winners)
    plt.yticks([0,1,2],labels)
    plt.xlabel('transparency level')
    #plt.colorbar(orientation='horizontal')
    for i in range(len(same_winners.index)):
        for j in range(len(same_winners.columns)):
            plt.text(j, i, str(same_winners.loc[same_winners.index[i], same_winners.columns[j]])+'\%',
                           ha="center", va="center", color="w")
    plt.title('Percent of Same Candidate Winning \nper Voting System Pair and Transparency Level')
    plt.savefig('../report_showerter-dberenberg/figs/same_winners.pdf',bbox_inches='tight')
    plt.clf()
