import numpy as np
import math
from sympy import *

logs = lambda x: math.log(x) if x > 0 else -np.inf

# Viterbi algorithm
def viterbi(obs,transition,emission):
    epsilon = 1e-6
    n = len(obs)
    k = len(transition)
    v = np.zeros([n,k], dtype = float)
    trace = {}
    # Initalize first row
    max_p = 0
    for x in range(100):
        print("Iteration:", x)
        max_p_curr = max_p
        for j in range(0,k):
            v[0,j] = logs(transition[0,0]) + logs(emission[1][obs[0]])
            trace[j+1] = [1]
        for i in range(1, n):
            trace_curr = {}
            for j in range(0,k):
                # loops through the states and stores the max probability in v[i,j]
                v[i,j], state = max((v[i-1,l]+logs(transition[l,j]),l) for l in range(0, k))
                v[i,j] = v[i,j] + logs(emission[j+1][obs[i]])
                # keep pointers to the path
                trace_curr[j+1] = trace[state+1] + [j+1]
            # keeps current trace
            trace = trace_curr
        max_p, state = max((v[n-1,l], l) for l in range(0,k))
        print(*obs, sep='-')
        print('| ' * len(obs))
        print(*trace[state+1], sep=' ')
        transition, emission = countTransitionEmission(obs, trace[state+1])
        transition, emission = normalize(transition, emission)



        if abs(max_p_curr - max_p) < epsilon:
            break
    #outputs

    return trace[state+1]

def countTransitionEmission(x,trace):

    transition = np.zeros([6,6], dtype = float)
    emission = {
        1: {'A':0.,'T':0.,'C':0.,'G':0.},
        2: {'A':0.,'T':0.,'C':0.,'G':0.},
        3: {'A':0.,'T':0.,'C':0.,'G':0.},
        4: {'A':0.,'T':0.,'C':0.,'G':0.},
        5: {'A':0.,'T':0.,'C':0.,'G':0.},
        6: {'A':0.,'T':0.,'C':0.,'G':0.}
    }
    k = len(transition)


    for i in range(0,len(trace)-1):
        #count transtion
        transition[trace[i]-1,trace[i+1]-1]+=1
        emission[trace[i]][x[i]]+=1
    #outputs
    print("transition\n", transition)
    print('emission\n', emission)
    return transition,emission

def normalize(transition,emission):
    #1
    #sol_tii = solve(Eq(((transition[5,0]+transition[0,0])/tii)-((transition[5,1]+transition[0,1])/(1-tii)),0),tii)
    sol_tii = (transition[0,0]+transition[5,0])/(transition[5,1]+transition[0,1]+transition[0,0]+transition[5,0])
    sol_tig= 1 - sol_tii
    transition[0,0] = sol_tii
    transition[5,0] = sol_tii
    transition[0,1] = sol_tig
    transition[5,1] = sol_tig
    print('sol_tii',sol_tii,'sol_tig',sol_tig)
    #2
    transition[4,2] = transition[4,2]/(transition[4,5]+transition[4,2])
    transition[4,5] = 1-transition[4,2]
    #print('sol_tgg',sol_tgg,'sol_tgi',sol_tgi)
    #3


    emission[1]['A'] = (emission[1]['A'])/(emission[1]['A']+emission[1]['T']+emission[1]['C']+emission[1]['G'])
    emission[1]['C'] = (emission[1]['C'])/(emission[1]['A']+emission[1]['T']+emission[1]['C']+emission[1]['G'])
    emission[1]['T'] = (emission[1]['T'])/(emission[1]['A']+emission[1]['T']+emission[1]['C']+emission[1]['G'])
    emission[1]['G'] = 1 - emission[1]['A'] - emission[1]['C'] - emission[1]['T']
    #print("sol_eia",sol_eia,"sol_eic",sol_eic,"sol_eit",sol_eit,"sol_eig",sol_eig)
    # sol_eia 0.0 sol_eic 0.0 sol_eit 0.0 sol_eig 1.0
    #4
    N = (emission[3]['A'])+(emission[4]['A'])+(emission[5]['A'])+(emission[3]['T'])+(emission[4]['T'])+(emission[5]['T'])+(emission[3]['C'])+(emission[4]['C'])+(emission[5]['C'])+(emission[3]['G'])+(emission[4]['G'])+(emission[5]['G'])
    sol_ega = ((emission[3]['A'])+(emission[4]['A'])+(emission[5]['A']))/N
    sol_egg = ((emission[3]['G'])+(emission[4]['G'])+(emission[5]['G']))/N
    sol_egt = ((emission[3]['T'])+(emission[4]['T'])+(emission[5]['T']))/N
    sol_egc = 1-sol_ega-sol_egg-sol_egt

    for i in range(3,6):
        emission[i]['A'] = sol_ega
        emission[i]['G'] = sol_egg
        emission[i]['T'] = sol_egt
        emission[i]['C'] = 1-sol_ega-sol_egg-sol_egt

    print('sol_egc',sol_egc,'sol_egt',sol_egt,'sol_ega',sol_ega,'sol_egg',sol_egg)
    return transition, emission
    # sol_eia 0.5 sol_eic 0.0 sol_eit 0.0 sol_eig 0.5
    # sol_egc 0.05882352941176472 sol_egt 0.3235294117647059 sol_ega 0.4117647058823529 sol_egg 0.20588235294117646

# Forward Algorithm
def forward(obs, transition, emission):
    n = len(obs)
    k = len(transition)
    f = np.zeros([n,k], dtype = float)

    # Initalize first row
    for j in range(0,k):
        f[0,j] = logs(transition[0,j]) + logs(emission[j+1][obs[0]])

    for i in range(1, n):
        for j in range(0,k):
            amax = max((f[i-1,l] + logs(transition[l,j])) for l in range(0, k))
            bl = sum(math.exp(f[i-1,l] + transition[l,j] - amax) for l in range(0, k))
            f[i,j] = amax + logs(bl) + logs(emission[j+1][obs[i]])

    likelihood = sum(math.exp(f[n-1,j]) for j in range(0,k))

    return f, logs(likelihood)

# Backward Algorithm
def backward(obs, transition, emission):
    n = len(obs)
    k = len(transition)
    b = np.zeros([n,k], dtype = float)

    # Initalize first row
    for j in range(0,k):
        b[n-1,j] = 1

    for i in range(n-2,-1,-1):
        for j in range(0,k):
            amax = max((logs(transition[j,l]) + logs(emission[l+1][obs[i+1]]) + b[i+1,l]) for l in range(0,k))
            bl = sum(math.exp((transition[j,l] + (emission[l+1][obs[i+1]]) + b[i+1,l]) - amax) for l in range(0,k))
            b[i,j] = amax + logs(bl)

    return b



if __name__ == '__main__':
    x1 = 'AAATTTTATTACGTTTAGTAGAAGAGAAAGGTAAACATGATGG'

    emission = {
        1: {'A':.1,'T':.2,'C':.3,'G':.4},
        2: {'A':1.,'T':0.,'C':0.,'G':0.},
        3: {'A':.4,'T':.3,'C':.2,'G':.1},
        4: {'A':.4,'T':.3,'C':.2,'G':.1},
        5: {'A':.4,'T':.3,'C':.2,'G':.1},
        6: {'A':0.,'T':1.,'C':0.,'G':0.}
    }

    transition = np.array([
        [.6,.4,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,1.,0.],
        [0.,0.,.6,0.,0.,.4],
        [.6,.4,0.,0.,0.,0.]
    ])
    trace = viterbi(x1,transition,emission)
    print(*x1, sep='-')
    print('| ' * len(x1))
    print(*trace, sep=' ')
    #path = viterbi(x1,transition,emission)
    #normalize(transition,emission)
