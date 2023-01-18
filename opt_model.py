import pandas as pd
import os
import numpy as np
import re
from sklearn import preprocessing
import gurobipy as gp
from gurobipy import GRB

#df: data
#firms: list of names(number) of the firms
#inputs: name of inputs
#outputs: name of outputs
#k: firm to calculate the counterfactual
#E_desired: efficiency one wants to obtain
#type: technology used, can be CRS or VRS
#nu: value of nus nu[0]*l0+nu[1]*l1+nu[2]*l2



def counterf(df,firms,inputs,outputs,k,E_desired,type,nu):

    #model
    m=gp.Model("cfbm")

    #parameters

    x0={}
    for f in firms:
        x0[f]=df[df.DMU==f][inputs]

    y0={}
    for f in firms:
        y0[f]=df[df.DMU==f][outputs]


    F_desired=1/E_desired 


    M_l0=float(x0[k][max(x0[k])]) #M for l0


    #bigMs (change depending on the scale of the data)
    M_1=10   #slack input
    M_2=10 #slack output
    M_4=10  #border 

    #variables
 
    beta={}
    for f in firms:
        beta[f]=m.addVar(lb=0,name='beta'+str(f)) 

    F=m.addVar(lb=1,name='inveff'+str(k))

    #counterfactual variable
    xk={}
    for i in inputs:
        xk[i]=m.addVar(lb=0,name='cf'+str(i)) 
    
    
    gamma={}
    for i in inputs:
        gamma[i]=m.addVar(lb=0,name="gamma_"+str(i))
    for o in outputs:
        gamma[o]=m.addVar(lb=0,name="gamma_"+str(o))

    

    u={}
    for i in inputs:
        u[i]=m.addVar(vtype=GRB.BINARY, name="milp_kkt_"+str(i))

    v={}
    for o in outputs:
        v[o]=m.addVar(vtype=GRB.BINARY, name="milp_kkt_"+str(o))

    w={}
    for f in firms:
        w[f]=m.addVar(vtype=GRB.BINARY, name="milp_kkt3_beta_"+str(f))
    

    if type=='VRS':
        chi=m.addVar(name='vrscond')

    xik={}
    for i in inputs:
        xik[i]=m.addVar(vtype=GRB.BINARY, name='l0_'+str(i))

    xik2={}
    for i in inputs:
        xik2[i]=m.addVar(lb=0, name='l1_'+str(i))

    #obj function
    m.setObjective(nu[0]*gp.quicksum(xik[i] for i in inputs)+nu[1]*gp.quicksum(xik2[i] for i in inputs)+nu[2]*gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE)
    
   
    #constraints

    #primal

    for i in inputs:
        m.addConstr(xk[i]>=gp.quicksum(beta[f]*float(x0[f][i]) for f in firms), "rin"+str(i))

    for o in outputs:
        m.addConstr(F*float(y0[k][o])<=gp.quicksum(beta[f]*float(y0[f][o]) for f in firms), "rout"+str(o)) 

    #dual
    if type=='VRS': 
        m.addConstr(gp.quicksum(gamma[o]*y0[k][o] for o in outputs)+chi==1, 'kkt1')
        for f in firms:
            m.addConstr(gp.quicksum(gamma[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma[o]*y0[f][o] for o in outputs)-chi>=0, 'kkt2')
        m.addConstr(F-gp.quicksum(beta[f] for f in firms)==0, 'vrscond')
    else:
        m.addConstr(gp.quicksum(gamma[o]*y0[k][o] for o in outputs)==1, 'kkt1') #>=1
        for f in firms:
            m.addConstr(gp.quicksum(gamma[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma[o]*y0[f][o] for o in outputs)>=0, 'kkt2')

    #slacks
    for i in inputs:
        m.addConstr(gamma[i]<=M_1*u[i])

    for i in inputs:
        m.addConstr(xk[i]-gp.quicksum(beta[f]*float(x0[f][i]) for f in firms)<=M_1*(1-u[i]))

    for o in outputs:
        m.addConstr(gamma[o]<=M_2*v[o])

    for o in outputs:
        m.addConstr(-F*float(y0[k][o])+gp.quicksum(beta[f]*float(y0[f][o]) for f in firms)<=M_2*(1-v[o]))

    #frontier definition 
    for f in firms:
        m.addConstr(beta[f]<=M_4*w[f])

    for f in firms:
        m.addConstr(gp.quicksum(gamma[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma[o]*y0[f][o] for o in outputs) <= M_4*(1-w[f]))        
    
    #imposing efficiency desired
    m.addConstr(F<=F_desired)

    #l0 norm linearization
    for i in inputs:
        m.addConstr((-M_l0*xik[i]<=xk[i]-x0[k][i]),"l01_"+str(i))
        m.addConstr((xk[i]-x0[k][i]<=M_l0*xik[i]),"l02_"+str(i))

    #l1 norm lineatization
    for i in inputs:
        m.addConstr(xik2[i]>=(xk[i]-x0[k][i]),"l11_"+str(i))
        m.addConstr(-xik2[i]<=(xk[i]-x0[k][i]),"l12_"+str(i))

 

    #m.write('cfbench.lp')

    m.setParam('OutputFlag', 0)
    m.optimize()

    if (m.status == 3) or (m.status==4):
        change={i:np.nan for i in inputs} 
        xk_sol={i:np.nan for i in inputs}
        E_sol=np.nan
        lambda_sol={i:np.nan for i in firms}
        fobj=np.nan
        dev=np.nan

    else:
    

        print('Runtime: '+ str(m.Runtime))
        print('Obj: '+str(m.ObjVal))

        fobj=m.ObjVal

        xk_sol={}
        for i in inputs:
            xk_sol[i]=xk[i].getAttr(GRB.Attr.X)

        change={}

        for i in inputs:
            change[i]=float(xk[i].getAttr(GRB.Attr.X)-x0[k][i])

        dev=sum((float(x0[k][i])-xk_sol[i])*(float(x0[k][i])-xk_sol[i]) for i in inputs)
    

        beta_sol={}
        for f in firms:
            beta_sol[f]=beta[f].getAttr(GRB.Attr.X)

        F_sol=F.getAttr(GRB.Attr.X)
        E_sol=1/F_sol

        lambda_sol={}
        for f in firms:
            lambda_sol[f]=beta_sol[f]*E_sol

      

    return change, xk_sol, E_sol,lambda_sol, fobj,dev


 
#change: vector of changes
#xk_sol: counterfactual explanation
#E_sol: efficiency of the counterfactual 
#lambda_sol: new value of lambdas 
#fobj: value of the objective function
#dev: total deviations 
   
    
    
#illustration
df=pd.DataFrame(data={'DMU':[1,2,3],'x1':[0.5,1.5,1.75],'x2':[1,0.5,1.25],'y1':[1,1,1]})
firms=df['DMU']
inputs=[i for i in df.columns if 'x' in i]
outputs=[i for i in df.columns if 'y' in i]
k=3
E_desired=0.8
type='CRS'
nu=[1,0,1]
change, xk_sol, E_sol,lambda_sol, fobj,dev = counterf(df,firms,inputs,outputs,k,E_desired,type,nu)
print(xk_sol)




