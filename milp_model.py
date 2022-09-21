import pandas as pd
import os
import numpy as np
import re
from sklearn import preprocessing
import gurobipy as gp
from gurobipy import GRB

#data

data=pd.DataFrame(data={'firm':[1,2,4,5],'x1':[1,2.5,1.5,1.5],'x2':[1.5,0.5,1,2.5],'y1':[1,1,1,1]}) #cuando esta cerca de una cara

firms=list(data['firm'])

#k: firm to calculate the counterfactual
#E_desired: efficiency one wants to obtain
#norm: distance used, can be l2, l0 or l2+l0
#type: technology used, can be CRS or VRS

def counterf(data,k,E_desired,norm,type): 

    #optimization model
    m=gp.Model("cfbm")

    #parameters

    firms=list(data['firm'])
    inputs=[i for i in data.columns if 'x' in i]
    outputs=[i for i in data.columns if 'y' in i]
    #w_inputs=[i for i in data.columns if 'w' in i]
    #w_outputs=[i for i in data.columns if 'p' in i]
    
    #normalizing the data
   
    #data[inputs] = data[inputs].apply(lambda x: (x - x.min()) / (x.max() - x.min())) 



    x0={}
    for f in firms:
        x0[f]=data[data.firm==f][inputs]

    y0={}
    for f in firms:
        y0[f]=data[data.firm==f][outputs]


    F_desired=1/E_desired 


  
    M_l0=float(x0[k][max(x0[k])]) 

    #adjust 
    
    M_1=10 #1e4
    M_2=10 # 1
    M_3=10 #F_deseada
    M_4=10 #10 


    #variables

    beta={}
    for f in firms:
        beta[f]=m.addVar(lb=0,name='beta'+str(f)) 


    F=m.addVar(lb=1,name='inveff'+str(k))

    #counterfactual variable
    xk={}
    for i in inputs:
        xk[i]=m.addVar(lb=0,name='cf'+str(i)) 
    
    #gamma auxiliar bilevel
    gamma={}
    for i in inputs:
        gamma[i]=m.addVar(lb=0,name="gamma_"+str(i))
    for o in outputs:
        gamma[o]=m.addVar(lb=0,name="gamma_"+str(o))
    
    mu={}
    for f in firms:
        mu[f]=m.addVar(lb=0,name="gamma_beta_"+str(f))
    mu['F']=m.addVar(lb=0, name="gamma_F")

    #u_i y v auxiliar milp kkt

    u={}
    for i in inputs:
        u[i]=m.addVar(vtype=GRB.BINARY, name="milp_kkt_"+str(i))

    v={}
    for o in outputs:
        v[o]=m.addVar(vtype=GRB.BINARY, name="milp_kkt_"+str(o))

    w={}
    for f in firms:
        w[f]=m.addVar(vtype=GRB.BINARY, name="milp_kkt3_beta_"+str(f))
    w[0]=m.addVar(vtype=GRB.BINARY,name="milp_kkt_F")

    if type=='VRS':
        chi=m.addVar(name='vrscond')

    xik={}
    for i in inputs:
        xik[i]=m.addVar(vtype=GRB.BINARY, name='l0_'+str(i))

    #obj function
    if norm=="l2":
        m.setObjective(gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE)
    elif norm=="l0":
        m.setObjective(gp.quicksum(xik[i] for i in inputs), GRB.MINIMIZE)
    elif norm=="l0l2":
        m.setObjective(gp.quicksum(xik[i] for i in inputs)+1e-5*gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE) 
 

    #constraints

    for i in inputs:
        m.addConstr(xk[i]>=gp.quicksum(beta[f]*float(x0[f][i]) for f in firms), "rin"+str(i))

    for o in outputs:
        m.addConstr(F*float(y0[k][o])<=gp.quicksum(beta[f]*float(y0[f][o]) for f in firms), "rout"+str(o)) 


    if type=='VRS':
        m.addConstr(gp.quicksum(gamma[o]*y0[k][o] for o in outputs)-mu['F']+chi==1, 'kkt1')
        for f in firms:
            m.addConstr(gp.quicksum(gamma[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma[o]*y0[f][o] for o in outputs)-mu[f]-chi==0, 'kkt2')
        m.addConstr(F-gp.quicksum(beta[f] for f in firms)==0, 'vrscond')
    else:
        m.addConstr(gp.quicksum(gamma[o]*y0[k][o] for o in outputs)-mu['F']==1, 'kkt1')
        for f in firms:
            m.addConstr(gp.quicksum(gamma[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma[o]*y0[f][o] for o in outputs)-mu[f]==0, 'kkt2')

    for i in inputs:
        m.addConstr(gamma[i]<=M_1*u[i])

    for i in inputs:
        m.addConstr(xk[i]-gp.quicksum(beta[f]*float(x0[f][i]) for f in firms)<=M_1*(1-u[i]))

    for o in outputs:
        m.addConstr(gamma[o]<=M_2*v[o])

    for o in outputs:
        m.addConstr(-F*float(y0[k][o])+gp.quicksum(beta[f]*float(y0[f][o]) for f in firms)<=M_2*(1-v[o]))

    
    m.addConstr(F<=M_3*w[0])

    m.addConstr(gp.quicksum(gamma[o]*y0[k][o] for o in outputs)-1 <= M_3*(1-w[0]))

    for f in firms:
        m.addConstr(beta[f]<=M_4*w[f])

    for f in firms:
        m.addConstr(gp.quicksum(gamma[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma[o]*y0[f][o] for o in outputs) <= M_4*(1-w[f]))        
    
        
    m.addConstr(F<=F_desired)

    


    #l0
    for i in inputs:
        m.addConstr((-M_l0*xik[i]<=xk[i]-x0[k][i]),"l01_"+str(i))
        m.addConstr((xk[i]-x0[k][i]<=M_l0*xik[i]),"l02_"+str(i))


    m.optimize()



    #print('Runtime: '+ str(m.Runtime))
    #print('Obj: '+str(m.ObjVal))

    fobj=m.ObjVal

    xk_sol={}
    for i in inputs:
        xk_sol[i]=xk[i].getAttr(GRB.Attr.X)

    change={}

    for i in inputs:
        change[i]=float(xk[i].getAttr(GRB.Attr.X)-x0[k][i])

    #print('x0: '+str(x0[k]))
    #print('x_sol: '+str(xk_sol))
    #print('cambio: '+str(change))
    


    beta_sol={}
    for f in firms:
        beta_sol[f]=beta[f].getAttr(GRB.Attr.X)

    F_sol=F.getAttr(GRB.Attr.X)
    E_sol=1/F_sol

    lambda_sol={}
    for f in firms:
        lambda_sol[f]=beta_sol[f]*E_sol


    u_sol={}
    for i in inputs:
        u_sol[i]=u[i].getAttr(GRB.Attr.X)
    v_sol={}
    for o in outputs:
        v_sol[o]=v[o].getAttr(GRB.Attr.X)

    w_sol={}
    for f in firms:
        w_sol[f]=w[f].getAttr(GRB.Attr.X)
    w_sol[0]=w[0].getAttr(GRB.Attr.X)

    gamma_sol={}
    for i in inputs:
        gamma_sol[i]=gamma[i].getAttr(GRB.Attr.X)
    for o in outputs:
        gamma_sol[o]=gamma[o].getAttr(GRB.Attr.X)

    mu_sol={}
    for f in firms:
        mu_sol[f]=mu[f].getAttr(GRB.Attr.X)
    mu_sol['F']=mu['F'].getAttr(GRB.Attr.X)

    #for i in inputs:
    #    print(xk_sol[i]-gp.quicksum(beta_sol[f]*float(x0[f][i]) for f in firms))

    #for f in firms:
    #    print(gp.quicksum(gamma_sol[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma_sol[o]*y0[f][o] for o in outputs))

    return change, xk_sol, E_sol,lambda_sol, fobj

    
E_desired=0.8
k=5
norm="l2"
type='CRS'
change, xk_sol, E_sol,lambda_sol, fobj = counterf(data,k,E_desired,norm,type)



#################PROBANDO EL NUEVO MODELO SIN BILEVEL

#necesito las firmas que son eficientes

#efi_pigdata={}
#for f in firms:
#    efi, lambdas_dados=calculate_dea(datos,f,[],'CRS')
#    if efi==1:
#        efi_pigdata[f]=efi


#import json
#with open('efi_pigdata.txt', 'w') as file:
#     file.write(json.dumps(efi_pigdata)) 

import json
with open('efi_pigdata.txt', 'r') as f:
  efi = json.load(f)

efi_firms=list(efi.keys())
efi_firms=list(map(int, efi_firms))

def counterf2(datos,efi_firms,k,E_deseada,norm):


    ##creo el modelo de optimization 

    #modelo
    m=gp.Model("cfbm")

    #parametros

    firms=list(datos['firm'])
    inputs=[i for i in datos.columns if 'x' in i]
    outputs=[i for i in datos.columns if 'y' in i]
    #w_inputs=[i for i in datos.columns if 'w' in i]
    #w_outputs=[i for i in datos.columns if 'p' in i]

    #los datos esta vez son las eficientes más la que quiero cambias
    #firms2=efi_firms
    #firms2.append(k)

    x0={}
    for f in efi_firms+[k]:
        x0[f]=datos[datos.firm==f][inputs]

    y0={}
    for f in efi_firms+[k]:
        y0[f]=datos[datos.firm==f][outputs]

    
    #los lambda nuevos de cada firma
    lambda_n={}
    for f in efi_firms:
        lambda_n[f]=m.addVar(lb=0,name='lambda'+str(f))


    #counterfactual variable
    xk={}
    for i in inputs:
        xk[i]=m.addVar(lb=0,name='cf'+str(i)) #counterf de los inputs
    

    #obj function
    if norm=="l2":
        m.setObjective(gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE)
    elif norm=="l0":
        m.setObjective(gp.quicksum(xik[i] for i in inputs), GRB.MINIMIZE)
    elif norm=="l0l2":
        m.setObjective(gp.quicksum(xik[i] for i in inputs)+1e-5*gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE) 

    #constraints

    for i in inputs:
        m.addConstr(E_deseada*xk[i]==gp.quicksum(lambda_n[f]*float(x0[f][i]) for f in efi_firms), "rin"+str(i))


    for o in outputs:
        m.addConstr(float(y0[k][o])==gp.quicksum(lambda_n[f]*float(y0[f][o]) for f in efi_firms), "rout"+str(o)) 

    m.optimize()

    fobj=m.ObjVal

    xk_sol={}
    for i in inputs:
        xk_sol[i]=xk[i].getAttr(GRB.Attr.X)

    cambio={}

    for i in inputs:
        cambio[i]=float(xk[i].getAttr(GRB.Attr.X)-x0[k][i])

    print('x0: '+str(x0[k]))
    print('x_sol: '+str(xk_sol))
    print('cambio: '+str(cambio))
    


    lambda_sol={}
    for f in efi_firms:
        lambda_sol[f]=lambda_n[f].getAttr(GRB.Attr.X)
    


    return cambio, xk_sol, lambda_sol, fobj


k=5
E_deseada=0.8
norm="l2"
efi_firms=[1,2,4]
cambio2, xk_sol2, lambda_sol2, fobj2= counterf2(datos,efi_firms,k,E_deseada,norm)


