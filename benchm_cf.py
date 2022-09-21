import pandas as pd
import os
import numpy as np
import re
from sklearn import preprocessing
import gurobipy as gp
from gurobipy import GRB

##leo los datos de pigdata de R

datos=pd.read_csv('pigdata.csv')

#datos=datos[['firm','x1','y4']] #me quedo con un solo input y un solo output
#datos=datos[0:3]

#datos=pd.DataFrame(data={'firm':[1,2,3],'x1':[1,2,2],'x2':[2,1,2],'y1':[1,1,1]})

datos2=pd.DataFrame(data={'firm':[1,2,3],'x1':[1,2,2],'x2':[1.5,0.5,2],'y1':[1,2,1.5]})
datos=datos2

datos3=pd.DataFrame(data={'firm':[1,2,3,4,5],'x1':[1,2,1.5,2,2.5],'x2':[1.5,1,1.25,2,2],'y1':[1,1,1,1,1]})
datos=datos3

datos4=pd.DataFrame(data={'firm':[1,2,3,4,5],'x1':[1,2,1.5,2,2.5],'x2':[1.5,0.5,1,2,2],'y1':[1,2,1.5,1.5,1]})
datos=datos4

datos5=pd.DataFrame(data={'firm':[1,2,4,5],'x1':[1,2,1.5,2.5],'x2':[1.5,0.5,1,2],'y1':[1,2,1.5,1]}) #distintos y
datos6=pd.DataFrame(data={'firm':[1,2,4,5],'x1':[1,2.5,1.5,2.5],'x2':[1.5,0.5,1,2],'y1':[1,1,1,1]}) #problema cuando frontera por tres
datos7=pd.DataFrame(data={'firm':[1,2,5],'x1':[1,2.5,2.5],'x2':[1.5,0.5,2],'y1':[1,1,1]}) #frontera solo con dos
datos8=pd.DataFrame(data={'firm':[1,2,4,5],'x1':[1,2.5,1.5,1.5],'x2':[1.5,0.5,1,2.5],'y1':[1,1,1,1]}) #cuando esta cerca de una cara

datos=datos6

firms=list(datos['firm'])

#voy a calcular la eficiencia de todas las empresas del pigdata (248) ¿cuantas son ineficientes?


def calculate_dea(datos,k,xk_sol,type):
    
    m=gp.Model("cfbm")

    #parametros

    firms=list(datos['firm'])
    inputs=[i for i in datos.columns if 'x' in i]
    outputs=[i for i in datos.columns if 'y' in i]
    #w_inputs=[i for i in datos.columns if 'w' in i]
    #w_outputs=[i for i in datos.columns if 'p' in i]

    x0={}
    for f in firms:
        x0[f]=datos[datos.firm==f][inputs]

    if xk_sol!=[]:
        for i in inputs:
            x0[k][i]=xk_sol[i]

    y0={}
    for f in firms:
        y0[f]=datos[datos.firm==f][outputs]
 

    l={}
    for f in firms:
        l[f]=m.addVar(lb=0,name='lambda'+str(f)) #lambda de cada firma
    

    
    Ek=m.addVar(lb=0,ub=1,name='eff')


    #obj function
    m.setObjective(Ek, GRB.MINIMIZE)

    #constraints
    
    for i in inputs:
        m.addConstr(Ek*x0[k][i]>=gp.quicksum(l[f]*x0[f][i] for f in firms), "rin"+str(i))


    for o in outputs:
        m.addConstr(float(y0[k][o])<=gp.quicksum(l[f]*y0[f][o] for f in firms), "rout"+str(o))

    if type=='VRS':
        m.addConstr(gp.quicksum(l[f] for f in firms)==1)


    #m.write('cfbench.lp')

    m.optimize()

    print('Obj: '+str(m.ObjVal))

    
    efi=Ek.getAttr(GRB.Attr.X)


    
    lambdas_sol={}
    for f in firms:
        lambdas_sol[f]=l[f].getAttr(GRB.Attr.X)


    return efi, lambdas_sol


xk_sol=[]
k=5
efi, lambdas_dados=calculate_dea(datos,k,xk_sol,'CRS')
xk_sol={'x1':1.5,'x2':1}


#inefi_pigdata={}
#for f in firms:
#    efi, lambdas_dados=calculate_dea(datos,f,[])
#    if efi<1:
#        inefi_pigdata[f]=efi

#inefi_pigdata

#199 empresas son ineficientes

#muyinefi={}#
#for f in in#efi_pigdata.keys():
#    if inefi_pigdata[f]<0.75:
#        muy#inefi[f]=inefi_pigdata[f]

#len(muyinefi.keys())

#hay 39 empresas con una eficiencia por debajo de 0.75

#import json
#with open('inefi_pigdata.txt', 'w') as file:
#     file.write(json.dumps(inefi_pigdata)) 

#xk_sol={}
#for i in inputs:
#    xk_sol[i]=1

#xk_sol={'x1','x2','x3','x4'}

def counterf(datos,k,E_deseada,norm,type):


   
    ##creo el modelo de optimization 

    #modelo
    m=gp.Model("cfbm")

    #parametros

    firms=list(datos['firm'])
    inputs=[i for i in datos.columns if 'x' in i]
    outputs=[i for i in datos.columns if 'y' in i]
    #w_inputs=[i for i in datos.columns if 'w' in i]
    #w_outputs=[i for i in datos.columns if 'p' in i]
    #normalizo

    #min_max_scaler = preprocessing.MinMaxScaler()
    #datos[inputs] = datos[inputs].apply(lambda x: (x - x.min()) / (x.max() - x.min())) #quito el output porque es cte
    #datos_s= min_max_scaler.fit_transform(datos, axis=1)) 
    #datos_s = pd.DataFrame(datos_s,columns=datos.columns)
    #boston=datos_s



    x0={}
    for f in firms:
        x0[f]=datos[datos.firm==f][inputs]

    y0={}
    for f in firms:
        y0[f]=datos[datos.firm==f][outputs]

    


    F_deseada=1/E_deseada #1/E_deseada


    

    M_l0=float(x0[k][max(x0[k])]) #los M para la l0

    M_1=10 #1e4
    M_2=10 # 1
    M_3=10 #F_deseada
    M_4=10 #10 
    #M=1e5 #ajustar

    #las variables
 

    #los beta de cada firma 
    beta={}
    for f in firms:
        beta[f]=m.addVar(lb=0,name='beta'+str(f)) 



    F=m.addVar(lb=1,name='inveff'+str(k))#calculo de la nueva eff F=1/E

    #counterfactual variable
    xk={}
    for i in inputs:
        xk[i]=m.addVar(lb=0,name='cf'+str(i)) #counterf de los inputs
    
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

    #m.setObjective(gp.quicksum(xik[i] for i in inputs)+0.1*gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE) 
    #m.setObjective(gp.quicksum(xik[i] for i in inputs), GRB.MINIMIZE)
    #m.setObjective(gp.quicksum((float(x0[k][i])-xk[i])*(float(x0[k][i])-xk[i]) for i in inputs), GRB.MINIMIZE) 

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
    
        
    m.addConstr(F<=F_deseada)

    


    #para la l0
    for i in inputs:
        m.addConstr((-M_l0*xik[i]<=xk[i]-x0[k][i]),"l01_"+str(i))
        m.addConstr((xk[i]-x0[k][i]<=M_l0*xik[i]),"l02_"+str(i))



    #m.write('cfbench.lp')


    m.optimize()



    print('Runtime: '+ str(m.Runtime))
    print('Obj: '+str(m.ObjVal))

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
    


    beta_sol={}
    for f in firms:
        beta_sol[f]=beta[f].getAttr(GRB.Attr.X)

    F_sol=F.getAttr(GRB.Attr.X)
    E_sol=1/F_sol

    lambda_sol={}
    for f in firms:
        lambda_sol[f]=beta_sol[f]*E_sol
    
    #for o in outputs:
    #    print(v[o].getAttr(GRB.Attr.X))


    #print([(f,l) for (f,l) in zip (lambdas_dados.keys(),lambdas_dados.values()) if l!=0])
    #print([(f,l) for (f,l) in zip (lambda_sol.keys(),lambda_sol.values()) if l!=0])


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

    for i in inputs:
        print(xk_sol[i]-gp.quicksum(beta_sol[f]*float(x0[f][i]) for f in firms))

    #for f in firms:
    #    print(gp.quicksum(gamma_sol[i]*x0[f][i] for i in inputs)-gp.quicksum(gamma_sol[o]*y0[f][o] for o in outputs))

    return cambio, xk_sol, E_sol,lambda_sol, fobj

    

E_deseada=0.8
k=5
norm="l2"
type='CRS'
cambio, xk_sol, E_sol,lambda_sol, fobj = counterf(datos,k,E_deseada,norm,type)


#con data={'firm':[1,2,3],'x1':[1,2,2],'x2':[2,1,2],'y1':[1,1,1]}
#firm 3 tiene Efi=0.75 si pido E=0.8
#con l0:            x0=[2,2], counterf=[2,1] y E_sol=1
#con l2:            x0=[2,2], counterf=[1.875,1.875] y E_sol=0.8
#con l0+0.1*l2 l    x0=[2,2], counterf=[2, 1.75] y E_sol=0.8

#con data={'firm':[1,2,3],'x1':[1,2,2],'x2':[1.5,1,2],'y1':[1,1,1]}
#firm 3 tiene Efi=0.66 si pido E=0.8
#con l0:            x0=[2,2], counterf=[1,2] y E_sol=1
#con l2:            x0=[2,2], counterf=[1.8,1.6] y E_sol=0.8
#con l0+0.1*l2 l    x0=[2,2], counterf=[2, 1.5] y E_sol=0.8

import json
with open('inefi_pigdata.txt', 'r') as f:
  inefi = json.load(f)


inefi2={}
for f in inefi.keys():
    if inefi[f]<0.7:
        inefi2[f]=inefi[f]



solucion=[]
cambios=[]
objetivo=[]
efi=[]

#empiezo con la euclidea
#deseo una eficiencia de 0.8

for f in list(inefi2.keys()):
    cambio, xk_sol, E_sol,lambda_sol, fobj = counterf(datos,int(f),0.7,"l2",type)
    solucion.append(xk_sol)
    cambios.append(cambio)
    objetivo.append(fobj)
    efi.append(E_sol)


#solucion_muyinefi=pd.DataFrame(solucion)

with open('counterf_07_l2.txt', 'w') as file:
     file.write(json.dumps(solucion)) 

with open('cambio_07_l2.txt', 'w') as file:
     file.write(json.dumps(cambios)) 

with open('objetivo_07_l2.txt', 'w') as file:
     file.write(json.dumps(objetivo)) 

with open('efi_07_l2.txt', 'w') as file:
     file.write(json.dumps(efi)) 



#efi 0.7
with open('counterf_07_l2.txt', 'r') as f:
  sol = json.load(f)


prueba=pd.DataFrame(sol)
prueba['firm']=inefi2.keys()
prueba['E0']=inefi2.values()

prueba.to_excel('07l2.xlsx')

with open('resultados_nuevos/cambio_1_l2.txt', 'r') as f:
  cambios = json.load(f)

l=len(cambios)

cambios=pd.DataFrame(cambios)

veces_cambio={}
for c in list(cambios.columns):
    veces_cambio[c]=sum(abs(cambios[c])>1e-3)/l

veces_cambio

inputs_changed=[]
for i in range(l):
    inputs_changed.append(sum([abs(a)>1e-3 for a in list(cambios.iloc[i])]))

l0_media=sum(inputs_changed)/l

deviations=[]
for i in range(l):
    deviations.append(np.linalg.norm(cambios.iloc[i]))


l2_media=sum(deviations)/l

summary=veces_cambio
summary['l0']=l0_media
summary['l2']=l2_media

summary




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

#cojo las empresas cuya lambda es distinta de cero al calcular el dea de k
xk_sol=[]
k=3
efi, lambdas_dados=calculate_dea(datos,k,xk_sol,'CRS')

efi_firms=[f for (f,l) in zip (lambdas_dados.keys(),lambdas_dados.values()) if l!=0]



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



    #para la l0
    #for i in inputs:
    #    m.addConstr((-M_l0*xik[i]<=xk[i]-x0[k][i]),"l01_"+str(i))
    #    m.addConstr((xk[i]-x0[k][i]<=M_l0*xik[i]),"l02_"+str(i))



    #m.write('cfbench.lp')


    m.optimize()



    print('Runtime: '+ str(m.Runtime))
    print('Obj: '+str(m.ObjVal))

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
    
    #for o in outputs:
    #    print(v[o].getAttr(GRB.Attr.X))


    #print([(f,l) for (f,l) in zip (lambdas_dados.keys(),lambdas_dados.values()) if l!=0])
    #print([(f,l) for (f,l) in zip (lambda_sol.keys(),lambda_sol.values()) if l!=0])


    return cambio, xk_sol, lambda_sol, fobj


k=5
E_deseada=0.8
norm="l2"
efi_firms=[1,2,4]
cambio2, xk_sol2, lambda_sol2, fobj2= counterf2(datos,efi_firms,k,E_deseada,norm)



print([(f,l) for (f,l) in zip (lambdas_dados.keys(),lambdas_dados.values()) if l!=0])

efi_firms
print([(f,l) for (f,l) in zip (lambda_sol.keys(),lambda_sol.values()) if l!=0])
