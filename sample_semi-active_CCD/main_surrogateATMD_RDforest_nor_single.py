#%%
import pandas as pd
import func_ATMD_Geo
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.util import load_logs
import pandas as pd
import math
import os
import numpy as np
from scipy.optimize import NonlinearConstraint
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
'''
this is main function of Active TMD and floating platform geometry optimization.

include:
    parameters boundary, 
    objective function,
    constraint function,
    code quick start,
    surrogate model creates and uses,
    data flow transmission, 
    bayes optimizer definition and quick restart,
    optimization loop,
    data output
    ...
'''
#%  initial parameter data, (can not change top five name sequence of data, or will affect the calculation of mass and inertia.)
## TMD can be tuned as the frequency from wave(0.0334Hz) to tower(0.397Hz) for all mass 

bound={                    ## Boundary of the parameters     
    "columns_D":(9,12),
    "damping_L":(58,71),     
    "support_LB":(23,31),
    "deep_DB":(12,18),
    "wide_W":(9,13),
    'damp5':(42000,155100),
    'damp7':(42000,155100),
    'damp9':(42000,155100),
    'damp11':(42000,155100),
    'damp13':(42000,155100),
    'damp15':(42000,155100),
    'damp17':(42000,155100),
    'damp19':(42000,155100),
    'damp21':(42000,155100),
    'damp23':(42000,155100),
    'damp25':(42000,155100),
    'damp11.4':(42000,155100),
    'TMD_mass':(18674,186740),
    'TMD_stiffness':(9000,1.2e6), 
}

bound_nor={                    ## Boundary of the parameters     
    "columns_D":(0,1),
    "damping_L":(0,1),     
    "support_LB":(0,1),
    "deep_DB":(0,1),
    "wide_W":(0,1),
    'damp5':(0,1),
    'damp7':(0,1),
    'damp9':(0,1),
    'damp11':(0,1),
    'damp13':(0,1),
    'damp15':(0,1),
    'damp17':(0,1),
    'damp19':(0,1),
    'damp21':(0,1),
    'damp23':(0,1),
    'damp25':(0,1),
    'damp11.4':(0,1),
    'TMD_mass':(0,1),
    'TMD_stiffness':(0,1), 
}

col=[
    "columns_D",
    "damping_L",
    "support_LB",
    "deep_DB",
    "wide_W",
    'damp5',
    'damp7',
    'damp9',
    'damp11',
    'damp13',
    'damp15',
    'damp17',
    'damp19',
    'damp21',
    'damp23',
    'damp25',
    'damp11.4',
    "TMD_mass",
    "TMD_stiffness",
    'platformTMD_mass',
    "platform_mass",
    'platTMDiner_x',
    'platTMDiner_y',
    'platTMDiner_z',
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "displace_V",
    "connection_point",
    "CM_plaftform",
    'CM_platformTMD',
    "CMz_TMD",
    "AEP",
    "DEL_TM",
    "Platform_ultPitch",
    'target',
    'flag',
    'flag_geom',
]

##    define the objective function for optimization  
def obj_fun(mass):
    value=-mass  ## mass(empty column)
    return value

def constraint_advanced(DEL_TM,pitch,AEP):
    return [AEP,DEL_TM,pitch]
constraint_lower=np.array([4.01e7,-np.inf,-np.inf])
constraint_upper=np.array([np.inf,8.05e4,7.387])
constraint=NonlinearConstraint(constraint_advanced,constraint_lower,constraint_upper)

##    define simple(geometry) constrain
def constraint_simple(para):
    con1=para['damping_L']/math.sin(math.pi/4)-(2*5.25/math.sin(math.pi/4)-2*5.25
            )-2*para['support_LB']/math.sin(math.pi/4)-para['columns_D']    ## greater or equal 0
    con2=para['damping_L']/2-para['wide_W']       ##greater than 5.24999
    con3=1.414*para['wide_W']-para['columns_D']   ##greater than 0
    con4=para['support_LB']/math.sin(math.pi/4)-para['columns_D']/2-(para['damping_L']/2
            -para['wide_W'])/math.sin(math.pi/4)+2.1746        ##greater than 0
    if con1>=0 and con2>5.249999 and con3>0 and con4>0:
        flag_simpcon=1
    else:
        flag_simpcon=0
    return flag_simpcon

data=pd.DataFrame(columns=col)   ## main data flow definition

##   surrogate model definition(geometry in Gaussian process and control para in KNN)
model_cons_AEP=GaussianProcessRegressor(
            kernel=RationalQuadratic(),
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=100,
            random_state=254,
            copy_X_train=False)
model_cons_pitch=GaussianProcessRegressor(
            kernel=RationalQuadratic(),
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=100,
            random_state=254,
            copy_X_train=False)
model_cons_DEL=GaussianProcessRegressor(
            kernel=RationalQuadratic(),
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=100,
            random_state=254,
            copy_X_train=False)
model_cons_target=GaussianProcessRegressor(
            kernel=RationalQuadratic(),
            alpha=1e-9,
            normalize_y=True,
            n_restarts_optimizer=100,
            random_state=254,
            copy_X_train=False)


data_initial=pd.read_excel('initial_data_total_clean.xlsx')
data_initialRDF=pd.read_excel('initial_data_total.xlsx')
#print(data_initial)
flag_continue=int(input("Load data?(1:NO, 2:YES):"))

x=data_initial.loc[:,'columns_D':'TMD_stiffness'].copy()
x=func_ATMD_Geo.normalization_x(x,bound)
y_AEP=data_initial.loc[:,'AEP'].copy()
y_pitch=data_initial.loc[:,'Platform_ultPitch'].copy()
y_DEL=data_initial.loc[:,'DEL_TM'].copy()
y_target=data_initial.loc[:,'target'].copy()

x_RDF=data_initialRDF.loc[:,'columns_D':'wide_W'].copy()
y_RDF=data_initialRDF.loc[:,'flag'].copy()
model_RDF=RandomForestClassifier(n_estimators=100,random_state=36)

x_sample_AEP,x_test_AEP,y_sample_AEP,y_test_AEP=train_test_split(x,y_AEP,test_size=0.1,shuffle=True,random_state=6)
model_cons_AEP=func_ATMD_Geo.surrogate_train_validate(x_sample_AEP,x_test_AEP,y_sample_AEP,y_test_AEP,
                                                                     model_cons_AEP,['AEP'])

x_sample_pitch,x_test_pitch,y_sample_pitch,y_test_pitch=train_test_split(x,y_pitch,test_size=0.1,shuffle=True,random_state=194)
model_cons_pitch=func_ATMD_Geo.surrogate_train_validate(x_sample_pitch,x_test_pitch,y_sample_pitch,y_test_pitch,
                                                                     model_cons_pitch,['Platform_ultPitch'])

x_sample_DEL,x_test_DEL,y_sample_DEL,y_test_DEL=train_test_split(x,y_DEL,test_size=0.1,shuffle=True,random_state=845)
model_cons_DEL=func_ATMD_Geo.surrogate_train_validate(x_sample_DEL,x_test_DEL,y_sample_DEL,y_test_DEL,
                                                                     model_cons_DEL,['DEL_TM'])


x_sample_target,x_test_target,y_sample_target,y_test_target=train_test_split(x,y_target,test_size=0.1,shuffle=True,random_state=848)
model_cons_target=func_ATMD_Geo.surrogate_train_validate(x_sample_target,x_test_target,y_sample_target,y_test_target,
                                                                       model_cons_target,['target'])

x_sample_RDF,x_test_RDF,y_sample_RDF,y_test_RDF=train_test_split(x_RDF,y_RDF,shuffle=True,random_state=63)
model_RDF=func_ATMD_Geo.RDFsurrogate_train_validate(x_sample_RDF,x_test_RDF,y_sample_RDF,y_test_RDF,model_RDF)

#%  Bayes optimizer definition and data setting
optimizer=BayesianOptimization(   
    f=None,
    pbounds=bound_nor,
    constraint=constraint,
    verbose=2,
    random_state=1,
    allow_duplicate_points=True
    )
utility=UtilityFunction('ucb',kappa=2.576,xi=0.0)


#mintar=[]
# bad_value=-3     ## when simple constraint exist conflict, set target value with -3(a bad value) to navigate optimizer(no used in this method)
# bad_constraint=[10.212e4,8.866]

if  flag_continue==2:
    data_previous=pd.read_excel('data_AGopt1.xlsx')
    data=data_previous
    for i in range(len(data)):
        x=data.loc[[i],'columns_D':'TMD_stiffness']
        x=func_ATMD_Geo.normalization_x(x,bound)
        y=data.loc[i,'target']
        c=data.loc[i,'AEP':'Platform_ultPitch']
        optimizer.register(params=x,target=y,constraint_value=c)
    print("New optimizer is now aware of {} points.".format(len(optimizer.space)))
    used=len(optimizer.space)
    
    while(1):                                ## drop the columns that no need
        col_first=data.columns
        if col_first[0]=='columns_D':
            break
        data.drop([col_first[0]],axis=1,inplace=True)
elif  flag_continue==1:
    data=data_initial.copy()
    for i in range(len(data)):
        x=data.loc[[i],'columns_D':'TMD_stiffness']
        x=func_ATMD_Geo.normalization_x(x,bound)
        y=data.loc[i,'target']
        c=data.loc[i,'AEP':'Platform_ultPitch']
        optimizer.register(params=x,target=y,constraint_value=c)
    print("New optimizer is now aware of {} points.".format(len(optimizer.space)))
    used=len(optimizer.space)
    while(1):                                ## drop the columns that no need
        col_first=data.columns
        if col_first[0]=='columns_D':
            break
        data.drop([col_first[0]],axis=1,inplace=True)
else:
    print('please input legal number!')
    exit()


for i in range(800):
    print('NO.'+str(i+used)+' sample simulation start!!!')

    next_point=optimizer.suggest(utility)

    point_frame=pd.DataFrame(next_point,index=[i+used],columns=x.columns)
    point_frame=func_ATMD_Geo.anti_normalization_x(point_frame,bound)
    data.loc[i+used]=point_frame.loc[i+used].copy()
    para=data.loc[i+used].copy()

    
    #% simple constrain validate
    flag_simpcon=constraint_simple(para)
    if flag_simpcon==1:
        print('No geometry conflict!!!')
        para['flag_geom']=1
    else:
        print('Geometry conflict exist.')
        target,y_predict_pitch,y_predict_DEL,y_pre_AEP=func_ATMD_Geo.surrogate_predict(point_frame,model_cons_pitch,
                                                                    model_cons_DEL,model_cons_target,bound,model_cons_AEP)
        constraints=[y_predict_DEL[0],y_predict_pitch[0],y_pre_AEP[0]]
        optimizer.register(params=next_point,target=target,constraint_value=constraints)
        para['flag']=0
        para['flag_geom']=0
        para['target']=target
        para['DEL_TM']=y_predict_DEL[0]
        para['Platform_ultPitch']=y_predict_pitch[0]
        para['AEP']=y_pre_AEP[0]
        data.loc[i+used]=para.copy()
        data.to_excel('data_AGopt.xlsx',sheet_name='1',index=True)
        continue

    #% random forest predict
    flag_pred=func_ATMD_Geo.RDFsurrogate_predict(point_frame,model_RDF)
    if flag_pred==0:
        print('High probability to error,skip.')
        target,y_predict_pitch,y_predict_DEL,y_pre_AEP=func_ATMD_Geo.surrogate_predict(point_frame,model_cons_pitch,
                                                                    model_cons_DEL,model_cons_target,bound,model_cons_AEP)
        constraints=[y_predict_DEL[0],y_predict_pitch[0],y_pre_AEP[0]]
        optimizer.register(params=next_point,target=target,constraint_value=constraints)
        para['flag']=0
        para['target']=target
        para['DEL_TM']=y_predict_DEL[0]
        para['Platform_ultPitch']=y_predict_pitch[0]
        para['AEP']=y_pre_AEP[0]
        data.loc[i+used]=para.copy()
        data.to_excel('data_AGopt.xlsx',sheet_name='1',index=True)
        continue

    #% calculate mass and moment of inertia
    para=func_ATMD_Geo.calculate_mass_inertia(para)
    data.loc[i+used]=para.copy()

    #% AQWA running

    num=str(i+used)

    flag_AQWA=func_ATMD_Geo.AQWA_run(num,para)  # flag_AQWA=1 AQWA run successfully 
    
    if flag_AQWA==1:
        print("AQWA simulation complete!!!")
    else:
        print('AQWA calcualtion fail.')
        target,y_predict_pitch,y_predict_DEL,y_pre_AEP=func_ATMD_Geo.surrogate_predict(point_frame,model_cons_pitch,
                                                                    model_cons_DEL,model_cons_target,bound,model_cons_AEP)
        constraints=[y_predict_DEL[0],y_predict_pitch[0],y_pre_AEP[0]]
        optimizer.register(params=next_point,target=target,constraint_value=constraints)
        para['flag']=0
        para['target']=target
        para['DEL_TM']=y_predict_DEL[0]
        para['Platform_ultPitch']=y_predict_pitch[0]
        para['AEP']=y_pre_AEP[0]
        data.loc[i+used]=para.copy()
        data.to_excel('data_AGopt.xlsx',sheet_name='1',index=True)
        continue

    #% transform .AH1 file
    flag_tran=func_ATMD_Geo.transform_AH(num)
    if flag_tran==1:
        print('Hydrodynamic file translation complete!!!')
    else:
        print('Hydrodynamic file translation fail.')

    #%  Openfast run
    para,flag_cp,flag_op,flag_postop,flag_650=func_ATMD_Geo.Openfast_run(num,para)

    if flag_cp==1 and flag_op==1 and flag_postop==1 and flag_650==1:
        print('Openfast running and postprocessing complete!!!')
    else:
        print('Geometry conflict exist.')
        target,y_predict_pitch,y_predict_DEL,y_pre_AEP=func_ATMD_Geo.surrogate_predict(point_frame,model_cons_pitch,
                                                                    model_cons_DEL,model_cons_target,bound,model_cons_AEP)
        constraints=[y_predict_DEL[0],y_predict_pitch[0],y_pre_AEP[0]]
        optimizer.register(params=next_point,target=target,constraint_value=constraints)
        para['flag']=0
        para['target']=target
        para['DEL_TM']=y_predict_DEL[0]
        para['Platform_ultPitch']=y_predict_pitch[0]
        para['AEP']=y_pre_AEP[0]
        data.loc[i+used]=para.copy()
        data.to_excel('data_AGopt.xlsx',sheet_name='1',index=True)
        continue
        
    target=obj_fun(para['platform_mass'])
    constraint_value=constraint_advanced(para['DEL_TM'],para['Platform_ultPitch'],para['AEP'])

    optimizer.register(params=next_point,target=target,constraint_value=constraint_value)

    para['target']=target
    para['flag']=1          ## flag will be set as 1 when all processes complete.
    data.loc[i+used]=para.copy()
    data_initial.loc[i+used]=para.copy()


    model_cons_AEP,x_sample_AEP,y_sample_AEP=func_ATMD_Geo.surrogate_update(model_cons_AEP,point_frame,para,x_sample_AEP,y_sample_AEP
                                                             ,i+used,'AEP',bound)
    model_cons_pitch,x_sample_pitch,y_sample_pitch=func_ATMD_Geo.surrogate_update(model_cons_pitch,point_frame,para,x_sample_pitch,y_sample_pitch
                                                             ,i+used,'Platform_ultPitch',bound)
    model_cons_DEL,x_sample_DEL,y_sample_DEL=func_ATMD_Geo.surrogate_update(model_cons_DEL,point_frame,para,x_sample_DEL,y_sample_DEL
                                                             ,i+used,'DEL_TM',bound)
    model_cons_target,x_sample_target,y_sample_target=func_ATMD_Geo.surrogate_update(model_cons_target,point_frame,para,x_sample_target,y_sample_target
                                                             ,i+used,'target',bound)
    #model_RDF,x_sample_RDF,y_sample_RDF=func_ATMD_Geo.RDFsurrogate_update(model_RDF,para,x_sample_RDF,y_sample_RDF,i+used)
    data.to_excel('data_AGopt.xlsx',sheet_name='1',index=True)
    data_initial.to_excel('initial_data_AGopt.xlsx',sheet_name='1',index=True)
#%%
