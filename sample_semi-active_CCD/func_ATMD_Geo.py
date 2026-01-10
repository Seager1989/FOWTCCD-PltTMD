'''

include whole functions used or unused in main_ATMD_Geo.py

'''

#%%
import numpy as np
import pandas as pd
import math
import shutil,os
import re
import subprocess
from pyFAST.input_output import FASTInputFile
from pyFAST.input_output import FASTOutputFile
import pyFAST.case_generation.runner as runner
from pCrunch import Analysis
import ROSCO_toolbox.ofTools.fast_io.output_processing as fp
from pyDOE import lhs
from sklearn.metrics import mean_squared_error, r2_score
'''
Columns sample
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
    "platform_mass",
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "displace_V",
    "connection_point",
    "CM_plaftform",
    "CMz_TMD",
    "AEP",
    "DEL_TM",
    "Platform_ultPitch",
    'flag',
    'target',
]
'''
#%%
def calculate_mass_inertia(para):
    """""
     This function is used to calculate masss, inertia, displace volume and connection point, update the parameter dataFramea 
     and output it

     para:
        is the parameter dataFrame

    """
    DD=6     #unit m
    TU=3
    TB=1.5
    mass_turbine=1.556104e6   # kg
    density_steel=259.2441  # kg/m**2
    density_concrete=1.1278e3  # kg/m**3

    # calculate displace volume and remaining buoyancy 
    v_column=(para[0]/2)**2*math.pi*(para[3])*4
    v_damplate=(para[1]**2-(para[1]-2*para[4])**2)*TB
    displace_V=v_column+v_damplate
    buoy=1025*displace_V   

    #calculate surface steel mass
    surf_columns=2*math.pi*(para[0]/2)*(para[3]+DD)*4
    surf_damplate=(para[1]**2-(para[1]-2*para[4])**2)*2 -para[0]**2*math.pi+(8*para[1]-8*para[4])*TB
    surf_supplate=((para[2]/math.sin(math.pi/4)*2*para[0]+math.pi/4*para[0]**2)
                   -para[0]**2+4*(15*15-math.pi*15*15/4))*2-para[0]**2*math.pi+((para[2]/math.sin(math.pi/4)-15)*2
                                                                                +0.5*math.pi*15+math.pi*0.5*para[0])*TU*4
    surf=surf_columns+surf_damplate+surf_supplate
    mass_steel=surf*density_steel  

    mass_ballast=buoy-mass_turbine-mass_steel  # remaining buoyancy
    platform_mass=mass_ballast+mass_steel  #calcalate platform mass

    # calculate center of mass
    mass_columns=mass_steel*(surf_columns/surf)
    mass_damplate=mass_steel*(surf_damplate/surf)
    mass_supplate=mass_steel*(surf_supplate/surf)
    CM_steel_columns=(DD-para[3])/2
    CM_steel_supplate=(DD+0.5*TU)
    CM_steel_damplate=-(para[3]+0.5*TB)
    CM_concrete=-(para[3]+0.5*TB+0.4)   #0.4 check in the report concrete ballast CM

    CM_steel=(mass_supplate*CM_steel_supplate+mass_columns*CM_steel_columns+mass_damplate*CM_steel_damplate)/mass_steel
    CM=(CM_steel*mass_steel+CM_concrete*mass_ballast)/platform_mass 

    #calculate steel moment of inertia
    Iz_columns=(1/4*para[0]**2+2*para[2]**2)*mass_columns
    Iz_damplate=1/6*density_steel*para[1]**4-1/6*density_steel*(para[1]-2*para[4])**4
    Iz_supplate=1/12*(para[0]**2+(para[2]/math.sin(math.pi/4)*2+para[0])**2)*mass_supplate
    Iz_steel=Iz_supplate+Iz_columns+Iz_damplate

    Ix_columns=mass_columns*(1/12*(6/4*para[0]**2+(para[3]+DD)**2)+para[2]**2+(CM_steel_columns-CM)**2)
    Ix_damplate=2*density_steel*para[1]**2*(1/12*(para[1]**2+TB**2)+(CM_steel_damplate-CM)**2)\
        -2*density_steel*(para[1]-2*para[4])**2*(1/12*((para[1]-2*para[4])**2+TB**2)+(CM_steel_damplate-CM)**2)
    Ix_supplate=mass_supplate*(1/12*(TU**2+(para[2]/math.sin(math.pi/4)*2+para[0])**2)+(CM_steel_supplate-CM)**2)
    Ix_steel=Ix_supplate+Ix_columns+Ix_damplate
    Iy_steel=Ix_steel

    #calculate conctrete moment of inertia(scaling)
    Iz_concrete=mass_ballast*((para[1]/65.25)*29.264)**2
    Ix_concrete=mass_ballast*(((para[1]/65.25)*21.0863)**2+(CM_concrete-CM)**2)
    Iy_concrete=Ix_concrete

    #calculate total monment of inertia(steel plus concrete)
    Iz=Iz_steel+Iz_concrete
    Ix=Ix_steel+Ix_concrete
    Iy=Iy_steel+Iy_concrete

    ##calculate connection points
    connection_point=  para[2]+(para[0]/2*math.sin(math.pi/4))            

    # set output paremeter
    newpara=para
    newpara["platform_mass"]=platform_mass
    newpara["inertia_x"]=Ix
    newpara["inertia_y"]=Iy
    newpara["inertia_z"]=Iz
    newpara["displace_V"]=displace_V
    newpara["connection_point"]=connection_point
    newpara[ "CM_plaftform"]=CM
    return newpara

def AQWA_run(num,para):
    """""
     This function is used to copy the AQWA files, rewrite the running file and run AQWA

     num:
        is the number of calculating as well as the name of root catalog
     para:
        is the parameter dataFrame

    """
    ## copy files, rewrite files and run AQWA, get .AH1 file
    connection_point=para['connection_point']
    connection_point_negative=-connection_point
    pat=num+'/MAT_DM_files'
    try:
        shutil.copytree('MAT_DM_files',pat)   ##  copy the ansys file
        shutil.copy('MAT_DM.wbpj',num)
        shutil.copy('runJob.py',num)
    except:
        flag=0
        print('AQWA files copy fail.')
        return flag
    
    old_dir=re.compile(r'c:/Users/29953/Desktop/WBtest/')
    dire=os.getcwd().replace('\\','/')
    new_dire=dire+'/'+num+'/'
    file_data=''
    param0='param0.Expression'
    new_para0='  param0.Expression = str("'+str(para[0])+'")\n'   # columns D
    param1='param1.Expression'
    new_para1='  param1.Expression = str("'+str(para[2])+'")\n'   # support plate LB
    param2='param2.Expression'
    new_para2='  param2.Expression = str("'+str(para[3])+'")\n'   # deep DB
    param3='param3.Expression'
    new_para3='  param3.Expression = str("-'+str(para[3])+'")\n'  # negetive deep DB
    param4='param4.Expression'
    new_para4='  param4.Expression = str("'+str(para[1])+'")\n'   # damping plate L
    param5='param5.Expression'
    new_para5='  param5.Expression = str("'+str(para[4])+'")\n'    # wide W
    param15='param15.Expression'
    new_para15='  param15.Expression = str("'+str(para[0])+'")\n'  # columns D
    param8='param8.Expression'
    new_para8='  param8.Expression = str("'+str(para['platform_mass'])+'")\n'    # platform mass(TMD)
    param9='param9.Expression'
    new_para9='  param9.Expression = str("'+str(para['inertia_y'])+'")\n'   # Iy
    param11='param11.Expression'
    new_para11='  param11.Expression = str("'+str(para['inertia_x'])+'")\n'  # Ix
    param13='param13.Expression'
    new_para13='  param13.Expression = str("'+str(para['inertia_z'])+'")\n'  # Iz
    param14='param14.Expression'
    new_para14='  param14.Expression = str("'+str(connection_point)+'")\n'           # point 1 y
    param12='param12.Expression'
    new_para12='  param12.Expression = str("'+str(connection_point)+'")\n'           # point 1 x
    param17='param17.Expression'
    new_para17='  param17.Expression = str("'+str(connection_point_negative)+'")\n'  # point 2 y
    param16='param16.Expression'
    new_para16='  param16.Expression = str("'+str(connection_point)+'")\n'           # point 2 x
    param19='param19.Expression'
    new_para19='  param19.Expression = str("'+str(connection_point)+'")\n'           # point 3 y
    param18='param18.Expression'
    new_para18='  param18.Expression = str("'+str(connection_point_negative)+'")\n'  # point 3 x
    param21='param21.Expression'
    new_para21='  param21.Expression = str("'+str(connection_point_negative)+'")\n'  # point 4 y
    param20='param20.Expression'
    new_para20='  param20.Expression = str("'+str(connection_point_negative)+'")\n'  # point 4 x
    param22='param22.Expression'
    new_para22='  param22.Expression = str("'+str(para['CM_plaftform'])+'")\n'       # CM of platform
    
    with open(new_dire+'/runJob.py',mode='rt',encoding='utf-16') as a:  ##rewrite the Workbench running file
        for line in a:
            if 'c:/Users/29953/Desktop/WBtest/' in line:
                line=old_dir.sub(new_dire,line)
            if param0 in line:
                line=new_para0
            if param1 in line:
                line=new_para1
            if param2 in line:
                line=new_para2
            if param3 in line:
                line=new_para3
            if param4 in line:
                line=new_para4
            if param5 in line:
                line=new_para5
            if param8 in line:
                line=new_para8
            if param9 in line:
                line=new_para9
            if param11 in line:
                line=new_para11
            if param12 in line:
                line=new_para12
            if param13 in line:
                line=new_para13
            if param14 in line:
                line=new_para14
            if param15 in line:
                line=new_para15
            if param16 in line:
                line=new_para16
            if param17 in line:
                line=new_para17
            if param18 in line:
                line=new_para18
            if param19 in line:
                line=new_para19
            if param20 in line:
                line=new_para20
            if param21 in line:
                line=new_para21
            if param22 in line:
                line=new_para22
            file_data += line

    with open(new_dire+'/runJob.py','w',encoding='utf-16') as a:
        a.write(file_data)    
    
    subprocess.run(['cmd.exe','/c'
                    'RunWb2','-F',new_dire+'/MAT_DM.wbpj','-B','-R',new_dire+'/runJob.py'],)  # run AQWA

    try:
        shutil.copy(new_dire+'MAT_DM_copy_files/dp0/AQW/AQW/AQ/Analysis/ANALYSIS.AH1',new_dire)  # copy .AH1 file to root file
        flag=1
    except:
        flag=0
    return flag


def transform_AH(num):
    """""
     This function is used to move the .AH1 file and run the HydroAQWA2FAST to transform AQWA file

     num:
        is the number of calculating as well as the name root catalog
    """
    # copy and run
    shutil.copy('HydroAQWA2FAST.exe',num)
    shutil.copy('InputforHydroAQWA2FAST.txt',num)
    os.chdir(num)
    AQWA2_dir='HydroAQWA2FAST.exe'
    try:
        subprocess.Popen(AQWA2_dir)
        flag=1
    except:
        flag=0
    os.chdir('../')
    return flag


#  run openfast and postprocess
def Openfast_run(num,para):
    """""
     This function is used to process the files of openfast(move the files, change the data in files),
     run in parallel and postprocession.

     active damping files and IEC-1A level turbulent wind files checked.
     weibull ultra pitch angle checked.
     TMD stop spring stiffness and damping checked.

     num:
        is the number of calculating as well as the name root catalog
     para:
        is the parameter dataFrame
    
    """
    DD=6     #unit m
    TU=3
    TB=1.5

    # caclulate the TMD position and stop position
    H=para['deep_DB']+DD+TU+TB
    
    stop_positive=DD+TU
    stop_negative=-para['deep_DB']
    TMD_z=(stop_positive+stop_negative)/2
    # copy openfast files and change values in files 
    try:
        shutil.copytree('DTU_TMD',num+'\DTU_TMD')
        shutil.copy('Cp_Ct_Cq.DTU10MW.txt',num)
        flag_cp=1
    except:
        flag_cp=0
        flag_runop=0
        flag_postop=0
        flag_650=0
        print('copy and move openfast files fail.')
        return para,flag_cp,flag_runop,flag_postop,flag_650
    
    os.chdir(num+'\DTU_TMD\Subcomponents')

    filename='DTU_10MW_NAUTILUS_GoM_HydroDyn_v11.4ms.dat'
    f=FASTInputFile(filename)
    f['PtfmVol0']=para['displace_V']
    f.write(filename)

    for i in range(11):
        filename = "DTU_10MW_NAUTILUS_GoM_HydroDyn_v"+str(i*2+5)+"ms.dat"
        f=FASTInputFile(filename)
        f['PtfmVol0'] = para['displace_V']
        f.write(filename)

    filename='DTU_10MW_NAUTILUS_GoM_ElastoDyn.dat'
    f=FASTInputFile(filename)
    f['PtfmMass']=para['platform_mass']
    f['PtfmRIner']=para['inertia_x']
    f['PtfmPIner']=para['inertia_y']
    f['PtfmYIner']=para['inertia_z']
    f['PtfmCMzt']=para['CM_plaftform']
    f.write('DTU_10MW_NAUTILUS_GoM_ElastoDyn.dat')

    filename='DTU_10MW_NAUTILUS_GoM_MoorDyn.dat'
    mor = FASTInputFile(filename)  
    mor['ConnectionProp'][4,2]=para['connection_point']
    mor['ConnectionProp'][4,3]=para['connection_point']
    mor['ConnectionProp'][5,2]=-para['connection_point']
    mor['ConnectionProp'][5,3]=para['connection_point']
    mor['ConnectionProp'][6,2]=-para['connection_point']
    mor['ConnectionProp'][6,3]=-para['connection_point']
    mor['ConnectionProp'][7,2]=para['connection_point']
    mor['ConnectionProp'][7,3]=-para['connection_point']
    mor.write('DTU_10MW_NAUTILUS_GoM_MoorDyn.dat')

    x_position=[1,1,-1,-1]
    y_position=[1,-1,1,-1]
    for i in range(4):
        TMD_Xposi=x_position[i]*para['support_LB']
        TMD_Yposi=y_position[i]*para['support_LB']
        for j in range(11):
            filename='StC-Sub-Zdof-'+str(i+1)+'-'+str(j*2+5)+'ms.dat'
            f=FASTInputFile(filename)
            f['StC_Z_M']=para['TMD_mass']
            f['StC_Z_K']=para['TMD_stiffness']
            f['StC_Z_C']=para['damp'+str(j*2+5)]
            f['StC_Z_PSP']=stop_positive
            f['StC_Z_NSP']=stop_negative
            f['StC_P_X']=TMD_Xposi
            f['StC_P_Y']=TMD_Yposi
            f['StC_P_Z']=TMD_z
            f.write(filename)

    for i in range(4):
        filename='StC-Sub-Zdof-'+str(i+1)+'-11.4ms.dat'
        f=FASTInputFile(filename)
        f['StC_Z_M']=para['TMD_mass']
        f['StC_Z_K']=para['TMD_stiffness']
        f['StC_Z_C']=para['damp11.4']
        f['StC_Z_PSP']=stop_positive
        f['StC_Z_NSP']=stop_negative
        f['StC_P_Z']=TMD_z
        f['StC_P_X']=x_position[i]*para['support_LB']
        f['StC_P_Y']=y_position[i]*para['support_LB']
        f.write(filename)

    os.chdir('..\..')

    # create paralle run .fst files
    Ncore = 12
    exe_dir='DTU_TMD/'
    fastfiles=[]
    for i in range(11):
        fastfile=os.path.join(exe_dir,"DTU_10MW_NAUTILUS_GoM_A15_v"+str(i*2+5)+"ms.fst")
        fastfiles.append(fastfile)
    fastfile=os.path.join(exe_dir,"DTU_10MW_NAUTILUS_GoM_A15_v11.4ms.fst")
    fastfiles.append(fastfile)

    FAST_EXE=os.path.join(exe_dir,'openfast_x64.exe')

    # run openfast parallelly and data postprocess
    if flag_cp==1:
        try:
            runner.run_cmds(fastfiles, FAST_EXE, parallel=True, showOutputs=False, 
                            nCores=None, showCommand=True, flags=[], verbose=True)
            flag_runop=1
        except:
            print('Openfast simulation fail.')
            flag_runop=0
            flag_postop=0
            flag_650=0
            os.chdir('../')
            return para,flag_cp,flag_runop,flag_postop,flag_650
        try:
            AEP,[DEL_aveBrF, DEL_aveBrM, DEL_aveTrF, DEL_aveTrM
                 ],BldDis,pltform_maxPitch,flag_650=platform_postprocessing(exe_dir)
            flag_postop=1
            if flag_650==0:
                print('Simulation time less than 650s.')
        except:
            print('Openfast postprocess fail.')
            flag_postop=0
            flag_650=0
            os.chdir('../')
            return para,flag_cp,flag_runop,flag_postop,flag_650
    # set para data
    para['CMz_TMD']=TMD_z
    para['AEP']=AEP
    para['DEL_TM']=DEL_aveTrM
    para['Platform_ultPitch']=pltform_maxPitch
    os.chdir('../')
    return para,flag_cp,flag_runop,flag_postop,flag_650

def platform_postprocessing(filepath):
    """""
     This function is used to postprocessing for openfast data, calculate AEP,DEL and platform pitch 
     motion angle based on Weibull ditribution.

     filepath:
        the root file catalog
    """
    this_dir            = os.path.dirname(__file__) 
    flag_650=1
    T = 650  #650
    #%% Initialize processing classes
    # postprocessing for calculating the average power for each wind input
    numbin = 12
    pr_bin = []
    for i in range(3):
    #p1=Analysis.Power_Production().prob_WindDist(windspeed=i+3.5,disttype='cdf')
      p=Analysis.Power_Production().prob_WindDist(windspeed = 2*i+5,disttype = 'pdf')*2   #p*Weibull distribution width 2m/s
      pr_bin.append(p)  #weibul distribution
    p=Analysis.Power_Production().prob_WindDist(windspeed=10.5,disttype='pdf')
    pr_bin.append(p)
    for i in range(numbin-5):
      p=Analysis.Power_Production().prob_WindDist(windspeed = 2*i+13,disttype = 'pdf')*2   #p*Weibull distribution width 2m/s
      pr_bin.append(p)  
    p=Analysis.Power_Production().prob_WindDist(windspeed=11.4,disttype='pdf')
    pr_bin.append(p)
    #%% Collect the fast outputs
    fastoutfiles = []
    for i in range(11):
        caseout=os.path.join(filepath,'DTU_10MW_NAUTILUS_GoM_A15_v'+str(i*2+5)+'ms.out') 
        fastoutfiles.append(caseout) 
    caseout=os.path.join(filepath,'DTU_10MW_NAUTILUS_GoM_A15_v11.4ms.out') 
    fastoutfiles.append(caseout) 
    fastposts=fp.output_processing()
    fastoutputs=fastposts.load_fast_out(filenames=fastoutfiles,tmin=50, tmax= 650)    
    #%% Mean power of each input
    P_bin=[]
    BldDis_V = []
    Pitch_bin=[]
    Pitch_std=[]
    for i in range(12):
        if 200 in fastoutputs[i]['Time']:  #  600s is the total length between tmax and tmin
            pwr=fastoutputs[i]['GenPwr'].mean()
            P_bin.append(pwr)
            blddis_i=fastoutputs[i]['TipDxc1'].max()
            BldDis_V.append(blddis_i)
            pitch=fastoutputs[i]['PtfmPitch'].mean()
            Pitch_bin.append(pitch)
            pit=fastoutputs[i]['PtfmPitch'].std()
            Pitch_std.append(pit)
        else:
            flag_650=0
    BldDis = [max(BldDis_V)]
    #%% AEP calculation
    P_ave=0
    Pitch_wellbull=0
    std_wellbull=0
    for i in range(numbin):
        P_ave += pr_bin[i]*P_bin[i]
    AEP=8760*P_ave

    for i in range(numbin):
        Pitch_wellbull += Pitch_bin[i]*pr_bin[i]
    for i in range(numbin):
        std_wellbull+= Pitch_std[i]*pr_bin[i]

    platform_Pitch=Pitch_wellbull+3*std_wellbull

    #%% Calculate the mean DEL: blade root M abd F; Tower root M and F;
    chan_infos=[['RootFxb1',4],['RootFyb1',4],['RootMxb1',4],['RootMyb1',4],['TwrBsFxt',4],['TwrBsFyt',4],['TwrBsMxt',4],['TwrBsMyt',4]]
    panalyze=Analysis.Loads_Analysis()
    # load DEL vector
    localDel=panalyze.get_DEL(fast_data=fastoutputs, chan_info = chan_infos, binNum = 100, t=T-50)
    # weibull weighted DEL
    DEL_aveBrFx=0
    DEL_aveBrFy=0
    DEL_aveBrMx=0
    DEL_aveBrMy=0
    DEL_aveTrFx=0
    DEL_aveTrFy=0
    DEL_aveTrMx=0
    DEL_aveTrMy=0
    for i in range(numbin):
        DEL_aveBrFx += pr_bin[i]*localDel['RootFxb1'][i]
        DEL_aveBrFy += pr_bin[i]*localDel['RootFyb1'][i]
        DEL_aveBrMx += pr_bin[i]*localDel['RootMxb1'][i]
        DEL_aveBrMy += pr_bin[i]*localDel['RootMyb1'][i]
        DEL_aveTrFx += pr_bin[i]*localDel['TwrBsFxt'][i]
        DEL_aveTrFy += pr_bin[i]*localDel['TwrBsFyt'][i]
        DEL_aveTrMx += pr_bin[i]*localDel['TwrBsMxt'][i]
        DEL_aveTrMy += pr_bin[i]*localDel['TwrBsMyt'][i]
    # Combine X and Y
    DEL_aveBrF = np.sqrt(DEL_aveBrFx**2 + DEL_aveBrFy**2)
    DEL_aveBrM = np.sqrt(DEL_aveBrMx**2 + DEL_aveBrMy**2)
    DEL_aveTrF = np.sqrt(DEL_aveTrFx**2 + DEL_aveTrFy**2)
    DEL_aveTrM = np.sqrt(DEL_aveTrMx**2 + DEL_aveTrMy**2)
    
    #%%
    return AEP, [DEL_aveBrF, DEL_aveBrM, DEL_aveTrF, DEL_aveTrM],BldDis,platform_Pitch,flag_650

def initial_DOE(bound,num,para_num):
    """
    This function is used to generate initial DOE data  

    bound:
        should be the dataFrame as the example in main.py
    num:
        is the number of samples
    para_num:
        is the number of the parameters
    """
    samp=lhs(para_num,samples=num,criterion='m')    # latin-hypercube generate
    samp=pd.DataFrame(samp)
    samp.columns=bound.columns[0:para_num]
    bound.loc[3]=(bound.loc[1]-bound.loc[0])/bound.loc[2]   # calculate total step
    #print(bound)
    for i in range(para_num):
       sa=samp.columns[i]
       samp[sa]=samp[sa]*bound.loc[3][sa]      # get step
    samp=round(samp)       #int
    for i in range(para_num):
       sa=samp.columns[i]
       samp[sa]=samp[sa]*bound.loc[2][sa]+bound.loc[0][sa]    # change step to value

    i=0
    while 1 :
        if i==len(samp):
           break
        test=samp.loc[i]
        constr1=test[2]/math.sin(math.pi/4)-test[0]/2-\
               (test[1]/2-test[4])/math.sin(math.pi/4)+2.1746 #圆柱在板内>0
        constr2=1.414*test[4]-test[0]                         #板宽度大于浮筒直径
        constr3=test[1]/2-test[4]                             #板成形
        constr4=test[1]/math.sin(math.pi/4)-(2*5.25/math.sin(math.pi/4)-2*5.25)-\
                2*test[2]/math.sin(math.pi/4)-test[0]         #浮筒在板内
        
        if  constr1>0 and constr2>0 and constr3>5.25 and constr4>0 :
            i=i+1
        else:
            newone=lhs(para_num,samples=2,criterion='m')   # generate new sample(ramdon)
            newone=pd.DataFrame(newone)
            newone.drop(index=1,inplace=True)
            newone.columns=bound.columns[0:para_num]
            for j in range(para_num):
                sa=newone.columns[j]
                newone[sa]=newone[sa]*bound.loc[3][sa]
            newone=round(newone)
            for j in range(para_num):
                sa=newone.columns[j]
                newone[sa]=newone[sa]*bound.loc[2][sa]+bound.loc[0][sa]
            samp.loc[i]=newone.loc[0]
    return samp

def Mean_error(y_test,y_pred):  
    '''
    this function is used to calculate the mean error of surrogate model prediction data
    '''
    error=0
    for i in range(len(y_test)):
       error += abs(y_test.loc[i]-y_pred.loc[i])
    mean_error=error/len(y_test)
    return mean_error

def Relative_error(y_test,y_pred):
    '''
    this function is used to calculate the relative error of surrogate model prediction data
    '''
    error=0
    for i in range(len(y_test)):
        error += abs(y_test.loc[i]-y_pred.loc[i])/abs(y_test.loc[i])
    relative_error=100*error/len(y_test)
    return relative_error

def surrogate_train_validate(x_sample,x_test,y_sample,y_test,model_cons,col):
    x_sample=x_sample.reset_index(drop=True)
    y_sample=y_sample.reset_index(drop=True)
    x_test=x_test.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    
    
    model_cons.fit(x_sample,y_sample)
  
  
    y_pred=model_cons.predict(x_test)
    y_pred=pd.DataFrame(y_pred,columns=col)
  
    mse=mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test,y_pred)
    print('mean squared error:',mse)
    print('r2:',r2)
    meanerror=Mean_error(y_test,y_pred)
    relaerror=Relative_error(y_test,y_pred)
    print('mean error',meanerror)
    print('relative error',relaerror)
    return model_cons

def RDFsurrogate_train_validate(x_sample,x_test,y_sample,y_test,model_RDF,):
    x_sample=x_sample.reset_index(drop=True)
    y_sample=y_sample.reset_index(drop=True)
    x_test=x_test.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    model_RDF.fit(x_sample,y_sample)
    y_pred=model_RDF.predict(x_test)
    y_pred=pd.DataFrame(y_pred,columns=['flag'])
    count=0
    for i in range(len(y_test)):
        if y_test.loc[i]==y_pred.loc[i,'flag']:
            count+=1

    print('right count',count)
    relative=100*count/len(y_test)
    print('Accuracy%',relative)
    return model_RDF


def surrogate_predict(point_frame,mod1_pitch,mod1_DEL,mod1_tar,bound,mod1_AEP):
    # y_predict_AEP=model_cons_AEP.predict(point_frame)
    # y_predict_AEP=pd.DataFrame(y_predict_AEP,columns=['AEP'])

    x=normalization_x(point_frame,bound)
    y_pre_AEP=mod1_AEP.predict(x.loc[:,'columns_D':'TMD_stiffness'])
    y_pre_pitch=mod1_pitch.predict(x.loc[:,'columns_D':'TMD_stiffness'])
    y_pre_DEL=mod1_DEL.predict(x.loc[:,'columns_D':'TMD_stiffness'])
    y_pre_target=mod1_tar.predict(x.loc[:,'columns_D':'TMD_stiffness'])
    
    y_pre_target=pd.DataFrame(y_pre_target,columns=['target'])
    target=y_pre_target.loc[0,'target']

    return target,y_pre_pitch,y_pre_DEL,y_pre_AEP

def RDFsurrogate_predict(point_frame,model_RDF):
    y_pred=model_RDF.predict(point_frame.loc[:,'columns_D':'wide_W'])
    return y_pred

def surrogate_update(mod1,point_frame,para,x_sample,y_sample,n,feature,bound):
    
    
    x=normalization_x(point_frame,bound)
    x_sample=x_sample.append(x)
    
   
    y_sample.loc[n]=para[feature].copy()
    mod1.fit(x_sample,y_sample)
    return mod1,x_sample,y_sample

def RDFsurrogate_update(mod,para,x_sample,y_sample,n):
    x_sample.loc[n]=para['columns_D':'wide_W'].copy()
    y_sample.loc[n]=para['flag'].copy()
    mod.fit(x_sample,y_sample)
    return mod,x_sample,y_sample
#%%
def calculate_mass_inertia_Awater(para):
    """""
     include ballast water
     This function is used to calculate masss, inertia, displace volume and connection point, update the parameter dataFramea 
     and output it

     para:
        is the parameter dataFrame

    """
    mass_water=1.2e6
    Ix_water=0.909e9
    Iy_water=Ix_water
    Iz_water=1.815e9
    DD=6     #unit m
    TU=3
    TB=1.5
    mass_turbine=1.556104e6   # kg
    density_steel=259.2441  # kg/m**2
    density_concrete=1.1278e3  # kg/m**3

    # calculate displace volume and remaining buoyancy 
    v_column=(para[0]/2)**2*math.pi*(para[3])*4
    v_damplate=(para[1]**2-(para[1]-2*para[4])**2)*TB
    displace_V=v_column+v_damplate
    buoy=1025*displace_V   

    #calculate surface steel mass
    surf_columns=2*math.pi*(para[0]/2)*(para[3]+DD)*4
    surf_damplate=(para[1]**2-(para[1]-2*para[4])**2)*2 -para[0]**2*math.pi+(8*para[1]-8*para[4])*TB
    surf_supplate=((para[2]/math.sin(math.pi/4)*2*para[0]+math.pi/4*para[0]**2)
                   -para[0]**2+4*(15*15-math.pi*15*15/4))*2-para[0]**2*math.pi+((para[2]/math.sin(math.pi/4)-15)*2
                                                                                +0.5*math.pi*15+math.pi*0.5*para[0])*TU*4
    surf=surf_columns+surf_damplate+surf_supplate
    mass_steel=surf*density_steel  

    mass_ballast=buoy-mass_turbine-mass_steel-mass_water  # remaining buoyancy
    platform_mass=mass_ballast+mass_steel+mass_water  #calcalate platform mass

    # calculate center of mass
    mass_columns=mass_steel*(surf_columns/surf)
    mass_damplate=mass_steel*(surf_damplate/surf)
    mass_supplate=mass_steel*(surf_supplate/surf)
    CM_steel_columns=(DD-para[3])/2
    CM_steel_supplate=(DD+0.5*TU)
    CM_steel_damplate=-(para[3]+0.5*TB)
    CM_concrete=-(para[3]+0.5*TB+0.4)   #0.4 check in the report concrete ballast CM
    CM_water=-para['deep_DB']+3.5/2

    CM_steel=(mass_supplate*CM_steel_supplate+mass_columns*CM_steel_columns+mass_damplate*CM_steel_damplate)/mass_steel
    CM=(CM_steel*mass_steel+CM_concrete*mass_ballast+CM_water*mass_water)/platform_mass 

    #calculate steel moment of inertia
    Iz_columns=(1/4*para[0]**2+2*para[2]**2)*mass_columns
    Iz_damplate=1/6*density_steel*para[1]**4-1/6*density_steel*(para[1]-2*para[4])**4
    Iz_supplate=1/12*(para[0]**2+(para[2]/math.sin(math.pi/4)*2+para[0])**2)*mass_supplate
    Iz_steel=Iz_supplate+Iz_columns+Iz_damplate

    Ix_columns=mass_columns*(1/12*(6/4*para[0]**2+(para[3]+DD)**2)+para[2]**2+(CM_steel_columns-CM)**2)
    Ix_damplate=2*density_steel*para[1]**2*(1/12*(para[1]**2+TB**2)+(CM_steel_damplate-CM)**2)\
        -2*density_steel*(para[1]-2*para[4])**2*(1/12*((para[1]-2*para[4])**2+TB**2)+(CM_steel_damplate-CM)**2)
    Ix_supplate=mass_supplate*(1/12*(TU**2+(para[2]/math.sin(math.pi/4)*2+para[0])**2)+(CM_steel_supplate-CM)**2)
    Ix_steel=Ix_supplate+Ix_columns+Ix_damplate
    Iy_steel=Ix_steel

    #calculate conctrete moment of inertia(scaling)
    Iz_concrete=mass_ballast*((para[1]/65.25)*29.264)**2
    Ix_concrete=mass_ballast*(((para[1]/65.25)*21.0863)**2+(CM_concrete-CM)**2)
    Iy_concrete=Ix_concrete

    #calculate total monment of inertia(steel plus concrete)
    Iz=Iz_steel+Iz_concrete+Iz_water
    Ix=Ix_steel+Ix_concrete+Ix_water
    Iy=Iy_steel+Iy_concrete+Iy_water

    ##calculate connection points
    connection_point=  para[2]+(para[0]/2*math.sin(math.pi/4))            

    # set output paremeter
    newpara=para
    newpara["platform_mass"]=platform_mass
    newpara["inertia_x"]=Ix
    newpara["inertia_y"]=Iy
    newpara["inertia_z"]=Iz
    newpara["displace_V"]=displace_V
    newpara["connection_point"]=connection_point
    newpara[ "CM_plaftform"]=CM
    return newpara

def normalization_x(x_sample,bound):
    y_sample=x_sample.copy()
    for key in bound:
        y_sample[key]=x_sample[key].map(lambda x: (x-bound.get(key)[0])/(bound.get(key)[1]-bound.get(key)[0]))

    return y_sample

def anti_normalization_x(x_sample,bound):
    for key in bound:
        x_sample[key]=x_sample[key].map(lambda x: x*(bound.get(key)[1]-bound.get(key)[0])+bound.get(key)[0])

    return x_sample
    
def calculate_mass_inertia_TMD(para):
    """""
     This function is used to calculate masss, inertia, displace volume and connection point, update the parameter dataFramea 
     and output it

     para:
        is the parameter dataFrame

    """
    DD=6     #unit m
    TU=3
    TB=1.5
    mass_turbine=1.556104e6   # kg
    density_steel=259.2441  # kg/m**2
    density_concrete=1.1278e3  # kg/m**3
    
    stop_positive=DD+TU
    stop_negative=-para['deep_DB']
    TMD_z=(stop_positive+stop_negative)/2
    
    # calculate displace volume and remaining buoyancy 
    v_column=(para[0]/2)**2*math.pi*(para[3])*4
    v_damplate=(para[1]**2-(para[1]-2*para[4])**2)*TB
    displace_V=v_column+v_damplate
    buoy=1025*displace_V   

    #calculate surface steel mass
    surf_columns=2*math.pi*(para[0]/2)*(para[3]+DD)*4
    surf_damplate=(para[1]**2-(para[1]-2*para[4])**2)*2 -para[0]**2*math.pi+(8*para[1]-8*para[4])*TB
    surf_supplate=((para[2]/math.sin(math.pi/4)*2*para[0]+math.pi/4*para[0]**2)
                   -para[0]**2+4*(15*15-math.pi*15*15/4))*2-para[0]**2*math.pi+((para[2]/math.sin(math.pi/4)-15)*2
                                                                                +0.5*math.pi*15+math.pi*0.5*para[0])*TU*4
    surf=surf_columns+surf_damplate+surf_supplate
    mass_steel=surf*density_steel  

    mass_ballast=buoy-mass_turbine-mass_steel-para['TMD_mass']*4  # remaining buoyancy
    platform_mass=mass_ballast+mass_steel  #calcalate platform mass
    platformTMD_mass=platform_mass+para['TMD_mass']*4

    # calculate center of mass
    mass_columns=mass_steel*(surf_columns/surf)
    mass_damplate=mass_steel*(surf_damplate/surf)
    mass_supplate=mass_steel*(surf_supplate/surf)
    CM_steel_columns=(DD-para[3])/2
    CM_steel_supplate=(DD+0.5*TU)
    CM_steel_damplate=-(para[3]+0.5*TB)
    CM_concrete=-(para[3]+0.5*TB+0.4)   #0.4 check in the report concrete ballast CM

    CM_steel=(mass_supplate*CM_steel_supplate+mass_columns*CM_steel_columns+mass_damplate*CM_steel_damplate)/mass_steel
    CM=(CM_steel*mass_steel+CM_concrete*mass_ballast)/platform_mass 
    CM_TMD=(CM_steel*mass_steel+CM_concrete*mass_ballast+4*para['TMD_mass']*TMD_z)/platformTMD_mass 

    #calculate steel moment of inertia
    Iz_columns=(1/4*para[0]**2+2*para[2]**2)*mass_columns
    Iz_damplate=1/6*density_steel*para[1]**4-1/6*density_steel*(para[1]-2*para[4])**4
    Iz_supplate=1/12*(para[0]**2+(para[2]/math.sin(math.pi/4)*2+para[0])**2)*mass_supplate
    Iz_steel=Iz_supplate+Iz_columns+Iz_damplate
    

    Ix_columns=mass_columns*(1/12*(6/4*para[0]**2+(para[3]+DD)**2)+para[2]**2+(CM_steel_columns-CM)**2)
    Ix_damplate=2*density_steel*para[1]**2*(1/12*(para[1]**2+TB**2)+(CM_steel_damplate-CM)**2)\
        -2*density_steel*(para[1]-2*para[4])**2*(1/12*((para[1]-2*para[4])**2+TB**2)+(CM_steel_damplate-CM)**2)
    Ix_supplate=mass_supplate*(1/12*(TU**2+(para[2]/math.sin(math.pi/4)*2+para[0])**2)+(CM_steel_supplate-CM)**2)
    Ix_steel=Ix_supplate+Ix_columns+Ix_damplate
    Iy_steel=Ix_steel

    #calculate conctrete moment of inertia(scaling)
    Iz_concrete=mass_ballast*((para[1]/65.25)*29.264)**2
    Ix_concrete=mass_ballast*(((para[1]/65.25)*21.0863)**2+(CM_concrete-CM)**2)
    Iy_concrete=Ix_concrete
  
    #calculate TMD moment of inertia
    Iz_TMD=4*para['TMD_mass']*(para['support_LB']^2+para['support_LB']^2)
    Ix_TMD=4*para['TMd_mass']*(TMD_z^2+para['support_LB']^2)
    Iy_TMD=Ix_TMD

    #calculate total monment of inertia(steel plus concrete)
    Iz=Iz_steel+Iz_concrete
    Ix=Ix_steel+Ix_concrete
    Iy=Iy_steel+Iy_concrete
    
    IZ=Iz_steel+Iz_concrete+Iz_TMD
    IX=Ix_steel+Ix_concrete+Ix_TMD
    IY=Iy_steel+Iy_concrete+Iy_TMD
    ##calculate connection points
    connection_point= para[2]+(para[0]/2*math.sin(math.pi/4))            

    # set output paremeter
    newpara=para
    newpara["platform_mass"]=platform_mass
    newpara['platformTMD_mass']=platformTMD_mass
    newpara["inertia_x"]=Ix
    newpara["inertia_y"]=Iy
    newpara["inertia_z"]=Iz
    newpara['platTMDiner_x']=IX
    newpara['platTMDiner_y']=IY
    newpara['platTMDiner_z']=IZ
    newpara["displace_V"]=displace_V
    newpara["connection_point"]=connection_point
    newpara["CM_plaftform"]=CM
    newpara['CM_platformTMD']=CM_TMD
    return newpara

def sensiticity_para_simple(para,num,per):
    para_y=para.copy()
    para_y.loc[0,num]=para.loc[0,num]*(1+per)
    return para_y

def AQWA_run_includeTMD(num,para):
    """""
     This function is used to copy the AQWA files, rewrite the running file and run AQWA(platform include TMD property)

     num:
        is the number of calculating as well as the name of root catalog
     para:
        is the parameter dataFrame

    """
    ## copy files, rewrite files and run AQWA, get .AH1 file
    connection_point=para['connection_point']
    connection_point_negative=-connection_point
    pat=num+'/MAT_DM_files'
    try:
        shutil.copytree('MAT_DM_files',pat)   ##  copy the ansys file
        shutil.copy('MAT_DM.wbpj',num)
        shutil.copy('runJob.py',num)
    except:
        flag=0
        print('AQWA files copy fail.')
        return flag
    
    old_dir=re.compile(r'c:/Users/29953/Desktop/WBtest/')
    dire=os.getcwd().replace('\\','/')
    new_dire=dire+'/'+num+'/'
    file_data=''
    param0='param0.Expression'
    new_para0='  param0.Expression = str("'+str(para[0])+'")\n'   # columns D
    param1='param1.Expression'
    new_para1='  param1.Expression = str("'+str(para[2])+'")\n'   # support plate LB
    param2='param2.Expression'
    new_para2='  param2.Expression = str("'+str(para[3])+'")\n'   # deep DB
    param3='param3.Expression'
    new_para3='  param3.Expression = str("-'+str(para[3])+'")\n'  # negetive deep DB
    param4='param4.Expression'
    new_para4='  param4.Expression = str("'+str(para[1])+'")\n'   # damping plate L
    param5='param5.Expression'
    new_para5='  param5.Expression = str("'+str(para[4])+'")\n'    # wide W
    param15='param15.Expression'
    new_para15='  param15.Expression = str("'+str(para[0])+'")\n'  # columns D
    param8='param8.Expression'
    new_para8='  param8.Expression = str("'+str(para['platformTMD_mass'])+'")\n'    # platform mass(TMD)
    param9='param9.Expression'
    new_para9='  param9.Expression = str("'+str(para['platTMDiner_y'])+'")\n'   # Iy
    param11='param11.Expression'
    new_para11='  param11.Expression = str("'+str(para['platTMDiner_x'])+'")\n'  # Ix
    param13='param13.Expression'
    new_para13='  param13.Expression = str("'+str(para['platTMDiner_z'])+'")\n'  # Iz
    param14='param14.Expression'
    new_para14='  param14.Expression = str("'+str(connection_point)+'")\n'           # point 1 y
    param12='param12.Expression'
    new_para12='  param12.Expression = str("'+str(connection_point)+'")\n'           # point 1 x
    param17='param17.Expression'
    new_para17='  param17.Expression = str("'+str(connection_point_negative)+'")\n'  # point 2 y
    param16='param16.Expression'
    new_para16='  param16.Expression = str("'+str(connection_point)+'")\n'           # point 2 x
    param19='param19.Expression'
    new_para19='  param19.Expression = str("'+str(connection_point)+'")\n'           # point 3 y
    param18='param18.Expression'
    new_para18='  param18.Expression = str("'+str(connection_point_negative)+'")\n'  # point 3 x
    param21='param21.Expression'
    new_para21='  param21.Expression = str("'+str(connection_point_negative)+'")\n'  # point 4 y
    param20='param20.Expression'
    new_para20='  param20.Expression = str("'+str(connection_point_negative)+'")\n'  # point 4 x
    param22='param22.Expression'
    new_para22='  param22.Expression = str("'+str(para['CM_platformTMD'])+'")\n'       # CM of platform
    
    with open(new_dire+'/runJob.py',mode='rt',encoding='utf-16') as a:  ##rewrite the Workbench running file
        for line in a:
            if 'c:/Users/29953/Desktop/WBtest/' in line:
                line=old_dir.sub(new_dire,line)
            if param0 in line:
                line=new_para0
            if param1 in line:
                line=new_para1
            if param2 in line:
                line=new_para2
            if param3 in line:
                line=new_para3
            if param4 in line:
                line=new_para4
            if param5 in line:
                line=new_para5
            if param8 in line:
                line=new_para8
            if param9 in line:
                line=new_para9
            if param11 in line:
                line=new_para11
            if param12 in line:
                line=new_para12
            if param13 in line:
                line=new_para13
            if param14 in line:
                line=new_para14
            if param15 in line:
                line=new_para15
            if param16 in line:
                line=new_para16
            if param17 in line:
                line=new_para17
            if param18 in line:
                line=new_para18
            if param19 in line:
                line=new_para19
            if param20 in line:
                line=new_para20
            if param21 in line:
                line=new_para21
            if param22 in line:
                line=new_para22
            file_data += line

    with open(new_dire+'/runJob.py','w',encoding='utf-16') as a:
        a.write(file_data)    
    
    subprocess.run(['cmd.exe','/c'
                    'RunWb2','-F',new_dire+'/MAT_DM.wbpj','-B','-R',new_dire+'/runJob.py'],)  # run AQWA

    try:
        shutil.copy(new_dire+'MAT_DM_copy_files/dp0/AQW/AQW/AQ/Analysis/ANALYSIS.AH1',new_dire)  # copy .AH1 file to root file
        flag=1
    except:
        flag=0
    return flag


def actballast(para,per):
    mass_water=1.2e6
    inertia_x_water=0.909e9
    inertia_y_water=inertia_x_water
    inertia_z_water=1.815e9

    para['inertia_x']=para['inertia_x']+inertia_x_water*per
    para['inertia_y']=para['inertia_y']+inertia_y_water*per
    para['inertia_z']=para['inertia_z']+inertia_z_water*per
    para['platform_mass']=para['platform_mass']+mass_water*per
    return para