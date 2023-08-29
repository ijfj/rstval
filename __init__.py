
def ncountraster(rst):
    ''' 
    Count of units by different values of a raster
    
    Input: 
    - rst: raster name
    
    Ouput: 
    - d: dictionary { raster value : occurrence count }
    '''
    from osgeo import gdal
    import numpy as np
    
    rst_link = gdal.Open(rst) 
    rst_np = np.array(rst_link.GetRasterBand(1).ReadAsArray().astype(int))
    
    values, counts=np.unique(rst_np, return_counts=True)
    d={}
    for i in range(values.shape[0]):
        d[values[i]]=counts[i]

    return d

def count3d(ref_v, cl_v, str_v, n_str, kv_d, ref_el=None, cl_el=None):
    ''' 

    Count of units with the value ref_el in ref_v and cl_el in cl_v.
    
    Input: 
    - ref_v: reference vector 
    - cl_v: classification vector
    - str_v: strata vector
    - n_str: number os strata
    - kv_d: dictionary with correspondence between the position in the vector and strata label
    - ref_el=None: element of the reference, when None all values of the reference count
    - cl_el=None: raster name, when None all values of the classification count
    
    Ouput: 
    - count: vector of n_str elements. Each position (i) corresponds to the number of occurrences ref_el and cl_el in the stratum kv_d[i]

    '''
    n=len(ref_v) 
    count=[0]*n_str
    for i in range(n):
        ind_count=list(kv_d.values()).index(str_v[i])
        if ((ref_el!=None) and (cl_el!=None)):
            if ((ref_v[i]==ref_el) and (cl_v[i]==cl_el)):  
                count[ind_count]+=1 
        elif (cl_el!=None):
            if (cl_v[i]==cl_el):  
                count[ind_count]+=1 
        elif (ref_el!=None): 
            if (ref_v[i]==ref_el):  
                count[ind_count]+=1 
        else: 
            count[ind_count]+=1 
    return count
    
def confmatrix3d_count(cl_v, ref_v, str_v):
    ''' 

    Construction of the confusion 3D-matrix
    
    Input: 
    - ref_v: reference vector 
    - cl_v: classification vector
    - str_v: strata vector
    
    Ouput: 
    - aconf: dataframe of pandas with the confusion matrix (lines/columns: map/classification). Each position is a vector with h elements, where h is the number of strata (different values in str_v)
    - kv_d: dictionary with correspondence between the position in the vectors in the confusion matrix and strata label

    '''
    import numpy as np
    import pandas as pd
    

    # number of classes 
    classes_cl=np.sort(np.unique(cl_v))
    classes_ref=np.sort(np.unique(ref_v))
    
    # verify if all classes are the same in cl and ref
    # np.array_equal(classes_cl,classes_ref))

    # vector position - label strata
    kv_str=np.unique(str_v)
    kv_str=np.sort(kv_str)
    n_str=len(kv_str)

    kv_d={}
    for i in range(n_str):
        kv_d[i]=kv_str[i]

    #Confusion Matrix
    # lines VS columns : map VS classification
    cmh={}

    for class_ref in classes_ref:
        cmh[class_ref]={} 
        for class_cl in classes_cl:
            cmh[class_ref][class_cl]=count3d(ref_v, cl_v, str_v, n_str, kv_d, ref_el=class_ref, cl_el=class_cl)

    cmh['t']={}
    for class_ref in classes_ref: 
        for class_cl in classes_cl: 
            cmh['t'][class_cl]=count3d(ref_v, cl_v, str_v, n_str, kv_d, cl_el=class_cl)
        cmh[class_ref]['t']=count3d(ref_v, cl_v, str_v, n_str, kv_d, ref_el=class_ref)
    
    cmh['t']['t']=count3d(ref_v, cl_v, str_v, n_str, kv_d)
            
    
    aconf=pd.DataFrame(cmh) 
    return aconf, kv_d


def count2d(ref_v, cl_v, ref_el=None, cl_el=None,):
    ''' 

    Count of units with the value ref_el in ref_v and cl_el in cl_v.
    
    Input: 
    - ref_v: reference vector 
    - cl_v: classification vector  
    - ref_el=None: element of the reference, when None all values of the reference count
    - cl_el=None: raster name, when None all values of the classification count
    
    Ouput: 
    - count: number of occurrences ref_el and cl_el

    '''
    n=len(ref_v)
    count=0
    for i in range(n): 
        if ((ref_el!=None) and (cl_el!=None)):
            if ((ref_v[i]==ref_el) and (cl_v[i]==cl_el)):  
                count+=1 
        elif (cl_el!=None):
            if (cl_v[i]==cl_el):  
                count+=1 
        elif (ref_el!=None): 
            if (ref_v[i]==ref_el):  
                count+=1 
        else: 
            count+=1 
    return count
    
def confmatrix2d_count(cl_v, ref_v):
    ''' 

    Construction of the confusion matrix with counts
    
    Input: 
    - ref_v: reference vector 
    - cl_v: classification vector
    - str_v: strata vector
    
    Ouput: 
    - aconf: dataframe of pandas with the confusion matrix (lines/columns: map/classification).

    '''
    import numpy as np
    import pandas as pd 

    # number of classes 
    classes_cl=np.sort(np.unique(cl_v))
    classes_ref=np.sort(np.unique(ref_v))
    
    # verify if all classes are the same in cl and ref
    # np.array_equal(classes_cl,classes_ref))

    #Confusion Matrix
    cm={}

    for class_ref in classes_ref:
        cm[class_ref]={} 
        for class_cl in classes_cl:
            cm[class_ref][class_cl]=count2d(ref_v, cl_v, ref_el=class_ref, cl_el=class_cl)

    cm['t']={}
    for class_ref in classes_ref: 
        for class_cl in classes_cl: 
            cm['t'][class_cl]=count2d(ref_v, cl_v, cl_el=class_cl)
        cm[class_ref]['t']=count2d(ref_v, cl_v, ref_el=class_ref)
    
    cm['t']['t']=count2d(ref_v, cl_v)
            
    
    aconf=pd.DataFrame(cm) 
    return aconf

def confmatrix2d_prop(cm_count, cl_counts):
    ''' 

    Construction of the confusion matrix with proportions
    
    Input: 
    - cm_count: confusion matrix with counts 
    - cl_counts: dictionary with the number of units or weight (value) by class (key)
    
    Ouput: 
    - cm: dataframe of pandas with the confusion matrix (lines/columns: map/classification).

    '''
    import numpy as np
    import pandas as pd 
    cm=cm_count.copy()
    N=sum(cl_counts.values())
    cl_ref=cm.columns.values.tolist()[:-1] #j
    cl_cl=cm.index.tolist()[:-1] #i
    
    for i in cl_cl:
        Ni=cl_counts[i]
        ni=cm.loc[i]['t']
        for j in cl_ref:
            nij=cm.loc[i][j] 
            cm.at[i,j]=Ni/N*nij/ni
    
    
    for i in  cl_cl:
        sumaux=0.0
        for j in cl_ref:
            sumaux += cm.loc[i][j]
        cm.at[i, 't']=sumaux

    
    for j in  cl_ref:
        sumaux=0.0
        for i in cl_cl:
            sumaux += cm.loc[i][j]
        cm.at['t', j]=sumaux

    cm.at['t', 't']=1.0
    return cm


def metrics3d_count(cm3d, str_kv, str_counts, areaunit=None):
    ''' 

    Computation of the estimated metrics when the sample is stratified and the strata are not (mandatory) be the classes
    
    Input: 
    - cm3d: confusion 3D-matrix with counts 
    - str_kv: dictionary with correspondence between the position in the vector and strata label
    - str_counts: dictionary with the number of units or weight (value) by label stratum (key)
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    
    Ouput: 
    - metrics: dataframe of pandas with the estimated metrics

    '''
    import numpy as np
    import pandas as pd 
    import math as m

    # number of units for each stratum (in the same order of the vectors in cm3d)
    areas=[0]*len(str_kv.keys())
    for i in range(len(areas)):
        areas[i]=str_counts[str_kv[i]]
    N=sum(areas)

    # metrics
    metrics={} 
    cl_ref=cm3d.columns.values.tolist()[:-1] #j
    cl_cl=cm3d.index.tolist()[:-1] #i
    
    # number of units for each class
    metrics['Nj']={}
    metrics['v_Nj']={}
    metrics['sd_Nj']={}
    if areaunit:
        metrics['Aj']={}
        metrics['v_Aj']={}
        metrics['sd_Aj']={}
    metrics['Pj']={}
    metrics['v_Pj']={}
    metrics['sd_Pj']={}
    for j in cl_ref:
        Nj=0
        vNj=0
        for k in range(len(areas)):
            Nk=areas[k]
            n_jk=cm3d[j]['t'][k]
            nk=cm3d['t']['t'][k]
            #if j==2:
            #    print(k, Nk, n_jk, nk)
            Nj += Nk * n_jk/nk
            
            yk    = n_jk/nk
            s2yk  = nk/(nk-1) * yk*(1-yk)
            vNj += Nk * Nk/nk * (1 - nk/Nk) * s2yk

        metrics['Nj'][j]    = Nj
        metrics['v_Nj'][j]  = vNj
        metrics['sd_Nj'][j] = m.sqrt(vNj)
        if areaunit:
            metrics['Aj'][j]    = Nj*areaunit
            metrics['v_Aj'][j]  = vNj*areaunit*areaunit
            metrics['sd_Aj'][j] = m.sqrt(vNj*areaunit*areaunit)
        metrics['Pj'][j]    = Nj/N
        metrics['v_Pj'][j]  = vNj/(N*N)
        metrics['sd_Pj'][j] = m.sqrt(vNj/(N*N))

    # user's accuracy
    metrics['UAi']={}
    metrics['v_UAi']={}
    metrics['sd_UAi']={}
    metrics['CEi']={}
    metrics['v_CEi']={}
    metrics['sd_CEi']={}
    for i in cl_cl:
        numUAi = 0.0
        denUAi = 0.0 

        for k in range(len(areas)):
            Nk=areas[k]
            niik=cm3d.loc[i][i][k]
            ni_k=cm3d.loc[i]['t'][k]
            nk=cm3d['t']['t'][k]
            
            numUAi += Nk * niik/nk
            denUAi += Nk * ni_k/nk
            #print('i=', i, 'k=', k, 'Nk=', Nk, 'niik=', niik, 'ni_k=', ni_k, 'nk=', nk)
        
        UAi = numUAi/denUAi
        
        #print(numUAi, denUAi)
        numvUAi = 0.0
        denvUAi = 0.0 

        for k in range(len(areas)):
            Nk=areas[k]
            niik=cm3d.loc[i][i][k]
            ni_k=cm3d.loc[i]['t'][k]
            nk=cm3d['t']['t'][k]

            yk=niik/nk
            xk=ni_k/nk
            s2yk= nk/(nk-1) * yk * (1-yk)
            s2xk= nk/(nk-1) * xk * (1-xk)
            sxyk= nk/(nk-1) * yk * (1-xk)

            numvUAi += Nk*Nk/nk * (1-nk/Nk) * (s2yk + UAi*UAi*s2xk - 2*UAi*sxyk)
            denvUAi += Nk*xk

        vUAi=numvUAi/(denvUAi*denvUAi)

        metrics['UAi'][i]    = UAi
        metrics['v_UAi'][i]  = vUAi
        metrics['sd_UAi'][i] = m.sqrt(vUAi)
        metrics['CEi'][i]    = 1-UAi
        metrics['v_CEi'][i]  = vUAi
        metrics['sd_CEi'][i] = m.sqrt(vUAi)

    #print(pd.DataFrame(metrics)[['UAi', 'v_UAi', 'sd_UAi', 'CEi', 'v_CEi', 'sd_CEi']])


    # producer's accuracy
    metrics['PAj']={}
    metrics['v_PAj']={}
    metrics['sd_PAj']={}
    metrics['OEj']={}
    metrics['v_OEj']={}
    metrics['sd_OEj']={}
    for j in cl_ref:
        numPAj = 0.0
        denPAj = 0.0 

        for k in range(len(areas)):
            Nk=areas[k]
            njjk=cm3d.loc[j][j][k]
            n_jk=cm3d.loc['t'][j][k]
            nk=cm3d['t']['t'][k]
            
            numPAj += Nk * njjk/nk
            denPAj += Nk * n_jk/nk
        
        PAj = numPAj/denPAj
        

        numvPAj = 0.0
        denvPAj = 0.0 

        for k in range(len(areas)):
            Nk=areas[k]
            njjk=cm3d.loc[j][j][k]
            n_jk=cm3d.loc['t'][j][k]
            nk=cm3d['t']['t'][k]

            yk=njjk/nk
            xk=n_jk/nk
            s2yk= nk/(nk-1) * yk * (1-yk)
            s2xk= nk/(nk-1) * xk * (1-xk)
            sxyk= nk/(nk-1) * yk * (1-xk)

            numvPAj += Nk*Nk/nk * (1-nk/Nk) * (s2yk + PAj*PAj*s2xk - 2*PAj*sxyk)
            denvPAj += Nk*xk

        vPAj=numvPAj/(denvPAj*denvPAj)

        metrics['PAj'][j]    = PAj
        metrics['v_PAj'][j]  = vPAj
        metrics['sd_PAj'][j] = m.sqrt(vPAj)
        metrics['OEj'][j]    = 1-PAj
        metrics['v_OEj'][j]  = vPAj
        metrics['sd_OEj'][j] = m.sqrt(vPAj)

    #print(pd.DataFrame(metrics)[['PAj', 'v_PAj', 'sd_PAj', 'OEj', 'v_OEj', 'sd_OEj']])

   
    # Overall accuracy
    metrics['OA']={}
    metrics['v_OA']={}
    metrics['sd_OA']={} 
    
    OA = 0.0
    vOA = 0.0
    for k in range(len(areas)):
        Nk=areas[k]
        nk=cm3d['t']['t'][k]
        sumniik=0
        for i in cl_cl:
            if j in cl_ref:
                 niik=cm3d.loc[i][i][k]
                 sumniik += niik

        yk   = sumniik/nk
        s2yk = nk/(nk-1) * yk * (1-yk)

        OA  += Nk/N * sumniik/nk
        vOA += Nk/N * Nk/N *(1-nk/Nk) * s2yk/nk 

    metrics['OA']['OA']    = OA
    metrics['v_OA']['OA']  = vOA
    metrics['sd_OA']['OA'] = m.sqrt(vOA) 

    #print(pd.DataFrame(metrics)[['OA', 'v_OA', 'sd_OA']])

    return pd.DataFrame(metrics)


def metrics2d_count_str(cmh2d_count, cl_counts, areaunit=None):
    '''  

    Computation of the estimated metrics when the sample is stratified and the strata are the classes using the confusion 2d-matrix with counts

    Input: 
    - cmh2d_count: confusion 2D-matrix with counts  
    - cl_counts: dictionary with the number of units or weight (value) by class (key)
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    
    Ouput: 
    - metrics: dataframe of pandas with the estimated metrics

    '''
    import numpy as np
    import pandas as pd 
    import math as m

    # number of units for each stratum
    #cl_counts
    N=sum(cl_counts.values())

    # metrics
    metrics={} 
    cl_ref=cmh2d_count.columns.values.tolist()[:-1] #j
    cl_cl=cmh2d_count.index.tolist()[:-1] #i
    
    # number of units for each class
    metrics['Nj']={}
    metrics['v_Nj']={}
    metrics['sd_Nj']={}
    if areaunit:
        metrics['Aj']={}
        metrics['v_Aj']={}
        metrics['sd_Aj']={}
    metrics['Pj']={}
    metrics['v_Pj']={}
    metrics['sd_Pj']={}
    
    for j in cl_ref:
        Nj=0
        vNj=0
        for k in cl_cl:
            Nk=cl_counts[k]
            nkj=cmh2d_count.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            
            Nj += Nk * nkj/nk
            
            yk    = nkj/nk
            s2yk  = nk/(nk-1) * yk*(1-yk)
            vNj += Nk * Nk/nk * (1 - nk/Nk) * s2yk

        metrics['Nj'][j]    = Nj
        metrics['v_Nj'][j]  = vNj
        metrics['sd_Nj'][j] = m.sqrt(vNj)
        if areaunit:
            metrics['Aj'][j]    = Nj*areaunit
            metrics['v_Aj'][j]  = vNj*areaunit*areaunit
            metrics['sd_Aj'][j] = m.sqrt(vNj*areaunit*areaunit)
        metrics['Pj'][j]    = Nj/N
        metrics['v_Pj'][j]  = vNj/(N*N)
        metrics['sd_Pj'][j] = m.sqrt(vNj/(N*N))



    #print(pd.DataFrame(metrics)[['Nj', 'v_Nj', 'sd_Nj', 'Pj', 'v_Pj', 'sd_Pj']])
    
    # user's accuracy
    metrics['UAi']={}
    metrics['v_UAi']={}
    metrics['sd_UAi']={}
    metrics['CEi']={}
    metrics['v_CEi']={}
    metrics['sd_CEi']={}
    for i in cl_cl:
        Ni=cl_counts[i]
        nii=cmh2d_count.loc[i][i]
        ni=cmh2d_count.loc[i]['t']

        UAi = nii/ni
        vUAi= (1-ni/Ni)* (UAi*(1-UAi))/(ni-1)

        metrics['UAi'][i]    = UAi
        metrics['v_UAi'][i]  = vUAi
        metrics['sd_UAi'][i] = m.sqrt(vUAi)
        metrics['CEi'][i]    = 1-UAi
        metrics['v_CEi'][i]  = vUAi
        metrics['sd_CEi'][i] = m.sqrt(vUAi)

    #print(pd.DataFrame(metrics)[['UAi', 'v_UAi', 'sd_UAi', 'CEi', 'v_CEi', 'sd_CEi']])

    # producer's accuracy
    metrics['PAj']={}
    metrics['v_PAj']={}
    metrics['sd_PAj']={}
    metrics['OEj']={}
    metrics['v_OEj']={}
    metrics['sd_OEj']={}
    for j in cl_ref:
        Nj=cl_counts[j]
        njj=cmh2d_count.loc[j][j]
        nj=cmh2d_count.loc[j]['t']
        
        numPAj = Nj*njj/nj

        denPAj = 0.0 

        for k in cl_cl:
            Nk=cl_counts[k]
            nkj=cmh2d_count.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            
            denPAj += Nk * nkj/nk
        
        PAj = numPAj/denPAj
        

        numvPAj = 0.0
        denvPAj = 0.0 

        for k in cl_cl:
            Nk=cl_counts[k]
            nkj=cmh2d_count.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            
            xk=nkj/nk
            
            if k!=j:
                s2xk= nk/(nk-1) * xk * (1-xk)
                numvPAj += Nk*Nk/nk * (1-nk/Nk) * PAj*PAj*s2xk
            else:
                yk=njj/nj
                s2yk= nk/(nk-1) * yk * (1-yk)
                numvPAj +=  Nk*Nk/nk * (1-nk/Nk) * (1-PAj)*(1-PAj) * s2yk
            
            denvPAj += Nk*xk

        vPAj=numvPAj/(denvPAj*denvPAj)

        metrics['PAj'][j]    = PAj
        metrics['v_PAj'][j]  = vPAj
        metrics['sd_PAj'][j] = m.sqrt(vPAj)
        metrics['OEj'][j]    = 1-PAj
        metrics['v_OEj'][j]  = vPAj
        metrics['sd_OEj'][j] = m.sqrt(vPAj)

    #print(pd.DataFrame(metrics)[['PAj', 'v_PAj', 'sd_PAj', 'OEj', 'v_OEj', 'sd_OEj']])

    # Overall accuracy
    metrics['OA']={}
    metrics['v_OA']={}
    metrics['sd_OA']={} 
    
    OA = 0.0
    vOA = 0.0
    for k in cl_ref:
        Nk=cl_counts[k]
        nkk=cmh2d_count.loc[k][k]
        nk=cmh2d_count.loc[k]['t']
        
        OA  += Nk/N * nkk/nk

        yk   = nkk/nk
        s2yk = nk/(nk-1) * yk * (1-yk)

        vOA += Nk/N * Nk/N *(1-nk/Nk) * s2yk/nk 

    metrics['OA']['OA']    = OA
    metrics['v_OA']['OA']  = vOA
    metrics['sd_OA']['OA'] = m.sqrt(vOA) 

    #print(pd.DataFrame(metrics)[['OA', 'v_OA', 'sd_OA']])

    return pd.DataFrame(metrics)


def metrics2d_prop_str(cmh2d_prop, cmh2d_count, N, areaunit=None):
    '''  

    Computation of the estimated metrics when the sample is stratified and the strata are the classes using the confusion 2d-matrix with proportions

    Input: 
    - cmh2d_prop: confusion 2D-matrix with proportions  
    - cmh2d_count: confusion 2D-matrix with counts  
    - N: number of units of the region of the interest
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    
    Ouput: 
    - metrics: dataframe of pandas with the estimated metrics

    '''
    import numpy as np
    import pandas as pd 
    import math as m

    # metrics
    metrics={}
    cl_ref=cmh2d_prop.columns.values.tolist()[:-1] #j
    cl_cl=cmh2d_prop.index.tolist()[:-1] #i

    # weights
    W={} 
    for i in cl_cl:
        W[i]=cmh2d_prop.loc[i]['t'] 
    
    
    
    # number of units for each class
    metrics['Nj']={}
    metrics['v_Nj']={}
    metrics['sd_Nj']={}
    if areaunit:
        metrics['Aj']={}
        metrics['v_Aj']={}
        metrics['sd_Aj']={}
    metrics['Pj']={}
    metrics['v_Pj']={}
    metrics['sd_Pj']={}
    
    for j in cl_ref:
        Nj=0
        vNj=0
        for k in cl_cl:
            pkj=cmh2d_prop.loc[k][j]
            
            Nj += N * pkj
            
            nk=cmh2d_count.loc[k]['t']  
            Wk=W[k]
            
            yk    = pkj/Wk
            s2yk  = nk/(nk-1) * yk*(1-yk)
            vNj += N*Wk*N*Wk/nk * (1 - nk/(N*Wk)) * s2yk

        metrics['Nj'][j]    = Nj
        metrics['v_Nj'][j]  = vNj
        metrics['sd_Nj'][j] = m.sqrt(vNj)
        if areaunit:
            metrics['Aj'][j]    = Nj*areaunit
            metrics['v_Aj'][j]  = vNj*areaunit*areaunit
            metrics['sd_Aj'][j] = m.sqrt(vNj*areaunit*areaunit)
        metrics['Pj'][j]    = Nj/N
        metrics['v_Pj'][j]  = vNj/(N*N)
        metrics['sd_Pj'][j] = m.sqrt(vNj/(N*N))


    #print(pd.DataFrame(metrics)[['Nj', 'v_Nj', 'sd_Nj', 'Pj', 'v_Pj', 'sd_Pj']])
    

    # user's accuracy
    metrics['UAi']={}
    metrics['v_UAi']={}
    metrics['sd_UAi']={}
    metrics['CEi']={}
    metrics['v_CEi']={}
    metrics['sd_CEi']={}
    for i in cl_cl:
        pii=cmh2d_prop.loc[i][i]
        Wi=cmh2d_prop.loc[i]['t']
        ni=cmh2d_count.loc[i]['t']
        
        UAi = pii/Wi

        vUAi= (1-ni/(N*Wi)) * (UAi*(1-UAi))/(ni-1)
        
        metrics['UAi'][i]    = UAi
        metrics['v_UAi'][i]  = vUAi
        metrics['sd_UAi'][i] = m.sqrt(vUAi)
        metrics['CEi'][i]    = 1-UAi
        metrics['v_CEi'][i]  = vUAi
        metrics['sd_CEi'][i] = m.sqrt(vUAi)

    #print(pd.DataFrame(metrics)[['UAi', 'v_UAi', 'sd_UAi', 'CEi', 'v_CEi', 'sd_CEi']])
    
    # producer's accuracy
    metrics['PAj']={}
    metrics['v_PAj']={}
    metrics['sd_PAj']={}
    metrics['OEj']={}
    metrics['v_OEj']={}
    metrics['sd_OEj']={}
    for j in cl_ref: 
        pjj=cmh2d_prop.loc[j][j]
        
        numPAj = pjj
        denPAj = 0.0 

        for k in cl_cl: 
            pkj=cmh2d_prop.loc[k][j] 
            denPAj += pkj
        
        PAj = numPAj/denPAj
        

        numvPAj = 0.0
        denvPAj = 0.0 

        for k in cl_cl: 
            Wk=W[k]
            pkj=cmh2d_prop.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            
            xk=pkj/Wk
            
            if k!=j:
                s2xk= nk/(nk-1) * xk * (1-xk)
                numvPAj += Wk*Wk / nk * (1-nk/(N*Wk)) * PAj*PAj*s2xk
            else:
                yk=pkj/Wk
                s2yk= nk/(nk-1) * yk * (1-yk)
                numvPAj += Wk*Wk/nk * (1-nk/(N*Wk))  * (1-PAj) *(1-PAj) * s2yk
            

        p_j=cmh2d_prop.loc['t'][j]
        denvPAj = p_j

        vPAj=numvPAj/(denvPAj*denvPAj)

        metrics['PAj'][j]    = PAj
        metrics['v_PAj'][j]  = vPAj
        metrics['sd_PAj'][j] = m.sqrt(vPAj)
        metrics['OEj'][j]    = 1-PAj
        metrics['v_OEj'][j]  = vPAj
        metrics['sd_OEj'][j] = m.sqrt(vPAj)

    #print(pd.DataFrame(metrics)[['PAj', 'v_PAj', 'sd_PAj', 'OEj', 'v_OEj', 'sd_OEj']])
    
    
    # Overall accuracy
    metrics['OA']={}
    metrics['v_OA']={}
    metrics['sd_OA']={} 
    
    OA = 0.0
    vOA = 0.0
    for k in cl_ref: 
        Wk=W[k]
        pkk=cmh2d_prop.loc[k][k]
        nk=cmh2d_count.loc[k]['t']
        
        OA  += pkk

        yk   = pkk/Wk
        s2yk = nk/(nk-1) * yk * (1-yk)

        vOA += Wk * Wk *(1-nk/(N*Wk)) * s2yk/nk 

    metrics['OA']['OA']    = OA
    metrics['v_OA']['OA']  = vOA
    metrics['sd_OA']['OA'] = m.sqrt(vOA) 

    #print(pd.DataFrame(metrics)[['OA', 'v_OA', 'sd_OA']])

    return pd.DataFrame(metrics)



def metrics2d_count_strprop(cmh2d_count, cl_counts, areaunit=None):
    '''  

    Computation of the estimated metrics when the sample is stratified and proportional by each class using the confusion 2d-matrix with counts.

    Input:  
    - cmh2d_count: confusion 2D-matrix with counts   
    - cl_counts: dictionary with the number of units or weight (value) by class (key)
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    
    Ouput: 
    - metrics: dataframe of pandas with the estimated metrics

    '''
    import numpy as np
    import pandas as pd 
    import math as m

    # number of units for each stratum
    #cl_counts
    N=sum(cl_counts.values())

    # metrics
    metrics={} 
    cl_ref=cmh2d_count.columns.values.tolist()[:-1] #j
    cl_cl=cmh2d_count.index.tolist()[:-1] #i
    
    # number of units for each class
    metrics['Nj']={}
    metrics['v_Nj']={}
    metrics['sd_Nj']={}
    if areaunit:
        metrics['Aj']={}
        metrics['v_Aj']={}
        metrics['sd_Aj']={}
    metrics['Pj']={}
    metrics['v_Pj']={}
    metrics['sd_Pj']={}
    
    for j in cl_ref:
        n=cmh2d_count.loc['t']['t']
        n_j=cmh2d_count.loc['t'][j]
        Nj=N*n_j/n

        vNj=0
        for k in cl_cl:
            Nk=cl_counts[k]
            nkj=cmh2d_count.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            
            yk    = nkj/nk
            s2yk  = nk/(nk-1) * yk*(1-yk)
            vNj += N/n * (1 - n/N) * Nk*s2yk
            
        metrics['Nj'][j]    = Nj
        metrics['v_Nj'][j]  = vNj
        metrics['sd_Nj'][j] = m.sqrt(vNj)
        if areaunit:
            metrics['Aj'][j]    = Nj*areaunit
            metrics['v_Aj'][j]  = vNj*areaunit*areaunit
            metrics['sd_Aj'][j] = m.sqrt(vNj*areaunit*areaunit)
        metrics['Pj'][j]    = Nj/N
        metrics['v_Pj'][j]  = vNj/(N*N)
        metrics['sd_Pj'][j] = m.sqrt(vNj/(N*N))



    #print(pd.DataFrame(metrics)[['Nj', 'v_Nj', 'sd_Nj', 'Pj', 'v_Pj', 'sd_Pj']])
    
    # user's accuracy
    metrics['UAi']={}
    metrics['v_UAi']={}
    metrics['sd_UAi']={}
    metrics['CEi']={}
    metrics['v_CEi']={}
    metrics['sd_CEi']={}
    for i in cl_cl: 
        nii=cmh2d_count.loc[i][i]
        ni=cmh2d_count.loc[i]['t']
        n=cmh2d_count.loc['t']['t']


        UAi = nii/ni
        vUAi= (1-n/N)* (UAi*(1-UAi))/(ni-1)

        metrics['UAi'][i]    = UAi
        metrics['v_UAi'][i]  = vUAi
        metrics['sd_UAi'][i] = m.sqrt(vUAi)
        metrics['CEi'][i]    = 1-UAi
        metrics['v_CEi'][i]  = vUAi
        metrics['sd_CEi'][i] = m.sqrt(vUAi)

    #print(pd.DataFrame(metrics)[['UAi', 'v_UAi', 'sd_UAi', 'CEi', 'v_CEi', 'sd_CEi']])

    # producer's accuracy
    metrics['PAj']={}
    metrics['v_PAj']={}
    metrics['sd_PAj']={}
    metrics['OEj']={}
    metrics['v_OEj']={}
    metrics['sd_OEj']={}
    for j in cl_ref: 
        njj=cmh2d_count.loc[j][j]
        nj=cmh2d_count.loc[j]['t']
        n_j=cmh2d_count.loc['t'][j]
        
        PAj = njj/n_j
        

        numvPAj = 0.0 

        for k in cl_cl:
            Nk=cl_counts[k]
            nkj=cmh2d_count.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            n=cmh2d_count.loc['t']['t']
            
            xk=nkj/nk
            
            if k!=j:
                s2xk= nk/(nk-1) * xk * (1-xk)
                numvPAj += (1-n/N) * N/n * PAj*PAj * Nk*s2xk
            else:
                yk=njj/nj
                s2yk= nk/(nk-1) * yk * (1-yk)
                numvPAj +=  (1-n/N) * Nk*Nk/nk * (1-PAj)*(1-PAj) * s2yk
            
        denvPAj = N/n*n_j

        vPAj=numvPAj/(denvPAj*denvPAj)

        metrics['PAj'][j]    = PAj
        metrics['v_PAj'][j]  = vPAj
        metrics['sd_PAj'][j] = m.sqrt(vPAj)
        metrics['OEj'][j]    = 1-PAj
        metrics['v_OEj'][j]  = vPAj
        metrics['sd_OEj'][j] = m.sqrt(vPAj)

    #print(pd.DataFrame(metrics)[['PAj', 'v_PAj', 'sd_PAj', 'OEj', 'v_OEj', 'sd_OEj']])

    # Overall accuracy
    metrics['OA']={}
    metrics['v_OA']={}
    metrics['sd_OA']={} 
    
    OA = 0.0
    vOA = 0.0
    for k in cl_ref: 
        nkk=cmh2d_count.loc[k][k]
        nk=cmh2d_count.loc[k]['t']
        n=cmh2d_count.loc['t']['t']
        
        OA  += nkk/n

        yk   = nkk/nk
        s2yk = nk/(nk-1) * yk * (1-yk)

        vOA += (1-n/N) /n /n * nk*s2yk

    metrics['OA']['OA']    = OA
    metrics['v_OA']['OA']  = vOA
    metrics['sd_OA']['OA'] = m.sqrt(vOA) 

    #print(pd.DataFrame(metrics)[['OA', 'v_OA', 'sd_OA']])

    return pd.DataFrame(metrics)


def metrics2d_prop_strprop(cmh2d_prop, cmh2d_count, N, areaunit=None):
    '''  

    Computation of the estimated metrics when the sample is stratified and proportional by each class using the confusion 2d-matrix with proportions.

    Input: 
    - cmh2d_prop: confusion 2D-matrix with proportions  
    - cmh2d_count: confusion 2D-matrix with counts   
    - N: number of units of the region of the interest
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    
    Ouput: 
    - metrics: dataframe of pandas with the estimated metrics

    '''
    import numpy as np
    import pandas as pd 
    import math as m

    # metrics
    metrics={}
    cl_ref=cmh2d_prop.columns.values.tolist()[:-1] #j
    cl_cl=cmh2d_prop.index.tolist()[:-1] #i

    # weights
    
    
    
    # number of units for each class
    metrics['Nj']={}
    metrics['v_Nj']={}
    metrics['sd_Nj']={}
    if areaunit:
        metrics['Aj']={}
        metrics['v_Aj']={}
        metrics['sd_Aj']={}
    metrics['Pj']={}
    metrics['v_Pj']={}
    metrics['sd_Pj']={}
    
    for j in cl_ref:
        p_j=cmh2d_prop.loc['t'][j]
        n=cmh2d_count.loc['t']['t']
        Nj=N*p_j
        
        vNj=0
        for k in cl_cl:
            pkj=cmh2d_prop.loc[k][j]
            nk=cmh2d_count.loc[k]['t']  
            Wk=cmh2d_prop.loc[k]['t']
            
            yk    = pkj/Wk
            s2yk  = nk/(nk-1) * yk*(1-yk)
            vNj += N*N/n * (1 - n/N) * Wk * s2yk

        metrics['Nj'][j]    = Nj
        metrics['v_Nj'][j]  = vNj
        metrics['sd_Nj'][j] = m.sqrt(vNj)
        if areaunit:
            metrics['Aj'][j]    = Nj*areaunit
            metrics['v_Aj'][j]  = vNj*areaunit*areaunit
            metrics['sd_Aj'][j] = m.sqrt(vNj*areaunit*areaunit)
        metrics['Pj'][j]    = Nj/N
        metrics['v_Pj'][j]  = vNj/(N*N)
        metrics['sd_Pj'][j] = m.sqrt(vNj/(N*N))


    #print(pd.DataFrame(metrics)[['Nj', 'v_Nj', 'sd_Nj', 'Pj', 'v_Pj', 'sd_Pj']])
    

    # user's accuracy
    metrics['UAi']={}
    metrics['v_UAi']={}
    metrics['sd_UAi']={}
    metrics['CEi']={}
    metrics['v_CEi']={}
    metrics['sd_CEi']={}
    for i in cl_cl:
        pii=cmh2d_prop.loc[i][i]
        Wi=cmh2d_prop.loc[i]['t']
        ni=cmh2d_count.loc[i]['t']
        n=cmh2d_count.loc['t']['t']
        
        UAi = pii/Wi

        vUAi= (1-n/N) * (UAi*(1-UAi))/(ni-1)
        
        metrics['UAi'][i]    = UAi
        metrics['v_UAi'][i]  = vUAi
        metrics['sd_UAi'][i] = m.sqrt(vUAi)
        metrics['CEi'][i]    = 1-UAi
        metrics['v_CEi'][i]  = vUAi
        metrics['sd_CEi'][i] = m.sqrt(vUAi)

    #print(pd.DataFrame(metrics)[['UAi', 'v_UAi', 'sd_UAi', 'CEi', 'v_CEi', 'sd_CEi']])
    
    # producer's accuracy
    metrics['PAj']={}
    metrics['v_PAj']={}
    metrics['sd_PAj']={}
    metrics['OEj']={}
    metrics['v_OEj']={}
    metrics['sd_OEj']={}
    for j in cl_ref: 
        pjj=cmh2d_prop.loc[j][j]
        p_j=cmh2d_prop.loc['t'][j]
        
        PAj = pjj/p_j
        

        numvPAj = 0.0 

        for k in cl_cl: 
            Wk=cmh2d_prop.loc[k]['t']
            pkj=cmh2d_prop.loc[k][j]
            nk=cmh2d_count.loc[k]['t']
            
            xk=pkj/Wk
            
            if k!=j:
                s2xk= nk/(nk-1) * xk * (1-xk)
                numvPAj += N*N/n * Wk * (1-n/N) * PAj*PAj*s2xk
            else:
                yk=pkj/Wk
                s2yk= nk/(nk-1) * yk * (1-yk)
                numvPAj += Wk*Wk *N*N/nk * (1-n/N) * (1-PAj) *(1-PAj) * s2yk
            

        denvPAj = N* p_j

        vPAj=numvPAj/(denvPAj*denvPAj)

        metrics['PAj'][j]    = PAj
        metrics['v_PAj'][j]  = vPAj
        metrics['sd_PAj'][j] = m.sqrt(vPAj)
        metrics['OEj'][j]    = 1-PAj
        metrics['v_OEj'][j]  = vPAj
        metrics['sd_OEj'][j] = m.sqrt(vPAj)

    #print(pd.DataFrame(metrics)[['PAj', 'v_PAj', 'sd_PAj', 'OEj', 'v_OEj', 'sd_OEj']])
    
    
    # Overall accuracy
    metrics['OA']={}
    metrics['v_OA']={}
    metrics['sd_OA']={} 
    
    OA = 0.0
    vOA = 0.0
    for k in cl_ref: 
        Wk=cmh2d_prop.loc[k]['t']
        pkk=cmh2d_prop.loc[k][k]
        nk=cmh2d_count.loc[k]['t']
        n=cmh2d_count.loc['t']['t']
        
        OA  += pkk

        yk   = pkk/Wk
        s2yk = nk/(nk-1) * yk * (1-yk)

        vOA += (1-n/N)/n * Wk * s2yk 

    metrics['OA']['OA']    = OA
    metrics['v_OA']['OA']  = vOA
    metrics['sd_OA']['OA'] = m.sqrt(vOA) 

    #print(pd.DataFrame(metrics)[['OA', 'v_OA', 'sd_OA']])

    return pd.DataFrame(metrics)


def metrics2d_count_withoutstr(cmh2d_count, N, areaunit=None):
    '''  

    Computation of the estimated metrics when the sample is random  using the confusion 2d-matrix with counts.

    Input:  
    - cmh2d_count: confusion 2D-matrix with counts   
    - N: number of units of the region of the interest
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    
    Ouput: 
    - metrics: dataframe of pandas with the estimated metrics

    '''
    import numpy as np
    import pandas as pd 
    import math as m

    # number of units for each stratum
    #cl_counts 
    n=cmh2d_count.loc['t']['t']

    # metrics
    metrics={} 
    cl_ref=cmh2d_count.columns.values.tolist()[:-1] #j
    cl_cl=cmh2d_count.index.tolist()[:-1] #i
    
    # number of units for each class
    metrics['Nj']={}
    metrics['v_Nj']={}
    metrics['sd_Nj']={}
    if areaunit:
        metrics['Aj']={}
        metrics['v_Aj']={}
        metrics['sd_Aj']={}
    metrics['Pj']={}
    metrics['v_Pj']={}
    metrics['sd_Pj']={}
    
    for j in cl_ref:
        n_j=cmh2d_count.loc['t'][j]
        
        Nj=N/n*n_j

        y=n_j/n 
        s2y=n/(n-1)*y*(1-y)
        vNj=N*N/n*(1-n/N)*s2y

        metrics['Nj'][j]    = Nj
        metrics['v_Nj'][j]  = vNj
        metrics['sd_Nj'][j] = m.sqrt(vNj)
        if areaunit:
            metrics['Aj'][j]    = Nj*areaunit
            metrics['v_Aj'][j]  = vNj*areaunit*areaunit
            metrics['sd_Aj'][j] = m.sqrt(vNj*areaunit*areaunit)
        metrics['Pj'][j]    = Nj/N
        metrics['v_Pj'][j]  = vNj/(N*N)
        metrics['sd_Pj'][j] = m.sqrt(vNj/(N*N))



    #print(pd.DataFrame(metrics)[['Nj', 'v_Nj', 'sd_Nj', 'Pj', 'v_Pj', 'sd_Pj']])
    
    # user's accuracy
    metrics['UAi']={}
    metrics['v_UAi']={}
    metrics['sd_UAi']={}
    metrics['CEi']={}
    metrics['v_CEi']={}
    metrics['sd_CEi']={}
    for i in cl_cl:
        nii=cmh2d_count.loc[i][i]
        ni=cmh2d_count.loc[i]['t']

        UAi = nii/ni

        vUAi= n/ni*(1-n/N)* (UAi*(1-UAi))/(n-1)

        metrics['UAi'][i]    = UAi
        metrics['v_UAi'][i]  = vUAi
        metrics['sd_UAi'][i] = m.sqrt(vUAi)
        metrics['CEi'][i]    = 1-UAi
        metrics['v_CEi'][i]  = vUAi
        metrics['sd_CEi'][i] = m.sqrt(vUAi)

    #print(pd.DataFrame(metrics)[['UAi', 'v_UAi', 'sd_UAi', 'CEi', 'v_CEi', 'sd_CEi']])

    # producer's accuracy
    metrics['PAj']={}
    metrics['v_PAj']={}
    metrics['sd_PAj']={}
    metrics['OEj']={}
    metrics['v_OEj']={}
    metrics['sd_OEj']={}
    for j in cl_ref: 
        njj=cmh2d_count.loc[j][j]
        n_j=cmh2d_count.loc['t'][j]
        
        PAj = njj/n_j

        x=n_j/n 
        vPAj =  (1-n/N) / x * (1-PAj)*PAj/ (n-1)
            

        metrics['PAj'][j]    = PAj
        metrics['v_PAj'][j]  = vPAj
        metrics['sd_PAj'][j] = m.sqrt(vPAj)
        metrics['OEj'][j]    = 1-PAj
        metrics['v_OEj'][j]  = vPAj
        metrics['sd_OEj'][j] = m.sqrt(vPAj)

    #print(pd.DataFrame(metrics)[['PAj', 'v_PAj', 'sd_PAj', 'OEj', 'v_OEj', 'sd_OEj']])

    # Overall accuracy
    metrics['OA']={}
    metrics['v_OA']={}
    metrics['sd_OA']={} 
    
    OA = 0.0
    for k in cl_ref: 
        nkk=cmh2d_count.loc[k][k] 
        
        OA  += nkk/n  

    vOA = (1-n/N) * OA*(1-OA)/(n-1)

    metrics['OA']['OA']    = OA
    metrics['v_OA']['OA']  = vOA
    metrics['sd_OA']['OA'] = m.sqrt(vOA) 

    #print(pd.DataFrame(metrics)[['OA', 'v_OA', 'sd_OA']])

    return pd.DataFrame(metrics)


def metrics(cl_v, ref_v, N, areaunit=None, 
            stratified=False, str_counts=None,
            strata_classes=True, str_v=None,  
            proportional_sampling=False, 
            proportions=False):
    '''  

    Computation of the estimated metrics when the sample is random using the confusion 2d-matrix with counts.

    Input:  
    - cl_v: classification vector
    - ref_v: reference vector 
    - N: number of units of the region of the interest
    - areaunit=None: area of an unit. When None, areas of classes are not estimated
    - stratified=False: boolean value. Stratified?
    - str_counts=None: when stratified=True, str_counts is a dictionary with counts by stratum. 
    - strata_classes=False: boolean value. Strata are the classes?
    - str_v=None: strata vector
    - proportional_sampling=False: boolean value. The sampling is selected proportionally by the size of each stratum?
    - proportions=False: boolean value. Use of confusion matrix with proportions? 
    
    Ouput: (Generally) 
    - cm* : confusion matrix with counts and/or proportions
    - strata : position in the vector and the label of the strata
    - report* : report with metrics

    '''
    import pandas as pd
    
    if stratified:
        if strata_classes:
            cm2d_count = confmatrix2d_count(cl_v, ref_v)
            if proportional_sampling:
                if proportions: 
                    cm2d_prop = confmatrix2d_prop(cm2d_count, str_counts)
                    report2d = metrics2d_prop_strprop(cm2d_prop, cm2d_count, N, areaunit=areaunit)
                    return cm2d_count, cm2d_prop, report2d
                else:
                    report2d = metrics2d_count_strprop(cm2d_count, str_counts, areaunit=areaunit)
                    return cm2d_count, report2d
            else:
                if proportions:
                    cm2d_prop = confmatrix2d_prop(cm2d_count, str_counts)
                    report2d = metrics2d_prop_str(cm2d_prop, cm2d_count, N, areaunit=areaunit)
                    return cm2d_count, cm2d_prop, report2d
                else:
                    report2d = metrics2d_count_str(cm2d_count, str_counts, areaunit=areaunit)
                    return cm2d_count, report2d
        else:
            cmh3d_count, str_kv= confmatrix3d_count(cl_v, ref_v, str_v)
            report3d = metrics3d_count(cmh3d_count, str_kv, str_counts, areaunit=areaunit)
            keys=str_kv.keys()
            values=str_kv.values()
            return cmh3d_count, pd.DataFrame({'position': keys, 'stratum': values}), report3d
    else:
        cm2d_count = confmatrix2d_count(cl_v, ref_v)
        report2d = metrics2d_count_withoutstr(cm2d_count, N, areaunit=areaunit)

        return cm2d_count, report2d
