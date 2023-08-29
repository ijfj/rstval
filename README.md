# rstval

    def metrics(cl_v, ref_v, N, areaunit=None, 
            stratified=False, str_counts=None,
            strata_classes=True, str_v=None,  
            proportional_sampling=False, 
            proportions=False)


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
