### 
### analysis.py
### 
### Knowledge Representation: Bayesian Network Project 2 
### 
### Uses 
### 

from BNReasoner import BNReasoner
import time
# Import use case
BN = BNReasoner('use_case/use_case.xml')
BN_strict = BNReasoner('use_case/use_case_strict.xml')

# Create lists for run time
BN_usecase_times = []
BN_strict_times = []

# Append the run time of every iteration for the free case
for i in range(10):
    start = time.time()
    var = BN.bn.get_all_variables().pop()
    cpt = BN.bn.get_cpt(var)
    vars = BN.bn.get_all_variables()
    while len(vars) > 1:
        cpt = BN.f_multiplication(BN.bn.get_cpt(vars.pop()), cpt)
    end = time.time()
    result = end - start
    BN_usecase_times.append(result)

# For the strict case
for i in range(10):
    start = time.time()
    var = BN_strict.bn.get_all_variables().pop()
    cpt = BN_strict.bn.get_cpt(var)
    vars = BN_strict.bn.get_all_variables()
    while len(vars) > 1:
        cpt = BN_strict.f_multiplication(BN_strict.bn.get_cpt(vars.pop()), cpt)
    end = time.time()
    result = end - start
    BN_strict_times.append(result)

# Compute averages of runtime
average1 = sum(BN_usecase_times) / len(BN_usecase_times)
average2 = sum(BN_strict_times) / len(BN_strict_times)

# Create dataframes of runtime
unstrict = pd.DataFrame(BN_usecase_times)
strict = pd.DataFrame(BN_strict_times)

# Save the files
unstrict.to_csv('use_case_time.csv')
strict.to_csv('strict_time.csv')

# Do Shapiro-Wilk test
shapiro(use_time)
shapiro(strict_time)

# Reload data if needed
use_time = pd.read_csv("use_case_time.csv", header=0, names=["use_case"])
strict_time = pd.read_csv("strict_time.csv", header=0, names=["strict_case"])

# Join dataframes
times = use_time.join(strict_time)

# Do t-test
ttest_ind(times['use_case'], times['strict_case'])

# Get SD
times['use_case'].std()
times['strict_case'].std()

# Get sem 
times['use_case'].sem()
times['strict_case'].sem()