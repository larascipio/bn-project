### 
### BNReasoner.py
### 
### Knowledge Representation: Bayesian Network Project 2 
### 
### Used for reasoning tasks in a Bayesian network, contains methods that a BayesianNet object.
### 

from typing import Union
from BayesNet import BayesNet
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import minmax_scale

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

        self.paths = [[]]

    def print_structure(self, bn):
        print(bn.draw_structure())

    def get_edges(self, bn):
        return bn.structure.edges

    def pruning(self, variables, evidence):
        """
        Node- and edge-prune the Bayesian network given a set of query variables Q and evidence e.
        """
        self.pruned_bn = copy.deepcopy(self.bn)
        self.variables = variables.union(evidence)
        self.evidence = evidence
        
        # Step 1: delete outgoing edges
        for v in self.evidence:
            children = self.pruned_bn.get_children(v)
            for c in children:
                self.pruned_bn.del_edge((v, c))

        while True:
            self.children = False
            # Step 2: delete leaf nodes recursively
            for v in self.pruned_bn.get_all_variables():
                if not self.pruned_bn.get_children(v) and v not in self.variables:
                    self.pruned_bn.del_var(v)
                    self.children = True
            if not self.children:
                break
        
        return self.pruned_bn

    def find_all_paths(self, bn, start, path) -> None:
        """
        Finds all possible (undirected) paths between a start and end node
        :bn: Network to traverse through
        :start: Node to start with.
        :path:  ?
        :return: A list of paths.
        """
        path.append(start)
        # Depth-first recursive search
        for node in bn.get_children(start) + bn.get_parents(start):
            if node not in path:
                self.find_all_paths(bn, node, path.copy())
        self.paths.append(path)

    def is_d_blocked(self, path, evidence):
        """
        Checks for every triple in a path if it is active, and subsequently if the path is d-blocked.
        :path: List of paths
        :evidence:
        :return: True if path is d-blocked and False if the path is not d-blocked.
        """
        # Check every triple in the path
        for i in range(0, len(path)-2):
            # Causal chain
            if path[i+1] in self.bn.get_children(path[i]) and path[i+2] in self.bn.get_children(path[i+1]):
                if path[i+1] in evidence:
                    # Triple is inactive, path is d_blocked
                    return True

            # Common cause
            if path[i] and path[i+2] in self.bn.get_children(path[i+1]):
                if path[i+1] in evidence:
                    # Triple is inactive, path is d_blocked
                    return True

            # Common effect
            if path[i] and path[i+2] in self.bn.get_parents(path[i+1]):
                if path[i+1] or self.bn.get_children(path[i+1]) not in evidence:
                    # Triple is inactive, path is d-blocked
                    return True

        # If no inactive triples are found: path is not d-blocked
        return False

    def is_d_separated(self, X, Y, evidence):
        """
        Determines given three sets of variables X, Y, and Z, whether X is d-separated of Y given Z.
        :return: True if X is d-separated from Y given evidence and False otherwise
        """
        for node in X:
            self.find_all_paths(self.bn, node, [])

        # Select all paths that end with a node in Y
        selected_paths = []
        for path in self.paths:
            if not path:
                continue
            if path[-1:][0] in Y:
                selected_paths.append(path)

        # Check for each path if d_blocked
        for path in selected_paths:
            if not self.is_d_blocked(path, evidence):
                # Active path is not d-separated
                print(f'{X} is not d-separated from {Y} given {evidence}')
                return False
        # If no paths are active: d-separated
        print(f'{X} is d-separated from {Y} given {evidence}')
        return True

    def is_independent(self, X, Y, evidence):
        """
        Determines given three sets of variables X, Y, and Z, whether X is independent of Y given Z.
        """
        if self.is_d_separated(X, Y, evidence):
            print(f'{X} is independent from {Y} given {evidence}')
            return True
        print(f'{X} is not independent from {Y} given {evidence}')
        return False

    def marginalize(self, f, var):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out
        :return A cpt with variables in ordering summed out
        """
        # Delete columns with variable to be summed out
        del f[var]
        
        # Find other variables that are in the table 
        columns = list(f.columns)
        columns.remove('p')
        
        # If only one variable left, return the summed probability
        if len(columns) == 0:
            return f['p'].sum()
        
        # Take the sum of variables with the same truth values
        summed_out_cpt = f.groupby(columns).aggregate({'p': 'sum'})
        summed_out_cpt.reset_index(inplace=True)
        
        return summed_out_cpt

    def max_out(self, factor, X) -> pd.DataFrame:
        """
        Computes the CPT in which X is maxed out.
        :factor: input CPT or other factor input
        :X: variables to max out
        :return: pd.DataFrame of a CPT with instantiation of X and maxed-out values
        """
        # Loop through every variable to max out
        for i in X:
            print(factor)
            # Select all columns except X
            columns = list(factor.columns.values)
            print(columns)
            # Max out variable
            max_cpt = factor.groupby(columns).max().reset_index()
            # Remember the variable instantiation (tuples)
            max_cpt['p'] = max_cpt[['p', X]].apply(tuple, axis=1)
            # Remove the maxed-out variable X
            max_cpt = max_cpt.drop(X, axis=1)

        return max_cpt
    
    def f_multiplication(self, f, g) -> pd.DataFrame:
        """
        Computes the multiplied factor h=fg, given two factors f and g.
        :input: f, g are factors (pd.DataFrame)
        :return: h=fg, a cpt table
        """
        # Get all variables in both factors
        Z = list(set(f.columns).union(set(g.columns)))
        # Get the overlap over variables
        overlap = list(set(f.columns).intersection(set(g.columns)))
        if 'p' in overlap:
            overlap.remove('p')
            Z.remove('p')

        h_Z = pd.DataFrame(columns=Z)

        # For every row of each factor
        for i in range(0, f.shape[0]):
            for j in range(0, g.shape[0]):

                # Check if rows are the same
                if f[overlap].iloc[i].tolist() == g[overlap].iloc[j].tolist():

                    # Create new table of variable instances
                    new = {}

                    for col in Z:
                        # Determine the variable instances
                        if col in list(f.columns.values):
                            new[col] = f[col].iloc[i]
                        else:
                            new[col] = g[col].iloc[j]

                    # Multiply the factors
                    new["p"] = f["p"].iloc[i] * g["p"].iloc[j]
                    h_Z = h_Z.append(new, ignore_index=True)
        return h_Z
    
    def min_degree(self, vars:str):
        """
        :vars: list of variables of graph which needs sorting
        :return: list of sorted variables
        Queue the variable in the graph with the smallest degree for ordering
        """

        # Create interaction graph
        int_graph = self.bn.get_interaction_graph()
        degree_ = dict((int_graph.degree()))
        ordering = []

        while len(vars) > 0:

            # Get the intersection of vars and nodes in degree
            degree = {}
            for i in list(set(degree_.keys()) & set(vars)):
                degree[i] = degree_[i]

            # Sort dict by value (amount of degrees)
            degree = dict(sorted(degree.items(), key=lambda x:x[1]))

            # Loop through every item in dict
            for key in degree:
                # Check neighbors
                neighbors = int_graph.neighbors(key)

                # Connect neighbor nodes to each other
                neighbor_edges = combinations(neighbors, 2)

                # Add these new connections to the list of edges
                for edge in neighbor_edges:
                    if int_graph.has_edge(edge[0], edge[1]) == False:
                        int_graph.add_edge(edge[0], edge[1])

                # Remove the node from graph
                vars.remove(key)
                int_graph.remove_node(key)
                ordering.append(key)

            # return the ordered variables
            return ordering
    
    def elimination(self, set_of_vars, *heuristic) -> pd.DataFrame:
        """
        :set_of_vars: list of variables to be eliminated
        :*heuristic: 'min_fil' or 'min_degree'
        :returns: a cpt out of which the set_of_vars is summed out.
        """

        # Get all the tables
        cpts = self.bn.get_all_cpts()

        # Order the variables
        # self.ordered_vars(set_of_vars)

        end_cpt = {}
        dependency = {}

        # Loop through the correct order of variables:
        for i in set_of_vars:
            print(i)
            # List of cpts to multiply
            multiply = []
            print("HUIDIGE end_cpt\n", end_cpt)
            if i in list(end_cpt.keys()):
                multiply.append(end_cpt)
            
            for cpt in cpts:
                if i in cpts[cpt]:
                    multiply.append(cpts[cpt])

            print(multiply)
            for elem in cpts.items():
                print(cpts.items())
                if elem in multiply:
                    print(elem)

            print("list of multiplications\n", multiply)
            if len(multiply) > 1:
                result = multiply.pop()
                print(result)
                for m in multiply:
                    print(m)
                    result = self.f_multiplication(result, m)
            elif len(multiply) == 1:
                result = multiply[0]
            print(result)

            # Marginalize the variable
            n_f = self.marginalize(result, i)
            print(n_f)
            
            # Multiply the factor to the existing factor
            if len(end_cpt) == 0:
                end_cpt = n_f
                
                
        return n_f
    
    def the_smallest(self, vars, graph) -> pd.DataFrame:
        """
        :vars: set of variables in a list
        :return: dictionary of sorted nodes (small->large)
        """
        n_graph = graph
        new_edges = {}

        for var in vars:
            new_edges[var] = 0
            neighbors = n_graph.neighbors(var)
            # Connect neighbor nodes to each other
            neighbor_edges = combinations(neighbors, 2)
            # Add these new connections to the list of edges
            for edge in neighbor_edges:
                if n_graph.has_edge(edge[0], edge[1]) == False:
                    n_graph.add_edge(edge[0], edge[1])
                    new_edges[var] += 1
            n_graph = graph

        # Sort variables by the least amount of edges.
        smallest = dict(sorted(new_edges.items(), key=lambda x:x[1]))
        return smallest
    
    def min_fill(self, vars):
        """
        Queue variable which deletion would add the fewest new edges to
        the graph
        :vars: set of unsorted variables
        :retun: list of sorted nodes for elimination
        """
        # Create interaction graph
        int_graph = self.bn.get_interaction_graph()
        degree_ = dict((int_graph.degree()))

        # Get the intersection of vars and nodes in degree
        degree = {}
        for i in list(set(degree_.keys()) & set(vars)):
            degree[i] = degree_[i]

        # Order of elimination of X
        ordering = []

        # List of variables to loop through
        var_list = list(degree.keys())

        while len(var_list) > 0:
            # Get the node and corresponding graph which creates the least amount of new edges
            node = self.the_smallest(var_list, int_graph)

            # Update graph and variable list
            int_graph.remove_node(list(node.keys())[0])
            var_list.remove(list(node.keys())[0])

            # Add variable to the ordering
            ordering.append(list(node.keys())[0])
        return ordering
    
    def ordering(self, set_of_Vars, heuristic):
        """
        (Hint: you get the interaction graph ”for free” from the BayesNet class.))
        :set_of_Vars: A set of variables X in the BN
        :returns: two lists of a good ordering for the elimination of X
        """
        if heuristic == 'min_degree':
            return self.min_degree(set_of_Vars)

        if heuristic == 'min_fill':
            return self.min_fill(set_of_Vars)
    
    def marg_dist(self, Q, e, *heuristic):
        """
        :Q: Query Variables, a list - ['C', 'D']
        :e: A dict of instances of variables - {'A': False}
        :heuristic: ordering of the elimination
        :returns: the dict
        Given query variables Q and possibly empty evidence e, compute the
        marginal distribution P(Q|e). Note that Q is a subset of the variables
        in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)
        """
        
        # Loop through the evidence and adjust its table
        for var, inst in e.items():
            table = self.bn.get_cpt(var)
            table = table[table[var] == inst]
            total_table = table[table[var]]
            prob_e *= table['p'].iloc[0]
            self.bn.update_cpt(var, table)
            print(self.bn.get_cpt(var))

            # Also adjust the tables of the children
            children = self.bn.get_children(var)
            for child in children:
                table_c = self.bn.get_cpt(child)
                table_c = table_c[table_c[var] == inst]

                self.bn.update_cpt(child, table_c)

        # Remove elements that should not be eleminated (Q)
        irrelevant_factors = self.bn.get_all_variables()
        
        for i in Q:
            irrelevant_factors.remove(i)
        print(irrelevant_factors)
        
        # Pick heuristic 
        heuristic = 'min_fill'
        
        marginalized_cpt = self.elimination(irrelevant_factors)

        return
    
    def map(self, Q, e):
        """
        :Q: A list of Variables
        :e: Evidence, variables with instantiation - {'A' : True}
        :return: The maximum a-posteriory instantiation and the value of the query variables Q
        """
        # Get the variables of the network
        all_vars = self.bn.get_all_variables()
        # Eliminate all variables that are not in the query or evidence
        eliminate = [i for i in all_vars if (i not in Q) and (i not in list(e.keys()))]
        joint_dist = self.elimination1(eliminate)

        # Get the maximum value of the instantiation
        for elem in list(e.keys()):
            print(elem)
            joint_dist = joint_dist[joint_dist[elem] == e[elem]]

        # Get the row of the max value
        # [Gets row and column name[gets the row number[gets the maximum value]]]
        return joint_dist.loc[joint_dist.iloc[joint_dist.argmax()]]

if __name__ == "__main__":
    # Create BN graph 
    test_file = 'testing/lecture_example.BIFXML'
    BN = BNReasoner(test_file)

    #### TESTING FUNCTIONS ####
    ## Pruning
    variables = {'Winter?', 'Rain?', 'Wet Grass?'}
    evidence = {'Rain?'}
    pruned_bn = BN.pruning(variables, evidence)

    ## D-separation
    # Not D-separated
    X_1 = {'Winter?'}
    Y_1 = {'Winter?', 'Rain?'}
    evidence_1 = {'Slippery Road?'}
    # BN.is_d_separated(X_1, Y_1, evidence_1)

    # D-separated (is also printed through independence)
    X_2 = {'Wet Grass?'}
    Y_2 = {'Slippery Road?'}
    evidence_2 = {'Rain?'}
    # BN.is_d_separated(X_2, Y_2, evidence_2)
    
    ## Independence
    # Independent 
    BN.is_independent(X_1, Y_1, evidence_1)
    # Not independent 
    BN.is_independent(X_2, Y_2, evidence_2)
    print()

    ## Marginalization 
    f = BN.bn.get_cpt('Sprinkler?')
    variable_f = 'Sprinkler'
    print(f'CPT of {variable_f} before marginalizing {variable_f}:\n{f}')
    print('CPT after marginalizing:')
    print(BN.marginalize(f, 'Sprinkler?'))
    print()

    ## Maxing out
    cpt = BN.bn.get_cpt('Rain?')
    print(BN.max_out(cpt, 'Rain?'))

    # # Multiplication
    g = BN.bn.get_cpt('Rain?')
    variable_g = 'Rain'
    multiplied = BN.f_multiplication(f, g)
    print(f'Multiplication of {variable_f} and {variable_g}:\n{multiplied}')
    print()

    ## Ordering with heuristics
    set_of_var = {'Winter?', 'Rain?', 'Wet Grass?', 'Sprinkler?', 'Slippery Road?'}
    # Min Degree Order
    ordered_list = BN.ordering(set_of_var, 'min_degree')
    print(f'List ordered with Min Degree heuristic:\n:{ordered_list}')
    # Min Fill Order
    ordered_list = BN.ordering(set_of_var, 'min_fill')
    print(f'List ordered with Min Fill heuristic:\n:{ordered_list}')

    ## Elimination 
    set_of_Vars = ['Sprinkler?','Rain?', 'Winter?']
    # print(BN.elimination(set_of_Vars))
