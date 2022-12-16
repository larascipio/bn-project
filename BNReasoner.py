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

    def get_structure(self, bn):
        return bn.draw_structure()

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

    def marginalize(self, factor, X) -> pd.DataFrame:
        """
        Computes the CPT from factor in which X is summed-out.
        :factor: factor from which to sum X out
        :X: variable to sum out
        :return: a marginalized CPT
        """
        # Get copy from input cpt
        # copy_cpt = copy.deepcopy(factor.get_cpt(X))
        # copy_cpt = copy.deepcopy(factor)
        copy_cpt = factor
        
        # Keep track of columns for later use 
        columns = list(copy_cpt.columns.values)
        if X in columns:
            columns.remove(X)
        else:
            return copy_cpt
        columns.remove('p')
        
        # Keep true and false values while removing the factor X
        true_values = copy_cpt[(copy_cpt[X] == True)].drop(X, axis=1)
        false_values = copy_cpt[(copy_cpt[X] == False)].drop(X, axis=1)

        # Take the sum of True and False values according to the other variables in the cpt
        summed_out_cpt = pd.concat([true_values, false_values]).groupby(columns)['p'].sum().reset_index()
        
        return summed_out_cpt

    def max_out(self, factor, X) -> pd.DataFrame:
        """
        Computes the CPT in which X is maxed out.
        :factor: input CPT or other factor input
        :X: variables to max out
        :return: pd.DataFrame of a CPT with instantiation of X and maxed-out values
        """
        # Get copy of the original CPT
        for i in X:
            # Select all columns except X
            columns = list(factor.columns.values)
            columns.remove(i)
            columns.remove('p')
            factor['Value'] = ""

            # Keep true and false values while removing the factor X
            true_values = factor[(factor[i] == True)] #.drop(i, axis=1)
            true_values['Value'] = i + "=" + true_values[i].astype(str)
            true_values.drop(i, axis=1)
            print(true_values)

            # true_values['p'] = true_values['p'].astype(str) + i + "=True"
            false_values = factor[(factor[i] == False)]#.drop(i, axis=1)
            false_values['Value'] = i + "=" + false_values[i].astype(str)
            # false_values.drop(i, axis=1)
            print(false_values)


            # # Max out variable
            # max_cpt = copy_cpt.groupby(columns).max().reset_index()
            # # Remember the variable instantiation (tuples)
            # max_cpt['p'] = max_cpt[['p', X]].apply(tuple, axis=1)
            # # Remove the maxed-out variable X
            # max_cpt = max_cpt.drop(X, axis=1)

            # # Max out variable
            print(columns)
            max_cpt = pd.concat([true_values, false_values]).groupby(columns)['p'].max().reset_index()

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
    
    def elimination(self, set_of_Vars, *heuristic):
        """
        :set_of_Vars: A set of variables X in the BN
        :returns a marginalized cpt in which a set of variables is eliminated:
        """
        # Copy to change current network 

        # Apply ordering is applicable
        # if heuristic == 'min_degree':
        #     ordered_vars = self.ordering(set_of_Vars, 'min_degree')
        # else:
        #     ordered_vars = self.ordering(set_of_Vars, 'min_fill')

        old_marg_cpt = None
        ordered_vars = set_of_Vars
        # print(ordered_vars)
        # Eliminate every variable 
        for var in ordered_vars:
            
            # Check if variable is not the first in the loop 
            if old_marg_cpt is not None:
                list_factors = [old_marg_cpt]
                
                
                # parent_cpt = copy.deepcopy(self.bn.get_cpt(var))
                # list_factors.append(parent_cpt)
            # Start with first node when variable is the first in the loop 
            else:
                parent_cpt = self.bn.get_cpt(var)
                list_factors = [parent_cpt]
            
            # Combine all children-nodes from the variable
            for child in self.bn.get_children(var):
                child_cpt = self.bn.get_cpt(child)
                list_factors.append(child_cpt)
            
            # Take first node to multiply
            new_cpt = list_factors.pop()
            length = len(list_factors)
            
            # Multiply all nodes with each other  
            while len(list_factors) > 0:
                new_cpt = self.f_multiplication(new_cpt, list_factors.pop())
        
            if length > 0:
                marg_cpt = self.marginalize(new_cpt, var)
            else:
                marg_cpt = new_cpt
            
            # Update variable cpt with new marginalized cpt 
            # self.bn.update_cpt(var, marg_cpt)

            # Keep marginalized cpt for next multiplication
            dict = self.bn.get_all_cpts()

            for i in dict:
                if var in dict[i]:
                    dict[i] = dict[i].drop(columns=[var])
            for i in dict:
                self.bn.update_cpt(i, dict[i])

            old_marg_cpt = marg_cpt
        
        
        return old_marg_cpt

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
        # Create a copy of the network
        # self.marge_bn = copy.deepcopy(self.bn)
        prob_e = 1
        # Loop through the evidence and adjust its table
        for var, inst in e.items():
            table = self.bn.get_cpt(var)
            table = table[table[var] == inst]
            total_table = table[table[var]]
            prob_e *= table['p'].iloc[0]
            # self.bn.update_cpt(var, table)
            # print(self.bn.get_cpt(var))

            # Also adjust the tables of the children
            children = self.bn.get_children(var)
            for child in children:
                table_c = self.bn.get_cpt(child)
                table_c = table_c[table_c[var] == inst]

                # self.bn.update_cpt(child, table_c)
        
        # joint_marg = set(Q) | set(list(e.keys()))
        # print(joint_marg)

        # Remove elements that should not be eleminated (Q)
        irrelevant_factors = self.bn.get_all_variables()
        
        for i in Q:
            irrelevant_factors.remove(i)
        
        # Pick heuristic 
        heuristic = 'min_fill'
        
        # irrelevant_factors = ['Wet Grass?']
        # Eliminate irrelevant factors of the query 
        # marginalized_cpt = self.elimination(irrelevant_factors, heuristic)
        
        marginalized_cpt = self.elimination(irrelevant_factors)
        # print(marginalized_cpt)
        exit()
        # print(self.bn.get_cpt('Winter?'))
        print(marginalized_cpt)
        marginalized_cpt['p'] = minmax_scale(marginalized_cpt['p'])
        print(marginalized_cpt)
        print(prob_e)
        # Calculate true and false values of Q
        true_Q = marginalized_cpt.iloc[-1]['p']
        prob_true = true_Q / prob_e
        print(prob_true)
        false_Q = marginalized_cpt['p'].sum() - true_Q
        prob_false = false_Q / prob_e
        print(prob_false)
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
    # Create test
    test_file = 'testing/lecture_example.BIFXML'
    BN = BNReasoner(test_file)
    # BN.get_structure()

    # Variables for testing
    X = {'Winter?'}
    Y = {'Winter?', 'Rain?'}
    evidence = {}
    f = BN.bn.get_cpt('Winter?')
    g = BN.bn.get_cpt('Rain?')

    # Functions
    # BN.pruning(variables, evidence)
    # BN.is_d_separated(X, Y, evidence)
    # BN.is_independent(X, Y, evidence)
    # BN.marginalize(f, 'Rain?')
    # print(BN.max_out(BN.bn, 'Sprinkler?'))
    # BN.f_multiplication(f, g)
    # BN.min_degree({'Winter?', 'Rain?', 'Wet Grass?', 'Sprinkler?', 'Slippery Road?'})
    # BN.min_fill({'Winter?', 'Rain?', 'Wet Grass?', 'Sprinkler?', 'Slippery Road?'})
    # BN.ordering({'Winter?', 'Rain?', 'Wet Grass?', 'Sprinkler?', 'Slippery Road?'})
    # set_of_Vars = ['Wet Grass?', 'Slippery Road?', 'Rain?', 'Winter?']
    set_of_Vars = ['Winter?']
    # BN.elimination(set_of_Vars)
    BN.marg_dist(set_of_Vars, {'Winter?': True})
    # BN.marg_dist(['Rain?'], {'Winter?': True}, "heuristic")