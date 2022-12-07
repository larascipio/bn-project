from typing import Union
from BayesNet import BayesNet
import copy


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
    
    def is_d_blocked(self, path, evidence):
        """
        Checks for every triple in a path if it is active, and subsequently if the path is d-blocked.
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

    def find_all_paths(self, bn, start, path):
        """
        Finds all possible (undirected) paths between a start and end node, 
        returns a list of paths.
        """
        path.append(start)
        # Depth first recursive search 
        for node in bn.get_children(start) + bn.get_parents(start):
            if node not in path:
                self.find_all_paths(bn, node, path.copy())
        paths.append(path)

    def is_d_separated(self, X, Y, evidence):
        """
        Determines given three sets of variables X, Y, and Z, whether X is d-separated of Y given Z. 
        """
        for node in X:
            self.find_all_paths(self.bn, node, [])
        
        # Select all paths that end with a node in Y
        selected_paths = []
        for path in paths:
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
        



if __name__ == "__main__":
    # Create test 
    test_file = 'testing/lecture_example.BIFXML'
    BN = BNReasoner(test_file)
    # BN.get_structure()
    X = {'Slippery Road?', 'Wet Grass?'}
    Y = {'Winter?'}
    evidence = {}
    # BN.pruning(variables, evidence)
    paths = [[]]
    BN.is_d_separated(X, Y, evidence)
    BN.is_independent(X, Y, evidence)

    # print(BN.get_edges())
    # #visualize
    # nx.draw(int_graph)
    # plt.show()