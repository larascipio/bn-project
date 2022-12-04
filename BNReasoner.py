from typing import Union
from BayesNet import BayesNet


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

    def get_structure(self):
        print(self.bn.draw_structure())

    def pruning(self, variables, evidence):
        self.variables = variables.union(evidence)
        self.evidence = evidence 
    
        # Step 1: del leaf nodes (now it still deletes all nodes)
        for v in self.bn.get_all_variables():
            if v not in self.variables:
                self.bn.del_var(v)

        # Step 2: del outgoing edges
        for node in self.evidence:
            children = self.bn.get_children(node)
            for c in children:
                self.bn.del_edge((node, c))
            
            

if __name__ == "__main__":
    # Create test 
    test_file = 'testing/lecture_example.BIFXML'
    BN = BNReasoner(test_file)
    # BN.get_structure()
    variables = {'Slippery Road?', 'Wet Grass?'}
    evidence = {'Rain?'}
    BN.pruning(variables, evidence)
