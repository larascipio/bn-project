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
        self.perform_once = False
        self.children = False 

        while True:
            # Step 1: del leaf nodes 
            for v in self.bn.get_all_variables():
                if not self.bn.get_children(v) and v not in self.variables:
                    self.children = True 
                    self.bn.del_var(v)
            if not self.children and self.perform_once:
                break
            
            self.children = False
            self.perform_once = True 
            # Step 2: del outgoing edges
            for v in self.evidence:
                children = self.bn.get_children(v)
                for c in children:
                    self.bn.del_edge((v, c))

        self.get_structure()
                
            

if __name__ == "__main__":
    # Create test 
    test_file = 'testing/lecture_example.BIFXML'
    BN = BNReasoner(test_file)
    # BN.get_structure()
    variables = {'Slippery Road?', 'Wet Grass?'}
    evidence = {'Rain?'}
    BN.pruning(variables, evidence)
