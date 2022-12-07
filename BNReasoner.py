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

    def get_edges(self):
        return self.bn.structure.edges

    def pruning(self, variables, evidence):
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
        
        print(self.get_structure(self.pruned_bn))
        return self.pruned_bn
        
    
    # def d_separation(self, (x,y), evidence):
    #     self.pruning()

        
        


if __name__ == "__main__":
    # Create test 
    test_file = 'testing/lecture_example.BIFXML'
    BN = BNReasoner(test_file)
    # BN.get_structure()
    variables = {'Slippery Road?', 'Wet Grass?'}
    evidence = {'Rain?'}
    BN.pruning(variables, evidence)
    # BN.d_separation(('Slippery Road?', 'Wet Grass?'), 'Rain')
    # print(BN.get_edges())
    # #visualize
    # nx.draw(int_graph)
    # plt.show()