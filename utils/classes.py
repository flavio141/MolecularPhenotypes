from Bio.PDB import Select, Polypeptide

class ChainSelector(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id
    
    def accept_residue(self, residue):
        return Polypeptide.is_aa(residue)
    
    def accept_atom(self, atom):
        return True