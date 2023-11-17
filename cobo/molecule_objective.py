import numpy as np
import torch 
import selfies as sf 
from cobo.utils.mol_utils.mol_utils import smiles_to_desired_scores
from cobo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
from cobo.utils.mol_utils.selfies_vae.data import collate_fn
from cobo.latent_space_objective import LatentSpaceObjective
from cobo.utils.mol_utils.mol_utils import GUACAMOL_TASK_NAMES
import pkg_resources
# make sure molecule software versions are correct: 
assert pkg_resources.get_distribution("selfies").version == '2.0.0'
assert pkg_resources.get_distribution("rdkit-pypi").version == '2022.3.1'
assert pkg_resources.get_distribution("molsets").version == '0.3.1'

class MoleculeObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        task_id='pdop',
        path_to_vae_statedict="../cobo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        smiles_to_selfies={},
    ):
        assert task_id in GUACAMOL_TASK_NAMES + ["logp"]

        self.dim                    = 256 # SELFIES VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.smiles_to_selfies      = smiles_to_selfies # dict to hold computed mappings form smiles to selfies strings

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
        )

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)
            self.smiles_to_selfies[smile] = selfie

        return decoded_smiles

    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        score = smiles_to_desired_scores([x], self.task_id).item()

        return score

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict) 
            self.vae.load_state_dict(state_dict, strict=True) 
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        self.vae.max_string_length = self.max_string_length

    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        X_list = []
        for smile in xs_batch:
            try:
                selfie = self.smiles_to_selfies[smile]
            except:
                selfie = sf.encoder(smile)
                self.smiles_to_selfies[smile] = selfie
            tokenized_selfie = self.dataobj.tokenize_selfies([selfie])[0]
            encoded_selfie = self.dataobj.encode(tokenized_selfie).unsqueeze(0)
            X_list.append(encoded_selfie)
        X = collate_fn(X_list)
        dict = self.vae(X.cuda())
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

        return z, vae_loss, dict['recon_loss_all'], dict['kldiv']