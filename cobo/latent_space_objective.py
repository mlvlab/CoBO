import numpy as np
import torch 


class LatentSpaceObjective:
    '''Base class for any latent space optimization task
        class supports any optimization task with accompanying VAE
        such that during optimization, latent space points (z) 
        must be passed through the VAE decoder to obtain 
        original input space points (x) which can then 
        be passed into the oracle to obtain objective values (y)''' 

    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        task_id=''
        ):

        self.xs_to_scores_dict = xs_to_scores_dict 
        self.num_calls = num_calls
        self.task_id = task_id
        self.vae = None
        self.initialize_vae()
        assert self.vae is not None

    def __call__(self, z):
        ''' Input 
                z: a numpy array or pytorch tensor of latent space points
            Output
                out_dict['valid_zs'] = the zs which decoded to valid xs 
                out_dict['decoded_xs'] = an array of valid xs obtained from input zs
                out_dict['scores']: an array of valid scores obtained from input zs
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        decoded_xs = self.vae_decode(z)
        scores = []
        for x in decoded_xs:
            if x in self.xs_to_scores_dict:
                score = self.xs_to_scores_dict[x]
            else:
                score = self.query_oracle(x)
                self.xs_to_scores_dict[x] = score
                if np.logical_not(np.isnan(score)):
                    self.num_calls += 1
            scores.append(score)

        scores_arr = np.array(scores)
        decoded_xs = np.array(decoded_xs)

        bool_arr = np.logical_not(np.isnan(scores_arr)) 
        decoded_xs = decoded_xs[bool_arr]
        scores_arr = scores_arr[bool_arr]
        valid_zs = z[bool_arr]

        out_dict = {}
        out_dict['scores'] = scores_arr
        out_dict['valid_zs'] = valid_zs
        out_dict['decoded_xs'] = decoded_xs
        return out_dict

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        raise NotImplementedError("Must implement vae_decode()")

    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")

    def initialize_vae(self):
        ''' Sets variable self.vae to the desired pretrained vae '''
        raise NotImplementedError("Must implement method initialize_vae() to load in vae for desired optimization task")

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
        raise NotImplementedError("Must implement method vae_forward() (forward pass of vae)")
