import torch
import random
import numpy as np
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
from cobo.cobo import CoBOState
from cobo.latent_space_objective import LatentSpaceObjective
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False

class Optimize(object):
    """
    Run CoBO Optimization
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run cobo). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
    """
    def __init__(
        self,
        task_id: str,
        seed: int=None,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        max_n_oracle_calls: int=70_000,
        learning_rte: float=0.001,
        acq_func: str="ts",
        bsz: int=10,
        num_initialization_points: int=10_000,
        init_n_update_epochs: int=20,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        update_e2e: bool=True,
        k: int=1_000,
        verbose: bool=True,
        lam_lip : float=100.0,
        lam_surr : int=1,
        lam_recon : int=1,
        lam_z : float=0.1,
    ):

        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e 
        self.lam_lip = lam_lip
        self.lam_surr = lam_surr
        self.lam_recon = lam_recon
        self.lam_z = lam_z
        self.set_seed()
        if wandb_project_name:
            self.wandb_project_name = wandb_project_name
        else:
            self.wandb_project_name = f"acuisition-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"

        self.load_train_data()
        self.initialize_objective()

        assert isinstance(self.objective, LatentSpaceObjective), "self.objective must be an instance of LatentSpaceObjective"
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert len(self.init_train_x) == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} xs, instead got {len(self.init_train_x)} xs"
        assert self.init_train_y.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} ys, instead got {self.init_train_y.shape[0]} ys"
        assert self.init_train_z.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} zs, instead got {self.init_train_z.shape[0]} zs"

        self.cobo_state = CoBOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            learning_rte=learning_rte,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose,
            lam_lip=lam_lip,
            lam_surr=lam_surr,
            lam_recon=lam_recon,
            lam_z=lam_z,
        )

    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self

    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_y (a tensor of corresponding latent space points)
        '''
        return self

    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self

    def create_wandb_tracker(self):                                         
        if self.track_with_wandb:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                name=f'CoBO',
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            ) 
            self.wandb_run_name = wandb.run.name
            self.wandb_run_id = wandb.run.id            
            base_dir = os.getcwd()
            wandb.save(base_dir + "./optimize.py", policy="now")
            wandb.save(base_dir + "../cobo/cobo.py", policy="now")
            wandb.save(base_dir + "../cobo/utils/utils.py", policy="now")
            wandb.save(base_dir + "../cobo/utils/bo_utils/turbo.py", policy="now")            
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self

    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            dict_log = {
                "best_found":self.cobo_state.best_score_seen,
                "n_oracle_calls":self.cobo_state.objective.num_calls,
                "total_number_of_e2e_updates":self.cobo_state.tot_num_e2e_updates,
                "best_input_seen":self.cobo_state.best_x_seen,
                "TR_length":self.cobo_state.tr_state.length
            }
            self.tracker.log(dict_log)

        return self

    def run_cobo(self): 
        ''' Main optimization loop
        '''
        self.create_wandb_tracker()
        
        step = 0
        while self.cobo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            if (self.cobo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                
                # ########################################################
                # # correlation analysis
                # from scipy.stats import pearsonr
                # def get_correlation(x,y):
                #     return pearsonr(x, y)[0]
                # with torch.no_grad():
                #     z = self.cobo_state.objective.vae_forward(self.cobo_state.top_k_xs)[0].detach().cpu()
                #     y = np.array(self.cobo_state.top_k_scores)
                    
                #     diffz = z - z[:,None]
                #     diffy = y - y[:,None]
                #     distz = torch.norm(diffz, p=2, dim=-1).reshape(-1).detach().cpu().numpy()
                #     disty = np.abs(diffy).reshape(-1)
                    
                #     corr = get_correlation(disty, distz)
                    
                #     self.tracker.log({"corr":corr})
                # ########################################################
                
                self.cobo_state.update_models_e2e(self.track_with_wandb, self.tracker)
                self.cobo_state.recenter()
                
                ########################################################
                # # smoothness analysis
                # save_path = f"./analysis_data"
                
                # if step % 10==0:
                #     state = self.cobo_state
                #     with torch.no_grad():
                #         topx = self.cobo_state.top_k_xs
                #         topz = state.objective.vae_forward(topx)[0]
                #         topy = self.cobo_state.top_k_scores
                        
                #     np.save(f"{save_path}/top_x_all_{self.lam_lip}_{step}.npy", topx)
                #     np.save(f"{save_path}/top_z_all_{self.lam_lip}_{step}.npy", topz.detach().cpu())
                #     np.save(f"{save_path}/top_y_all_{self.lam_lip}_{step}.npy", topy)
                # step+=1
                ########################################################
                
            else: 
                self.cobo_state.update_surrogate_model()
            
            self.cobo_state.acquisition()
            if self.cobo_state.tr_state.restart_triggered:
                self.cobo_state.initialize_tr_state()

            if self.cobo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.cobo_state.new_best_found = False

        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        self.log_topk_table_wandb()

        return self 

    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end 
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.cobo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.cobo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.cobo_state.objective.num_calls}")

        return self

    def log_topk_table_wandb(self):
        ''' After optimization finishes, log
            top k inputs and scores found
            during optimization '''
        if self.track_with_wandb:
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.cobo_state.top_k_scores):
                data_list.append([ score, str(self.cobo_state.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})
            self.tracker.finish()

        return self

    def done(self):
        return None

def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    os.environ["WANDB_MODE"]="offline"    
    fire.Fire(Optimize)
