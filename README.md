# Advancing Bayesian Optimization via Learning Correlated Latent Space (CoBO)
Official PyTorch Implementation for Advancing Bayesian Optimization via Learning Correlated Latent Space (CoBO) ([arxiv](https://arxiv.org/pdf/2310.20258.pdf)).
> Seunghun Lee*, Jaewon Chu*, Sihyeon Kim*, Juyeon Ko, Hyunwoo J. Kim, In Advanced in Neural Information Processing Systems (NeurIPS 2023).

<img src="https://github.com/mlvlab/CoBO/assets/29230924/6cffa825-4c02-4f08-acd2-31e546c4458d" width="80%" height="80%">

## Setup
We provide setup script file and environment file.

To setup the project, you can use the provided YAML file by running the following command:

```
conda env create -f requirements.yml
```


Or, for a shell script setup, run:

```
sh setup.sh
```

## Run
This repository uses tasks from the [GuacaMol](https://github.com/BenevolentAI/guacamol) benchmark. Run a task with:

```
python3 scripts/molecule_optimization.py --task_id [TASK] run_cobo done
```

Available **[TASK]** codes include:
- **med1**: Median molecules 1
- **pdop**: Perindopril MPO
- **adip**: Amlodipine MPO
- **rano**: Ranolazine MPO
- **osmb**: Osimertinib MPO
- **zale**: Zaleplon MPO
- **valt**: Valsartan SMARTS
- **med2**: Median molecules 2
- **siga**: Sitagliptin MPO
- **dhop**: Deco Hop
- **shop**: Scaffold Hop
- **fexo**: Fexofenadine MPO

For more tasks, see the [GuacaMol](https://github.com/BenevolentAI/guacamol) benchmark page.


## Citation
```
@inproceedings{lee2023advancing,
  title={Advancing Bayesian Optimization via Learning Correlated Latent Space},
  author={Lee, Seunghun and Chu, Jaewon and Kim, Sihyeon and Ko, Juyeon and Kim, Hyunwoo J},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgements
This repository is based on [lolbo](https://github.com/nataliemaus/lolbo).

## License
Code is released under MIT License.
