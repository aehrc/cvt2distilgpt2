from argparse import Namespace
from pathlib import Path
from ray import tune
from ray.tune import CLIReporter
from ray_lightning.tune import get_tune_ddp_resources
from transmodal.command_line_arguments import get_clargs
from transmodal.config import get_config
from transmodal.slurm import SlurmCluster
from transmodal.ext.collect_env_details import main as collect_env_details
from transmodal.tune.callbacks import GarbageCollection
from transmodal.trainer import create_trainer
from transmodal.tune.trial_progress import ModifiedTrialProgressCallback
from transmodal.utils import (
    checkpoint_exists,
    get_best_ckpt,
    get_ckpt_path,
    get_dataset,
    get_paths,
    get_transmodal,
    get_trials,
    get_workstation_config,
    gpu_clean_up,
    trial_dirname_string,
    write_test_ckpt_path,
)
import importlib
import os
import ray
import torch
import yaml


def objective(config):

    # Get the dataset
    dataset, config = get_dataset(config)

    # Trainer
    trainer = create_trainer(**config)

    # Multimodal model instance
    transmodal = get_transmodal(config)

    # Train
    trainer.fit(transmodal, datamodule=dataset)

    # Ensure that all GPUs are free
    del dataset, config, trainer, transmodal
    gpu_clean_up(verbose=True)


def main(clargs, *args):

    # Print environment details
    collect_env_details()

    # Get the model configuration
    config = get_config(clargs)

    # Hyperparameter optimisation
    if clargs.hyper:
        ray.init(
            address='auto',
            _node_ip_address=os.environ['IP_HEAD_ADDR'].split(':')[0],
            _redis_password=os.environ['REDIS_PASSWORD'],
        )
        resume = clargs.resumable if checkpoint_exists(os.path.join(config['exp_dir'], 'hyperparameter_search')) \
            else False
        sampler = getattr(
            importlib.import_module(config['sampler']['module']), config['sampler']['definition']
        )(**config['sampler']['kwargs']) if 'sampler' in config else None
        pruner = getattr(
            importlib.import_module(config['pruner']['module']), config['pruner']['definition']
        )(**config['pruner']['kwargs']) if 'pruner' in config else None

        print(f'Resources per trial: {config["gpus_per_trial"]}')

        analysis = tune.run(
            objective,
            config=config,
            search_alg=sampler,
            scheduler=pruner,
            num_samples=config['num_trials'],
            local_dir=config['exp_dir'],
            resources_per_trial=get_tune_ddp_resources(
                num_workers=config['gpus_per_trial'], use_gpu=True
            ),
            metric=config['monitor'],
            mode=config['monitor_mode'],
            resume=resume,
            name='hyperparameter_search',
            trial_dirname_creator=trial_dirname_string,
            callbacks=[ModifiedTrialProgressCallback(), GarbageCollection()],
            progress_reporter=CLIReporter(metric_columns={}, parameter_columns={}),
            verbose=2,
            # fail_fast=True,
            # _remote=True,
        )
        print(f'Best trial was: {analysis.get_best_trial(scope="all")}')
        print(f'Best hyperparameters found were: {analysis.get_best_config(scope="all")}')

    # Train
    if clargs.train:
        objective(config)

    # Load checkpoint for testing
    if clargs.test:

        if clargs.debug:
            transmodal = get_transmodal(config)
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if 'ckpt_path' in config:
                ckpt_path = config['ckpt_path']
            elif clargs.test_epoch > -1:
                ckpt_path = get_ckpt_path(config, load_epoch=clargs.test_epoch)
            else:
                ckpt_path = get_best_ckpt(config['exp_dir'], config['monitor_mode'])
            print('Testing checkpoint: {}.'.format(ckpt_path))
            write_test_ckpt_path(ckpt_path, clargs.exp_dir)
            transmodal = get_transmodal(config).load_from_checkpoint(checkpoint_path=ckpt_path)

        dataset, config = get_dataset(config)
        trainer = create_trainer(**config)
        dataset.setup(stage='test')
        trainer.test(transmodal, test_dataloaders=dataset.test_dataloader())

if __name__ == '__main__':

    # Get command line arguments
    clargs_base = get_clargs()

    # Determine the visibility of devices
    if clargs_base.cuda_visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = clargs_base.cuda_visible_devices
        print(f'CUDA_VISIBLE_DEVICES: {clargs_base.cuda_visible_devices}')

    # Get paths
    paths = get_paths(clargs_base.task)

    # Load the jobs
    with open(os.path.join('task', clargs_base.task, 'jobs.yaml')) as f:
        jobs = yaml.load(f, Loader=yaml.FullLoader)

    for ver, jargs in jobs.items():

        for trial in range(*get_trials(clargs_base, jargs)):

            clargs_dict = vars(clargs_base)
            clargs_dict.update(
                {**paths, **jargs, **get_workstation_config(jargs['num_gpus']), 'ver': ver, 'trial': trial},
            )
            clargs_dict['exp_dir'] = os.path.join(paths['exp_dir'], clargs_base.task, ver)
            if clargs_dict['trial'] >= 0:
                clargs_dict['exp_dir'] = os.path.join(clargs_dict['exp_dir'], str(trial))
            Path(clargs_dict['exp_dir']).mkdir(parents=True, exist_ok=True)
            clargs = Namespace(**clargs_dict)

            if clargs.sbatch:

                # Initialise Slurm cluster configuration
                cluster = SlurmCluster(
                    args=clargs,
                    log_path=clargs.exp_dir,
                    python_cmd='python3',
                    per_experiment_nb_nodes=clargs.num_nodes,
                    per_experiment_nb_cpus=clargs.num_workers,
                    per_experiment_nb_gpus=clargs.num_gpus,
                    begin=clargs.begin,
                    memory_mb_per_node=clargs.memory,
                    job_time=clargs.time_limit,
                )

                # Source the virtualenv
                cluster.add_command('source ' + clargs.venv_path)

                # Ray Cluster commands
                if clargs.hyper:
                    cluster.add_slurm_cmd(cmd='gpus-per-task', value=clargs.num_gpus)
                    cluster.add_slurm_cmd(cmd='tasks-per-node', value=1)
                    cluster.add_command('. ./transmodal/slurm_ray_cluster.sh')
                    cluster.per_experiment_nb_cpus = clargs.num_workers * clargs.num_gpus
                    cluster.per_experiment_nb_gpus = 0
                    cluster.no_srun = True
                else:
                    cluster.add_slurm_cmd(
                        cmd='tasks-per-node',
                        value=clargs.num_gpus if clargs.num_gpus > 0 else 1,
                    )

                # Request the quality of service for the job
                if clargs.qos:
                    cluster.add_slurm_cmd(cmd='qos', value=clargs.qos)

                # Email job status
                cluster.notify_job_status(
                    email=None, on_done=True, on_fail=True
                )

                job_display_name = '_'.join([clargs.ver, clargs.task])
                if clargs.trial > -1:
                    job_display_name = '_'.join([job_display_name, f'trial_{clargs.trial}'])

                # Submit job to Slurm workload manager
                cluster.optimize_parallel_cluster_gpu(
                    train_function=main,
                    job_display_name=job_display_name + '_',
                    enable_auto_resubmit=clargs.resumable,
                )

            else:
                main(clargs)
