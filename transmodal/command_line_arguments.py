from argparse import ArgumentParser

def str_to_bool(s):
    return s.lower() in ('yes', 'true', 't', '1')

def str_to_ray_resume(s):
    if s in ('LOCAL', 'REMOTE', 'PROMPT', 'ERRORED_ONLY'):
        return s
    return s.lower() in ('yes', 'true', 't', '1')

def str_to_none(s):
    if s == 'None':
        s = None
    return s

def get_clargs():
    '''
    Gets the job configuration from command line arguments.

    Returns:
        Dictionary containing the model's configuration.
    '''

    parser = ArgumentParser()

    parser.add_argument(
        '--debug',
        default=False,
        type=str_to_bool,
        help='Training, validation, and testing are completed using one mini-batch',
    )
    parser.add_argument(
        '--delete-previous',
        '--delete_previous',
        default=False,
        type=str_to_bool,
        help='Delete the checkpoints and logs from previous sessions',
    )
    parser.add_argument('--ver', type=str, help='Model version')
    parser.add_argument('--task', type=str, help='The name of the task')
    parser.add_argument('--venv-path', '--venv_path', type=str, help='Path to the bin/activate of the virtualenv')

    # Slurm & hardware arguments
    parser.add_argument('--num-workers', '--num_workers', default=0, type=int, help='Number of workers for each DataLoader')
    parser.add_argument('--num-gpus', '--num_gpus', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--num-nodes', '--num_nodes', default=1, type=int, help='Number of nodes for the job')
    parser.add_argument('--memory', default='0', type=str, help='Minimum amount of memory')
    parser.add_argument('--time-limit', '--time_limit', default='0-02:00:00', type=str, help='Job time limit')
    parser.add_argument('--sbatch', default=False, type=str_to_bool, help='Submit job to the Slurm manager')
    parser.add_argument('--accelerator', default='ddp', type=str, help='Distributed backend')
    parser.add_argument('--qos', default=None, type=str, help='Quality of service')
    parser.add_argument('--begin', default='now', type=str,
                        help='When to begin the Slurm job, e.g. now+1hour')
    parser.add_argument('--slurm-cmd-path', '--slurm_cmd_path', type=str)
    parser.add_argument('--resumable', default=False, type=str_to_ray_resume,
                        help='Enable resubmission of job if incomplete')
    parser.add_argument('--cuda-visible-devices',
                        '--cuda_visible_devices',
                        default=None,
                        type=str_to_none,
                        help='Manually restrict CUDA devices'
                        )
    parser.add_argument('--exp-dir', '--exp_dir', default=None, type=str, help='Slurm command line argument for exp_dir')
    parser.add_argument('--dataset-dir', '--dataset_dir', default=None, type=str, help='Slurm command line argument for dataset_dir')
    parser.add_argument('--ckpt-zoo-dir', '--ckpt_zoo_dir', default=None, type=str, help='Slurm command line argument for ckpt_zoo_dir')
    parser.add_argument('--total-gpus', '--total_gpus', default=None, type=int, help='Slurm command line argument for total_gpus')
    parser.add_argument('--total-memory', '--total_memory', default=None, type=int, help='Slurm command line argument for total_memory')
    parser.add_argument('--trial', default=-1, type=int, help='Slurm command line argument for the trial number')

    # Training & hyperparameter optimisation arguments
    parser.add_argument('--train', default=False, type=str_to_bool, help='Perform training')
    parser.add_argument('--hyper', default=False, type=str_to_bool, help='Perform hyperparameter optimisation')
    parser.add_argument('--resume-training', '--resume_training', default=False, type=str_to_bool,
                        help='Resume training from the last epoch')
    parser.add_argument('--resume-epoch', '--resume_epoch', default=0, type=int, help='Epoch to resume training from')

    # Inference & test arguments
    parser.add_argument(
        '--test',
        default=False,
        type=str_to_bool,
        help='Evaluate the model on the test set',
    )
    parser.add_argument(
        '--test-val-set',
        '--test_val_set',
        default=False,
        type=str_to_bool,
        help='Evaluate on the validation set',
    )
    parser.add_argument(
        '--skip-test-set',
        '--skip_test_set',
        default=False,
        type=str_to_bool,
        help='Avoid using test set during testing. This is if only the validation set is to be tested during the test '
             'stage',
    )
    parser.add_argument(
        '--examine-infer',
        '--examine_infer',
        default=False,
        type=str_to_bool,
        help='Evaluate inference for a mini-batch of testing examples',
    )
    parser.add_argument(
        '--test-epoch',
        '--test_epoch',
        default=-1,
        type=int,
        help='Test the model using the specified epoch',
    )
    parser.add_argument(
        '--shuffle',
        default=False,
        type=str_to_bool,
        help='Shuffle the test set (used for examine-infer)',
    )
    # PyCharm arguments
    parser.add_argument('--mode', default=str)

    return parser.parse_args()