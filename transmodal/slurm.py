from subprocess import call
import datetime
import os
import signal
import sys
import traceback

class SlurmCluster(object):

    RUN_CMD = 'sbatch'
    def __init__(
            self,
            args=None,
            log_path=None,
            python_cmd='python3',
            enable_log_err=True,
            enable_log_out=True,
            per_experiment_nb_nodes=1,
            per_experiment_nb_cpus=1,
            per_experiment_nb_gpus=1,
            memory_mb_per_node=2000,
            job_time="15:00",
            no_srun=False,
            begin="now",
    ):
        self.args = args
        self.log_path = log_path

        self.enable_log_err = enable_log_err
        self.enable_log_out = enable_log_out
        self.slurm_files_log_path = None
        self.err_log_path = None
        self.out_log_path = None
        self.modules = []
        self.script_name = os.path.realpath(sys.argv[0])
        self.job_time = job_time
        self.minutes_to_checkpoint_before_walltime = 6
        self.per_experiment_nb_gpus = per_experiment_nb_gpus
        self.per_experiment_nb_cpus = per_experiment_nb_cpus
        self.per_experiment_nb_nodes = per_experiment_nb_nodes
        self.memory_mb_per_node = memory_mb_per_node
        self.email = None
        self.notify_on_end = False
        self.notify_on_fail = False
        self.job_name = None
        self.python_cmd = python_cmd
        self.gpu_type = None
        self.on_gpu = False
        self.call_load_checkpoint = False
        self.commands = []
        self.slurm_commands = []
        self.no_srun = no_srun
        self.begin = begin

        self.is_from_slurm_object = bool(vars(args)["slurm_cmd_path"])

    def add_slurm_cmd(self, cmd=None, value=None):
        self.slurm_commands.append((cmd, value))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def load_modules(self, modules):
        self.modules = modules

    def notify_job_status(self, email, on_done, on_fail):
        self.email = email
        self.notify_on_end = on_done
        self.notify_on_fail = on_fail

    def optimize_parallel_cluster_gpu(
            self,
            train_function,
            job_name="",
            enable_auto_resubmit=False,
            job_display_name=None
    ):
        if job_display_name is None:
            job_display_name = job_name

        self.optimize_parallel_cluster_internal(train_function, job_name, job_display_name,
                                                  enable_auto_resubmit, on_gpu=True)

    def optimize_parallel_cluster_internal(
            self,
            train_function,
            job_name,
            job_display_name,
            enable_auto_resubmit,
            on_gpu
    ):
        self.job_name = job_name
        self.job_display_name = job_display_name
        self.on_gpu = on_gpu
        self.enable_auto_resubmit = enable_auto_resubmit

        # layout logging structure
        self.layout_logging_dir()

        if self.is_from_slurm_object:

            # Script is called by slurm: it's an experiment.
            self.run_experiment(train_function)
        else:

            scripts_path = os.path.join(self.log_path, 'slurm_out_logs')
            self.schedule_experiment(self.get_max_trial_version(scripts_path))

    def schedule_experiment(self, exp_i):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp = 'session_{}_{}'.format(exp_i, timestamp)

        # generate command
        slurm_cmd_script_path = os.path.join(self.slurm_files_log_path, '{}.sh'.format(timestamp))
        slurm_cmd = self.build_slurm_command(slurm_cmd_script_path, timestamp, exp_i, self.on_gpu)
        self.save_slurm_cmd(slurm_cmd, slurm_cmd_script_path)

        # run script to launch job
        print('\nLaunching experiment...')
        result = call('{} {}'.format(self.RUN_CMD, slurm_cmd_script_path), shell=True)
        if result == 0:
            print(f'Launched experiment {slurm_cmd_script_path}.')
        else:
            print('Launch failed...')

    def slurm_time_to_seconds(self, job_time):
        seconds = 0
        time_component = job_time
        if '-' in job_time:
            days, time_component = job_time.split('-')
            seconds += int(days) * 24 * 60 * 60

        time_components = time_component.split(':')
        if len(time_components) == 3:
            hours, minutes, secs = time_components
            time_seconds = int(secs) + (int(minutes) * 60) + (int(hours) * 60 * 60)
            seconds += time_seconds

        elif len(time_components) == 2:
            minutes, secs = time_components
            time_seconds = int(secs) + (int(minutes) * 60)
            seconds += time_seconds

        elif len(time_components) == 1:
            secs = time_components[0]
            seconds += int(secs)

        return seconds

    def call_resume(self):

        job_id = os.environ['SLURM_JOB_ID']
        cmd = 'scontrol requeue {}'.format(job_id)

        print(f'\nRequeing job {job_id}...')
        result = call(cmd, shell=True)
        if result == 0:
            print(f'Requeued job {job_id}.')
        else:
            print('Requeue failed...')

        os._exit(0)

    def sig_handler(self, signum, frame):
        print(f"Caught signal: {signum}")
        self.call_resume()

    def term_handler(self, signum, frame):
        print("Bypassing sigterm.")

    def run_experiment(self, train_function):
        if self.enable_auto_resubmit:
            print('Setting signal to automatically requeue the job before timeout.')
            signal.signal(signal.SIGUSR2, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)
        else:
            print("Automatic requeuing has not been set. The job will not be requeued after timeout.")

        try:
            train_function(self.args, self)

        except Exception as e:
            print('Caught exception in worker thread', e)

            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            raise SystemExit


    def save_slurm_cmd(self, slurm_cmd, slurm_cmd_script_path):
        with open(slurm_cmd_script_path, mode='w') as file:
            file.write(slurm_cmd)

    def get_max_trial_version(self, path):
        files = os.listdir(path)
        version_files = [f for f in files if 'session_' in f]
        if len(version_files) > 0:
            # regex out everything except file version for ve
            versions = [int(f_name.split('_')[1]) for f_name in version_files]
            max_version = max(versions)
            return max_version + 1
        else:
            return 0

    def layout_logging_dir(self):

        # format the logging folder path
        slurm_out_path = os.path.join(self.log_path, self.job_name)
        self.log_path = slurm_out_path

        # if we have a test tube name, make the folder and set as the logging destination
        if not os.path.exists(slurm_out_path):
            os.makedirs(slurm_out_path)

        # when err logging is enabled, build add the err logging folder
        if self.enable_log_err:
            err_path = os.path.join(slurm_out_path, 'slurm_err_logs')
            if not os.path.exists(err_path):
                os.makedirs(err_path)
            self.err_log_path = err_path

        # when out logging is enabled, build add the out logging folder
        if self.enable_log_out:
            out_path = os.path.join(slurm_out_path, 'slurm_out_logs')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.out_log_path = out_path

        # place where slurm files log to
        self.slurm_files_log_path = os.path.join(slurm_out_path, 'slurm_scripts')

        if not os.path.exists(self.slurm_files_log_path):
            os.makedirs(self.slurm_files_log_path)

    def get_args(self, args):
        params = []
        for k, v in vars(args).items():
            # Place everything in quotes except boolean variables
            if self.should_escape(v):
                cmd = '--{} \"{}\"'.format(k, v)
            else:
                cmd = '--{} {}'.format(k, v)
            params.append(cmd)
        return ' '.join(params)

    def should_escape(self, v):
        v = str(v)
        return '[' in v or ';' in v or ' ' in v

    def build_slurm_command(self, slurm_cmd_script_path, timestamp, exp_i, on_gpu):
        sub_commands = ['#!/bin/bash']

        # Job name
        sub_commands.append('#SBATCH --job-name={}'.format('{}session_{}'.format(self.job_display_name, exp_i)))

        # Standard output path
        if self.enable_log_out:
            out_path = os.path.join(self.out_log_path, '{}_%j.out'.format(timestamp))
            sub_commands.append('#SBATCH --output={}'.format(out_path))

        # Standard error path
        if self.enable_log_err:
            err_path = os.path.join(self.err_log_path, '{}_%j.err'.format(timestamp))
            sub_commands.append('#SBATCH --error={}'.format(err_path))

        # Time limit
        sub_commands.append('#SBATCH --time={}'.format(self.job_time))

        # Begin time
        if self.begin != "now":
            sub_commands.append('#SBATCH --begin={}'.format(self.begin))

        # GPUs
        if self.per_experiment_nb_gpus > 0 and on_gpu:
            if self.gpu_type is not None:
                sub_commands.append('#SBATCH --gres=gpu:{}:{}'.format(self.gpu_type, self.per_experiment_nb_gpus))
            else:
                sub_commands.append('#SBATCH --gres=gpu:{}'.format(self.per_experiment_nb_gpus))

        # CPUs per task
        if self.per_experiment_nb_cpus > 0:
            sub_commands.append('#SBATCH --cpus-per-task={}'.format(self.per_experiment_nb_cpus))

        # Number of nodes
        sub_commands.append('#SBATCH --nodes={}'.format(self.per_experiment_nb_nodes))

        # Minimum amount of memory per node
        sub_commands.append('#SBATCH --mem={}'.format(self.memory_mb_per_node))

        # Signal command to catch job termination
        sub_commands.append(f'#SBATCH --signal=USR2@{self.minutes_to_checkpoint_before_walltime * 60}')

        # Subscribe to email if requested
        mail_type = []
        if self.notify_on_end:
            mail_type.append('END')
        if self.notify_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0:
            sub_commands.append('#SBATCH --mail-type={}'.format(','.join(mail_type)))
            sub_commands.append('#SBATCH --mail-user={}'.format(self.email))

        # Add custom sbatch commands
        for (cmd, value) in self.slurm_commands:
            if value:
                sub_commands.append('#SBATCH --{}={}'.format(cmd, value))
            else:
                sub_commands.append('#SBATCH --{}'.format(cmd))

        # Load modules
        for module in self.modules:
            sub_commands.append('module load {}'.format(module))

        # Remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # Add additional commands
        for cmd in self.commands:
            sub_commands.append(cmd)

        # Add run command
        args = self.get_args(self.args)
        args = '{} --{} {}'.format(args, "slurm_cmd_path", slurm_cmd_script_path)

        if self.no_srun:
            cmd = '{} {} {}'.format(self.python_cmd, self.script_name, args)
        else:
            cmd = 'srun {} {} {}'.format(self.python_cmd, self.script_name, args)
        sub_commands.append(cmd)

        # Build full command with empty lines in between
        return '\n'.join(sub_commands)
