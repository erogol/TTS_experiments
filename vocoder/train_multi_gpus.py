import os, sys
import math
import time
import subprocess
import argparse
import torch
import torch.distributed as dist

import pathlib
FILE_PATH = pathlib.Path(__file__).parent.absolute()


def main():
    """
    Call train.py as a new process and pass command arguments
    Example:
       ```CUDA_VISIBLE_DEVICES="0,1,3,4" python train_distributed.py --config_path model/config.json```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help='Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
        required='--continue_path' not in sys.argv
    )
    args = parser.parse_args()

    # OUT_PATH = create_experiment_folder(CONFIG.output_path, CONFIG.run_name,
                                        # True)
    # stdout_path = os.path.join(OUT_PATH, "process_stdout/")

    num_gpus = torch.cuda.device_count()
    group_id = time.strftime("%Y_%m_%d-%H%M%S")

    # set arguments for train.py
    command = [os.path.join(FILE_PATH, 'train.py')]
    command.append('--continue_path={}'.format(args.continue_path))
    command.append('--restore_path={}'.format(args.restore_path))
    command.append('--config_path={}'.format(args.config_path))
    command.append('--group_id=group_{}'.format(group_id))
    command.append('')

    # run processes
    processes = []
    for i in range(num_gpus):
        my_env = os.environ.copy()
        my_env["PYTHON_EGG_CACHE"] = "/tmp/tmp{}".format(i)
        command[-1] = '--rank={}'.format(i)
        stdout = None if i == 0 else open(os.devnull, 'w')
        p = subprocess.Popen(['python3'] + command, stdout=stdout, env=my_env)
        processes.append(p)
        print(command)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()
