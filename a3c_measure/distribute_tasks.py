import os, sys
import json
from params import HParams
from six.moves import shlex_quote

def new_cmd(session, name, cmd, logdir, shell):
  if isinstance(cmd, (list, tuple)):
    cmd = " ".join(shlex_quote(str(v)) for v in cmd)
  return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd), logdir, session, name, logdir)

def create_commands(session, args):

  task_counter = 0
  logdir = args.logdir
  args.num_workers += 1
  num_workers = args.num_workers

  base_cmd=['CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py']
  monitor_cmd=['CUDA_VISIBLE_DEVICES=', sys.executable, 'monitor.py']

  for arg in vars(args):
    val = getattr(args, arg)
    if arg in ['job_name', 'task', 'render', 'dry_run', 'verbosity', 'remotes', 'mode']:
      continue
    if type(val) is not str:
      val = str(val)
    base_cmd.extend(['--{}'.format(arg), val])
    monitor_cmd.extend(['--{}'.format(arg), val])

  print ("Base : {}".format(base_cmd))

  # Parameter Server
  args.job_name = 'ps'
  cmds_map = [new_cmd(session, "ps", base_cmd + ["--job-name", "ps"], logdir, "bash")]

  # Workers
  for worker_idx in range(num_workers-1):
    print ("worker : {}".format(worker_idx))
    cmds_map += [new_cmd(session, "w-{}".format(worker_idx), 
      base_cmd + ["--job-name", "worker", "--task", str(worker_idx)], logdir, "bash")]

  # Monitor
  cmds_map += [new_cmd(session, "pe", monitor_cmd + ["--job-name", "monitor", "--task", str(num_workers-1)], logdir, "bash")]

  tensorboard_cmd = new_cmd(session, "tb", ["tensorboard", "--logdir", logdir, "--port", "6006"], logdir, "bash")
  cmds_map += [tensorboard_cmd]

  windows = [v[0] for v in cmds_map]
  notes = []
  cmds = [
    "mkdir -p {}".format(logdir),
    "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), logdir),
  ]

  notes += ["Point your browser to http://localhost:6006 to see Tensorboard"]

  cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(args.logdir)]
  notes += ["Use `tail -f {}/*.out` to watch process output".format(logdir)]
  notes += ["Run `source {}/kill.sh` to kill the job".format(args.logdir)]

  for window, cmd in cmds_map:
    cmds += [cmd]

  return cmds, notes


def run():
  args = HParams

  # Save parameters in the folder as the backup.

  if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

  dparams={}
  for arg in vars(args):
    val = getattr(args, arg)
    dparams[arg] = val
  with open("/".join([args.logdir, "hparams.json"]), "w") as f:
    print ("Save the hyper-parameters.")
    json.dump(dparams, f)

  cmds, notes = create_commands('a3c', args)

  if args.dry_run:
    print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
  else:
    print("Executing the following commands:")
  print("\n".join(cmds))
  print("")
  if not args.dry_run:
    if args.mode == "tmux":
      os.environ["TMUX"] = ""
    os.system("\n".join(cmds))
  print('\n'.join(notes))

if __name__ == "__main__":
    run()


