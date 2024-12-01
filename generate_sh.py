import argparse
import torch as th
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments',add_help=False)
    parser.add_argument('--seednum', '-n', type=int, default=1)
    parser.add_argument('--seedindex', type=int, default=None)
    parser.add_argument('--comment', '-c',type=str, default="std")
    parser.add_argument("--hovering","-h", type=int, default=1)
    parser.add_argument("--landing", "-l", type=int, default=1)
    parser.add_argument("--tracking", "-t", type=int, default=1)
    parser.add_argument("--racing", "-r", type=int, default=1)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--ppo","-p", type=int, default=1)
    parser.add_argument("--mtd","-m", type=int, default=1)
    parser.add_argument("--bptt","-b", type=int, default=1)
    parser.add_argument("--shac","-s", type=int, default=1)
    parser.add_argument("--sac","-a", type=int, default=0)
    return parser


# generate .sh file
def main():
    seed_list = th.as_tensor([42, 0, 1, 2, 3])

    args = parse_args().parse_args()
    all_command = ""

    if args.seedindex is not None:
        seed_list = seed_list[args.seedindex:args.seedindex+1]
    else:
        seed_list = seed_list[:args.seednum]

    def add_command_for_task(name):
        command = ""
        if args.ppo:
            command += f"python examples/{name}/ppo.py --seed {seed} -c {args.comment}\n"
        if args.mtd:
            command += f"python examples/{name}/mtd.py --seed {seed} -c {args.comment}\n"
        if args.bptt:
            command += f"python examples/{name}/bptt.py --seed {seed} -c {args.comment}\n"
        if args.shac:
            command += f"python examples/{name}/shac.py --seed {seed} -c {args.comment}\n"
        if args.sac:
            command += f"python examples/{name}/sac.py --seed {seed} -c {args.comment}\n"

        return command
    for seed in seed_list:
        if args.hovering:
            all_command += add_command_for_task("hovering")
        if args.landing:
            all_command += add_command_for_task("landing")
        if args.tracking:
            all_command += add_command_for_task("tracking")
        if args.racing:
            all_command += add_command_for_task("racing")

    with open("run.sh", "w") as f:
        f.write(all_command)

    if args.run:
        os.system("chmod +x run.sh")
        os.system("./run.sh")


if __name__ == "__main__":
    main()