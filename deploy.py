# from agent import Agent
# from agent2 import Agent
from agent3 import Agent

from argparse import ArgumentParser


def main(args):
    go_1 = Agent(path=args.model, server=args.server, port=args.port, joystick=not args.no_joystick)
    go_1.init_pose()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='use a command server')
    parser.add_argument('--no-joystick', dest='no_joystick', action='store_true', help='disable joystick')
    parser.add_argument('-p', '--port', type=int, default=9292, help='port to receive commands from')
    parser.add_argument('-m', '--model', type=str, default='weights/rough_150000.pt')  # 'weights/asym_full_model_10000.pt'
    arguments = parser.parse_args()

    main(arguments)
