import os
import argparse

from mosec import Server
from collections import OrderedDict
from backend_model import (InCoderModel,
                           PolyCoderModel,
                           CodeLlamaModel,
                           StarCoder2Model
                           )
from backend_t5 import T5

MODEL_MAPPING_NAMES = OrderedDict([
    ("polycoder", PolyCoderModel),
    ("incoder", InCoderModel),
    ("starcoder2", StarCoder2Model),
    ("codellama", CodeLlamaModel)
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="gpt2",
        help=
        "The model to use. You can choose one of [santacoder, gpt2, gptneo, gptj, llama, wenzhong, skywork, damo, chatglm, alpaca, dolly].",
    )
    parser.add_argument("--gpu",
                        type=str,
                        required=False,
                        default='0',
                        help="Set os.environ['CUDA_VISIBLE_DEVICES'].")

    parser.add_argument("--port", help="mosec args.")
    parser.add_argument("--timeout", help="mosec args.")
    parser.add_argument("--debug", action="store_true", help="mosec args.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.model == 't5':
        server = Server()
        server.append_worker(T5)
        server.run()
    else:
        sniffer_model = MODEL_MAPPING_NAMES[args.model]
        server = Server()
        server.append_worker(sniffer_model, num=1, env=[{"CUDA_VISIBLE_DEVICES": args.gpu}])
        server.run()