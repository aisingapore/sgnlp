import json
from typing import *


def sequential_weighted_avg(x, weights):
    """Return a sequence by weighted averaging of x (a sequence of vectors).
    Args:
        x: batch * len2 * hdim
        weights: batch * len1 * len2, sum(dim = 2) = 1
    Output:
        x_avg: batch * len1 * hdim
    """
    return weights.bmm(x)


def load_dict(fname: str) -> Dict:
    """
    Loading a dictionary from a json file

    :param fname:
    :return:
    """
    with open(fname, "r") as fp:
        data = json.load(fp)
    return data


def process_ctx_quac(ctx: str) -> str:
    """
    process the context passage of QuAC for ease of use

    :param ctx:
    :return:
    """
    ctx = ctx[:-13]
    ctx = "CANNOTANSWER " + ctx
    return ctx


def process_para_json(paragraph_json: Dict) -> Dict:
    """
    Process a paragraph json

    :param paragraph_json:
    :return:
    """
    paragraph_json["context"] = process_ctx_quac(paragraph_json["context"])
    for qa in paragraph_json["qas"]:
        for answer in qa["prev_ans"]:
            if answer["text"] == "CANNOTANSWER":
                answer["answer_start"] = 0
            else:
                answer["answer_start"] += 13
    return paragraph_json
