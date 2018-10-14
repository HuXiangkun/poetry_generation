from flask import Flask, render_template, request
from generate import Poetry
import json

import pickle


app = Flask(__name__)

vocab = pickle.load(open("../data/vocab.pkl", "rb"))
id2word = pickle.load(open("../data/id2word.pkl", "rb"))

p5 = Poetry(5, len(vocab))
p7 = Poetry(7, len(vocab))


def generate(name, sent_len):
    str1 = None
    str2 = None
    err = None
    w1 = name[-2]
    w2 = name[-1]

    origin_w1 = None
    origin_w2 = None
    if w1 not in vocab.keys():
        origin_w1 = w1
        w1 = "#"
    if w2 not in vocab.keys():
        origin_w2 = w2
        w2 = "#"

    w1_id = [vocab[w1]]
    w2_id = [vocab[w2]]
    if sent_len == 5:
        sent1, sent2 = p5.generate(w1_id, w2_id)
    else:
        sent1, sent2 = p7.generate(w1_id, w2_id)
    str1 = ""
    str2 = ""
    if origin_w1:
        str1 += origin_w1
    else:
        str1 += w1
    if origin_w2:
        str2 += origin_w2
    else:
        str2 += w2
    for w in sent1[0][1:]:
        str1 += id2word[w]
    for w in sent2[0][1:]:
        str2 += id2word[w]
    return str1, str2, err


@app.route("/5", methods=['GET', 'POST'])
def poetry5():
    sent1 = None
    sent2 = None
    err = None
    if request.method == "POST":
        name = request.form.get("name")
        if len(name) < 2:
            err = "名字必须大于等于2个字。"
        else:
            sent1, sent2, err = generate(name, 5)
    return render_template("poetry5.html", sent1=sent1, sent2=sent2, err=err)


@app.route("/7", methods=['GET', 'POST'])
def poetry7():
    sent1 = None
    sent2 = None
    err = None
    if request.method == "POST":
        name = request.form.get("name")
        if len(name) < 2:
            err = "名字必须大于等于2个字。"
        else:
            sent1, sent2, err = generate(name, 7)
    return render_template("poetry7.html", sent1=sent1, sent2=sent2, err=err)


@app.route("/")
def root():
    return "Hello, World!"


@app.route("/poetry_5/<name>", methods=['GET'])
def get_poetry5(name):
    sent1 = None
    sent2 = None
    err = None
    if len(name) < 2:
        err = "名字必须大于等于2个字。"
    else:
        sent1, sent2, err = generate(name, 5)
    ret = {}
    if err:
        ret["error"] = err
    else:
        ret["s1"] = sent1
        ret["s2"] = sent2
    return json.dumps(ret)


@app.route("/poetry_7/<name>", methods=['GET'])
def get_poetry7(name):
    sent1 = None
    sent2 = None
    err = None
    if len(name) < 2:
        err = "名字必须大于等于2个字。"
    else:
        sent1, sent2, err = generate(name, 7)
    ret = {}
    if err:
        ret["error"] = err
    else:
        ret["s1"] = sent1
        ret["s2"] = sent2
    return json.dumps(ret)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
