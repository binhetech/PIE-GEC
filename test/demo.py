import sys

sys.path.append("../word_edit")
sys.path.append("../resources")

from pie import PieModel
import time


def demo():
    text = "With the developments of the TV, there are an problems. Joel is the writer who have liveed in the beijing city."
    tStart = time.time()
    result = pie.correct(text, doSentToken=False, doCleanUp=False)
    print("time={}, result={}".format(time.time() - tStart, result))


def test(pie, inFile="./scratch/conll-2014/official-2014.combined.m2.src",
         outFile="./scratch/conll-2014/official-2014.combined.m2.src.tgt"):
    input_files = open(inFile).readlines()
    output = []
    tStart = time.time()
    for text in input_files:
        result = pie.correct(text, doSentToken=False, doCleanUp=False)
        output.append(result["correction"] + "\n")
    open(outFile, "w").writelines(output)
    print("time={}, timePerSent={}".format(time.time() - tStart, (time.time() - tStart) / len(input_files)))


if __name__ == "__main__":
    pie = PieModel(vocab_file="../resources/configs/vocab.txt",
                 bert_config_file="../resources/configs/bert_config.json",
                 path_inserts="../resources/models/conll/common_inserts.p",
                 path_deletes="../resources/models/conll/common_deletes.p",
                 path_multitoken_inserts="../resources/models/conll/common_multitoken_inserts.p",
                 predict_checkpoint="../resources/models/pie_model.ckpt",
                 output_dir="../resources/models", max_seq_length=128, use_tpu=False, numGpu=2, inferMode="conll",
                 doSpellCheck=True)
    demo()
    # test(pie)
