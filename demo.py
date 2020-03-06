from pie import PieModel
import time


def demo():
    text = "With the development of the TV, there is an problems. Joel is the writer who hve liveed in beijing city."
    tStart = time.time()
    result = pie.correct(text, doSentToken=False, doCleanUp=False)
    print("time={}, result={}".format(time.time() - tStart, result))


def test(pie):
    input_files = open("./scratch/conll-2014/official-2014.combined.m2.src").readlines()[:2]
    output = []
    tStart = time.time()
    for text in input_files:
        result = pie.correct(text, doSentToken=False, doCleanUp=False)
        output.append(result["correction"] + "\n")
    open("./scratch/conll-2014/official-2014.combined.m2.src.tgt", "w").writelines(output)
    print("time={}, timePerSent={}".format(time.time() - tStart, (time.time() - tStart) / len(input_files)))


if __name__ == "__main__":
    pie = PieModel()
    test(pie)
