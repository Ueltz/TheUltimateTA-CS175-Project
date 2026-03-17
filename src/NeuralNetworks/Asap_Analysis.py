from shared_data_prep import load_all_data
import numpy as np

def simple_load():
    data = load_all_data()

    test = data["asap2"]



    test = np.array(test[["raw_score", "prompt_id"]], dtype=np.float32)

    return test 

def find_distributions_of_scorers_in_set(y):
    scores = []

    for i in range(7):
        scores.append([0, 0, 0, 0, 0, 0, 0])

    for i in range(len(y)):
        set = int(y[i][1]) - 100
        score = int(y[i][0])
        print(str(set) + " " + str(score))
        scores[set][score] += 1

    
    for i in range(len(scores)):
        #print("Prompt " + str(i) + " distribution of scores:\n" + str(scores[i]) + "\n")
        sum = 0
        total = 0
        for j in range(len(scores[i])):
            sum += scores[i][j] * j
            total += scores[i][j]

        print("Prompt " + str(i) + " average score:\n" + str(sum / total) + "\n")


    return scores

def main():
    find_distributions_of_scorers_in_set(simple_load())

if __name__=="__main__":
    main()