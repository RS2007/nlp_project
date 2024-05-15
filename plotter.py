from evaluation import Evaluation
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.evaluator = Evaluation()

    def plot_results(self,doc_IDs_ordered,query_ids,qrels,out_folder,key,iterations=11):
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        # print(bm25_results)

        for k in range(1, iterations+1):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print(
                "Precision, Recall and F-score @ "
                + str(k)
                + " : "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )
            MAP = self.evaluator.meanAveragePrecision(
            doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
            doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +
            str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # # Plot the metrics and save plot
        plt.plot(range(1, iterations+1), precisions, label="Precision")
        plt.plot(range(1, iterations+1), recalls, label="Recall")
        plt.plot(range(1, iterations+1), fscores, label="F-Score")
        plt.plot(range(1, iterations+1), MAPs, label="MAP")
        plt.plot(range(1, iterations+1), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(out_folder + "eval_plot_"+key+".png")
        plt.show()
        pass
