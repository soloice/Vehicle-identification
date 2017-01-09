import numpy as np
import codecs


class MyDataBase:
    def __init__(self, car_ref):
        self.ref = car_ref
        self.top_k = 200

    def retrieve_on_batch(self, car_query):
        print car_query.shape, self.ref.shape
        print '# queries:', len(car_query), '# ref:', len(self.ref)

        # Assume: self.ref: m-by-n array, car_query: p-by-n array
        xx, zz = np.sum(self.ref ** 2, axis=1), np.sum(car_query ** 2, axis=1)
        # dist2: p-by-m array, where dist2[i, j] is the square Euclidean distance between car_query[i] and self.ref[j]
        dist2 = -2.0 * np.dot(car_query, self.ref.T) + zz.reshape((-1, 1)) + xx.reshape((1, -1))
        del xx, zz
        query_result = dist2.argsort()[:, :self.top_k]
        return query_result

    def printXML(self, query_name_list, ref_name_list, file_name='../data/res/try.txt'):
        len1 = len(query_name_list)  # number of queries
        len2 = len(ref_name_list[0])  # number of predictions for each query
        assert self.top_k == len2
        f = codecs.open(file_name, "w", "gb2312")
        f.write("<?xml version=\"1.0\" encoding=\"gb2312\"?>")
        f.write("\n")
        f.write('<Message Version="1.0">')
        f.write("\n")
        f.write('    <Info evaluateType="6" mediaFile="VehicleRetrieval" />')
        f.write("\n")
        f.write("    <Items>")
        f.write("\n")
        for i in xrange(len1):
            f.write('        <Item imageName="' + query_name_list[i] + '">')
            f.write("\n")
            f.write("            ")
            for j in xrange(len2):
                f.write(ref_name_list[i][j] + " ")
            f.write("\n        </Item>\n")
        f.write("    </Items>\n")
        f.write("</Message>\n")

    def calc_MAP(self, query_result, query_answer):
        # query_result: predicted; query_answer: ground truth
        # print query_result.shape, query_answer[0]
        assert len(query_answer) == len(query_result)
        nb_queries = len(query_result)

        # query_answer[i] is a list/dict/set of indices,
        # such that query_answer[i] contains all indices of cars of the same vehicleID with query[i] in the ref set
        rel = np.array([[1. if x in query_answer[row] else 0. for x in query_result[row]]
                           for row in xrange(len(query_result))])
        len_each_answer = np.array([len(answer) for answer in query_answer])
        correct = np.cumsum(rel, axis=1)
        # print correct
        P = correct / np.arange(1, query_result.shape[-1] + 1)
        # print 'Shape(P)', P.shape
        # AP = P.mean(axis=1)
        AP = np.sum(P * rel, axis=1) / np.min(np.c_[len_each_answer, np.zeros(nb_queries) + self.top_k], axis=1)
        MAP = AP.mean()
        return MAP, AP
