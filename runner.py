import argparse

from src.system import ScaleNLP
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank_path", default='models/bert_ranking', type=str)
    parser.add_argument("--ranking_batchsize", default=8, type=int)
    parser.add_argument("--qa_model", default="models/bert_qa", type=str)
    parser.add_argument("--qa_batchsize", default=1, type=int)
    parser.add_argument("--number_paragraphs", default=100, type=int)

    parser.add_argument("--index", default="../index", type=str)
    parser.add_argument("--number_docs", default=10, type=int)
    parser.add_argument("--max_sequence", default=384, type=int)
    parser.add_argument("--max_query", default=64, type=int)
    parser.add_argument("--max_answer", default=200, type=int)
    parser.add_argument("--top", default=50, type=int)
    parser.add_argument("--stride", default=256, type=int)
    parser.add_argument("--lower", default=True)
    parser.add_argument("--device", default="cpu", type=str)

    opt = parser.parse_args()

    ir_system = ScaleNLP(opt)

    query = "What are the risk associated with covid for young adults"

    search_results = ir_system.query_processor(query)

    for (idx, res) in enumerate(search_results):
        print(res)
        print("*"*88)





