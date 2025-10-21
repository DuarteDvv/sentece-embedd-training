from sentence_transformers.evaluation import InformationRetrievalEvaluator

def create_ir_evaluator(dataset, name="eval"):
   
    queries = {}  # {query_id: texto_da_query}
    corpus = {}   # {doc_id: texto_do_documento}
    relevant_docs = {}  # {query_id: set(doc_ids_relevantes)}
    
    for idx, item in enumerate(dataset):
        query_id = f"q_{idx}"
        doc_id = f"d_{idx}"
        
        # anchor é a query
        queries[query_id] = item['anchor']
        
        # positive é o documento relevante
        corpus[doc_id] = item['positive']
        
        # mapear qual documento é relevante para cada query
        relevant_docs[query_id] = {doc_id}
    
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
        score_functions={"cosine": lambda x, y: (x @ y.T).cpu()}
    )
    
    return evaluator