def prepare_input(passages, scores, m=1, label='positive'):

    batch_queries = batch[label]
    batch_scores = batch[f'{label}_score']
    texts_tgr = []
    texts_src = []

    for j in range(m):
        labels += [1]
        try:
            texts_tgt += [batch_queries[j]]
            scores += [batch_scores[j]]
        except: # sometimes #available is less than #specififed 
            offset = int(j % len(queries))
            texts_tgt += [queries[offset]]
            scores += [batch_scores[offset]]

        printed_scores = round(scores[-1]*100)
        texts_src += [self.prefix.format(printed_score, p)]

    return texts_src, texts_tgt
