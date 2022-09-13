#Preprocessing
import numpy as np

from lemminflect import getLemma
from spacy.matcher import PhraseMatcher

#NLP nltk
from nltk.corpus import wordnet as wn

#Cosine computation
from scipy.spatial import distance

### ––– ###

## extract text from spacy token
extract_text_from_token = lambda token: token.text

## Extract plain text from word net item
extract_text_from_wn_item = lambda item: item.name().split('.', 1)[0]

class MetaphorChecker(object):

    def __init__(self, sentence, nlp, wordvec) -> None:
        self.lowered_sentence = sentence.lower()
        self.chopped = sentence.split(' ')
        self.nlp = nlp
        self.wordvec = wordvec
    
    def __get_word_in_wv_filter__(self):
        return lambda word: word in self.wordvec.wv.index_to_key

    def __get_word_by_pos_filter__(self, pos):
        return lambda token: token.pos_ == pos

    def __extract_verbs__(self):
        doc = self.nlp(self.lowered_sentence)
        
        verb_tokens = list(filter(self.__get_word_by_pos_filter__('VERB'), doc))
        verb_strings = list(map(extract_text_from_token, verb_tokens))
        return list(filter(self.__get_word_in_wv_filter__(), verb_strings))

    def __create_pmatcher__(self, verbs):
        verb_docs = [self.nlp(verb) for verb in verbs]
        verb_matcher = PhraseMatcher(self.nlp.vocab)
        verb_matcher.add('TARGETS', verb_docs)

    def __remove_word_from_chopped__(self, word):
        dummy = self.chopped.copy()
        dummy.remove(word)
        return dummy

    def __get_meaning_of_verb__(self, verb):
        direct_syn = wn.synsets(verb, pos = 'v') #list of synsets for one verb
        
        # Get candidate words from wordnet
        if len(direct_syn) == 0:
            verb_meanings = [getLemma(verb, upos = 'VERB')[0]]
        else:
            verb_meanings = []
            for syn in direct_syn:
                lemmas = syn.lemmas()
                for l in lemmas:
                    hyponyms = l.synset().hyponyms()
                    if len(hyponyms) > 0:
                        verb_meanings.extend(list(map(extract_text_from_wn_item, hyponyms)))
                    else:
                        verb_meanings.append(extract_text_from_wn_item(l))
        
        # Filter candidate words in wordnet
        verb_meanings = list(filter(self.__get_word_in_wv_filter__(), verb_meanings))
        verb_meanings_deduped = np.unique(np.array(verb_meanings))

        return verb_meanings_deduped.tolist()

    def __find_best_fit_synonym_idx__(self, verb_syns, verb_context):
        comparison = [distance.cosine(syn, verb_context) for syn in verb_syns]        
        best_fit_index = comparison.index(np.amax(np.array(comparison)))

        return best_fit_index
    
    def check(self):
        # Generate verb meanings
        verbs = self.__extract_verbs__()
        
        verbs_meanings = []
        for verb in verbs:
            verbs_meanings.append(self.__get_meaning_of_verb__(verb))

        verbs_meaning_vecs = [[self.wordvec.wv[c] for c in ac] for ac in verbs_meanings]

        # Generate verb context meaning
        verbs_contexts = [] # list of lists containing verb-specific contexts
        for verb in verbs:
            verbs_contexts.append(list(filter(self.__get_word_in_wv_filter__(), self.__remove_word_from_chopped__(verb))))

        # if len(ctx) = 0, mean = NaN
        verbs_contexts_mean = [[np.mean([self.wordvec.wv[word] for word in ctx], axis = 0)] for ctx in verbs_contexts]
        
        # Compare verb and verb context meaning and determine metaphoricity
        results = []
        for i in range(0, len(verbs)):
            verb_string = verbs[i]
            verb_meanings = verbs_meanings[i]
            verb_meaning_vecs = verbs_meaning_vecs[i]
            verb_context_mean = verbs_contexts_mean[i]
            if verb_context_mean is not np.nan and len(verb_meaning_vecs) > 0:
                best_fit_syn_idx = self.__find_best_fit_synonym_idx__(verb_meaning_vecs, verb_context_mean)
                best_fit_syn = verb_meanings[best_fit_syn_idx]
                similarity = self.wordvec.wv.similarity(best_fit_syn, verb_string)
                
                if similarity > 0.45: #Different threshold?
                    results.append([verb_string, 'LITERAL', similarity, best_fit_syn])
                else:
                    results.append([verb_string, 'METAPHORICAL', similarity, best_fit_syn])
            else:
                results.append([verb_string, 'DEFAULT LITERAL'])
        
        return results
