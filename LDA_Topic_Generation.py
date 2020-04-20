import pandas as pd
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

#NLTK for names removal
import nltk
from nameparser.parser import HumanName
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Load spacy object
nlp = spacy.load('en', disable=['parser', 'ner'])

#Mallet path
mallet_path = 'C:/mallet-2.0.8/bin/mallet'


class LDA_Modeling:
    """
    Parameters:
    path : Path to the data file [string]
    text_colu : Text column on which LDA has to be applied [string]. 
    
    User input:
    During the processing and topic generation, user will be interrupted to check the top keywords and insist of any keywords are not
    significant for analysis. Those words will be dropped and the LDA will run again for analysis. The topic numbers will be finalized
    user and the dataframe will be generated with those topics.
    
    Returns:
    LDA_model : LDA model is saved as pickle
    topic_df : dataframe with labelled topics
    """
    
    def __init__(self, path, text_col):
       
        self.path = path
        self.text_col = text_col
        self.text=[]
        self.data_lemmatized = []
        if self.path.split('.')[1]=='csv':
            self.df = pd.read_csv(self.path)
        elif 'xl' in self.path.split('.')[1]:
            self.df = pd.read_excel(self.path)
            
            
    def sent_to_words(sentences):
        """
        Parameters:
        sentences : List of text documents [string]

        Returns: Words in each sentence as gensim list object
        """
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
   
    
    def remove_stopwords(texts):
        """
        Parameters:
        texts : List of list of words in each text document [string]

        Returns: list of list of words in each document with stopwords removed. Retains nouns, adverb, adjective, verb.
        """
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    
    def make_bigrams(texts):
        """
        Parameters:
        texts : List of list of words in each text document [string]

        Returns: list of list of words and bigrams wherever applicable in each document.
        """
        return [bigram_mod[doc] for doc in texts]

    
    def make_trigrams(texts):
        """
        Parameters:
        texts : List of list of words in each text document [string]

        Returns: list of list of words and trigrams wherever applicable in each document.
        """
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    
    def lemmatization(texts, allowed_postags=['NOUN','ADJ', 'VERB', 'ADV']):
        """
        Parameters:
        texts : List of list of words in each text document [string]

        Returns: 
        texts_out : list of list of lemmatized words wherever applicable in each document.
        """
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    
    def word_rmv_lemma(lemmatized_texts, new_word_list):
        """
        Parameters:
        texts : List of list of lemmatized words in each text document [string]
        new_word_list : List of new words you wish to drop from the lemmatized words. These words will be 
                        appened to the below list of words. Pass [] if no new word is to be added.

        Returns: list of list of words with unwanted words removed in each document.
        """
        list_words = ['incident', 'ticket', 'day', 'hour', 'email', 'nfid', 'hours/days', 'issue', 'resol', \
                      'close', 'please', 'fix', 'escalation', 'time', 'add', 'attachment', 'thank', 'mark', \
                      'count', 'provide', 'jira', 'check', 'update', 'escalate', 'asap', 'advise', 'jessie', \
                      'date', 'attach', 'screen', 'shot', 'shoot', 'screenshot', 'status', 'show', 'roadm']
        if len(new_word_list)!=0:
            list_words.extend(new_word_list)
        else:
            pass
        new_data = []
        for x in lemmatized_texts:
            new_note=[]
            for i in x:
                if True in [word in i.lower() for word in list_words]:
                    pass
                else:
                    new_note.append(i)
            new_data.append(new_note)
        return new_data
    
    
    def preprocessor(self, new_data):
        """
        Parameters:
        new_data : List of new words you wish to drop from the lemmatized words. These words will be 
                    appened to the below list of words. Pass [] if no new word is to be added, the text
                    data will be return after all processing on it.

        Returns: list of list of lemmatized words and without words listed in the input for each document.
        """
        if len(new_data)==0:
            self.df[self.text_col].fillna('', inplace=True)
            self.text = self.df[self.text_col].values.tolist()
            data_words=[]
            data_words = list(LDA_Modeling.sent_to_words(self.text))
            #Stopwords
            stop_words = stopwords.words('english')
            data_words_nostops = LDA_Modeling.remove_stopwords(data_words)
            # Do lemmatization keeping only noun, adj, vb, adv
            self.data_lemmatized = LDA_Modeling.lemmatization(data_words_nostops, allowed_postags=['NOUN','ADJ', 'VERB', 'ADV'])
            data = LDA_Modeling.word_rmv_lemma(self.data_lemmatized, [])
        else:
            data = new_data
        return data
    
    
    def make_corpus(data):
        """
        Parameters:
        data : Lemmatized and unwanted words removed list of list of words.

        Returns: 
        id2word : Dictionary of words based on input text data.
        corpus: list of corpus for each document
        """
        # Create Dictionary
        id2word = corpora.Dictionary(data)
  
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data]
        return id2word, corpus
        
    
    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts in all documents
        limit : Max num of topics

        Returns:
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values


    def format_topics_sentences(ldamodel, corpus, texts, lemmdata):
        """
        Parameters:
        ldamodel : The LDA Mallet model with optimal number of topics based on coherence score.
        data : Lemmatized and unwanted words removed list of list of words.
        corpus : Gensim corpus
        texts : List of input texts in all documents
        lemmdata : List of lemmatized texts in all documents using 'preprocessor' function.

        Returns:
        sent_topics_df : Dataframe with all text documents labelled using the LDA mallet optimal model.
        Dataframe columns - topic number, percentage contribution of the topic compared to other topics, 
        Top 20 keywords, cleaned Text column and the lemmatized version of the cleaned text.
        """
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num,topn = 20)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), \
                                                                      topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        lemm_text=[]
        for i in lemmdata:
            lemm_text.append(" ".join(i))

        # Add original text to the end of the output
        contents = pd.Series(texts)
        lemm_col = pd.Series(lemm_text)
        sent_topics_df = pd.concat([sent_topics_df, contents, lemm_col], axis=1)
        sent_topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Lemmatized_txt']
        return sent_topics_df
    
    
    def get_keywords(optimal_model, num_topics, num_words):
        """
        Parameters:
        optimal_model : The LDA Mallet model with optimal number of topics based on coherence score.
        num_topics : Number of topics t for this optimal model based on coherence score plot.
        num_words : number of top keywords N user wants to analyse.(8/10/20/30...)

        Returns:
        kw : [t X 2] Dataframe with topic numbers and their respective top N keywords.
        """    
        t=0
        lst=[]
        mainlst=[]
        kw=pd.DataFrame(columns=['Topic', 'Keywords'])
        while(t!=num_topics):
            lst=[]
            kw = kw.append({'Topic': t}, ignore_index=True)
            for i, j in optimal_model.show_topic(t, topn = num_words):
                if len(lst)<=20:
                    lst.append(i)
            mainlst.append(str(lst))
            t+=1
        kw['Keywords'] = pd.Series(mainlst)
        kw['Topic']=kw['Topic'].astype(int)
        return kw
    
    def modeling(self, data2=[]):
        """
        Parameters:
        data2 : Should be default to [] when executing for the first time. When additional words 
                have to be removed as the user agrees, new drop words will be added to data2 automatically.

        Returns:
        final_df : Dataframe with labelled text documents, top keywords, text, percentage distribution
        df_keywords : Top N Keywords per topics returned by get_keywords function.
        optimal_model : Optimal model chosen by the user.
        """        
        self.data = LDA_Modeling.preprocessor(self, data2)
        id2word, corpus = LDA_Modeling.make_corpus(self.data)
        model_list, coherence_values = LDA_Modeling.compute_coherence_values(dictionary=id2word, corpus=corpus, \
                                                                texts=self.data, start=3, limit=20, step=1)
        # Show graph
        limit=20; start=3; step=1;
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
        
        #Show topic number wise scores
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        
        #Print keywords for each topic
        ui1 = int(input("Enter value of t (topics) for keywords: "))
        optimal_model=model_list[ui1-3]
        ui2 = int(input("Enter number of words you want to analyse/topic: "))
        
        #Show Topicwise keywords
        df_keywords = LDA_Modeling.get_keywords(optimal_model, ui1, ui2)
        print(df_keywords)
        
        #Remove unnecessary words
        ui3 = input("Do you want to remove words?(y/n) ")
        if ui3.lower()=='y':
            um1 = print("Please enter words separated by spaces & press ENTER: ")
            ui4 = str(input())
            ulist = ui4.split(' ')
            new_data = LDA_Modeling.word_rmv_lemma(self.data_lemmatized, ulist)
            final_df, df_keywords, optimal_model = LDA_Modeling.modeling(self, new_data)
        else:
            #Get labeled dataframe
            final_df = pd.DataFrame()
            final_df = LDA_Modeling.format_topics_sentences(optimal_model, corpus, self.text, self.data)
        return final_df, df_keywords, optimal_model