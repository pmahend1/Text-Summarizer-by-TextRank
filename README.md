# TextRank
TextRank implementation in Spark using Python

## Table of Contents
 - [Introduction](#introduction)
 - [Dataset](#dataset)
    - [Data Collection & Preparation](#data-collection-presentation)
 - [Methods](#methods)
    - [Text Rank](#textrank)
 - [Demo](#demo)
 - [Evaluation](#evaluation)
 - [Installation](#installation)
 - [Execution Instruction](#execution-instruction)
 - [Source Code](#source-code)
 - [References](#references)

## Introduction
This project aims at summarizing long texts into shorter texts by making use of TextRank algorithm like PageRank algorithm except that its applied to nodes of texts instead of hyperlinks. 
The result of this project is to perform an unsupervised method of sentence and keyword extraction. The most important components used in this project include PageRank equation with weighted graph, Graph ,WordNet Python library and Python-Spark(PySpark) tools.
 

## Dataset
### Data Collection & Preparation
The data set used in this project has a collection of twelve text documents which have been generated randomly using Random Text Generator software. The dataset contains documents of exactly 500 words split across different paragraphs. The content of each document varies and the organization of the words inside each document is also different.

#### Sample Data

See here

## Methods

### 1. Text Summarization

    * Take input text and use sentence tokenizer them to split them into sentences.

    * Each sentence will be a node of our graph.

    * Edges are links between sentences.

    * Since its a continuous text , for simplification we take nC2 as the no of undirected edges. We store these nC2 pair of sentences(nodes) as edge parameters.

    * Similarity acts as the edge weight between the sentences. We  made use of wordnet library to get similarity between sentences.
        For sentences, using synset for tagged words we recursively find similarity between words of sentences.
        For each iteration get the best score add it best score
        Take average of the sum


    * So for a text with n sentences we will have n nodes and nC2*2 edges (undirected graph graph) , and weight being Levenshtein distance or some other similarity measure as mentioned above.

    * We apply PageRank equation for the above generated graph by converting it into directed graph. We run the pagerank equation for 20 times 
    
    * After that we order the sentences based decreasing order of text rank score.

    * Finally extract sentences about W words or less(should end with dot) as required for summarization where W is user passed parameter or default is 100

### 2. Top N keywords extraction

    * Take input text and split them by words by using WordNet API for tags and word tokenizing

    * Each unique keyword will be a node of our graph.

    * We need to get rid of stop words so that they do not appear as key words.

    * Select only the unique words in the context. Let this be U. Edges are links between words.

    * Take the same approach we did for sentence extraction for weights of edges.

    * So for a text with U unique words(excluding stop words) we will have U nodes and UC2 edges(undirected) , and weight being similarity distance from Wordnet

    * Once we get graph we apply page rank equation for weighted graph by converting it into directed graph and for 20 iterations.

    * After that we order the words based decreasing order of text rank score.

    * Finally extract top N words as set/required by user or default is 5.


#### Sample Output of TextRank


##Demo


## Observations



## Evaluation



## Installation

### Dependencies

* Spark with Python 2.7
* [Install pip](http://pip.readthedocs.org/en/stable/installing/)
* NLTK Library
* Apache Spark

#### Install NLTK , Networkx and punkt 

```bash
sudo pip install -U nltk
pip install networkx
```

#### Download Stopwords from nltk data source

```python
 #Pythonic way
 import nltk
 nltk.download('punkt')
 nltk.download('averaged_perceptron_tagger')

```
    
### Execution Instruction

#### Summarization using TextRank (Need to change)
    
    $ spark-submit textrank.py <inputfile_path_in_hdfs> <summary_words_length> <no_of_keywords> 

    <summary_words_length> <no_of_keywords>  are not mandatory
        
#### Output Interpretation

Outputs for the collected test text files are uploaded in sample output directory.
They are run with default 100 word summary and no.of keywords = 5 
    
## Source Code


## References
1. TextRank: Bringing Order into Texts, Rada Mihalcea and Paul Tarau, University of North Texas 
   https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

2. WikiPedia: https://en.wikipedia.org/wiki/Automatic_summarization#TextRank_and_LexRank

3. WordNet API: http://nlpforhackers.io/wordnet-sentence-similarity/

4. GitHub: https://github.com/davidadamojr/TextRank
