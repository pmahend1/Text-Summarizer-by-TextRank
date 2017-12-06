# TextRank
TextRank implementation in Spark using Python

## Table of Contents
 - [Introduction](#introduction)
 - [Dataset](#dataset)
    - [Data Collection and Preparation](#data-collection-and-preparation)
    - [Test Data](#test-data)
 - [Methods](#methods)
    - [1.Text Summarization](#1-text-summarization)
    - [2.Top N keywords extraction](#2-top-n-keywords-extraction)
 - [Results](#results)
    - [Sample Output of Summary](#sample-output-of-summary)
    - [Sample Output of Top N Keywords](#sample-output-of-top-n-keywords)
    - [Output Interpretation and Observations](#output-interpretation-and-observations)
 - [Evaluation](#evaluation)
 - [Installation](#installation)
 - [Execution Instructions](#execution-instructions)
 - [Source Code](#source-code)
 - [Issues and Workarounds](#issues-and-workarounds)
 - [References](#references)

## Introduction
This project aims at summarizing long texts into shorter texts by making use of TextRank algorithm. It is similar to PageRank algorithm except that its applied to nodes of texts instead of hyperlinks and takes edge weights into consideration.

The result of this project is to perform an unsupervised method of summary and keyword extraction. The most important components used in this project include PageRank equation with weighted graph, Graph ,WordNet Python library and Python-Spark(PySpark) tools.
 

## Dataset
### Data Collection and Preparation
The data set used in this project has a collection of ten text documents which have been generated randomly using Random Text Generator software. The dataset contains documents of exactly 500 words split across different paragraphs. The content of each document varies and the organization of the words inside each document is also different.

#### Test Data

Look at directory [inputfiles](https://github.com/pmahend1/TextRank---Text-Summarizer/tree/master/inputfiles)

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

## Results
Results of this project are as expected. Output files can be found in [outputfiles](https://github.com/pmahend1/TextRank---Text-Summarizer/tree/master/outputfiles)

### Sample Output of Summary

```Creeping fourth. Gathered bring. Of don't two darkness also moveth, midst fifth bring make deep abundantly called appear our life days air let place man you're waters abundantly won't itself together you after. Saw face place fly own from fowl rule place, first greater brought beast void first behold fruitful lights they're fowl green evening whales fifth, in cattle spirit it fruitful abundantly gathered moveth. Isn't that over saying fourth good days itself dominion, together she'd it kind so a midst fifth male there multiply itself sixth place days. Be seasons that without living creeping herb that days own their female spirit greater. Second was also is all gathering itself winged light.```

### Sample Output of Top N Keywords
```
1 creature
2 let
3 male
4 beast
5 day
6 sea
7 spirit
```

#### Output Interpretation and Observations

Outputs for the collected test text files are uploaded in outputfiles.
They are run with default 100 word summary and no.of keywords = 5

The sentences are edges of undirected graph initially. Based on text rank score they are sorted , sub stringed and write to output.


- Some key observations are the order of sentences can change based on Text rank score achieved. Sentence similarity also has keyrole in this.
- Program run seems to be efficient. 
- Some keywords may not be highly semantic. WordNet similarity has effect on it.

## Evaluation
It is observed that text rank can be efficiently used to extract summary and top keywords of a long text. We see that for text documents of around 500 words(around 2500-3000 charactes) program took around 60 seconds to complete in pseudo distribured installation of Spark on Ubuntu 16.0.4 LTS.

Sentence similarity was better achieved  by making use of WordNet library. Since this is unsupervised method , results may not be always semantically make sence depending upon the input parameters passed and the original flow of the text.

This model can be used as baseline for further Machine Learning/Artificial intelligence on Natural language processing to get results that are
somewhat similar to human feedback. It can be good approach to use this for getting summary of similar texts that are clustered before.(news articles with similar content)


## Installation

### Dependencies

* Spark with Python 2.7
* [pip](http://pip.readthedocs.org/en/stable/installing/)
* NLTK Library
* Networkx library
* Apache Spark

#### Install NLTK , Networkx and punkt 

```bash
sudo pip install -U nltk
pip install networkx
```
[networkx](https://pypi.python.org/pypi/networkx/2.0)

#### Download Stopwords from nltk data source

```python
 #Pythonic way
 import nltk
 nltk.download('punkt')
 nltk.download('averaged_perceptron_tagger')

```
    
### Execution Instructions

Copy text file into HDFS
```
hadoop fs -put <testfile.txt> <hdfs_folder_path>
ex : hadoop fs -put Test2.txt /user/username/Project 

hadoop fs -ls /user/username/Project
```

Copy english.pickle file into same folder where textrank.py is saved and will be run


#### Submitting PySpark program

```    
$ spark-submit textrank.py <hdfs_file_path> <summary_words_length> <no_of_keywords> 

Note : <summary_words_length> <no_of_keywords>  are not mandatory
```

*ex :* 
Specifying both words summary words length and no. of keywords desired.  
`spark-submit textrank.py /user/Project/TextRank/Test2.txt 130 10`  

Specifying summary words length only(keywords retrieved will be 5 by default).  
`spark-submit textrank.py /user/Project/TextRank/Test2.txt 120`  

Not specifying summary words length or no. of keywords desired(Default of 100 summary words and 5 top keywords will be returned).  
`spark-submit textrank.py /user/Project/TextRank/Test2.txt`  
        

    
## Source Code

Will be uploaded in future.  

## Issues and Workarounds

2 reduce(collect) operations on RDD after input file read and sentence RDD conversion are leading to following errors.
However workarounds are used in local pseudo distributed installation of Spark and the code works as expected and resulting output for as expected. Please note we dont have complete access to install or make changes in cluster nodes of public education cluster. If we run the program on DSBA cluster it fails in the middle at the get_summary phase with error *'tokenizers/punkt/english.pickle' not found.  Please
use the NLTK Downloader to obtain the resource:nltk.download()*
  


### Issue 1

When flatMap or map operation is called with get_summary() method on sentencesTokensRDD that gives pickle.PicklingError: args[0] from \__newobj__   

**Probable Reasons and approach used to rectify it:**   
* Could be an error with PySpark source code **/lib/pyspark.zip/pyspark/cloudpickle.py**  
    - Looked if there are updates for cloudpickle.py recently. There is an update recently. So tried replacing cloudpickle.py in local with latest code ; but that yielded incompatibility issues due to Python3 and Python2. So we should see by installing latest stable version of Spark.

*Result* - using latest version of cloudpickle.py did not work. Have to check installing latest version of Spark

* Worker nodes dont have NLTK-Punkt, networkx libraries installed or dont have access to it. Or may be DSBA cluster does not have installation of these  in cluster nodes or master.

    - setup_environment(s) function is called from RDD which will import/install dependent libraries on worker node

*Result* - Error still exists. 

* Default Spark runs with Python2 and running Python3 could fix the issue.
    -  Ran PySpark program with Python3 by running `export PYSPARK_PYTHON=python3` on terminal before submitting program and some minor changes to code as required for Python3 compatibility.  

*Result* - Error still exists. 

* Tokenizer list not present in worker nodes
    - Publish the tokenizer list to nodes. Save english.pickle file in same directory from where spark program is run [reference](#https://stackoverflow.com/questions/46878186/pickling-error-with-spark-submit-pickle-picklingerror-args0-from-newobj?noredirect=1&lq=1#46879838)  

```sent_detector = nltk.data.load('english' + '.pickle')  # DSBA cluster
    print("sent_detector", str(sent_detector))
    sc.broadcast(sent_detector)
```
* Also submitted question on [StackOverflow](https://stackoverflow.com/questions/47616036/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class).   
*Result* - No workable solution yet

* Raised bug with Apache Spark [SPARK-22711](https://issues.apache.org/jira/browse/SPARK-22711)  
*Result* - Yet to get response.

**Workaround**  
Instead RDD1.flatMap() and RDD2.collect() run function(RDD2.collect()). [Reference](https://stackoverflow.com/questions/44911539/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class-with-hado?noredirect=1&lq=1#45081066)

`summary = get_summary(sentencesTokensRDD.collect(),summary_length=summary_limit)`



### Issue 2
RuntimeError: maximum recursion depth exceeded in cmp when RDD.collect() or RDD.saveAsTextFile() is called.  

```
    keywordsRDD = inputFile.flatMap(lambda k: get_keywords(k))
    print("Keywords are..")
    keywordsRDD.saveAsTextFile("textrank_keywords")
```

**Probable Reasons and approach used to rectify it**

* Could be issue with worker nodes not having all the libraries. Seems DSBA cluster does not have NLTK-PUNKT libraries.  
        - setup_environment(s) function is called from RDD which will import/install dependent libraries on worker nodes 
        - also tried import nltk nltk.download('punkt') on DSBA terminal. It gives network error.  
*Result* - Error still exists.

* Python recursion limit is less. 
        - Change recursion limit to max value `sys.setrecursionlimit(32627)`  
*Result* - Error still exists.

**Workaround:**  
Instead of RDD.flatMap() and then RDD.collect , run function(RDD.collect()). [Reference](https://stackoverflow.com/questions/44911539/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class-with-hado?noredirect=1&lq=1#45081066)

`keywords = get_keywords(str(inputFile.collect()),number_of_words=keywordLimit)`



## References
1. TextRank: Bringing Order into Texts, Rada Mihalcea and Paul Tarau, University of North Texas 
   https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

2. WikiPedia: https://en.wikipedia.org/wiki/Automatic_summarization#TextRank_and_LexRank

3. WordNet API: http://nlpforhackers.io/wordnet-sentence-similarity/

4. GitHub: https://github.com/davidadamojr/TextRank
