
SemEval-2017 Task 4: Sentiment Analysis in Twitter
Codalab Submission and Scoring Instructions
http://alt.qcri.org/semeval2017/task4/

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

The data enclosed is a compilation of annotated sentiment datasets for SemEval-2017 task 4, to be used for development with CodaLab. The files are grouped per subtask and language, and this full README is included for each subtask.

Here are the subtasks for SemEval-2017 task 4:

A: Message Polarity Classification
B: Topic-Based Message Polarity Classification, two-point scale
C: Topic-Based Message Polarity Classification, five-point scale
D: Topic-Based Tweet Quantification, two-point scale
E: Topic-Based Tweet Quantification, five-point scale

4A English:

baseline: baseline-A-english.txt
input: SemEval2017-task4-dev.subtask-A.english.INPUT.txt
gold answers: twitter-2016test-A-English.txt
scorer: SemEval2017_task4_test_scorer_subtaskA.pl

4A Arabic:

baseline: baseline-A-arabic.txt
input: SemEval2017-task4-dev.subtask-A.english.INPUT.txt
gold answers: SemEval2017-task4-dev.subtask-A-Arabic.txt
scorer: SemEval2017_task4_test_scorer_subtaskA.pl

4B English:

baseline: baseline-B-english.txt
input: SemEval2017-task4-dev.subtask-BD.english.INPUT.txt
gold answers: twitter-2016test-BD-English.txt
scorer: SemEval2017_task4_test_scorer_subtaskB.pl

4B Arabic:

baseline: baseline-B-arabic.txt
input: SemEval2017-task4-dev.subtask-BD.arabic.INPUT.txt
gold answers: SemEval2017-task4-dev.subtask-BD.arabic.txt
scorer: SemEval2017_task4_test_scorer_subtaskB.pl

4C-4E: Coming soon!

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Scorer Usage Instructions:

 (a) outside of CodaLab
     perl SemEval2016_task4_test_scorer_subtask[A-E].pl <GOLD_FILE> <INPUT_FILE>
 (b) with CodaLab, i.e., $codalab=1 (certain formatting is expected, this should be only used by the task organizers, but we give it here for completeness)
     perl SemEval2016_task4_test_scorer_subtask[A-E].pl <INPUT_FILE> <OUTPUT_DIR>

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

CodaLab Instructions:

Please test your system early using the development set in order to ensure successful submission. During the testing phase, you will only be allowed to submit twice. 

Prior to submitting, you will need to register for the task and to accept the terms and conditions.

In order to make a submission, compress your file as a zip (do NOT zip a folder), and submit at the page located below under "Participate" -> "Submit/View Results" -> Click "Submit"

The competitions are located here:

4A English: https://competitions.codalab.org/competitions/15885
4A Arabic: https://competitions.codalab.org/competitions/15887
4B English: https://competitions.codalab.org/competitions/15888
4B Arabic: https://competitions.codalab.org/competitions/15889

4C-4E: Coming Soon!

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Contact:

E-mail: semevaltweet@googlegroups.com

Sara Rosenthal, IBM Research 
Noura Farra, Columbia University 
Preslav Nakov, Qatar Computing Research Institute, HBKU 

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Summary of the subtasks:

Subtask A: Message Polarity Classification.
Given a message, classify whether the message is of positive, negative, or neutral sentiment.

Subtasks B-C: Topic-Based Message Polarity Classification.
Given a message and a topic, classify the message on
B) two-point scale: positive or negative sentiment towards that topic
C) five-point scale: sentiment conveyed by that tweet towards the topic on a five-point scale.

Subtasks D-E: Tweet quantification.
Given a set of tweets about a given topic, estimate the distribution of the tweets across
D) two-point scale: positive and negative classes
E) five-point scale: the five classes of a five-point scale.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


References:

Preslav Nakov, Sara Rosenthal, Svetlana Kiritchenko, Saif M. Mohammad, Zornitsa Kozareva, Alan Ritter, Veselin Stoyanov, Xiaodan Zhu. Developing a successful SemEval task in sentiment analysis of Twitter and other social media texts. Language Resources and Evaluation 50(1): 35-65 (2016).

Preslav Nakov, Alan Ritter, Sara Rosenthal, Fabrizio Sebastiani, and Veselin Stoyanov. SemEval-2016 Task 4: Sentiment Analysis in Twitter. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval'2016), June 16-17, 2016, San Diego, California, USA.

Sara Rosenthal, Preslav Nakov, Svetlana Kiritchenko, Saif M Mohammad, Alan Ritter, and Veselin Stoyanov. SemEval-2015 Task 10: Sentiment Analysis in Twitter. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval'2015), pp.451-463, June 4-5, 2016, Denver, Colorado, USA.

Sara Rosenthal, Preslav Nakov, Alan Ritter, Veselin Stoyanov. SemEval-2014 Task 9: Sentiment Analysis in Twitter. In Proceedings of International Workshop on Semantic Evaluation (SemEval’14), pp.73-80, August 23-24, 2014, Dublin, Ireland.

Preslav Nakov, Sara Rosenthal, Zornitsa Kozareva, Veselin Stoyanov, Alan Ritter, Theresa Wilson. SemEval-2013 Task 2: Sentiment Analysis in Twitter. In Proceedings of the Second Joint Conference on Lexical and Computational Semantics (*SEM'13), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval'2013). pp. 312-320, June 17-19, 2013, Atlanta, Georgia, USA.
