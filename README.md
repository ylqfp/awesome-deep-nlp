# awesome-deep-nlp

## Attention  

1. ICLR15, Neural Machine Translation by Jointly Learning to Align and Translate , Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.   

1. ACL15, Encoding Source Language with Convolutional Neural Network for Machine Translation , Fandong Meng, Zhengdong Lu, Mingxuan Wang, Hang Li, Wenbin Jiang, Qun Liu

1. ACL15, A Hierarchical Neural Autoencoder for Paragraphs and Documents , Jiwei Li, Minh-Thang Luong, Dan Jurafsky  

1. EMNLP15, A Neural Attention Model for Sentence Summarization, Alexander M. Rush, Sumit Chopra and Jason Weston  

1. EMNLP15 short, Not All Contexts Are Created Equal: Better Word Representations with Variable Attention6, Wang Ling.

1. NAACL15 , Two/Too Simple Adaptations of Word2Vec for Syntax Problems  

1. NIPS14, Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu. Recurrent Models of Visual Attention.

1. ICLR15, Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate.  

1. ACL15, Fandong Meng, Zhengdong Lu, Mingxuan Wang, et al. Encoding Source Language with Convolutional Neural Network for Machine Translation.  

1. ACL15, Jiwei Li, Minh-Thang Luong, Dan Jurafsky. A Hierarchical Neural Autoencoder for Paragraphs and Documents. 

1. EMNLP15, Alexander M. Rush, Sumit Chopra, Jason Weston. A Neural Attention Model for Sentence Summarization. 

1. EMNLP15, Wang Ling, Lin Chu-Cheng, Yulia Tsvetkov, et al. Not All Contexts Are Created Equal: Better Word Representations with Variable Attention

1. End-to-End Attention-based Large Vocabulary Speech Recognition

1. NIPS14 ws, End-to-end Continuous Speech Recognition using Attention-based Recurrent NN: First Results http://arxiv.org/abs/1412.1602

1. NIPS15, Attention-Based Models for Speech Recognition 

1. EMNLP15, Effective Approaches to Attention-based Neural Machine Translation 

1. ICML15, Kelvin Xu, Jimmy Ba, Ryan Kiros, et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. 

1. Karol Gregor, Ivo Danihelka, Alex Graves, et al. DRAW: A Recurrent Neural Network For Image Generation. 

1. NIPS15, Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, et al. Teaching Machines to Read and Comprehend. 

1. NIPS15, Lei Jimmy Ba, Roger Grosse, Ruslan Salakhutdinov, Brendan Frey. Learning Wake-Sleep Recurrent Attention Models.


## IR

1. **DSSM**   


	Learning deep structured semantic models for web search using clickthrough data, cikm2013 [Paper](http://www.msr-waypoint.net/pubs/198202/cikm2013_DSSM_fullversion.pdf) [code1](https://github.com/mranahmd/dssm-wemb-theano) [code2](https://github.com/outstandingcandy/dssm)
2. **CDSSM**   
A latent semantic model with convolutional-pooling structure for information retrieval, msr, CIKM2014
3. **ARC-I**  
convolutional neural network architectures for matching natural language sentences, NIPS2014
3. **ARC-II**  
convolutional neural network architectures for matching natural language sentences, NIPS2014
3. **RAE**  
Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection, NIPS2011
3. **Deep Match**  
A deep architecture for matching short texts, NIPS,2013
3. **CNTN**  
Convolutional Neural Tensor Network Architecture for Community-based Question Answering, IJCAI2015
3. **CNNPI**  
convolutional neural network for paraphrase identification, NAACL2015
3. **MultiGranCNN**  
MultiGranCNN: An architecture for general matching of text chunks on multiple levels of granularity, ACL2015

3. **CLSTM**   
	Contextual LSTM (CLSTM) models for Large scale NLP tasks, Google, Arxiv201602
3. **CLSM**   
A latent semantic model with convolutional-pooling structure for information retrieval, cikm2014
4. **Recurrent-DSSM**  
Palangi, H., Deng, L., Shen, Y., Gao, J., He, X., Chen, J., Song, X., and Ward, R. Learning sequential semantic representations of natural language using recurrent neural networks. In ICASSP, 2015.

4. **LSTM-DSSM**  
SEMANTIC MODELLING WITH LONG-SHORT-TERM MEMORY FOR INFORMATION RETRIEVAL,ICLR2016, workshop

4. **DCNN: Dynamic convolutional neural network**  
a convolutional neural network for modeling sentences, acl2014
convolutional neural network architectures for matching natural language sentences, nips2014, noah
3. **BRAE: bilingually-constrained recursive auto-encoders**  
bilingually-constrained phrase embeddings for machine translation, acl2014, long paper
6. **LSTM-RNN**  
Deep sentence embedding using lstm networks: analysis and application to information retrieval, 201602
7. **SkipThought**  
Skip thought vectors, 
9. **Bidirectional LSTM-RNN**  
Bi-directional LSTM Recurrent Neural Network for Chinese Word Segmentation, 201602, Arxiv
10. **MV-DNN**  
A multi-view deep learning approach for cross domain user modeling in recommendation systems, WWW2015

## LSTM - A Search Space Odyssey  
Study: Vanilla LSTM and 8 variants. Each one differs from the vanilla LSTM by a single change.
Conclusion: Our results show that none of the variants can improve upon the standard LSTM architecture significantly, and demonstrate the forget gate and the output activation function to be its most critical components.

## Knowledge Graph

## Deep Generative Models

## awesome-deep-machine-learning
## Kernel

1.
Bayesian Nonparametric Kernel-Learning, NIPS2015  
针对问题：1.kernel需要predine，对quality of the finite sample estimator 有影响。-> data-driven kernel function. 2. N*N Gram矩阵需要计算，无法应用于大规模数据集。  

Random features have been recently
shown to be an effective way to scale
kernel methods to large datasets.
Roughly speaking, random feature
techniques like random kitchen sinks
(RKS) [18] work as follows.  Bochners theorem states that a continuous shift-invariant kernel K(x, y) = k(x − y) is a
positive definite function if and only if k(t) is the Fourier transform of a non-negative measure
ρ(ω).

