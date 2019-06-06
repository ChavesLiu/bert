# BERT

>**\*\*\*\*\* New May 31st, 2019: Whole Word Masking Models \*\*\*\*\***

**\*\*\*\*\* 2019年5月31日:全字掩蔽模型 \*\*\*\*\***

This is a release of several new models which were the result of an improvement
the pre-processing code.

这是几个新模型的版本，它们是预处理代码改进的结果。

In the original pre-processing code, we randomly select WordPiece tokens to
mask. For example:

在原始的预处理代码中，我们随机选择要屏蔽的字元标记。例如:

>`Input Text: the man jumped up , put his basket on phil ##am ##mon ' s head`
`Original Masked Input: [MASK] man [MASK] up , put his [MASK] on phil
[MASK] ##mon ' s head`

`输入文本: the man jumped up , put his basket on phil ##am ##mon ' s head`
`原来遮掩的输入: [MASK] man [MASK] up , put his [MASK] on phil [MASK] ##mon ' s head`

The new technique is called Whole Word Masking. In this case, we always mask
*all* of the the tokens corresponding to a word at once. The overall masking
rate remains the same.

这种新技术被称为全字屏蔽。在本例中，我们总是同时屏蔽*所有*对应于一个单词。总掩蔽率保持不变。

>`Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK]
[MASK] ' s head`

`全字掩盖的输入: the man [MASK] up , put his basket on [MASK] [MASK]
[MASK] ' s head`

>The training is identical -- we still predict each masked WordPiece token
independently. The improvement comes from the fact that the original prediction
task was too 'easy' for words that had been split into multiple WordPieces.

训练是相同的——我们仍然独立地预测每个蒙面单词标记。改进的原因在于，原来的任务对于被分成多个单词的单词预测来说太“简单”了。

>This can be enabled during data generation by passing the flag
`--do_whole_word_mask=True` to `create_pretraining_data.py`.

这可以通过向`create_pretraining_data.py`传递参数`--do_whole_word_mask=True`在数据生成期间启用。 

>Pre-trained models with Whole Word Masking are linked below. The data and
training were otherwise identical, and the models have identical structure and
vocab to the original models. We only include BERT-Large models. When using
these models, please make it clear in the paper that you are using the Whole
Word Masking variant of BERT-Large.

全字掩蔽的预先训练模型见下面链接。除此之外，数据和训练是相同的，模型的结构和词汇量也与原始模型相同。我们只包括BERT-Large。当使用这些模型时，请在论文中明确指出，您使用的是全字屏蔽变体的BERT-Large。

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

Model                                    | SQUAD 1.1 F1/EM | Multi NLI Accuracy
---------------------------------------- | :-------------: | :----------------:
BERT-Large, Uncased (Original)           | 91.0/84.3       | 86.05
BERT-Large, Uncased (Whole Word Masking) | 92.8/86.7       | 87.07
BERT-Large, Cased (Original)             | 91.5/84.8       | 86.09
BERT-Large, Cased (Whole Word Masking)   | 92.9/86.7       | 86.46

**\*\*\*\*\* 2019年2月7日:TfHub模块 \*\*\*\*\***

>BERT has been uploaded to [TensorFlow Hub](https://tfhub.dev). See
`run_classifier_with_tfhub.py` for an example of how to use the TF Hub module, 
or run an example in the browser on [Colab](https://colab.sandbox.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb).

BERT已经被上传到 [TensorFlow Hub](https://tfhub.dev). 看`run_classifier_with_tfhub.py` 例如，如何使用TF Hub模块， 或用浏览器[Colab](https://colab.sandbox.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)上运行一个示例.

**\*\*\*\*\* 2018年11月23日: Un-normalized multilingual model(未规范化语言模型) + Thai(泰语) +
Mongolian(蒙古语) \*\*\*\*\***

>We uploaded a new multilingual model which does *not* perform any normalization
on the input (no lower casing, accent stripping, or Unicode normalization), and
additionally inclues Thai and Mongolian.

我们上传了一个新的多语言模型， 它“不”对输入执行任何标准化(没有小写、重音剥离或Unicode规范化)，此外还包括泰语和蒙古语。

>**It is recommended to use this version for developing multilingual models,
especially on languages with non-Latin alphabets.**

**建议使用此版本开发多语言模型，特别是在非拉丁字母的语言上。**

>This does not require any code changes, and can be downloaded here:

这不需要任何代码更改，可以在此处下载

>*   **[`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters

*   **[`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104种语言，12层，768隐藏层，12个头，110m参数

**\*\*\*\*\* 2018年11月15日: SOTA SQuAD 2.0 System \*\*\*\*\***

>We released code changes to reproduce our 83% F1 SQuAD 2.0 system, which is
currently 1st place on the leaderboard by 3%. See the SQuAD 2.0 section of the
README for details.

我们发布了代码修改，重现了SQuAD2.0系统83%的F1，目前在积分榜上以3%的优势排名第一。有关详情，请参阅辩论席的SQuAD2.0部分。

> **\*\*\*\*\* New November 5th, 2018: Third-party PyTorch and Chainer versions of
> BERT available \*\*\*\*\***
>NLP researchers from HuggingFace made a
[PyTorch version of BERT available](https://github.com/huggingface/pytorch-pretrained-BERT)
which is compatible with our pre-trained checkpoints and is able to reproduce
our results. Sosuke Kobayashi also made a
[Chainer version of BERT available](https://github.com/soskek/bert-chainer)
(Thanks!) We were not involved in the creation or maintenance of the PyTorch
implementation so please direct any questions towards the authors of that
repository.

 **\*\*\*\*\*2018年11月5日:第三方PyTorch和Chainer版本的BERT可用\*\*\*\*\***

来自HuggingFace的NLP研究人员制作了一个PyTorch版本的BERT，它与我们预训练过的断点兼容，能够重现我们的结果。[可用BERT的PyTorch版本](https://github.com/huggingface/pytorch-pretrained-BERT)。Sosuke Kobayashi做的[可用BERT的Chainer版本](https://github.com/soskek/bert-chainer)。

(谢谢!)我们没有参与PyTorch的创建或维护实现，请直接向作者提问。

> **\*\*\*\*\* New November 3rd, 2018: Multilingual and Chinese models available
> \*\*\*\*\***
>
> *   **[`BERT-Base, Multilingual`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
> (Not recommended, use `Multilingual Cased` instead)**: 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
> *   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
> Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110Mparameters

**\*\*\*\*\*2018年11月3日，多语种中文车型上市\*\*\*\*\***

我们提供了两个BERT模型:

*   **[`BERT-Base, Multilingual`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
    (不推荐用 `Multilingual Cased` 加载)**: 102种语言，12层，768隐藏层，12个头，110m参数
*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    中文简体和繁体，12层，768隐藏层，12头，110M参数

>We use character-based tokenization for Chinese, and WordPiece tokenization for
all other languages. Both models should work out-of-the-box without any code
changes. We did update the implementation of `BasicTokenizer` in
`tokenization.py` to support Chinese character tokenization, so please update if
you forked it. However, we did not change the tokenization API.

我们中文使用的是基于字符的标记，对所有其他语言使用单词tokenization。这两种模型都应该解压即用，无需任何代码更改。我们在`tokenization.py`中更新了`BasicTokenizer`的实现。支持中文字符化，所以如果你如果fork了请更新。但是，我们没有更改了tokenization化API。

>For more, see the
[Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).

有关更多信息，请参见
[多语言README](https://github.com/google-research/bert/blob/master/multilingual.md).

**\*\*\*\*\* End new information \*\*\*\*\***

## 介绍

>**BERT**, or **B**idirectional **E**ncoder **R**epresentations from
**T**ransformers, is a new method of pre-training language representations which
obtains state-of-the-art results on a wide array of Natural Language Processing
(NLP) tasks.

**BERT**, or **B**idirectional **E**ncoder **R**epresentations 来源于**T**ransformers, 是一种新的预训练语言表达的方法，是解决自然语言处理(NLP)任务最先进的方法。

>Our academic paper which describes BERT in detail and provides full results on a
number of tasks can be found here:

我们的学术论文对BERT做了详细的描述，并提供了一些任务的结果，可以在这里找到:
[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

>To give a few numbers, here are the results on the
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) question answering
task:
>SQuAD v1.1 Leaderboard (Oct 8th 2018) | Test EM  | Test F1
------------------------------------- | :------: | :------:
1st Place Ensemble - BERT             | **87.4** | **93.2**
2nd Place Ensemble - nlnet            | 86.0     | 91.7
1st Place Single Model - BERT         | **85.1** | **91.8**
2nd Place Single Model - nlnet        | 83.5     | 90.1

下面是关于[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)的问答任务的一些数字结果:
SQuAD v1.1 排行榜(2018年10月8日) | Test EM  | Test F1
------------------------------------- | :------: | :------:
第一名组合模型 - BERT             | **87.4** | **93.2**
第二名组合模型 - nlnet            | 86.0     | 91.7
第一名单一模型 - BERT         | **85.1** | **91.8**
第二名单一模型 - nlnet        | 83.5     | 90.1


以及一些自然语言推理任务:

System                  | MultiNLI | Question NLI | SWAG
----------------------- | :------: | :----------: | :------:
BERT                    | **86.7** | **91.1**     | **86.3**
OpenAI GPT (Prev. SOTA) | 82.2     | 88.1         | 75.0

>Plus many other tasks.

加上许多其他任务。

>Moreover, these results were all obtained with almost no task-specific neural
network architecture design.

而且，这些结果都是在几乎没有针对特定任务的神经网络架构的情况下得到的。

>If you already know what BERT is and you just want to get started, you can
[download the pre-trained models](#pre-trained-models) and
[run a state-of-the-art fine-tuning](#fine-tuning-with-bert) in only a few
minutes.

如果你已经知道BERT，现在你只是想开始使用，你可以下载[预训练模型](#预训练模型) 
和 [运行最先进的微调模型fine-tuning](#fine-tuning-with-bert) 只需要几分钟。

## 什么是BERT？

>BERT is a method of pre-training language representations, meaning that we train
a general-purpose "language understanding" model on a large text corpus (like
Wikipedia), and then use that model for downstream NLP tasks that we care about
(like question answering). BERT outperforms previous methods because it is the
first *unsupervised*, *deeply bidirectional* system for pre-training NLP.

BERT是一种预训练的语言表达的方法，这意味着我们在一个大的文本语料库（如维基百科）上训练一个通用的“语言理解”模型，然后将该模型用于我们关心的下游NLP任务（如问答）。BERT优于以前的方法，因为它是第一个*无监督*、*深度双向*的预训练NLP系统。

>*Unsupervised* means that BERT was trained using only a plain text corpus, which
is important because an enormous amount of plain text data is publicly available
on the web in many languages.

*无监督*意味着BERT只使用纯文本语料库进行训练，这一点很重要，因为大量纯文本数据在网络上以多种语言公开。

>Pre-trained representations can also either be *context-free* or *contextual*,
and contextual representations can further be *unidirectional* or
*bidirectional*. Context-free models such as
[word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) or
[GloVe](https://nlp.stanford.edu/projects/glove/) generate a single "word
embedding" representation for each word in the vocabulary, so `bank` would have
the same representation in `bank deposit` and `river bank`. Contextual models
instead generate a representation of each word that is based on the other words
in the sentence.

预训练的表示方法可以是*上下文无关的*或者是*上下文相关的*，上下文表示还可以是*单向的*或*双向的*。上下文无关的模型，例如:[word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) 或者[GloVe](https://nlp.stanford.edu/projects/glove/)为词汇表中的每个单词生成一个单独的`word embedded`来表示，这样`bank`在`bank deposit`和`river bank`中将具有相同的表示。上下文模型生成基于句子中其他单词的每个单词的表示。

>BERT was built upon recent work in pre-training contextual representations —
including [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432),
[Generative Pre-Training](https://blog.openai.com/language-unsupervised/),
[ELMo](https://allennlp.org/elmo), and
[ULMFit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
— but crucially these models are all *unidirectional* or *shallowly
bidirectional*. This means that each word is only contextualized using the words
to its left (or right). For example, in the sentence `I made a bank deposit` the
unidirectional representation of `bank` is only based on `I made a` but not
`deposit`. Some previous work does combine the representations from separate
left-context and right-context models, but only in a "shallow" manner. BERT
represents "bank" using both its left and right context — `I made a ... deposit`
— starting from the very bottom of a deep neural network, so it is *deeply
bidirectional*.

BERT是建立在预训练包括上下文表示[半监督序列学习](https://arxiv.org/abs/1511.01432),[生成预训练](https://blog.openai.com/language-unsupervised/),[ELMo](https://allennlp.org/elmo),[ULMFit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html),但至关重要的是，这些模型都是“单向的”或“浅双向的”。这意味着每个单词只使用其左边(或右边)的单词进行上下文化。例如：在`I made a bank deposit`这句话中，`bank`的单向表示仅基于`I made a`，而不是`deposit`。以前的一些工作确实结合了来自单独的左上下文和右上下文模型的表示，但只是以一种`浅`的方式。BERT用`bank`的左右上下文来表示`bank`,`I made a ... deposit` — 从一个深度神经网络的最底部开始，所以它是`深度双向的`;

>BERT uses a simple approach for this: We mask out 15% of the words in the input,
run the entire sequence through a deep bidirectional
[Transformer](https://arxiv.org/abs/1706.03762) encoder, and then predict only
the masked words. For example:

BERT用了一种简单的方法:我们屏蔽掉输入中的15%的单词，通过深层双向[Transformer](https://arxiv.org/abs/1706.03762)的编码器，然后仅仅预测屏蔽掉的词，例如：

```
Input: the man went to the [MASK1] . he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
```

>In order to learn relationships between sentences, we also train on a simple
task which can be generated from any monolingual corpus: Given two sentences `A`
and `B`, is `B` the actual next sentence that comes after `A`, or just a random
sentence from the corpus?

为了学习句子之间的关系，我们还训练了一个简单的任务，这个任务可以从任何单语语料库生成:给定两个句子`A`和`B`，`B`是在`A`之后的下一个句子，或是语料库中的一个随机句子?

```
Sentence A: the man went to the store .
Sentence B: he bought a gallon of milk .
Label: IsNextSentence
```

```
Sentence A: the man went to the store .
Sentence B: penguins are flightless .
Label: NotNextSentence
```

>We then train a large model (12-layer to 24-layer Transformer) on a large corpus
(Wikipedia + [BookCorpus](http://yknzhu.wixsite.com/mbweb)) for a long time (1M
update steps), and that's BERT.

我们用BERT在一个大的训练语料(Wikipedia + [BookCorpus](http://yknzhu.wixsite.com/mbweb))，训练了一个大模型(12层到24层Transformer)；花费了很长的时间（更新了一百万个迭代的参数）

>Using BERT has two stages: *Pre-training* and *fine-tuning*.

用BERT有两步：*预训练*和*微调*

>**Pre-training** is fairly expensive (four days on 4 to 16 Cloud TPUs), but is a
one-time procedure for each language (current models are English-only, but
multilingual models will be released in the near future). We are releasing a
number of pre-trained models from the paper which were pre-trained at Google.
Most NLP researchers will never need to pre-train their own model from scratch.

**预训练** 是相当高成本的(使用4到16个云TPU用了4天),但是对于每种语言都是一次性的过程(目前的模型只使用英语，但是多语言模型将在不久的将来发布)

>**Fine-tuning** is inexpensive. All of the results in the paper can be
replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU,
starting from the exact same pre-trained model. SQuAD, for example, can be
trained in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of
91.0%, which is the single system state-of-the-art.

**微调** 是成本低的方案。本文的所有结果，只需要单个云TPU运行最多1小时可以得到；从相同的预训练模型开始，例如，SQuAD在单个TPU进行30分钟左右的训练，使Dev F1分数可达91.0%，这是目前单系统最优的水平；

>The other important aspect of BERT is that it can be adapted to many types of
NLP tasks very easily. In the paper, we demonstrate state-of-the-art results on
sentence-level (e.g., SST-2), sentence-pair-level (e.g., MultiNLI), word-level
(e.g., NER), and span-level (e.g., SQuAD) tasks with almost no task-specific
modifications.

BERT的另一个重要方面是它可以很容易地适应许多类型的NLP任务。在本文中，我们展示了在几乎没有特定任务修改的情况下，句子级(例如SST -2)、句子对级(例如MultiNLI)、单词级(例如NER)和span-level(例如SQuAD)任务的最新结果。

>## What has been released in this repository?

## 这个代码库中发布了什么?

>We are releasing the following:
*   TensorFlow code for the BERT model architecture (which is mostly a standard
    [Transformer](https://arxiv.org/abs/1706.03762) architecture).
*   Pre-trained checkpoints for both the lowercase and cased version of
    `BERT-Base` and `BERT-Large` from the paper.   
*   TensorFlow code for push-button replication of the most important
    fine-tuning experiments from the paper, including SQuAD, MultiNLI, and MRPC.

我们发布了以下内容:
*  BERT模型架构的TensorFlow代码(主要是一个标准[Transformer](https://arxiv.org/abs/1706.03762)体系结构)。
*  预训练的checkpoints的小写和大小写版本:`BERT-Base`和`BERT-Large`。
*  TensorFlow代码本文中最重要的微调实验包括SQuAD、MultiNLI和MRPC.

>All of the code in this repository works out-of-the-box with CPU, GPU, and Cloud
TPU.

本代码库中的所有代码都可以在CPU、GPU和云TPU上直接解压运行。

>## Pre-trained models
## 预训练模型

>We are releasing the `BERT-Base` and `BERT-Large` models from the paper.
`Uncased` means that the text has been lowercased before WordPiece tokenization,
e.g., `John Smith` becomes `john smith`. The `Uncased` model also strips out any
accent markers. `Cased` means that the true case and accent markers are
preserved. Typically, the `Uncased` model is better unless you know that case
information is important for your task (e.g., Named Entity Recognition or
Part-of-Speech tagging).

我们在paper中发布了`BERT-Base`和`BERT-Large`模型。`Uncased`表示在单词符号化之前文本已经小写，例如，`John Smith`变成了`john smith`。`Uncased`模型也去掉了任何口音标记。`Cased`表示保留了混合真实的大小写和重音符号通常，除非您知道这种情况的信息对于您的任务非常重要，否则`Uncased`模型更好，(例如，命名实体识别或词性标注)。

>These models are all released under the same license as the source code (Apache
2.0).

这些模型都是在与源代码(Apache 2.0)相同的许可下发布的。

>For information about the Multilingual and Chinese model, see the
[Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).

有关多语言和中文模型的信息，请参见
[Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).

>**When using a cased model, make sure to pass `--do_lower=False` to the training
scripts. (Or pass `do_lower_case=False` directly to `FullTokenizer` if you're
using your own script.)**

**使用大小写混合模型时，请确保将`——do_lower=False`传递给训练脚本。(如果使用自己的脚本，则直接将`do_lower_case=False`传递给`FullTokenizer`)**

>The links to the models are here (right-click, 'Save link as...' on the name):

模型的链接见下(右键单击链接，'链接存储为...'):

>*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads , 110M parameters
*   **[`BERT-Large, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
    (Not recommended, use `Multilingual Cased` instead)**: 102 languages,
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M


*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
    12层layer,768个隐藏层,12个头,110M参数;
    
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)**:
    24层layer,1024个隐藏层,16个头,340M参数;

*   **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**:
    12层layer,768个隐藏层,12个头,110M参数;

*   **[`BERT-Large, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)**:
    24层layer,1024个隐藏层,16个头,340M参数;
    
*   **[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104种语言,12层layer,768个隐藏层,12个头,110M参数;
    
*   **[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
    不推荐使用`Multilingual Cased`:102种语言,12层layer,768个隐藏层,12个头,110M参数;

*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    简体繁体，12层，768隐藏，12头，110M参数

>Each .zip file contains three items:
*   A TensorFlow checkpoint (`bert_model.ckpt`) containing the pre-trained
    weights (which is actually 3 files).
*   A vocab file (`vocab.txt`) to map WordPiece to word id.
*   A config file (`bert_config.json`) which specifies the hyperparameters of
    the model.

每个.zip文件包含三个项目：

*   一个TensorFlow 模型checkpoint断点文件(`bert_model.ckpt`)，包含预训练的参数(实际上是3个文件)。
*   一个词典文件(`vocab.txt`)，将词映射为词id。
*   一个配置文件(`bert_config.json`)，它指定模型的超参数。


>## Fine-tuning with BERT
## 关于BERT微调

>**Important**: All results on the paper were fine-tuned on a single Cloud TPU,
which has 64GB of RAM. It is currently not possible to re-produce most of the
`BERT-Large` results on the paper using a GPU with 12GB - 16GB of RAM, because
the maximum batch size that can fit in memory is too small. We are working on
adding code to this repository which allows for much larger effective batch size
on the GPU. See the section on [out-of-memory issues](#out-of-memory-issues) for
more details.

**重点**:本文的所有结果均在单个云TPU上进行的微调,具备64GB的内存.目前不可能再生产大部分的`BERT-Large`的结果，在本文中使用的GPU在12GB~16GB的内存，所以使用可以装入内存的最大的bath进行处理，我们正在努力将代码添加到这个代码库中，这样可以在GPU上实现更大的有效批处理大小。有关详情请参见[内存不足问题](#内存不足问题)一节。

>This code was tested with TensorFlow 1.11.0. It was tested with Python2 and
Python3 (but more thoroughly with Python2, since this is what's used internally
in Google).

这段代码是用TensorFlow 1.11.0测试的。用Python2和Python3(但是完全适配的是Python2，因为这是谷歌内部使用的版本)。

>The fine-tuning examples which use `BERT-Base` should be able to run on a GPU
that has at least 12GB of RAM using the hyperparameters given.

使用`BERT-Base`的微调示例应该能够在使用给定超参数的GPU上运行，GPU至少有12GB的RAM。

>### Fine-tuning with Cloud TPUs
### 云TPU微调

>Most of the examples below assumes that you will be running training/evaluation
on your local machine, using a GPU like a Titan X or GTX 1080.

假设你用像Titan X或GTX 1080这样GPU的本地计算机，可直接训练和评估下面这些例子。

>However, if you have access to a Cloud TPU that you want to train on, just add
the following flags to `run_classifier.py` or `run_squad.py`:

但是，如果您想要训练他，使用云TPU，只需将以下参数添加到`run_classifier.py`或`run_squad.py`;

```
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

>Please see the
[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)
for how to use Cloud TPUs. Alternatively, you can use the Google Colab notebook
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".

请看[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)了解如何使用云计算TPUs。或者，您可以使用谷歌Colab笔记本"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".

>On Cloud TPUs, the pretrained model and the output directory will need to be on
Google Cloud Storage. For example, if you have a bucket named `some_bucket`, you
might use the following flags instead:

使用云TPUs，使用谷歌云存储预训练模型要输出文件夹参数，例如，你有一个bucket叫`some_bucket`,您可以使用以下标志代替:

```
  --output_dir=gs://some_bucket/my_output_dir/
```

>The unzipped pre-trained model files can also be found in the Google Cloud
Storage folder `gs://bert_models/2018_10_18`. For example:

未解压的预训练模型文件，能够在谷歌云中找到存储文件夹`gs://bert_models/2018_10_18`。例如:

```
export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
```

>### Sentence (and sentence-pair) classification tasks

### 句子(和句子对)分类任务

>Before running this example you must download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`. Next, download the `BERT-Base`
checkpoint and unzip it to some directory `$BERT_BASE_DIR`.

在运行此示例[这个脚本](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)之前，必须下载[GLUE data](https://gluebenchmark.com/tasks)解压到某个目录`$GLUE_DIR`。接下来，下载`BERT-Base`断点，并将其解压缩到某个目录`$BERT_BASE_DIR`。

>This example code fine-tunes `BERT-Base` on the Microsoft Research Paraphrase
Corpus (MRPC) corpus, which only contains 3,600 examples and can fine-tune in a
few minutes on most GPUs.

这个示例代码基于微软研究释义语料库(Microsoft Research ase Corpus, MRPC)对`BERT-Base`进行了微调，该语料库只包含3600个示例，在大多数gpu上只需几分钟就可以进行微调。

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

>You should see output like this:

您应该看到这样的输出:

```
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
```

>This means that the Dev set accuracy was 84.55%. Small sets like MRPC have a
high variance in the Dev set accuracy, even when starting from the same
pre-training checkpoint. If you re-run multiple times (making sure to point to
different `output_dir`), you should see results between 84% and 88%.

这意味着开发集的准确率为84.55%。像MRPC这样的小集合在开发集精度上有很大的差异，即使是从相同的训练前断点开始时也是如此。如果您多次重新运行(确保指向不同的`output_dir`)，您应该会看到84%到88%之间的结果。

>A few other pre-trained models are implemented off-the-shelf in
`run_classifier.py`, so it should be straightforward to follow those examples to
use BERT for any single-sentence or sentence-pair classification task.

在`run_classifier.py`中还实现了其他一些预先训练的模型。因此，遵循这些示例使用BERT进行任何单句或句子对分类任务应该是很简单的。

>Note: You might see a message `Running train on CPU`. This really just means
that it's running on something other than a Cloud TPU, which includes a GPU.

注意:您可能会看到一条消息`正在CPU上训练`。这实际上只是意味着它运行在云TPU(包括GPU)之外的其他东西上。

>#### Prediction from classifier

#### 从分类器预测

>Once you have trained your classifier you can use it in inference mode by using
the --do_predict=true command. You need to have a file named test.tsv in the
input folder. Output will be created in file called test_results.tsv in the
output folder. Each line will contain output for each sample, columns are the
class probabilities.

一旦您训练了分类器，就可以使用--do_predict=true命令在推理模式下使用它。您需要一个名为test.tsv的文件在输入文件夹中。输出将在输出文件夹中创建名为test_results.tsv的文件。每一行将包含每个示例的输出，列是分类的概率。

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output/
```

>### SQuAD 1.1

### SQuAD 1.1

>The Stanford Question Answering Dataset (SQuAD) is a popular question answering
benchmark dataset. BERT (at the time of the release) obtains state-of-the-art
results on SQuAD with almost no task-specific network architecture modifications
or data augmentation. However, it does require semi-complex data pre-processing
and post-processing to deal with (a) the variable-length nature of SQuAD context
paragraphs, and (b) the character-level answer annotations which are used for
SQuAD training. This processing is implemented and documented in `run_squad.py`.

斯坦福问答数据集(SQuAD)是一个流行的问答基准数据集。BERT(在发布时)几乎不需要修改特定于任务的网络架构或增加数据，就可以获得最先进的结果。然而，它确实需要有点复杂的数据预处理和后处理来处理(a)队际内容各段长度不一的性质，和(b)用于队际训练的字符级阅读理解。此处理是在`run_squad.py`中实现并记录。

>To run on SQuAD, you will first need to download the dataset. The
[SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to
link to the v1.1 datasets any longer, but the necessary files can be found here:

要运行SQuAD数据集，您首先需要下载数据集。[SQuAD website](https://rajpurkar.github.io/squadexplorer/)似乎不再链接到v1.1数据集，但必要的文件可以在这里找到:

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

>Download these to some directory `$SQUAD_DIR`.

下载这些到某个文件夹`$SQUAD_DIR`。

>The state-of-the-art SQuAD results from the paper currently cannot be reproduced
on a 12GB-16GB GPU due to memory constraints (in fact, even batch size 1 does
not seem to fit on a 12GB GPU using `BERT-Large`). However, a reasonably strong
`BERT-Base` model can be trained on the GPU with these hyperparameters:

由于内存的限制，目前在12GB-16GB GPU上无法复现论文中最先进的阵容结果(事实上，即使是批量大小为1的GPU也无法在12GB的GPU上使用`BERT-Large`)。然而，一个相当强大的`BERT-Base`模型可以在这些GPU上使用这些参数运行:

```shell
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

>The dev set predictions will be saved into a file called `predictions.json` in
the `output_dir`:

开发集预测将保存到文件夹`output_dir`一个名为`predictions.json`的文件中:

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

>Which should produce an output like this:

它应该产生这样的输出:

```shell
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

>You should see a result similar to the 88.5% reported in the paper for
`BERT-Base`.

基于`BERT-Base`您应该会看到类似于论文中报告的88.5%的结果

>If you have access to a Cloud TPU, you can train with `BERT-Large`. Here is a
set of hyperparameters (slightly different than the paper) which consistently
obtain around 90.5%-91.0% F1 single-system trained only on SQuAD:

如果您可以访问云TPU，您可以使用`BERT-Large`进行训练。这里是一组超参数(与本文略有不同)，单系统训练只针对SQuAD始终得到F1约90.5%-91.0%:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

>For example, one random run with these parameters produces the following Dev
scores:

例如，一个带有这些参数的随机运行，在Dev集会产生以下分数:

```shell
{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
```

>If you fine-tune for one epoch on
[TriviaQA](http://nlp.cs.washington.edu/triviaqa/) before this the results will
be even better, but you will need to convert TriviaQA into the SQuAD json
format.

如果在此之前对[TriviaQA](http://nlp.cs.washington.edu/triviaqa/)进行一个epoch的微调，结果会更好，但是需要将TriviaQA转换为SQuAD json格式。

>### SQuAD 2.0

### SQuAD 2.0

>This model is also implemented and documented in `run_squad.py`.

这个模型也在`run_squad.py`中实现和记录。

>To run on SQuAD 2.0, you will first need to download the dataset. The necessary
files can be found here:

要在SQuAD 2.0上运行，首先需要下载数据集。所需文件可在此找到:

*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

>Download these to some directory `$SQUAD_DIR`.

将他们下载到某文件夹`$SQUAD_DIR`.

>On Cloud TPU you can run with BERT-Large as follows:

在云TPU你能够运行BERT-Large如下:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
```

>We assume you have copied everything from the output directory to a local
directory called ./squad/. The initial dev set predictions will be at
./squad/predictions.json and the differences between the score of no answer ("")
and the best non-null answer for each question will be in the file
./squad/null_odds.json

我们假设您已经将所有内容从输出目录复制到名为 ./squad/ 的本地目录中。最初的开发集预测将在 ./squad/predictions.json。每个问题的无答案("")和最佳非空答案之间的差异将在./squad/null_odds.json文件中

>Run this script to tune a threshold for predicting null versus non-null answers:

运行此脚本来调优预测空答案和非空答案的阈值:

```shell
python $SQUAD_DIR/evaluate-v2.0.py \
    $SQUAD_DIR/dev-v2.0.json \ 
    ./squad/predictions.json \
    --na-prob-file ./squad/null_odds.json
```

>Assume the script outputs "best_f1_thresh" THRESH. (Typical values are between
-1.0 and -5.0). You can now re-run the model to generate predictions with the
derived threshold or alternatively you can extract the appropriate answers from
./squad/nbest_predictions.json.

假设脚本输出"best_f1_thresh"THRESH。(典型值介于-1.0和-5.0之间)。现在，您可以重新运行模型，以生成带有派生阈值的预测，或者从./squad/nbest_predictions.json中提取适当的答案。

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
```

>### Out-of-memory issues

### 内存不足问题

>All experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of
device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely
to encounter out-of-memory issues if you use the same hyperparameters described
in the paper.

本文的所有实验都在一个拥有64GB设备内存的云TPU上进行了微调。因此，当使用12GB-16GB RAM的GPU时，如果使用与本文描述的相同的超参数，很可能会遇到内存不足的问题。

>The factors that affect memory usage are:
*   **`max_seq_length`**: The released models were trained with sequence lengths
    up to 512, but you can fine-tune with a shorter max sequence length to save
    substantial memory. This is controlled by the `max_seq_length` flag in our
    example code.
*   **`train_batch_size`**: The memory usage is also directly proportional to
    the batch size.
*   **Model type, `BERT-Base` vs. `BERT-Large`**: The `BERT-Large` model
    requires significantly more memory than `BERT-Base`.
*   **Optimizer**: The default optimizer for BERT is Adam, which requires a lot
    of extra memory to store the `m` and `v` vectors. Switching to a more memory
    efficient optimizer can reduce memory usage, but can also affect the
    results. We have not experimented with other optimizers for fine-tuning.

影响内存使用的因素有:

*   **`max_seq_length`**:发布的模型的最大序列长度达到512，但是您可以使用更短的最大序列长度进行
    微调，以节省大量内存。这由示例代码中的`max_seq_length`标志控制。

*   **`train_batch_size`**:内存使用量也与批大小成正比。

*   **模型类型,`BERT-Base`对比`BERT-Large`**:`BERT-Large`模型比`BERT-Base`需要更多的内存。

*   **优化器**: BERT的默认优化器是Adam，它需要大量额外的内存来存储`m`和`v`向量。切换到更高效的
    内存优化器可以减少内存使用，但也会影响结果。我们还没有试验过其他用于微调的优化器。


>Using the default training scripts (`run_classifier.py` and `run_squad.py`), we
benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with
TensorFlow 1.11.0:

使用默认的训练脚本(`run_classifier.py`和`run_squad.py`)，我们使用TensorFlow 1.11.0测试了单个Titan X GPU (12GB RAM)上的最大批处理大小:

System       | Seq Length | Max Batch Size
------------ | ---------- | --------------
`BERT-Base`  | 64         | 64
...          | 128        | 32
...          | 256        | 16
...          | 320        | 14
...          | 384        | 12
...          | 512        | 6
`BERT-Large` | 64         | 12
...          | 128        | 6
...          | 256        | 2
...          | 320        | 1
...          | 384        | 0
...          | 512        | 0

> Unfortunately, these max batch sizes for `BERT-Large` are so small that they
> will actually harm the model accuracy, regardless of the learning rate used. We
> are working on adding code to this repository which will allow much larger
> effective batch sizes to be used on the GPU. The code will be based on one (or
> both) of the following techniques:

不幸的是，这些`BERT-Large`的最大批处理大小是如此之小，以至于它们实际上会损害模型的准确性，而不管使用的学习率如何。我们正在向这个代码库添加代码，这将允许GPU上使用更大的有效批处理大小。守则将基于下列一项(或两项)技术:

>*    **Gradient accumulation**: The samples in a minibatch are typically
     independent with respect to gradient computation (excluding batch
     normalization, which is not used here). This means that the gradients of
     multiple smaller minibatches can be accumulated before performing the weight
     update, and this will be exactly equivalent to a single larger update.
>*    [**Gradient checkpointing**](https://github.com/openai/gradient-checkpointing):
     The major use of GPU/TPU memory during DNN training is caching the
     intermediate activations in the forward pass that are necessary for
     efficient computation in the backward pass. "Gradient checkpointing" trades
     memory for compute time by re-computing the activations in an intelligent
     way.

*   **梯度累积**:对于梯度计算，小型批中的样本通常是独立的(不包括这里没有使用的批标准化)。这意味着在执行权值更新之前，可以累积多个较小的小批的梯度，这与单个较大的更新完全相同。

*   [**梯度断点**](https://github.com/openai/gradient-checkpointing):在DNN训练中，GPU/TPU内存的主要用途是缓存前向传递的中间激活，这对于后向传递的高效计算是必要的。“梯度断点”通过以智能的方式重新计算激活，用内存交换计算时间。

**However, this is not implemented in the current release.**

**但是，这在当前版本没有实现**

## Using BERT to extract fixed feature vectors (like ELMo)

##利用BERT提取固定特征向量 (与 ELMo 相似)

> In certain cases, rather than fine-tuning the entire pre-trained model
 end-to-end, it can be beneficial to obtained *pre-trained contextual
 embeddings*, which are fixed contextual representations of each input token
 generated from the hidden layers of the pre-trained model. This should also
 mitigate most of the out-of-memory issues.

在某些情况下，与其端到端微调整个预训练模型，还不如获得*预训练上下文嵌入*，这是由预训练模型的隐含层生成的每个输入词的固定上下文表示形式。这也应该可以缓解大多数内存不足的问题。

> As an example, we include the script `extract_features.py` which can be used
> like this:

我们有`extract_features.py`脚本可以这样使用，例子如下：

```shell
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > /tmp/input.txt

python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

> This will create a JSON file (one line per line of input) containing the BERT
 activations from each Transformer layer specified by `layers` (-1 is the final
 hidden layer of the Transformer, etc.)

这将创建一个JSON文件(每行输入一行)，其中包含由`layers`指定的每个Transformer层的BERT激活(-1是Transformer的最后一个隐藏层，等等)。

> Note that this script will produce very large output files (by default, around
 15kb for every input token).

注意，这个脚本将生成非常大的输出文件(默认情况下，每个输入词大约产生15kb)。

> If you need to maintain alignment between the original and tokenized words (for
 projecting training labels), see the [Tokenization](#tokenization) section
 below.

如果您需要在原始单词和标记词之间保持对齐(用于投射训练标签)，请参阅下面的[#tokenization](#tokenization)一节。

> **Note:** You may see a message like `Could not find trained model in model_dir:
 /tmp/tmpuB5g5c, running initialization to predict.` This message is expected, it
 just means that we are using the `init_from_checkpoint()` API rather than the
 saved model API. If you don't specify a checkpoint or specify an invalid
 checkpoint, this script will complain.

**注意:**您可能会看到这样一条消息:`在model_dir:/tmp/tmpuB5g5c中找不到经过训练的模型，正在运行初始化以进行预测。`这条消息是预期的，它只是意味着我们正在使用`init_from_checkpoint()`API，而不是保存的模型API。如果没有指定断点或指定无效的断点，此脚本将打印出提示。

## Tokenization

> For sentence-level tasks (or sentence-pair) tasks, tokenization is very simple.
> Just follow the example code in `run_classifier.py` and `extract_features.py`.
> The basic procedure for sentence-level tasks is:
>1.  Instantiate an instance of `tokenizer = tokenization.FullTokenizer`
>2.  Tokenize the raw text with `tokens = tokenizer.tokenize(raw_text)`.
>3.  Truncate to the maximum sequence length. (You can use up to 512, but you
>    probably want to use shorter if possible for memory and speed reasons.)
>4.  Add the `[CLS]` and `[SEP]` tokens in the right place.

对于句子级任务(或句子对任务)，标记化非常简单。只需遵循`run_classifier.py`和`extract_features.py`中的示例代码即可。句子级任务的基本步骤是:
1.  实例化`tokenizer = tokeniz.fulltokenizer`的实例 
2.  使用`token = tokenizer.tokenize(raw_text)`对原始文本进行标记。
3.  截断到最大序列长度。(最多可以使用512个，但出于内存和速度的考虑，您可能希望使用更短的内存。)
4.  在正确的位置添加`[CLS]`和`[SEP]`标记。


> Word-level and span-level tasks (e.g., SQuAD and NER) are more complex, since
> you need to maintain alignment between your input text and output text so that
> you can project your training labels. SQuAD is a particularly complex example
> because the input labels are *character*-based, and SQuAD paragraphs are often
> longer than our maximum sequence length. See the code in `run_squad.py` to show
> how we handle this.

单词级和词下标级任务(例如，SQuAD和NER)更加复杂，因为您需要保持输入文本和输出文本之间的对齐，以便您可以投射您的训练标签。SQuAD是一个特别复杂的例子，因为输入标签是基于`字符`的，SQuAD段落通常比我们的最大序列长。参见`run_squad.py`中的代码。来展示我们是如何处理的。

> Before we describe the general recipe for handling word-level tasks, it's
> important to understand what exactly our tokenizer is doing. It has three main
> steps:

在我们描述处理单词级任务的一般方法之前，理解我们的记号赋予器究竟在做什么是很重要的。它有三个主要步骤:

>1.  **Text normalization**: Convert all whitespace characters to spaces, and
    (for the `Uncased` model) lowercase the input and strip out accent markers.
    E.g., `John Johanson's, → john johanson's,`.
2.  **Punctuation splitting**: Split *all* punctuation characters on both sides
    (i.e., add whitespace around all punctuation characters). Punctuation
    characters are defined as (a) Anything with a `P*` Unicode class, (b) any
    non-letter/number/space ASCII character (e.g., characters like `$` which are
    technically not punctuation). E.g., `john johanson's, → john johanson ' s ,`
3.  **WordPiece tokenization**: Apply whitespace tokenization to the output of
    the above procedure, and apply[WordPiece](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py)
    tokenization to each token separately. (Our implementation is directly based
    on the one from `tensor2tensor`, which is linked). E.g., `john johanson ' s
    , → john johan ##son ' s ,`

1.  **文本规范化**:将所有空白字符转换为空格，并(对于`Uncased`模型)小写输入，去掉重音符号。例如:`John Johanson's, → john johanson's,`。
2.  **标点符号分裂**:将*所有*标点符号两遍加分隔符(即，在所有标点符号周围加空格)。标点符号被定义为(a)任何带有`P*`Unicode类的字符，(b)任何非字母/数字/空格ASCII字符(例如，像`$`这样的字符在技术上不是标点符号)。例如:`john johanson's, → john johanson ' s ,`
3.  **词块的标记**:对上述过程的输出应用空格tokenization，并分别对每个令牌应用[WordPiece](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py) tokenization。(我们的实现直接基于`tensor2tensor`中的一个，链接如上)。例john johanson ' s    , → john johan ##son ' s ,`

>The advantage of this scheme is that it is "compatible" with most existing
English tokenizers. For example, imagine that you have a part-of-speech tagging
task which looks like this:

该方案的优点是它与大多数现有的英语分词器`兼容`。例如，假设您有一个词性标注任务，如下所示:

```
Input:  John Johanson 's   house
Labels: NNP  NNP      POS NN
```

>The tokenized output will look like this:

标记输出将如下所示:

```
Tokens: john johan ##son ' s house
```

>Crucially, this would be the same output as if the raw text were `John Johanson's house` (with no space before the `'s`).

关键是，这将是相同的输出，就像原始文本是`John Johanson's house`(没有空格前的`'S`)。

>If you have a pre-tokenized representation with word-level annotations, you can
simply tokenize each input word independently, and deterministically maintain an
original-to-tokenized alignment:

如果您有一个带有单词级注释的标记化表示，您可以简单地对每个输入单词进行单独的标记，并确定地保持原始到标记的对齐:

```python
### Input
import tokenization
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
    orig_to_tok_map.append(len(bert_tokens))
    bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
```

>Now `orig_to_tok_map` can be used to project `labels` to the tokenized
representation.

现在`orig_to_tok_map`可以使用`labels`去标记化的表示。

>There are common English tokenization schemes which will cause a slight mismatch
between how BERT was pre-trained. For example, if your input tokenization splits
off contractions like `do n't`, this will cause a mismatch. If it is possible to
do so, you should pre-process your data to convert these back to raw-looking
text, but if it's not possible, this mismatch is likely not a big deal.

有一些常见的英语标记化方案会在BERT的预训练方式之间造成轻微的不匹配。例如，如果您的输入标记化将“do n't”之类的缩写分开，这将导致不匹配。如果可以做到这一点，您应该对数据进行预处理，将其转换回原始文本，但是如果不可能，这种不匹配可能不是什么大问题。

## Pre-training with BERT

> We are releasing code to do "masked LM" and "next sentence prediction" on an
> arbitrary text corpus. Note that this is *not* the exact code that was used for
> the paper (the original code was written in C++, and had some additional
> complexity), but this code does generate pre-training data as described in the
> paper.

我们正在发布代码来对任意文本语料库执行`masked LM`和`下一个句子预测`。注意，这不是本文使用的确切代码(原始代码是用c++编写的，还有一些额外的复杂性)，但是这段代码确实生成了本文中描述的训练前数据。

> Here's how to run the data generation. The input is a plain text file, with one
> sentence per line. (It is important that these be actual sentences for the "next
> sentence prediction" task). Documents are delimited by empty lines. The output
> is a set of `tf.train.Example`s serialized into `TFRecord` file format.

下面是如何生成运行数据。输入是一个纯文本文件，每行一个句子。(重要的是，这些是`下一个句子预测`任务的实际句子)。文档由空行分隔。输出是一组`tf.train.Example`。示例被序列化为`TFRecord`文件格式。

> You can perform sentence segmentation with an off-the-shelf NLP toolkit such as
> [spaCy](https://spacy.io/). The `create_pretraining_data.py` script will
> concatenate segments until they reach the maximum sequence length to minimize
> computational waste from padding (see the script for more details). However, you
> may want to intentionally add a slight amount of noise to your input data (e.g.,
> randomly truncate 2% of input segments) to make it more robust to non-sentential
> input during fine-tuning.

您可以使用现成的NLP工具包执行句子分割，比如[spaCy](https://spacy.io/)。`create_pretraining_data.py`脚本将连接段落，直到它们达到最大的序列长度，以最小化填充造成的计算浪费(有关详细信息，请参阅脚本)。然而，您可能希望有意地在输入数据中添加少量的噪声(例如，随机截断2%的输入段)，使其在微调期间对非语句输入更加健壮。

> This script stores all of the examples for the entire input file in memory, so
> for large data files you should shard the input file and call the script
> multiple times. (You can pass in a file glob to `run_pretraining.py`, e.g.,
> `tf_examples.tf_record*`.)

这个脚本将整个输入文件的所有示例存储在内存中，因此对于大型数据文件，您应该对输入文件进行切分并多次调用脚本。(您可以将一个文件glob传递给`run_pretraining.py`，例如，`tf_examples.tf_record*`。)

> The `max_predictions_per_seq` is the maximum number of masked LM predictions per
> sequence. You should set this to around `max_seq_length` * `masked_lm_prob` (the
> script doesn't do that automatically because the exact value needs to be passed
> to both scripts).

`max_predictions_per_seq`是每个序列的最大masked LM预测数。您应该将其设置为`max_seq_length` *`masked_lm_prob`左右(脚本不会自动这样做，因为需要将确切的值传递给两个脚本)。

```shell
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

> Here's how to run the pre-training. Do not include `init_checkpoint` if you are
> pre-training from scratch. The model configuration (including vocab size) is
> specified in `bert_config_file`. This demo code only pre-trains for a small
> number of steps (20), but in practice you will probably want to set
> `num_train_steps` to 10000 steps or more. The `max_seq_length` and
> `max_predictions_per_seq` parameters passed to `run_pretraining.py` must be the
> same as `create_pretraining_data.py`.

下面是如何进行预训练。如果你是从头开始训练，不指定`init_checkpoint`即可。模型配置(包括vocab大小)在`bert_config_file`中指定。这个示例代码只对少量步骤(20)进行了预训练，但实际上您可能希望将`num_train_steps`设置为10000步或更多。传递给`run_pretraining.py`的`max_seq_length`和`max_predictions_per_seq`参数。必须与`create_pretraining_data.py`相同。



```shell
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

> This will produce an output like this:

这将产生这样的输出:

```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```

Note that since our `sample_text.txt` file is very small, this example training
will overfit that data in only a few steps and produce unrealistically high
accuracy numbers.

注意，因为我们的`sample_text.txt`文件非常小，这个例子的训练将在短短几个步骤内过拟合，并产生不切实际的高准确率结果。



>### Pre-training tips and caveats

### 预训练提示和警告

>*   **If using your own vocabulary, make sure to change `vocab_size` in
    `bert_config.json`. If you use a larger vocabulary without changing this,
    you will likely get NaNs when training on GPU or TPU due to unchecked
    out-of-bounds access.**
*   If your task has a large domain-specific corpus available (e.g., "movie
    reviews" or "scientific papers"), it will likely be beneficial to run
    additional steps of pre-training on your corpus, starting from the BERT
    checkpoint.
*   The learning rate we used in the paper was 1e-4. However, if you are doing
    additional steps of pre-training starting from an existing BERT checkpoint,
    you should use a smaller learning rate (e.g., 2e-5).
*   Current BERT models are English-only, but we do plan to release a
    multilingual model which has been pre-trained on a lot of languages in the
    near future (hopefully by the end of November 2018).
*   Longer sequences are disproportionately expensive because attention is
    quadratic to the sequence length. In other words, a batch of 64 sequences of
    length 512 is much more expensive than a batch of 256 sequences of
    length 128. The fully-connected/convolutional cost is the same, but the
    attention cost is far greater for the 512-length sequences. Therefore, one
    good recipe is to pre-train for, say, 90,000 steps with a sequence length of
    128 and then for 10,000 additional steps with a sequence length of 512. The
    very long sequences are mostly needed to learn positional embeddings, which
    can be learned fairly quickly. Note that this does require generating the
    data twice with different values of `max_seq_length`.
*   If you are pre-training from scratch, be prepared that pre-training is
    computationally expensive, especially on GPUs. If you are pre-training from
    scratch, our recommended recipe is to pre-train a `BERT-Base` on a single
    [preemptible Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing), which
    takes about 2 weeks at a cost of about $500 USD (based on the pricing in
    October 2018). You will have to scale down the batch size when only training
    on a single Cloud TPU, compared to what was used in the paper. It is
    recommended to use the largest batch size that fits into TPU memory.

*  **如果使用您自己的词汇表，必须在`bert_config.json`中更改`vocab_size`。如果你使用一个更大的词汇表而不改变这一点，由于未检查的越界访问，你可能会在GPU或TPU上训练得到NaNs的结果**
*  如果您的任务有一个大型的特定领域的语料库可用(例如，“电影评论”或“科学论文”)，那么从BERT断点开始，在语料库上运行额外的预训练步骤可能是有益的。
*  我们在论文中使用的学习率为1e-4。但是，如果您正在从现有的BERT断点开始进行额外的预训练步骤，则应该使用较小的学习率(例如，2e-5)。
*  目前的BERT模型只使用英语，但我们确实计划在不久的将来(希望在2018年11月底之前)发布一个多语言模型，该模型已经对很多语言进行了预训练。
*  较长的序列代价过高，因为注意力是序列长度的二次方。换句话说，一批长度为512的64个序列比一批长度为128的256个序列代价要高得多。全连接/卷积的代价是相同的，但是对于512长度的序列，注意力代价要大得多。因此，一个好的方法是预先训练90000步，序列长度为128，然后再训练10000步，序列长度为512。非常长的序列是学习位置嵌入最需要的，这样可以很快学会的。注意，这确实需要使用两次不同的`max_seq_length`值生成数据。
*  如果您是从头开始进行预培训，请准备好预培训在计算上代价非常高，尤其是在gpu上。如果您是从头开始进行预培训，我们推荐的方法是在单个[preemptible Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing)上预培训一个`BERT-Base`，大约需要两周时间，成本约为500美元(基于2018年10月的定价)。与本文中使用的方法相比，当只在单个云TPU上进行培训时，您必须缩小批处理大小。建议使用适合TPU内存的最大批处理大小。


### Pre-training data

### 预训练数据

>We will **not** be able to release the pre-processed datasets used in the paper.
For Wikipedia, the recommended pre-processing is to download
[the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2),
extract the text with
[`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply
any necessary cleanup to convert it into plain text.

我们**不能**发布论文中使用的预处理数据集。推荐的预处理是下载维基百科[最新的存储](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2),提取文本使用`WikiExtractor.py`(https://github.com/attardi/wikiextractor)，然后使用任何必要的清理工作将它转换成纯文本。

>Unfortunately the researchers who collected the
[BookCorpus](http://yknzhu.wixsite.com/mbweb) no longer have it available for
public download. The
[Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
is a somewhat smaller (200M word) collection of older books that are public
domain.

不幸的是，收集[BookCorpus](http://yknzhu.wixsite.com/mbweb)的研究人员不再提供公共下载。[Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)是一个较小的(2亿字)公共领域的旧书集合。

>[Common Crawl](http://commoncrawl.org/) is another very large collection of
text, but you will likely have to do substantial pre-processing and cleanup to
extract a usable corpus for pre-training BERT.

[常见的爬虫数据Common Crawl](http://commoncrawl.org/)是另一个非常大的文本集合，但是您可能必须进行大量的预处理和数据清洗，才能提取一个可用的语料库，用于BERT的预培训。

>### Learning a new WordPiece vocabulary

### 学习一个新的词汇表

>This repository does not include code for *learning* a new WordPiece vocabulary.
The reason is that the code used in the paper was implemented in C++ with
dependencies on Google's internal libraries. For English, it is almost always
better to just start with our vocabulary and pre-trained models. For learning
vocabularies of other languages, there are a number of open source options
available. However, keep in mind that these are not compatible with our
`tokenization.py` library:

此代码库不包含用于`学习`新单词词汇表的代码。原因是本文使用的代码是用c++实现的，依赖于谷歌的内部库。对于英语来说，从我们的词汇和预先训练的模型开始学习总是更好的。对于学习其他语言的词汇表，有许多开放源码选项可用。然而，请记住，这些与我们的`tokenization.py`并不兼容的库:

*   [Google's SentencePiece library](https://github.com/google/sentencepiece)

*   [tensor2tensor's WordPiece generation script](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)

*   [Rico Sennrich's Byte Pair Encoding library](https://github.com/rsennrich/subword-nmt)

>## Using BERT in Colab

##在Colab中使用BERT

>If you want to use BERT with [Colab](https://colab.research.google.com), you can
get started with the notebook
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".

如果您想将BERT与[Colab](https://colab.research.google.com)一起使用，可以从“[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)”开始。

>**At the time of this writing (October 31st, 2018), Colab users can access a
Cloud TPU completely for free.** Note: One per user, availability limited,
requires a Google Cloud Platform account with storage (although storage may be
purchased with free credit for signing up with GCP), and this capability may not
longer be available in the future. Click on the BERT Colab that was just linked
for more information.

**撰写本文时(2018年10月31日)，Colab用户可以完全免费访问云TPU。**注意:每个用户只有一个可用性有限的谷歌云平台存储帐户(尽管注册GCP时可以免费购买存储帐户)，而且这个功能将来可能不再可用。点击刚才BERT Colab链接获取更多信息。

>## FAQ

## 常见问题解答

>#### Is this code compatible with Cloud TPUs? What about GPUs?

>Yes, all of the code in this repository works out-of-the-box with CPU, GPU, and
Cloud TPU. However, GPU training is single-GPU only.

####这段代码与云TPUs兼容的吗?那gpu兼容吗?

是的，这个代码库中的所有代码都可以使用CPU、GPU和云TPU解压即可用。但是，GPU训练是单GPU的。

>#### I am getting out-of-memory errors, what is wrong?

>See the section on [out-of-memory issues](#out-of-memory-issues) for more
information.

####为什么报“我的内存不足”错误?

有关信息，请参见[内存不足问题](#内存不足问题)一节信息。

>#### Is there a PyTorch version available?

>There is no official PyTorch implementation. However, NLP researchers from
HuggingFace made a
[PyTorch version of BERT available](https://github.com/huggingface/pytorch-pretrained-BERT)
which is compatible with our pre-trained checkpoints and is able to reproduce
our results. We were not involved in the creation or maintenance of the PyTorch
implementation so please direct any questions towards the authors of that
repository.

####有PyTorch版本吗?

没有正式的PyTorch实现。然而，来自HuggingFace的NLP研究人员制作了一个[PyTorch版本的BERT可用](https://github.com/huggingface/pytorch-pretraining-BERT)，它与我们预先训练过的断点兼容，并且能够重现我们的结果。我们没有参与实现PyTorch版本的创建或维护，所以请向该代码库的作者提出相关问题。

>#### Is there a Chainer version available?

There is no official Chainer implementation. However, Sosuke Kobayashi made a
[Chainer version of BERT available](https://github.com/soskek/bert-chainer)
which is compatible with our pre-trained checkpoints and is able to reproduce
our results. We were not involved in the creation or maintenance of the Chainer
implementation so please direct any questions towards the authors of that
repository.

####有Chainer版本可用吗?

没有正式的Chainer实现。然而，所幸(Sosuke Kobayashi)制作了一个[Chainer版本的BERT](https://github.com/soskek/bert-chainer)，它与我们预先训练过的断点兼容，并且能够重现我们的结果。我们没有参与实现PyTorch版本的创建或维护，所以请向该代码库的作者提出相关问题。


>#### Will models in other languages be released?

>Yes, we plan to release a multi-lingual BERT model in the near future. We cannot
make promises about exactly which languages will be included, but it will likely
be a single model which includes *most* of the languages which have a
significantly-sized Wikipedia.

####其他语言的模型会发布吗?

是的，我们计划在不久的将来发布一个多语言的BERT模型。我们不能确切地承诺哪些语言将被包括在内，但它很可能是一个单一的模型，其中包含有*大多数*语言的大型维基百科。

>#### Will models larger than `BERT-Large` be released?

>So far we have not attempted to train anything larger than `BERT-Large`. It is
possible that we will release larger models if we are able to obtain significant
improvements.

####会发布比`BERT-Large`更大的版本吗?

到目前为止，我们还没有尝试过训练比`BERT-Large`更大的数据。如果我们能够获得显著的改进，我们可能会发布更大的模型。

>#### What license is this library released under?

>All code *and* models are released under the Apache 2.0 license. See the
`LICENSE` file for more information.

#### 这个库是根据什么许可证发布的?

所有代码*和*模型都是在Apache 2.0许可下发布的。有关更多信息，请参阅`LICENSE`文件。


>#### How do I cite BERT?

>For now, cite [the Arxiv paper](https://arxiv.org/abs/1810.04805):

####我该怎么引用BERT呢?

目前，参考 [the Arxiv paper](https://arxiv.org/abs/1810.04805):


```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

If we submit the paper to a conference or journal, we will update the BibTeX.

如果我们将论文提交会议或报纸，我们将更新BibTeX。

>## Disclaimer

>This is not an official Google product.

##免责声明

这不是一个正式的谷歌产品。

## Contact information

For help or issues using BERT, please submit a GitHub issue.

For personal communication related to BERT, please contact Jacob Devlin
(`jacobdevlin@google.com`), Ming-Wei Chang (`mingweichang@google.com`), or
Kenton Lee (`kentonl@google.com`).

##联系信息

有关使用BERT的帮助或问题，请提交GitHub问题。

用于与BERT相关的个人交流，请联系Jacob Devlin (`jacobdevlin@google.com`)、Ming-Wei Chang (`mingweichang@google.com`)或Kenton Lee (`kentonl@google.com`)。
