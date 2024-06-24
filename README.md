# kinyarwanda_ft_llm
a repository about fine tuning LLMs in Kinyarwanda


## Introduction

Recent advancements in artificial intelligence have introduced us to Large Language Models (LLMs), which can generate high-quality text and (with some effort and tweaking) provide factually accurate answers to natural language questions. 

While larger models perform exceptionally well with English text, they also show promising results with "low-resource languages"—languages with fewer textsresources. For example, GPT-4 can generate coherent text based on instructions and translate English text into Kinyarwanda relatively well.

However, these models can be expensive. There is a need to explore similar performance can be achieved by fine-tuning smaller, open-source models.

## Background

Research indicates that smaller open-source language models can be "taught" new languages and perform relatively well. For instance:

- Kuulmets et al. (2024) successfully "taught" LLaMA2-7B Estonian, reporting good results across various tasks.
- Xu et al. (2023) demonstrated that fine-tuning LLaMA2 (7B and 13B) on monolingual data, followed by fine-tuning with a small set of high-quality parallel texts, improves machine translation.

Other people,  have reported encouraging results when fine-tuning open-source LLMs on "smaller" languages (e.g. the Trelis Research youtube channel using Irish) - although these findings have not undergone rigorous academic testing.

## Project Objective

In this repository, we explore the possibility of fine-tuning open-source LLMs to support Kinyarwanda.

Most probably, better results could be achieved by leveraging the structure/morphology of Kinyarwanda (see the work by Nzeyimana & Niyongabo (2022)), 
but but in our initial explorations this will not be explored. 


## References

- Kuulmets, H., Purason, T., Luhtaru, A., & Fishel, M. (2024). Teaching LLaMA a New Language Through Cross-Lingual Knowledge Transfer. ArXiv, abs/2404.04042.
- Xu, H., Kim, Y., Sharaf, A., & Awadalla, H.H. (2023). A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models. ArXiv, abs/2309.11674.
- Trelis Research - [YouTube Video](https://youtu.be/bo49U3iC7qY?si=nwe89atcgeiybwSv)
- Nzeyimana, A., & Niyongabo Rubungo, A. (2022). KinyaBERT: a Morphology-aware Kinyarwanda Language Model. ArXiv, abs/2203.08459.






