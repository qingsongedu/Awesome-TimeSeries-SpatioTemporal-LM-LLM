# Large (Language) Models and Foundation Models (LLM, LM, FM) for Time Series and Spatio-Temporal Data

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/qingsongedu/Awesome-TimeSeries-AIOps-LM-LLM)
[![Visits Badge](https://badges.pufler.dev/visits/qingsongedu/Awesome-TimeSeries-AIOps-LM-LLM)](https://badges.pufler.dev/visits/qingsongedu/Awesome-TimeSeries-AIOps-LM-LLM)
<!-- ![Forks](https://img.shields.io/github/forks/qingsongedu/Awesome-TimeSeries-AIOps-LM-LLM) -->


A professionally curated list of **Large (Language) Models and Foundation Models (LLM, LM, FM) for Temporal Data (Time Series, Spatio-temporal, and Event Data)** with awesome resources (paper, code, data, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

We will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

For general **AI for Time Series (AI4TS)** Papers, Tutorials, and Surveys at the **Top AI Conferences and Journals**, please check [This Repo](https://github.com/qingsongedu/awesome-AI-for-time-series-papers). 


## Survey paper

[**Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**](https://arxiv.org/abs/2310.10196)  

**Authors**: Ming Jin, Qingsong Wen*, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li, Shirui Pan*, Vincent S. Tseng (IEEE Fellow), Yu Zheng (IEEE Fellow), Lei Chen (IEEE Fellow), Hui Xiong (IEEE Fellow)

🌟 If you find this resource helpful, please consider to star this repository and cite our survey paper:

```
@article{jin2023lm4ts,
  title={Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook}, 
  author={Ming Jin and Qingsong Wen and Yuxuan Liang and Chaoli Zhang and Siqiao Xue and Xue Wang and James Zhang and Yi Wang and Haifeng Chen and Xiaoli Li and Shirui Pan and Vincent S. Tseng and Yu Zheng and Lei Chen and Hui Xiong},
  journal={arXiv preprint arXiv:2310.10196},
  year={2023}
}
```

## LLMs for Time Series
#### General Time Series Analysis
* Time-LLM: Time Series Forecasting by Reprogramming Large Language Models, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2310.01728)
* TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2310.04948)
* TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2308.08241)
* LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2308.08469)
* The first step is the hardest: Pitfalls of Representing and Tokenizing Temporal Data for Large Language Models, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2309.06236)
* PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2210.08964)
* One Fits All: Power General Time Series Analysis by Pretrained LM, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/abs/2302.11939) [\[official code\]](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
* Large Language Models Are Zero-Shot Time Series Forecasters, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/abs/2310.07820) [\[official code\]](https://github.com/ngruver/llmtime)


  
#### Transportation Application
* Where Would I Go Next? Large Language Models as Human Mobility Predictors, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2308.15197)
* Leveraging Language Foundation Models for Human Mobility Forecasting, in *SIGSPATIAL* 2022, [\[paper\]](https://arxiv.org/abs/2209.05479)
#### Finance Application
* Temporal Data Meets LLM -- Explainable Financial Time Series Forecasting, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2306.11025)
* BloombergGPT: A Large Language Model for Finance, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2303.17564)
* WeaverBird: Empowering Financial Decision-Making with Large Language Model, Knowledge Base, and Search Engine, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2308.05361)[\[official-code\]](https://github.com/ant-research/fin_domain_llm)
* Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models, in *arXiv* 2023, [\[paper\]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788)
* Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2306.12659)
* The Wall Street Neophyte: A Zero-Shot Analysis of ChatGPT Over MultiModal Stock Movement Prediction Challenges, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2304.05351)

#### Healthcare Application 
* Large Language Models are Few-Shot Health Learners, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2305.15525)
* Health system-scale language models are all-purpose prediction engines, in *Nature* 2023, [\[paper\]](https://www.nature.com/articles/s41586-023-06160-y)
* A large language model for electronic health records, in *NPJ Digit. Med.* 2022, [\[paper\]](https://www.nature.com/articles/s41746-022-00742-2)

#### Event Analysis
* Drafting Event Schemas using Language Models, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2305.14847)
* Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/abs/2305.16646), [\[official-code\]](https://github.com/iLampard/ep_llm)



## PFMs for Time Series  
##### General Time Series Analysis
* SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/abs/2302.00861)
* A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, in *ICLR* 2023, [\[paper\]](https://arxiv.org/abs/2211.14730) [\[official code\]](https://github.com/yuqinie98/PatchTST)
* Contrastive Learning for Unsupervised Domain Adaptation of Time Series, in *ICLR* 2023, [\[paper\]](https://arxiv.org/abs/2206.06243)
* TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting, in *KDD* 2023, [\[paper\]](https://arxiv.org/abs/2306.09364)
* Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency, in *NeurIPS* 2022, [\[paper\]](https://arxiv.org/abs/2206.08496) [\[official code\]](https://github.com/mims-harvard/TFC-pretraining)
* Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting, in *KDD* 2022, [\[paper\]](https://arxiv.org/abs/2206.09113)
* TS2Vec: Towards Universal Representation of Time Series, in *AAAI* 2022, [\[paper\]](https://arxiv.org/abs/2106.10466) [\[official code\]](https://github.com/yuezhihan/ts2vec)
* Voice2Series: Reprogramming Acoustic Models for Time Series Classification, in *ICML* 2021. [\[paper\]](https://arxiv.org/abs/2106.09296) [\[official code\]](https://github.com/huckiyang/Voice2Series-Reprogramming)

#### Event Analysis 
* Prompt-augmented Temporal Point Process for Streaming Event Sequence, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/pdf/2310.04993.pdf) [\[official code\]](https://github.com/yanyanSann/PromptTPP)

## LLMs for Spatio-Temporal Graphs
* Language knowledge-Assisted Representation Learning for Skeleton-based Action Recognition, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2305.12398.pdf)
* Chatgpt-Informed Graph Neural Network for Stock Movement Prediction, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2306.03763.pdf)

## PFMs for Spatio-Temporal Graphs
#### General Purposes
* When do Contrastive Learning Signals help Spatio-Temporal Graph Forecasting? in *SIGSPATIAL* 2023. [\[paper\]](https://arxiv.org/pdf/2108.11873.pdf)
* Mining Spatio-Temporal Relations via Self-Paced Graph Contrastive Learning, in *KDD* 2022. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539422)

#### Climate
* Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks, in *Nature* 2023. [\[paper\]](https://www.nature.com/articles/s41586-023-06185-3)
* ClimaX: A Foundation Model for Weather and Climate, in *ICML* 2023. [\[paper\]](https://arxiv.org/abs/2301.10343) [\[official code\]](https://github.com/microsoft/ClimaX)
* GraphCast: Learning Skillful Medium-Range Global Weather Forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2212.12794)
* FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operator, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2202.11214)
* Accurate Medium-Range Global Weather Forecasting with 3d Neural Networks, in *Nature* 2023. [\[paper\]](https://www.nature.com/articles/s41586-023-06185-3)
* W-MAE: Pre-Trained Weather Model with Masked Autoencoder for Multi-Variable Weather Forecasting, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2304.08754.pdf)
* FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.02948)


#### Transportation
* Pre-trained Bidirectional Temporal Representation for Crowd Flows Prediction in Regular Region, in *IEEE Access* 2019. [\[paper\]](https://ieeexplore.ieee.org/document/8854786)
* Trafficbert: Pretrained Model with Large-Scale Data for Long-Range Traffic Flow Forecasting, in *Expert Systems with Applications* 2021. [\[paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0957417421011179)
* Building Transportation Foundation Model via Generative Graph Transformer, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2305.14826.pdf)

## LLMs for Video Data
* Zero-shot Video Question Answering via Frozen Bidirectional Language Models, in *NeurIPS* 2022. [\[paper\]](https://arxiv.org/pdf/2206.08155.pdf)
* Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners, in *NeurIPS* 2022. [\[paper\]](https://arxiv.org/pdf/2205.10747.pdf)
* VideoLLM: Modeling Video Sequence with Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2305.13292.pdf)
* VALLEY: Viode Assitant with Large Language Model Enhanced Ability, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2306.07207.pdf)
* Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning, in *CVPR* 2023. [\[paper\]](https://arxiv.org/pdf/2302.14115.pdf)
* Retrieving-to-Answer: Zero-Shot Video Question Answering with Frozen Large Language Models, in *NeurIPS* 2022. [\[paper\]](https://openreview.net/pdf?id=9uRS5ysgb9)
* VideoChat: Chat-Centric Video Understanding, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2305.06355.pdf)
* MovieChat: From Dense Token to Sparse Memory for Long Video Understanding, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2307.16449.pdf)
* Language Models are Causal Knowledge Extractors for Zero-shot Video Question Answering, in *CVPR* 2023.  [\[paper\]](https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/papers/Su_Language_Models_Are_Causal_Knowledge_Extractors_for_Zero-Shot_Video_Question_CVPRW_2023_paper.pdf)
* Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2306.02858.pdf)
* Learning Video Representations from Large Language Models, in *CVPR* 2023. 
[\[paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Learning_Video_Representations_From_Large_Language_Models_CVPR_2023_paper.pdf)
* Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2306.05424.pdf)
* Traffic-Domain Video Question Answering with Automatic Captioning, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2307.09636.pdf)
* LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling, in *CVPR* 2023. [\[paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_LAVENDER_Unifying_Video-Language_Understanding_As_Masked_Language_Modeling_CVPR_2023_paper.pdf)

## PFMs for Video Data
* OmniVL: One Foundation Model for Image-Language and Video-Language Tasks, in *NeurIPS* 2022. [\[paper\]](https://openreview.net/pdf?id=u4ihlSG240n)
* Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2306.04362.pdf)
* PAXION: Patching Action Knowledge in Video-Language Foundation Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2305.10683.pdf)
* mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video, in *ICML* 2023. [\[paper\]](https://openreview.net/pdf?id=DSOmy0ScK6)






#### Datasets
##### Traffic Application
* METR-LA [\[link\]](https://zenodo.org/record/5146275)
* PEMS-BAY [\[link\]](https://zenodo.org/record/5146275)
* PEMS04 [\[link\]](https://github.com/Davidham3/ASTGCN)
* SUTD-TrafficQA [\[link\]](https://github.com/sutdcv/SUTD-TrafficQA)
* TaxiBJ [\link\]](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)
* BikeNYC [\[link\]](https://citibikenyc.com/system-data)
* TaxiNYC [\[link\]](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* Mobility [\[link\]](https://github.com/gmu-ggs-nsf-abm-research/mobility-trends)
* LargeST [\[link\]](https://github.com/liuxu77/LargeST)

##### Healthcare Application
* PTB-XL [\[link\]](https://physionet.org/content/ptb-xl/1.0.3/)
* NYUTron [\[link\]](https://datacatalog.med.nyu.edu/dataset/10633)
* UF Health clinical corpus [\[link\]](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/gatortron_og)
* i2b2-2012 [\[link\]](https://www.i2b2.org/NLP/TemporalRelations/)
* MIMIC-III [\[link\]](https://physionet.org/content/mimiciii/1.4/)
* CirCor DigiScope [\[link\]](https://physionet.org/content/circor-heart-sound/1.0.3/)

##### Weather Application
* SEVIR [\[link\]](https://registry.opendata.aws/sevir/)
* Shifts-Weather Prediction [\[link\]](https://github.com/Shifts-Project/shifts)
* AvePRE [\[link\]](https://disc.gsfc.nasa.gov/)
* SurTEMP [\[link\]](https://disc.gsfc.nasa.gov/)
* SurUPS [\[link\]](https://disc.gsfc.nasa.gov/)
* ERA5 reanalysis data [\[link\]](https://github.com/pangeo-data/WeatherBench)
* CMIP6 [\[link\]](https://cds.climate.copernicus.eu/cdsapp\#!/dataset/projections-cmip6?tab=overview)

##### Finance Application
* Finance (Employment) [\[link\]](https://github.com/ashfarhangi/AA-Forecast/tree/main/dataset)
* StockNet [\[link\]](https://github.com/yumoxu/stocknet-dataset)
* EDT [\[link\]](https://github.com/Zhihan1996/TradeTheEvent/tree/main/data\#edt-dataset)
* NASDAQ-100 [\[link\]](https://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)

##### Video Application
* TGIF-QA [\[link\]](https://github.com/YunseokJANG/tgif-qa)
* MSR-VTT [\[link\]](https://drive.google.com/file/d/1pWym3bMNW_WrOZCi5Ls-wKFpaPMbLOio/view)
* WebVid [\[link\]](https://maxbain.com/webvid-dataset/)
* MSVD [\[link\]](https://www.microsoft.com/en-us/download/)
* DiDeMo [\[link\]](https://github.com/LisaAnne/TemporalLanguageRelease)
* COCO [\[link\]](https://cocodataset.org/\#home)

##### Event Analysis Application
* Amazon [\[link\]](https://github.com/ant-research/EasyTemporalPointProcess)
* Taobao [\[link\]](https://github.com/ant-research/EasyTemporalPointProcess)
* Retweet [\[link\]](https://github.com/ant-research/EasyTemporalPointProcess)
* StackOverflow [\[link\]](https://github.com/ant-research/EasyTemporalPointProcess)
* Taxi [\[link\]](https://github.com/ant-research/EasyTemporalPointProcess)

##### Other General Application
* ETT [\[link\]](https://github.com/zhouhaoyi/ETDataset)
* M4 [\[link\]](https://github.com/Mcompetitions/M4-methods)
* Electricity [\[link\]](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
* Alibaba Cluster Trace [\[link\]](https://github.com/alibaba/clusterdata)
* TSSB [\[link\]](https://github.com/ermshaua/time-series-segmentation-benchmark)
* UCR TS Classification Archive [\[link\]](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
 


## Related LLM/LM/FM Resources
#### Survey
* A Survey of Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2303.18223) [\[link\]](https://github.com/RUCAIBox/LLMSurvey)
* Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.13712) [\[link\]](https://github.com/Mooler0410/LLMsPracticalGuide)  
* LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.01933) [\[link\]](https://github.com/AGI-Edgerunners/LLM-Adapters)
* Beyond One-Model-Fits-All: A Survey of Domain Specialization for Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2305.18703)
* Large AI Models in Health Informatics: Applications, Challenges, and the Future, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2303.11568) [\[link\]](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)
* FinGPT: Open-Source Financial Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.06031) [\[link\]](https://github.com/AI4Finance-Foundation/FinGPT) 
* On the Opportunities and Challenges of Foundation Models for Geospatial Artificial Intelligence, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.06798)


#### Github
* Awesome LLM. [\[link\]](https://github.com/Hannibal046/Awesome-LLM)
* Open LLMs. [\[link\]](https://github.com/eugeneyan/open-llms)
* Awesome LLMOps. [\[link\]](https://github.com/tensorchord/Awesome-LLMOps)
* Awesome Foundation Models. [\[link\]](https://github.com/uncbiag/Awesome-Foundation-Models)
* Awesome Graph LLM. [\[link\]](https://github.com/XiaoxinHe/Awesome-Graph-LLM)
* Awesome Multimodal Large Language Models. [\[link\]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

## Related Resources
#### Surveys of Time Series
* Transformers in Time Series: A Survey, in *IJCAI* 2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2002.12478)
* Time-series forecasting with deep learning: a survey, in *Philosophical Transactions of the Royal Society A* 2021. [\[paper\]](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0209)
* A review on outlier/anomaly detection in time series data, in *CSUR* 2021. [\[paper\]](https://arxiv.org/abs/2002.04236)
* Deep learning for time series classification: a review, in *Data Mining and Knowledge Discovery* 2019. [\[paper\]](https://link.springer.com/article/10.1007/s10618-019-00619-1?sap-outbound-id=11FC28E054C1A9EB6F54F987D4B526A6EE3495FD&mkt-key=005056A5C6311EE999A3A1E864CDA986)
* A Survey on Time-Series Pre-Trained Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2305.10716) [\[link\]](https://github.com/qianlima-lab/time-series-ptms)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)

#### Surveys of AIOps
* AIOps: real-world challenges and research innovations, in *ICSE* 2019. [\[paper\]](https://web.eecs.umich.edu/~ryanph/paper/aiops-icse19-briefing.pdf)
* A Survey of AIOps Methods for Failure Management, in *TIST* 2021. [\[paper\]](https://jorge-cardoso.github.io/publications/Papers/JA-2021-025-Survey_AIOps_Methods_for_Failure_Management.pdf)
* AI for IT Operations (AIOps) on Cloud Platforms: Reviews, Opportunities and Challenges, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.04661)


#### LLM/LM/FM Papers for AIOps
* Empowering Practical Root Cause Analysis by Large Language Models for Cloud Incidents, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2305.15778)
* Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2301.03797)
* OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2310.07637)

## Citation
#### If you find this repository helpful for your work, please kindly cite:

```bibtex
@article{jin2023lm4ts,
  title={Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook}, 
  author={Ming Jin and Qingsong Wen and Yuxuan Liang and Chaoli Zhang and Siqiao Xue and Xue Wang and James Zhang and Yi Wang and Haifeng Chen and Xiaoli Li and Shirui Pan and Vincent S. Tseng and Yu Zheng and Lei Chen and Hui Xiong},
  journal={arXiv preprint arXiv:2310.10196},
  year={2023}
}
```



