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

**Authors**: Ming Jin, Qingsong Wen, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li, Shirui Pan, Vincent S. Tseng (IEEE Fellow), Yu Zheng (IEEE Fellow), Lei Chen (IEEE Fellow), Hui Xiong (IEEE Fellow)

🌟 If you find this resource helpful, please consider to star this repository and cite our survey paper:

```
@article{jin2023lm4tssurvey,
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
* PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2210.08964)
* One Fits All: Power General Time Series Analysis by Pretrained LM, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/abs/2302.11939) [\[official code\]](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
* Large Language Models Are Zero-Shot Time Series Forecasters, in *NeurIPS* 2023, [\[paper\]](https://arxiv.org/abs/2310.07820) [\[official code\]](https://github.com/ngruver/llmtime)


  
#### Transportation Application
* Leveraging Language Foundation Models for Human Mobility Forecasting, in *SIGSPATIAL* 2022, [\[paper\]](https://arxiv.org/abs/2209.05479)
* Where Would I Go Next? Large Language Models as Human Mobility Predictors, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2308.15197)
#### Finance Application
* Temporal Data Meets LLM -- Explainable Financial Time Series Forecasting, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2306.11025)
* BloombergGPT: A Large Language Model for Finance, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2303.17564)
* WeaverBird: Empowering Financial Decision-Making with Large Language Model, Knowledge Base, and Search Engine, in *arXiv* 2023, [\[paper\]](https://arxiv.org/abs/2308.05361)
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

## LLMs for Spatio-Temporal Data
#### 



#### 

#### Weather Forecasting
* Accurate medium-range global weather forecasting with 3D neural networks, in *Nature* 2023. [\[paper\]](https://www.nature.com/articles/s41586-023-06185-3)
* ClimaX: A foundation model for weather and climate, in *ICML* 2023. [\[paper\]](https://arxiv.org/abs/2301.10343) [\[official code\]](https://github.com/microsoft/ClimaX)
* FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.02948)
* GraphCast: Learning skillful medium-range global weather forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2212.12794)
* FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operator, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2202.11214)


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
@article{jin2023survey,
  title={Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook}, 
  author={Ming Jin and Qingsong Wen and Yuxuan Liang and Chaoli Zhang and Siqiao Xue and Xue Wang and James Zhang and Yi Wang and Haifeng Chen and Xiaoli Li and Shirui Pan and Vincent S. Tseng and Yu Zheng and Lei Chen and Hui Xiong},
  journal={arXiv preprint arXiv:2310.10196},
  year={2023}
}
```



