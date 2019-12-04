# attentive modality hopping for speech emotion recognition

## This repository contains the source code used in the following paper,

**ATTENTIVE MODALITY HOPPING MECHANISM FOR SPEECH EMOTION
RECOGNITION**, <a href="http://arxiv.org/abs/1912.00846">[paper]</a>

----------

### [requirements]
	tensorflow==1.14 (tested on cuda-10.1, cudnn-7.6)
	python==3.7
	scikit-learn>=0.20.0
	nltk>=3.3
    

### [download data corpus]
- IEMOCAP <a href="https://sail.usc.edu/iemocap/">[link]</a>
<a href="https://link.springer.com/article/10.1007/s10579-008-9076-6">[paper]</a>
- download IEMOCAP data from its original web-page (license agreement is required)


### [preprocessed-data schema (our approach)]
- Please refer our previous work that use same preprocessing (audio, text): <a href="https://github.com/david-yoon/multimodal-speech-emotion">[click] </a>
- If you want to download the "preprocessed corpus" from us directly, please send us an email after getting the license from IEMOCAP team.
- We cannot provide ASR-processed transcription due to the license issue (commercial API), however, we assume that it is moderately easy to extract ASR-transcripts from the audio signal by oneself. (we used google-cloud-speech-api)



### [source code]
- repository contains code for following models  (TBD soon)
	 > Attentive Modality Hopping (AMH) <br>


----------

### [training]
- refer to the "model/train_reference_script.sh"
- Results will be displayed in the console. <br>
- The final test result will be stored in "./TEST_run_result.txt" <br>



----------


### [cite]
- Please cite our paper, when you use our code | model | dataset
>
