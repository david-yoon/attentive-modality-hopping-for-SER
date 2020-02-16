# attentive-modality-hopping-for-SER

This repository contains the source code used in the following paper,

**Attentive Modality Hopping Mechanism for Speech Emotion Recognition**, <a href="http://arxiv.org/abs/1912.00846">[paper]</a>

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


### [preprocessing (our approach)]
- Please refer our previous work that use same preprocessing (audio, text): <a href="https://github.com/david-yoon/multimodal-speech-emotion">[click] </a>
- For video modality:
	- We first split each video frame into two sub-frames so that each segment contains only one actor.
	- Then we crop the center of each frame with size 224*224 to focus on the actor and to remove background in the video frame.
	-  Finally, we extract 2,048-dimensional visual features from each video data using pretrained ResNet-101 at a frame rate of 3 per second.
- Format of the data for our experiments:
	> Audio: [#samples, 1000, 120] - (#sampels, sequencs(max 10s), dims) <br>
	> Text (index) : [#samples, 128] - (#sampels, sequencs(max)) <br>
	> Video: [#samples, 32, 2048] - (#sampels, sequencs (max 10.6s), dims) <br>
- Emotion Classes :

  |    class   | #samples |
  |:----------:|----------:|
  |    angry   |    1,103 |
  |   excited  |    1,041 |
  |    happy   |      595 |
  |     sad    |    1,084 |
  | frustrated |    1,849 |
  |  surprise  |      107 |
  |   neutral  |    1,708 |

- If you want to use the same processed-data of our experiments, please drop us an email with the IEMOCAP license.
- We cannot provide ASR-processed transcription due to the license issue (commercial API); however, we assume that it is moderately easy to extract ASR-transcripts from the audio signal by oneself (we used google-cloud-speech-API).




### [source code]
- repository contains code for following models
	 > Attentive Modality Hopping (AMH) <br>


----------

### [training]
- refer to the "model/train_reference_script.sh"
- Results will be displayed in the console. <br>
- The final test result will be stored in "./TEST_run_result.txt" <br>



----------


### [cite]
- Please cite our paper, when you use our code | model | dataset
	> @article{yoon2019attentive,<br>
  title={Attentive Modality Hopping Mechanism for Speech Emotion Recognition},<br>
  author={Yoon, Seunghyun and Dey, Subhadeep and Lee, Hwanhee and Jung, Kyomin},<br>
  journal={arXiv preprint arXiv:1912.00846},<br>
  year={2019}<br>
}
