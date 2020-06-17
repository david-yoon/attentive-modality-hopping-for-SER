# attentive-modality-hopping-for-SER

This repository contains the source code used in the following paper,

**Attentive Modality Hopping Mechanism for Speech Emotion Recognition**, <a href="http://arxiv.org/abs/1912.00846">[paper]</a>

----------
### [Notice]
I recently found that I use the "precision" metric for the model evaluation.
When I change the metric from "precision" to "accuracy," models show similar performance for the "weighted" case. However, models show lower performance for the "unweighted" case.
This behavior is similarly observed for other models (MHA, MDRE). 

I already revised the source code. You can change the metric at the "project_config.py."

	USE_PRECISION = True   --> "precision" metric
	USE_PRECISION = False  --> "accuracy" metric


Precision (previously misreported as accuracy)

|   Model   | Modality | Weighted    | Unweighted  |
|:---------:|:--------:|-------------|-------------|
|  MDRE[9]  |    A+T   | 0.557 ± 0.018 | 0.536 ± 0.030 |
|  MDRE[9]  |    T+V   | 0.585 ± 0.040 | 0.561 ± 0.046 |
|  MDRE[9]  |    A+V   | 0.481 ± 0.049 | 0.415 ± 0.047 |
|  MHA[12]  |    T+V   | 0.583 ± 0.025 | 0.555 ± 0.040 |
|  MHA[12]  |    T+V   | 0.590 ± 0.017 | 0.560 ± 0.032 |
|  MHA[12]  |    A+V   | 0.490 ± 0.049 | 0.434 ± 0.060 |
|  MDRE[9]  |   A+T+V  | 0.602 ± 0.033 | 0.575 ± 0.046 |
| AMH(ours) |   A+T+V  | 0.624 ± 0.022 | 0.597 ± 0.040 |


Accuracy (revised results)

|   Model   | Modality | Weighted    | Unweighted |
|:---------:|:--------:|-------------|-------------|
|  MDRE[9]  |    A+T   | 0.498 ± 0.059 | 0.418 ± 0.077 |
|  MDRE[9]  |    T+V   | 0.579 ± 0.015 | 0.524 ± 0.021 |
|  MDRE[9]  |    A+V   | 0.477 ± 0.025 | 0.376 ± 0.024 |
|  MHA[12]  |    T+V   | 0.543 ± 0.026 | 0.491 ± 0.028 |
|  MHA[12]  |    T+V   | 0.580 ± 0.019 | 0.526 ± 0.024 |
|  MHA[12]  |    A+V   | 0.471 ± 0.047 | 0.371 ± 0.042 |
|  MDRE[9]  |   A+T+V  | 0.564 ± 0.043 | 0.490 ± 0.056 |
| AMH(ours) |   A+T+V  | 0.617 ± 0.016 | 0.547 ± 0.025 |


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
