# VehiclesTracking
This repository contains the project of the intnership made at T3Lab (https://www.t3lab.it/) for the master's degree at Unibo a.y. 2023/2024. The work aims to analyze the traffic on the roads in the area of Emilia-Romagna IT, in order to achieve stastistcs regarding the traffic flow over time. Having that kind of information one could be able, for example, to retrieve the intersections where there is the higher number of congestions in a week, and consequently intevene to improve the quality of the streets. To analyze the videos coming from the traffic cameras, Deep Learning techniques are exploited.  

# Execution Instructions
The notebook `vehicleTracking.ipynb` has been used to test various SOTA alghoritms and also to generate the example outputs in `data/`. The most accurate model for the case of study is Yolo Nas. 

## Load Model and Fine Tune
The fine tuning of the model Yolo NAS S from super gradients is executed in the notebook `Train_yolo_nas_on_custom_dataset.ipynb`. This notebook comes from Roboflow and uses a dataset created on that platform for the fine tuning. The dataset is available at https://universe.roboflow.com/unibo-sgsxs/vehicle-detect-from-video. The model obtained is  `models/average_model.pth`, and this model improves the weakness of the standard model. To test the model, the script `src/yoloNasRetrained.py` can be used.

## Speed Up with TensorRt
To quantize the model, it has to be firstly converted to ONNX. This can be done with few lines of code as explained in super gradient's documentation. The model is available in ` models/average_model.onnx`. The model can be tested with the notebook `onnxExecution.ipynb`, then the tensorrt engine can be build with the script `src/buildEngine.py` that enables FP16 precision. The engine is dependent on the hardware where is constructed and the building operation requires time; the script saves the engine to avoid to repeat this operation. To execute the engine, refer to the notebook `tensorRtExecution.ipynb`.

## Int8 Calibration
An attempt to calibrate the model with the notebook `calibration.ipynb` is failed with error; to reproduce the experiment, the dataset for the calibration has to be in a folder `calibration/`. Other tryies by referring to super gradients documentation also didn't lead to good results. The only model that has been quantized succesfully with Int8 precision is ResNet that is not suitable for this problem.

# Conclusions
The obtained model can run at about 20 FPS that is 2x of the execution with super gradients at FP32 precision. To obtain a real-time application, Int8 calibration is needed, as it should speed up the model by 6x. Future works can aim to achieve that calibration and then analyze the videos by using zones as in the notebook `vehicleTracking.ipynb` and extract statistics from the traffic.