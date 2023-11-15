from super_gradients.training import models
import supervision as sv
import cv2
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torchvision
from collections import defaultdict
from tracker.byte_tracker import BYTETracker

import time

classes = ['1', '2', '3', '5', '7']  # average_model_v2 classes (the correct ones)
names = ['None', 'Bycicle', 'Car', 'Motorcycle', 'None', 'Bus', 'None', 'Truck']

model = models.get(
	'yolo_nas_s',
	num_classes=len(classes),
	checkpoint_path='average_model.pth').to('cuda')

tracker = BYTETracker(model)

# img = Image.open('/home/gspina/Desktop/moto1.jpeg')

info = sv.VideoInfo.from_video_path('/home/gspina/Scrivania/lm/tirocinio/data/FromYoutube/cutted.mp4')
cap = cv2.VideoCapture('/home/gspina/Scrivania/lm/tirocinio/data/FromYoutube/cutted.mp4')
# print(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(info)
S = ((int)(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), (int)(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('/home/gspina/Desktop/prova.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), S)

vid_frame_count = 0
#results = []
track_history = defaultdict(lambda: [])

t0 = time.time()
while cap.isOpened():
	success, frame = cap.read()

	if success:
		vid_frame_count += 1

		res = model.predict(frame, conf=0.5)[0]

		detections = sv.Detections(xyxy=res.prediction.bboxes_xyxy,confidence=res.prediction.confidence,class_id=res.prediction.labels.astype(int))
		'''
		if res is not None:
			#res = demo_postprocess(res, S)#postprocess(res, 5, conf_thre=0.5)
			online_targets = tracker.update(res, (S[0], S[1], S[0], S[1], 0, 0), S)
			online_tlwhs = []
			online_ids = []
			online_scores = []
			for t in online_targets:
				tlwh = t.tlwh
				x,y,w,h = tlwh
				tid = t.track_id
				vertical = tlwh[2] / tlwh[3] > 1.6
				if tlwh[2] * tlwh[3] > 0 and not vertical:#self.args.min_box_area and not vertical:
					online_tlwhs.append(tlwh)
					online_ids.append(tid)
					online_scores.append(t.score)
				track = track_history[tid]
				bb_center = (float(x+w/2), float(y+h/2))
				track.append(bb_center)#(float(x), float((y))))
				if len(track) > 30:
					track.pop(0)
     
				points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
				cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
		'''		
		box_annotator = sv.BoxAnnotator()

		labels = [f"{names[int(classes[class_id])]} {confidence:0.2f}" for confidence, class_id in zip(res.prediction.confidence, res.prediction.labels.astype(int))]

		annotated_frame = box_annotator.annotate(scene= frame, detections=detections, labels= labels)

		# out.write(annotated_frame)

		if True:
			if vid_frame_count == 1:
				cv2.namedWindow('Prova', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Prova', annotated_frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
            			break
t1=time.time()
print(f'Execution time: {t1-t0}')
cap.release()
out.release()
cv2.destroyAllWindows()
'''
tran = T.Compose([
     T.PILToTensor(),
     T.ConvertImageDtype(torch.float),
])

img = np.array(img).astype(np.uint8)
img = cv2.resize(img, (640,640), cv2.INTER_LINEAR)

#img = img / 255.0

#img = tran(img)
#img = torch.permute(img,(1,2,0)).numpy()
print(img.shape)

results = list(model.predict(img, conf=0.3))[0]

detections = sv.Detections(
	xyxy = results.prediction.bboxes_xyxy,
	confidence = results.prediction.confidence,
	class_id = results.prediction.labels.astype(int)
	)

box_annotator = sv.BoxAnnotator()

labels = [f"{class_id} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

img_annotated = box_annotator.annotate(scene=img, detections=detections, labels=labels)

cv2.imshow('a', img_annotated)
'''
