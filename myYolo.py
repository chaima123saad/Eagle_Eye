import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
video_path = 'C:/Users/Chaima/Desktop/Eagle_Eye/tunisia.mp4' 

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'C:/Users/Chaima/Desktop/Eagle_Eye/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while True:
        ret, frame = cap.read() 
        if not ret:
            break 
        results = model(frame)

        predictions = results.pred[0]

        for *box, conf, cls in predictions:
            class_name = results.names[int(cls)] 
            print(f"Detected {class_name} with confidence {conf:.2f}")

            if conf >= 0.5: 
                x1, y1, x2, y2 = [int(coord) for coord in box]
                color = (0, 255, 0) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release() 
    cv2.destroyAllWindows()

print(f"Output video saved to: {output_path}")
