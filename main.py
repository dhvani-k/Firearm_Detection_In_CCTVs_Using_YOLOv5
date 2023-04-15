import numpy as np
import cv2


def detection(img_height, img_width, outputLayers, confidencevalueThreshold, NMSvalueThreshold):
    confidence_values = []
    classIDs = []
    box_dimesnions = []

    for output in outputLayers:
        for inspect in output:
            score_values = inspect[5:]
            classID = np.argmax(score_values)
            confidence_val = score_values[classID]
            if confidence_val > confidencevalueThreshold:
                box_dimension = inspect[0:4] * np.array([img_width, img_height, img_width, img_height])
                (originX, originY, width, height) = box_dimension.astype('int')
                x = int(originX - (width / 2))
                y = int(originY - (height / 2))

                box_dimesnions.append([x, y, int(width), int(height)])
                confidence_values.append(float(confidence_val))
                classIDs.append(classID)

    # Apply Non Maxima Suppression
    inspectionNMS = cv2.dnn.NMSBoxes(box_dimesnions, confidence_values, confidencevalueThreshold, NMSvalueThreshold)

    return inspectionNMS, box_dimesnions, classIDs, confidence_values


def image_detection(NNnet):
    imageconfidenceThreshold = 0.5
    imageNMSThreshold = 0.3

    image = cv2.imread('img/man-firing-45-cal-pistol.jpg')
    (img_height, img_width) = image.shape[:2]
    cv2.imshow('Input Image', image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    layerName = NNnet.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in NNnet.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    NNnet.setInput(blob)
    layersOutputs = NNnet.forward(layerName)

    inspectionNMS, box_dimesnions, _, _ = detection(img_height, img_width, layersOutputs, imageconfidenceThreshold,
                                                    imageNMSThreshold)

    if len(inspectionNMS) > 0:
        for i in inspectionNMS.flatten():
            (x, y) = (box_dimesnions[i][0], box_dimesnions[i][1])
            (w, h) = (box_dimesnions[i][2], box_dimesnions[i][3])

            cv2.rectangle(image, (x, y), (x + w, y + h), 3, 2)
            text = 'firearm'
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3, 2)

    cv2.imshow('Detection result', image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def video_detection(NNnet, labels):
    videoconfidenceThreshold = 0.5
    videoNMSThreshold = 0.3

    video = cv2.VideoCapture('test_video.mp4')
    writer_pointer = None
    (img_width, img_height) = (None, None)

    try:
        video_prop = cv2.CAP_PROP_FRAME_COUNT
        total_frames = int(video.get(video_prop))
        print("[INFO] Total number of frames in the input video : {} ".format(total_frames))
    except:
        print("[ERROR] Cannot detect the frames from the input video")

    outputLayer = NNnet.getLayerNames()
    outputLayer = [outputLayer[i[0] - 1] for i in NNnet.getUnconnectedOutLayers()]

    counter = 0
    while True:
        (ret, frame) = video.read()
        if not ret:
            break
        if img_width is None or img_height is None:
            (img_height, img_width) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        NNnet.setInput(blob)
        layersOutputs = NNnet.forward(outputLayer)

        inspectionNMS, box_dimesnions, classIDs, confidence_values = detection(img_height, img_width, layersOutputs,
                                                                               videoconfidenceThreshold,
                                                                               videoNMSThreshold)

        if (len(inspectionNMS) > 0):
            for i in inspectionNMS.flatten():
                (x, y) = (box_dimesnions[i][0], box_dimesnions[i][1])
                (w, h) = (box_dimesnions[i][2], box_dimesnions[i][3])

                color_scheme = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_scheme, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidence_values[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_scheme, 2)

                if writer_pointer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer_pointer = cv2.VideoWriter('Detection_Ouput_Video.avi', fourcc, 30,
                                                     (frame.shape[1], frame.shape[0]),
                                                     True)
        if writer_pointer is not None:
            writer_pointer.write(frame)
            print("Writing frame", counter + 1)
            counter = counter + 1

    writer_pointer.release()
    video.release()


def realtime_detection(NNnet):
    realtimeconfidenceThreshold = 0.1
    realtimeNMSThreshold = 0.05

    video_capture = cv2.VideoCapture(0)

    (img_width, img_height) = (None, None)

    outputLayer = NNnet.getLayerNames()
    outputLayer = [outputLayer[i[0] - 1] for i in NNnet.getUnconnectedOutLayers()]

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if img_width is None or img_height is None:
            (img_height, img_width) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        NNnet.setInput(blob)
        layersOutputs = NNnet.forward(outputLayer)

        inspectionNMS, box_dimesnions, classIDs, confidence_values = detection(img_height, img_width, layersOutputs,
                                                                               realtimeconfidenceThreshold,
                                                                               realtimeNMSThreshold)

        if len(inspectionNMS) > 0:
            for i in inspectionNMS.flatten():
                (x, y) = (box_dimesnions[i][0], box_dimesnions[i][1])
                (w, h) = (box_dimesnions[i][2], box_dimesnions[i][3])

                color_scheme = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_scheme, 2)
                text = 'firearm'
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_scheme, 2)

        cv2.imshow('Output', frame)
        if (cv2.waitKey(1000) & 0xFF == ord('q')):
            break

    # Finally when video capture is over, release the video capture and destroyAllWindows
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    modelConfigurationValue = './cfg/yolov3.cfg'
    modelWeightValues = './yolov3_training_last.weights'
    labelsFile = './classes.names'

    labels = open(labelsFile).read().strip().split('\n')

    np.random.seed(10)

    NNnet = cv2.dnn.readNetFromDarknet(modelConfigurationValue, modelWeightValues)

    image_detection(NNnet)

    video_detection(NNnet, labels)

    realtime_detection(NNnet)


main()
