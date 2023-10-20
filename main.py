# This is a sample Python script.
from collections import defaultdict

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torchreid
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

# def preprocess_image(image_path, target_size=(128, 64), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     """
#     Preprocess an image for a Torchreid model, such as MobileNet trained on DukeMTMC-reID.
#
#     Args:
#         image_path (str): Path to the image file.
#         target_size (tuple): Target size for resizing the image (height, width).
#         mean (tuple): RGB mean values for normalization.
#         std (tuple): RGB standard deviation values for normalization.
#
#     Returns:
#         torch.Tensor: Preprocessed image tensor ready to be fed into the model.
#     """
#     # Read the image using OpenCV
#     image = cv2.imread(image_path)
#
#     # Resize the image to the target size
#     image = cv2.resize(image, target_size)
#
#     # Convert the image from BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Normalize the image
#     image = (image / 255.0 - mean) / std
#
#     # Convert the image to a PyTorch tensor
#     image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
#
#     # Add a batch dimension (assuming single-image input)
#     image = image.unsqueeze(0)
#
#     return image


track_history = defaultdict(lambda: [])
persons = {}
person_id_counter = [100]


def preprocess_frame3(frame, need=False):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preprocessed_frame = preprocess(frame)
    return preprocessed_frame.unsqueeze(0)

def preprocess_frame2(frame, need=False):
    if need:
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            # transforms.CenterCrop((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((256, 128)),
            # transforms.CenterCrop((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # make the same but without toPilImage

    preprocessed_frame = preprocess(frame)
    return preprocessed_frame.unsqueeze(0)




def preprocess_frame(frame, target_size=(256, 128), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Resize the frame to the target size
    frame = cv2.resize(frame, target_size)

    # Convert the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame
    frame = (frame / 255.0 - np.array(mean)) / np.array(std)

    # Convert the frame to a PyTorch tensor
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

    # Add a batch dimension (assuming single-frame input)
    frame = frame.unsqueeze(0)

    return frame


def setReidModel():
    model = torchreid.models.build_model(
        name='mobilenetv2_x1_4',  # Replace with your model architecture
        num_classes=702,  # Replace with the number of classes in your dataset
    )
    device = torch.device('cpu')
    model.to(device)
    checkpoint = torch.load(r'C:\Users\Ektomo\PycharmProjects\pythonProject2\model\model.pth.tar-32',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_ideal_embeddings(paths, model):
    for id, path in paths:
        img = Image.open(path)
        data = np.array(img)
        pre_frame = preprocess_frame3(data)
        with torch.no_grad():
            cur_embedding = model(pre_frame)
            if id in persons:
                persons[id].append(cur_embedding)
            else:
                persons[id] = [cur_embedding]


def track_reid(model):
    yolo_model = YOLO('yolov8x.pt')
    yolo_classes = ["person"]

    # Load the checkpoint

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize and normalize)
        results = yolo_model.track(frame, persist=True, classes=[0])
        for result in results:

            boxes = result.boxes.xywh.cpu()
            id_box = result.boxes.id
            # annotated_frame = results[0].plot()
            if id_box is not None:
                print(type(id_box))
                cur_id_box = -1
                x, y, w, h = boxes[0]
                cropped_image = frame[int(x - w / 2):int(x + w / 2), int(y - h / 2):int(y + h / 2)]

                pre_frame = preprocess_frame3(cropped_image, True)

                with torch.no_grad():
                    cur_embedding = model(pre_frame)
                # cur_embedding = cur_embedding.detach().numpy()
                found_match = False
                for person_id, data in persons.items():
                    # ideal_embeddings = data['embeddings']
                    print(len(data), person_id)
                    for embed in data:
                        similarity = np.dot(cur_embedding, embed.T) / (
                                np.linalg.norm(cur_embedding) * np.linalg.norm(embed))
                        print("Похоже на", similarity)
                        if similarity > 0.72:
                            found_match = True
                            cur_id_box = person_id
                            # data['embedding'] = cur_embedding
                            break
                    if found_match:
                        break
                # if not found_match:
                #     person_id_counter[0] += 1
                #     person_id = person_id_counter[0]
                #     cur_id_box = person_id
                #     persons[person_id] = [cur_embedding]

                if cur_id_box:
                    # track_ids = cur_id_box

                    # for box, track_id in zip(boxes, cur_id_box):
                    # fixme Only plot the track for the person with ID 5 for this example
                    # if track_id != 5:
                    #     continue

                    track = track_history[cur_id_box]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)
                    # draw bbox
                    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 0, 255), 2)
                    # draw ID
                    cv2.putText(frame, str(cur_id_box), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame with Bounding Box', frame)

        # Exit the loop if a key is pressed (e.g., press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_path = r'C:\Users\Ektomo\PycharmProjects\pythonProject2\reid'
    model_reid = setReidModel()
    ideal_paths = []
    for i in range(1, 7):
        for j in range(1, 5):
            ideal_paths.append((i, base_path + f"\\{i}_{j}.jpg"))
    get_ideal_embeddings(ideal_paths,
                         model_reid)
    track_reid(model_reid)
