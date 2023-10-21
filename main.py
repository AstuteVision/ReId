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
waiting_class = 0
true_positive = [0]
false_positive = 0
true_negative = [0]
false_negative = [0]


# третий вариант, как будто самый быстрый, сюда в случае снятия вектора передаем ndarray и frame
def preprocess_frame3(frame, need=False):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128), antialias=None),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preprocessed_frame = preprocess(frame)
    return preprocessed_frame.unsqueeze(0)


# второй вариант, как будто работает дольше, в этом случае при снятии векторов передаем сразу img
def preprocess_frame2(frame, need=False):
    if need:
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128), antialias=None),
            # transforms.CenterCrop((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((256, 128), antialias=None),
            # transforms.CenterCrop((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # make the same but without toPilImage

    preprocessed_frame = preprocess(frame)
    return preprocessed_frame.unsqueeze(0)


# первый вариант, не используем
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
    checkpoint = torch.load(r'C:\Users\Ektomo\PycharmProjects\pythonProject2\model\model.pth.tar-50',
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


def track_reid(reid_model):
    yolo_model = YOLO('yolov8x.pt')

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
                for idx in range(len(boxes)):
                    print(type(id_box))
                    cur_id_box = result.boxes[idx].id
                    x, y, w, h = boxes[idx]
                    temp_w = int(x + w / 2)
                    temp_h = int(y + h / 2)

                    cropped_image = frame[int(y - h / 2):temp_h, int(x - w / 2):temp_w]
                    # cropped_image = frame[int(y): int(y+h), int(x): int(x+h)]
                    pre_frame = preprocess_frame3(cropped_image, True)

                    track = track_history[cur_id_box]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    with torch.no_grad():
                        cur_embedding = reid_model(pre_frame)
                    # cur_embedding = cur_embedding.detach().numpy()
                    found_match = False
                    matrix_percent = {}
                    for person_id, data in persons.items():
                        # ideal_embeddings = data['embeddings']
                        print(person_id)
                        matrix_percent[person_id] = []
                        cur_similarity = 0
                        find_person = -1
                        for embed in data:
                            similarity = np.dot(cur_embedding, embed.T) / (
                                    np.linalg.norm(cur_embedding) * np.linalg.norm(embed))
                            matrix_percent[person_id].append(similarity)

                            print("Похоже на", similarity)
                            # if similarity > 0.75:
                            #     found_match = True
                            #     cur_id_box = person_id
                            #     break
                        mean_similarity = np.mean(matrix_percent[person_id])
                        print('mean_similaruty for', person_id, mean_similarity)

                        if mean_similarity > cur_similarity:
                            cur_similarity = mean_similarity
                            find_person = person_id

                        if cur_similarity > 0.70:
                            found_match = True
                            cur_id_box = find_person
                            # break
                        # if found_match:
                        #     break

                    if not found_match:
                        cur_id_box = -1

                    if waiting_class == cur_id_box:
                        true_positive[0] += 1
                    else:
                        false_negative[0] += 1

                    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 0, 255), 2)
                    # draw ID
                    cv2.putText(frame, str(cur_id_box), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)

        cv2.imshow('Frame with Bounding Box', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Печатаем результат на выходе
            print("Метрика", f'recall: {true_positive[0] / true_positive[0] + false_negative[0]}')
            cap.release()
            cv2.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Путь к файлам с идеальными фото
    base_path = r'C:\Users\Ektomo\PycharmProjects\pythonProject2\reid'
    # Устанавливаем модель, по хорошему для тестов сюда можно положить имя в параметры функции
    model_reid = setReidModel()
    ideal_paths = []
    # Указываем какой класс ждем
    waiting_class = 4
    # i - количество классов, j - количество изображений в классе(по умолчанию делаем 4)
    for i in range(1, 7):
        for j in range(1, 5):
            ideal_paths.append((i, base_path + f"\\{i}_{j}.jpg"))
    # Создаем массив идеальных векторных представлений
    get_ideal_embeddings(ideal_paths,
                         model_reid)
    # Трекаем с созданной моделькой
    track_reid(model_reid)
