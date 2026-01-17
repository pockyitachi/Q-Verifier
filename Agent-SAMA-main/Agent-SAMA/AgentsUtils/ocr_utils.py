import math
import os
import cv2
import torch
import concurrent.futures
import dashscope
import numpy as np
from PIL import Image, ImageDraw
import os
from constant import TEMP_DIR, QWEN_API_KEY, CAPTION_MODEL

# Use environment variable if available, otherwise fall back to constant
QWEN_API_KEY_RESOLVED = os.environ.get("QWEN_API_KEY", QWEN_API_KEY)
from dashscope import MultiModalConversation


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def crop_image(img, position):
    """
    Applies a perspective crop based on 4-point quadrilateral.
    """
    position = position.tolist()

    # Sort by x coordinate
    position.sort(key=lambda p: p[0])

    # Ensure top-left and top-right are correctly ordered
    if position[0][1] > position[1][1]:
        position[0], position[1] = position[1], position[0]
    if position[2][1] > position[3][1]:
        position[2], position[3] = position[3], position[2]

    x1, y1 = position[0]
    x2, y2 = position[2]
    x3, y3 = position[3]
    x4, y4 = position[1]

    corners = np.float32([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])

    img_width = euclidean_distance((x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2, (y2 + y3) / 2)
    img_height = euclidean_distance((x1 + x2) / 2, (y1 + y2) / 2, (x4 + x3) / 2, (y4 + y3) / 2)

    corners_trans = np.float32([
        [0, 0],
        [img_width - 1, 0],
        [0, img_height - 1],
        [img_width - 1, img_height - 1]
    ])

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst


def calculate_size(box):
    """
    Returns the area of the bounding box.
    """
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two boxes.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = calculate_size(box1)
    box2_area = calculate_size(box2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def in_box(box, target):
    """
    Returns True if 'box' is entirely inside 'target'.
    """
    return all([
        box[0] > target[0],
        box[1] > target[1],
        box[2] < target[2],
        box[3] < target[3]
    ])


def remove_boxes(boxes_filt, size, iou_threshold=0.5):
    """
    Removes boxes that are either too large or have high overlap (IoU) with other boxes.

    Args:
        boxes_filt (list): List of bounding boxes [x1, y1, x2, y2].
        size (tuple): Image size as (width, height).
        iou_threshold (float): Overlap threshold above which boxes are considered redundant.

    Returns:
        list: Filtered list of boxes.
    """
    boxes_to_remove = set()
    max_area = 0.05 * size[0] * size[1]
    n = len(boxes_filt)

    # Precompute sizes to avoid redundant calculation
    box_sizes = [calculate_size(box) for box in boxes_filt]

    for i in range(n):
        if box_sizes[i] > max_area:
            boxes_to_remove.add(i)
            continue  # Skip overlap check for large boxes

        for j in range(n):
            if i == j:
                continue  # Do not compare a box with itself
            if j in boxes_to_remove or i in boxes_to_remove:
                continue  # Skip boxes already marked for removal
            if box_sizes[j] > max_area:
                boxes_to_remove.add(j)
                continue  # Skip IoU check if box j is too large

            iou = calculate_iou(boxes_filt[i], boxes_filt[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    return [box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove]


def det(input_image_path, caption, groundingdino_model, box_threshold=0.05, text_threshold=0.5):
    """
    Detects objects in an image using GroundingDINO with a given text caption.

    Args:
        input_image_path (str): Path to the image file.
        caption (str): Natural language prompt.
        groundingdino_model (callable): GroundingDINO model callable.
        box_threshold (float): Minimum confidence score for boxes.
        text_threshold (float): Minimum confidence for associating text.

    Returns:
        List[List[int]]: List of bounding boxes in [x1, y1, x2, y2] format.
    """
    # Load image and get size
    image = Image.open(input_image_path)
    width, height = image.size

    # Format the caption
    caption = caption.lower().strip()
    if not caption.endswith('.'):
        caption += '.'

    # Prepare correct input dict for GroundingDINO v1.0.0
    inputs = {
        'IMAGE_PATH': input_image_path,
        'TEXT_PROMPT': caption,
        'BOX_TRESHOLD': box_threshold,
        'TEXT_TRESHOLD': text_threshold
    }

    # Run the model
    result = groundingdino_model(inputs)
    boxes_filt = result['boxes']  # normalized [cx, cy, w, h] format

    # Convert to absolute [x1, y1, x2, y2] format
    scale = torch.tensor([width, height, width, height])
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu().int().tolist()

    # Filter out overlapping and oversized boxes
    filtered_boxes = remove_boxes(boxes_filt, (width, height))

    coordinates = []
    for box in filtered_boxes:
        coordinates.append([box[0], box[1], box[2], box[3]])

    return coordinates


def sort_box_clockwise(coordinates):
    arr = np.array(coordinates).reshape((4, 2))
    sum_ = np.sum(arr, axis=0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sorted_points = arr[np.argsort(theta)]
    sorted_points = sorted_points.reshape((4, -1))
    if sorted_points[0][0] > centroid[0]:
        sorted_points = np.concatenate([sorted_points[3:], sorted_points[:3]])
    sorted_points = sorted_points.reshape((4, 2)).astype(np.float32)
    return sorted_points


def run_ocr(image_path, ocr_detection, ocr_recognition):
    text_data = []
    coordinates = []

    image = cv2.imread(image_path)
    detection_result = ocr_detection(image)
    polygons = detection_result['polygons']

    for i in range(polygons.shape[0]):
        points = sort_box_clockwise(polygons[i])
        cropped = crop_image(image, points)

        try:
            text = ocr_recognition(cropped)['text'][0]
        except:
            continue

        flat = [int(e) for e in list(points.reshape(-1))]
        box = [flat[0], flat[1], flat[4], flat[5]]

        text_data.append(text)
        coordinates.append(box)

    return text_data, coordinates


def get_all_files_in_folder(folder_path):
    return [file_name for file_name in os.listdir(folder_path)]


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    radius = 10

    for x, y in coordinates:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')

    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image_path, box, index, temp_file=TEMP_DIR):
    image = Image.open(image_path)
    x1, y1, x2, y2 = map(int, box)

    if x1 >= x2 - 10 or y1 >= y2 - 10:
        return

    cropped = image.crop((x1, y1, x2, y2))
    save_path = os.path.join(temp_file, f"{index}.jpg")
    cropped.save(save_path)


def merge_text_blocks(
        text_list,
        coordinates_list,
        x_distance_threshold=45,
        y_distance_min=-20,
        y_distance_max=30,
        height_difference_threshold=20,
):
    merged_text_blocks = []
    merged_coordinates = []

    # Sort based on top-left (y, x) position
    sorted_indices = sorted(
        range(len(coordinates_list)),
        key=lambda i: (coordinates_list[i][1], coordinates_list[i][0]),
    )
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merged_flags = [False] * num_blocks

    for i in range(num_blocks):
        if merged_flags[i]:
            continue

        anchor = i
        group_texts = [sorted_text_list[anchor]]
        group_coords = [sorted_coordinates_list[anchor]]

        for j in range(i + 1, num_blocks):
            if merged_flags[j]:
                continue

            x1_anchor, y1_anchor, x2_anchor, y2_anchor = sorted_coordinates_list[anchor]
            x1_j, y1_j, x2_j, y2_j = sorted_coordinates_list[j]

            x_diff_avg = (abs(x1_anchor - x1_j) + abs(x2_anchor - x2_j)) / 2
            y_diff = y1_j - y2_anchor

            h_anchor = y2_anchor - y1_anchor
            h_j = y2_j - y1_j
            height_diff = abs(h_anchor - h_j)

            if (
                    x_diff_avg < x_distance_threshold
                    and y_distance_min <= y_diff < y_distance_max
                    and height_diff < height_difference_threshold
            ):
                group_texts.append(sorted_text_list[j])
                group_coords.append(sorted_coordinates_list[j])
                merged_flags[anchor] = True
                anchor = j
                merged_flags[anchor] = True

        merged_text = "\n".join(group_texts)
        x1 = min(coord[0] for coord in group_coords)
        y1 = min(coord[1] for coord in group_coords)
        x2 = max(coord[2] for coord in group_coords)
        y2 = max(coord[3] for coord in group_coords)

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([x1, y1, x2, y2])

    return merged_text_blocks, merged_coordinates


def generate_local(tokenizer, model, image_file, query):
    query_tokens = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query_tokens, history=None)
    return response


def process_image(image_path, query, caption_model=CAPTION_MODEL):
    dashscope.api_key = QWEN_API_KEY_RESOLVED
    image_uri = "file://" + image_path
    messages = [{
        'role': 'user',
        'content': [
            {'image': image_uri},
            {'text': query},
        ]
    }]

    response = MultiModalConversation.call(model=caption_model, messages=messages)

    try:
        return response['output']['choices'][0]['message']['content'][0]["text"]
    except (KeyError, IndexError, TypeError):
        return "This is an icon."


def generate_api(images, query, caption_model=CAPTION_MODEL):
    icon_map = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, image, query, caption_model=caption_model): index
            for index, image in enumerate(images)
        }

        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            icon_map[index + 1] = future.result()

    return icon_map
