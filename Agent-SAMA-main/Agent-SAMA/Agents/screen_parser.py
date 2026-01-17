from PIL import Image
from AgentsUtils.tool_utils import get_screenshot
from AgentsUtils.ocr_utils import run_ocr, det, get_all_files_in_folder, draw_coordinates_on_image, crop, \
    merge_text_blocks, generate_local, generate_api
from constant import TEMP_DIR, DEFAULT_PERCEPTION_ARGS, CAPTION_CALL_METHOD, CAPTION_MODEL
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os


def load_perception_models(
        device="cuda",
        caption_call_method=CAPTION_CALL_METHOD,
        caption_model=CAPTION_MODEL,
        groundingdino_model="AI-ModelScope/GroundingDINO",
        groundingdino_revision="v1.0.0",
        ocr_detection_model="iic/cv_resnet18_ocr-detection-db-line-level_damo",
        ocr_recognition_model="iic/cv_convnextTiny_ocr-recognition-document_damo",
):
    ### Load caption model ###
    if caption_call_method == "local":
        if caption_model == "qwen-vl-chat":
            model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
            vlm_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device,
                                                             trust_remote_code=True).eval()
            vlm_model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif caption_model == "qwen-vl-chat-int4":
            qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
            vlm_model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True,
                                                             use_safetensors=True).eval()
            vlm_model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True,
                                                                           do_sample=False)
        else:
            print(
                "If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
            exit(0)
        vlm_tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
    elif caption_call_method == "api":
        vlm_model = None
        vlm_tokenizer = None
        pass
    else:
        print("You must choose the caption model call function from \"local\" and \"api\"")
        exit(0)

    ### Load ocr and icon detection model ###
    groundingdino_dir = snapshot_download(groundingdino_model, revision=groundingdino_revision)
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
    ocr_detection = pipeline(Tasks.ocr_detection, model=ocr_detection_model)  # dbnet (no tensorflow)
    ocr_recognition = pipeline(Tasks.ocr_recognition, model=ocr_recognition_model)

    print("INFO: Loaded perception models:")
    print("\t- Caption model method:", caption_call_method, "| caption vlm model:", caption_model)
    print("\t- Grounding DINO model:", groundingdino_model)
    print("\t- OCR detection model:", ocr_detection_model)
    print("\t- OCR recognition model:", ocr_recognition_model)
    return ocr_detection, ocr_recognition, groundingdino_model, vlm_model, vlm_tokenizer


class ScreenParser:
    def __init__(self, adb_path, perception_args=DEFAULT_PERCEPTION_ARGS):
        self.ocr_detection, self.ocr_recognition, self.groundingdino_model, \
            self.vlm_model, self.vlm_tokenizer = load_perception_models(**perception_args)
        self.adb_path = adb_path

    def get_perception_infos(self, screenshot_file, temp_file=TEMP_DIR):
        get_screenshot(self.adb_path)

        width, height = Image.open(screenshot_file).size

        text, coordinates = run_ocr(screenshot_file, self.ocr_detection, self.ocr_recognition)
        text, coordinates = merge_text_blocks(text, coordinates)

        # Ensure coordinates are integers
        coordinates = [
            [int(c[0]), int(c[1]), int(c[2]), int(c[3])]
            for c in coordinates
        ]

        center_list = [[(c[0] + c[2]) / 2, (c[1] + c[3]) / 2] for c in coordinates]
        draw_coordinates_on_image(screenshot_file, center_list)

        perception_infos = []
        for i in range(len(coordinates)):
            perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
            perception_infos.append(perception_info)

        coordinates = det(screenshot_file, "icon", self.groundingdino_model)

        # Ensure icon coordinates are integers
        coordinates = [
            [int(c[0]), int(c[1]), int(c[2]), int(c[3])]
            for c in coordinates
        ]

        for i in range(len(coordinates)):
            perception_info = {"text": "icon", "coordinates": coordinates[i]}
            perception_infos.append(perception_info)

        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            if perception_infos[i]['text'] == 'icon':
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)

        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i], temp_file=temp_file)

        images = get_all_files_in_folder(temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
            if CAPTION_CALL_METHOD == "local":
                for i in range(len(images)):
                    image_path = os.path.join(temp_file, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = generate_local(self.vlm_tokenizer, self.vlm_model, image_path, prompt)
                    icon_map[i + 1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(temp_file, images[i])
                icon_map = generate_api(images, prompt, caption_model=CAPTION_MODEL)
            for i, j in zip(image_id, range(1, len(image_id) + 1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] = "icon: " + icon_map[j]

        for i in range(len(perception_infos)):
            x1, y1, x2, y2 = perception_infos[i]['coordinates']
            perception_infos[i]['coordinates'] = [
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            ]

        return perception_infos, width, height
