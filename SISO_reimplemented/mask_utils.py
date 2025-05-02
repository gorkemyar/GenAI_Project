import torchvision.transforms as T
import torch
from general_utils import image_to_tensor

def detect_best_bounding_box(object_detector, image, class_label):
    if image.ndim == 4:
        image = image.squeeze(0)
    image_pil = T.ToPILImage()(image.cpu())

    detections = object_detector(image_pil, candidate_labels=[class_label])

    if len(detections) == 0:
        return [[[0, 0, image_pil.width, image_pil.height]]]  # fallback full image

    best_det = max(detections, key=lambda x: x["score"])  # select best
    box = best_det["box"]

    return [[[box["xmin"], box["ymin"], box["xmax"], box["ymax"]]]]

def get_mask(object_detector, sam_processor, sam_model, image, class_label):
    image_tensor = image_to_tensor(image).to(sam_model.device)
    bounding_box = detect_best_bounding_box(object_detector, image_tensor, class_label)
    
    inputs = sam_processor(image, input_boxes=bounding_box, return_tensors="pt").to(sam_model.device)
  
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    return masks[0]