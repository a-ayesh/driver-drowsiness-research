import torch
import torchvision.transforms as transforms

class SegmentImage(object):
    def __init__(self, mtcnn, image_size=(299, 299)):
        super(SegmentImage, self).__init__()
        self.mtcnn = mtcnn
        self.image_size = image_size
        self.resize_transform = transforms.Resize(image_size)

    def __call__(self, image):
        # Convert image to PIL
        
        image_pil = transforms.ToPILImage()(image)
        

        # Detect face and landmarks
        boxes, _, landmarks = self.mtcnn.detect(image_pil, landmarks=True)

        if landmarks is not None:
            # Calculate midpoint between left and right eyes
            left_eye_x, left_eye_y = landmarks[0][0]
            right_eye_x, right_eye_y = landmarks[0][1]
            midpoint_x = (left_eye_x + right_eye_x) // 2
            midpoint_y = (left_eye_y + right_eye_y) // 2

            # Crop region corresponding to both eyes
            eye_crop_width = 299  # Adjust according to your preference
            eye_crop_height = 299  # Adjust according to your preference
            eyes_image = image_pil.crop((midpoint_x - eye_crop_width//2, midpoint_y - eye_crop_height//2, midpoint_x + eye_crop_width//2, midpoint_y + eye_crop_height//2))
            eyes_image = eyes_image.resize(self.image_size)  # Resize to specified image size
            
            # Convert PIL image to tensor
            eyes_image = transforms.ToTensor()(eyes_image)

            # Crop region corresponding to the mouth
            mouth_left_x, mouth_left_y = landmarks[0][3]
            mouth_right_x, mouth_right_y = landmarks[0][4]
            mouth_crop_width = 299  # Adjust according to your preference
            mouth_crop_height = 299  # Adjust according to your preference
            
            mouth_image = image_pil.crop((mouth_left_x - mouth_crop_width//2, mouth_left_y - mouth_crop_height//2, mouth_right_x + mouth_crop_width//2, mouth_right_y + mouth_crop_height//2))
            mouth_image = mouth_image.resize(self.image_size)  # Resize to specified image size
            
            # Convert PIL image to tensor
            mouth_image = transforms.ToTensor()(mouth_image)

            return eyes_image, mouth_image

        else:
            image = self.resize_transform(image_pil)
            image = transforms.ToTensor()(image)
            return image , image
