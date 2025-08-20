# Used in InternVL series 
# https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b/blob/main/build_mlp.py#L76
# Also used in Phi-3.5
# https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L208

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
)
from transformers.utils import TensorType, is_vision_available, logging

from transformers import AutoImageProcessor

logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image

import torch
import torchvision


# https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b/blob/main/build_mlp.py#L76
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.select_feature = 'patch'
        self.load_model()

    def load_model(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def resize_pos(self):
        print ('Dummy Resized')

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, glb_GN, sub_GN):
        """
        Grouped Normalization -- 
        glb_GN = nn.Parameter(torch.zeros([1, 1, 4096]))
        sub_GN = nn.Parameter(torch.zeros([1, 1, 1, 4096]))        
        """
        
        
        if not self.is_loaded:
            self.load_model()
        assert type(images) is list
        shapes = []
        input_imgs = []
        """
        To handle detailed local information and global context simultaneously
        1. Processing a Batch of Images
          Loops over images, img.shape is assumed to be (Batch, Channels, H, W)
          Calculates downsampled shapes [H//336, W//336], split into tiles of 336×336 pixels.
        2. Splitting and Reshaping Image Tensors (Image splitting)
          Sub-image Extraction: reshaped into patches (tiles) of 336×336, splits for localized feature extraction
          Global Image Downsampling: whole image is downsampled using bicubic interpolation to 336×336
        3. Feature Extraction Stage
          Both downsampled global image and tiled sub-images concatenated and image_features calculated
          Spatial grid of 24×24 patches. (N=24)
        4. For Each Original Image: Extracts global features, processes sub-image features 
          Reshapes into grids corresponding to original spatial structure & integration with normalization or positional embeddings
          Same with Patch Feature Arrangement; maintaining spatial and semantic structure.
        5. Sequential Feature Handling
        Slice off processed portions: image_features = image_features[1+h*w:]
        """      
        for img in images:
            _, C, H, W = img.shape
            shapes.append([H//336, W//336])
            sub_img = img.reshape(1,3,H//336,336,W//336,336).permute(0,2,4,1,3,5).reshape(-1,3,336,336).contiguous()
            glb_img = torch.nn.functional.interpolate(img.float(), size=(336,336), mode='bicubic',).to(sub_img.dtype)
            input_imgs.append(glb_img)
            input_imgs.append(sub_img)
        input_imgs = torch.cat(input_imgs, dim=0)

        image_forward_outs = self.vision_tower(input_imgs.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(input_imgs.dtype) ### B*?, N, C
        _, N, C = image_features.shape
        H = int(math.sqrt(N))
        assert N == 24 ** 2

        output_imgs = []
        output_len = []

        """
        The following is where pixel shuffle / channel wise packing happens:
        1. Looping Over Spatial Shapes
          Iterates over a list of shapes, where each shape is [height, width].
          B_ is the area (number of patches) for the current spatial shape.
        2. Global Image Feature Processing
          Selects the first image feature. Rearrange spatial patches or prepare for further feature aggregation.
          The final shape increases the channel dimension for concatenating or "unfolding" patches.
        3. Incorporating Global GroupNorm Features
          Repeats a GroupNorm (GN) tensor to match the spatial dimensions. 
          Concatenates GN features with reshaped global image features along the channel axis.
        4. Subregion Image Feature Processing
          Processes all subregion image features. Expands spatial resolution (h*12, w*12)
        5. Concatenates the processed glb_img, a GN tensor (glb_GN), and sub_img for the current shape
        """
        for [h, w] in shapes:
            B_ = h*w
            glb_img = image_features[:1] ### 1, N, C
            glb_img = glb_img.reshape(1,H,H,C).reshape(1,H//2,2,H//2,2,C).contiguous().permute(0,1,3,2,4,5).reshape(1,H//2,H//2,4*C).contiguous()
            temp_glb_GN = sub_GN.repeat(1, H//2, 1, 1)
            glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1,-1,4*C)
            
            sub_img = image_features[1:1+B_] ### ?, N, C
            sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//2,2,H//2,2,C).contiguous().permute(0,1,3,2,4,5).reshape(B_,-1,4*C).contiguous()
            sub_img = sub_img.reshape(1, h, w, 12, 12, -1).permute(0,1,3,2,4,5).reshape(1,h*12,w*12,4*C)
            temp_sub_GN = sub_GN.repeat(1, h*12, 1, 1)
            sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1,-1,4*C)

            output_imgs.append(torch.cat([glb_img, glb_GN, sub_img], dim=1))
            temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
            assert temp_len == output_imgs[-1].shape[1]
            output_len.append(temp_len)

            image_features = image_features[1+h*w:]

        output_imgs = torch.cat(output_imgs, dim=1)

        return output_imgs, output_len

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


