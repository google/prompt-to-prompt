import os
from typing import Optional, Union, Tuple, List, Dict
import torch.nn.functional as nnf
import abc
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input, Path, BaseModel

import ptp_utils
import seq_aligner


LOW_RESOURCE = False
MAX_NUM_WORDS = 77
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5


class ModelOutput(BaseModel):
    original_sd: Path
    with_prompt_to_prompt: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        model_id = "CompVis/stable-diffusion-v1-4"
        cache_dir = "diffusion-cache"
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to("cuda")
        self.tokenizer = self.ldm_stable.tokenizer
        self.controller = AttentionStore()

    @torch.inference_mode()
    def predict(
        self,
        original_prompt: str = Input(
            description="Input prompt used for the orinigal image",
            default="pink bear riding a bicycle",
        ),
        prompt_edit_type: str = Input(
            description="Choose the type of the prompt editing. See below for more information. If you are generating the original output, leave this empty.",
            choices=["Replacement", "Refinement", "Re-weight"],
            default=None,
        ),
        edited_prompt: str = Input(
            description="Prompted used for editing the original sd output image. If prompt_edit_type above is not set, then this field will be ignored. \
            See below for more information for how to edit the prompt from the original prompt. For Re-weight, just provided words in proginal_prompt with new weights.",
            default=None,
        ),
        local_edit: str = Input(
            description="Enable local editing. Provide the in the format of 'words_in_original_prompt | words_in_edited_prompt', and the rest content will be preserved.",
            default=None,
        ),
        cross_replace_steps: float = Input(
            description="Cross attention replace steps", ge=0, le=1, default=0.8
        ),
        self_replace_steps: float = Input(
            description="Self attention replace steps", ge=0, le=1, default=0.4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed for original output. But make sure to use the same seed for original-updated prompt pair.",
            default=8888,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        # sanity check
        if edited_prompt is not None:
            assert prompt_edit_type is not None, "Please select a prompt_edit_type."
        if prompt_edit_type is not None:
            assert edited_prompt is not None, "Please provide edited_prompt."

        local = None
        if edited_prompt is not None and local_edit is not None:
            assert "|" in local_edit, "Please provide valid local_edit information"
            local = [x.strip() for x in local_edit.split("|")]
            assert (
                len(local) == 2 and local[0] in original_prompt
            ), "Please provide valid local_edit information, make sure words exist in the prompts"

        words, weights = None, None
        if prompt_edit_type == "Re-weight":
            assert "|" in edited_prompt, "Please provide edited_prompt for Re-weight"
            words, weights = [x.strip() for x in edited_prompt.split("|")]
            words = [x.strip() for x in words.split(",")]
            assert all(
                [x in original_prompt for x in words]
            ), "All words for Re-weight should be in the original prompt"
            weights = [float(x.strip()) for x in weights.split(",")]
            assert len(words) > 0 and len(words) == len(
                weights
            ), "Please provide edited_prompt for Re-weight"

        if seed is None:
            print(
                f"Seed is not set, generating random seed. Note that you should assign same seed to the original-edited prompt pair for editing a generated image."
            )
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        g_cpu = torch.Generator().manual_seed(seed)
        prompts = [original_prompt]
        images, x_t = run_and_display(
            self.ldm_stable, prompts, self.controller, latent=None, generator=g_cpu
        )

        if prompt_edit_type is None:
            print(
                "Only original prompt provided, generation without prompt-to-prompt..."
            )
            output_path = "/tmp/out.png"
            pil_img = Image.fromarray(images[0])
            pil_img.save(output_path)
            return ModelOutput(original_sd=Path(output_path))

        # generating original-edited image pair
        prompts = (
            [original_prompt, original_prompt]
            if prompt_edit_type == "Re-weight"
            else [original_prompt, edited_prompt]
        )
        lb = LocalBlend(prompts, (local[0], local[1]), self.tokenizer) if local is not None else None
        if prompt_edit_type == "Replacement":
            controller = AttentionReplace(
                prompts,
                self.tokenizer,
                NUM_DIFFUSION_STEPS,
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
                local_blend=lb,
            )

        elif prompt_edit_type == "Refinement":
            controller = AttentionRefine(
                prompts,
                self.tokenizer,
                NUM_DIFFUSION_STEPS,
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
                local_blend=lb,
            )

        else:
            equalizer = get_equalizer(original_prompt, words, weights, self.tokenizer)
            controller = AttentionReweight(
                prompts,
                self.tokenizer,
                NUM_DIFFUSION_STEPS,
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
                equalizer=equalizer,
                local_blend=lb,
            )

        images, _ = run_and_display(self.ldm_stable, prompts, controller, latent=x_t)
        ori_output_path, p2p_output_path = "/tmp/out_ori.png", "/tmp/out_p2p.png"
        pil_img0, pil_img1 = Image.fromarray(images[0]), Image.fromarray(images[1])
        pil_img0.save(ori_output_path)
        pil_img1.save(p2p_output_path)
        return ModelOutput(
            original_sd=Path(ori_output_path),
            with_prompt_to_prompt=Path(p2p_output_path),
        )


class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [
            item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS)
            for item in maps
        ]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(
        self,
        prompts: List[str],
        words: [List[List[str]]],
        tokenizer,
        threshold=0.3,
        device="cuda:0",
    ):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        tokenizer,
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Tuple[float, float]]
        ],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        device="cuda",
    ):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(
            num_steps * self_replace_steps[1]
        )
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        tokenizer,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        device="cuda",
    ):
        super(AttentionReplace, self).__init__(
            prompts,
            tokenizer,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            device,
        )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        tokenizer,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        device="cuda",
    ):
        super(AttentionRefine, self).__init__(
            prompts,
            tokenizer,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace
            )
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
        self,
        prompts,
        tokenizer,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
        device="cuda:0",
    ):
        super(AttentionReweight, self).__init__(
            prompts,
            tokenizer,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select, values, tokenizer):
    # if type(word_select) is int or type(word_select) is str:
    #     word_select = (word_select,)
    # equalizer = torch.ones(len(values), 77)
    # values = torch.tensor(values, dtype=torch.float32)
    # for word in word_select:
    #     inds = ptp_utils.get_word_inds(text, word, tokenizer)
    #     equalizer[:, inds] = values

    equalizer = torch.ones(1, MAX_NUM_WORDS)
    for word, value in zip(word_select, values):
        values = torch.tensor([value], dtype=torch.float32)
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        print(inds)
        equalizer[:, inds] = values
    return equalizer


def run_and_display(
    ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None
):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(
            prompts,
            EmptyControl(),
            latent=latent,
            run_baseline=False,
            generator=generator,
        )
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(
        ldm_stable,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        low_resource=LOW_RESOURCE,
    )
    # ptp_utils.view_images(images)
    return images, x_t
