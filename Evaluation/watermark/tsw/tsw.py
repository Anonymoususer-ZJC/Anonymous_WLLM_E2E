from functools import partial

from torch.nn import functional as F
from torch import Tensor
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer, OPTForCausalLM

from utils.transformers_config import TransformersConfig
from watermark.base import BaseWatermark
from watermark.tsw.model import *


class TSWConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        self.prefix_length = 1
        self.z_threshold = 4.0
        self.ckpt = '/home/zhoujicheng/data/WLLM/TSW//init_0.25_1.75_default.pth'  # TODO:
        self.opt_embed_matrix = OPTForCausalLM.from_pretrained("facebook/opt-1.3b",
                                                               torch_dtype=torch.float16).cuda().get_input_embeddings().weight
        self.opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")
        # llm cfg
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.gen_kwargs = transformers_config.gen_kwargs
        self.device = transformers_config.device


class TSWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config):
        self.config = config

        ####
        self.seeding_scheme = "simple_1"
        self.hash_key = 15485863
        self.vocab_size = 50272

        checkpoint = torch.load(self.config.ckpt)
        layer_delta = sum(
            1 for key in checkpoint['delta_state_dict'] if "weight" in key)  # Counting only weight keys as layers
        layer_gamma = sum(
            1 for key in checkpoint['gamma_state_dict'] if "weight" in key)  # Counting only weight keys as layers

        embed_matrix = self.config.opt_embed_matrix
        self.gamma_network = GammaNetwork(input_dim=embed_matrix.shape[1], layers=layer_gamma).cuda()
        self.delta_network = DeltaNetwork(input_dim=embed_matrix.shape[1], layers=layer_delta).cuda()

        self.delta_network.load_state_dict(checkpoint['delta_state_dict'])
        self.gamma_network.load_state_dict(checkpoint['gamma_state_dict'])

        for name, param in self.delta_network.named_parameters():
            param.requires_grad = False
        for name, param in self.gamma_network.named_parameters():
            param.requires_grad = False
        self.delta_network.eval()
        self.gamma_network.eval()

        self.gamma_list = torch.empty(0, dtype=torch.float).cuda()
        self.delta_list = torch.empty(0, dtype=torch.float).cuda()
        self.embed_matrix = embed_matrix.float()
        ####

        self.tgt_tokenizer = self.config.generation_tokenizer
        self.opt_tokenizer = self.config.opt_tokenizer
        if self.tgt_tokenizer != self.opt_tokenizer:
            self.vocab_size = len(self.tgt_tokenizer)

    def reset_gamma_delta_list(self):
        self.gamma_list = torch.empty(0, dtype=torch.float).cuda()
        self.delta_list = torch.empty(0, dtype=torch.float).cuda()

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[
                       -1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(int(self.hash_key * prev_token))
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # Always use ids given by OPT model
        # since our gamma/delta network is trained on the embedding matrix of OPT model

        # seed the rng using the previous tokens/prefix according to the seeding_scheme

        self._seed_rng(input_ids)

        gamma = self.gamma_network(self.embed_matrix[input_ids[-1].item()])
        delta = self.delta_network(self.embed_matrix[input_ids[-1].item()])

        self.gamma_list = torch.cat([self.gamma_list, gamma])
        self.delta_list = torch.cat([self.delta_list, delta])

        greenlist_size = int(self.vocab_size * gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device='cuda', generator=self.rng)

        greenlist_ids = vocab_permutation[:greenlist_size]

        return greenlist_ids, gamma, delta

    def _calc_greenlist_mask(self, logits: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        self.rng = torch.Generator(device='cuda')

        if self.tgt_tokenizer != self.opt_tokenizer:
            llama_str = self.tgt_tokenizer.batch_decode(input_ids[:, -5:], add_special_tokens=False)
            ids_opt = self.opt_tokenizer(llama_str, add_special_tokens=False)['input_ids']

        for b_idx in range(input_ids.shape[0]):
            if self.tgt_tokenizer != self.opt_tokenizer:
                greenlist_ids, gamma, delta = self._get_greenlist_ids(torch.tensor(ids_opt[b_idx]).cuda())
            else:
                greenlist_ids, gamma, delta = self._get_greenlist_ids(input_ids[b_idx])

            green_tokens_mask = self._calc_greenlist_mask(logits=scores[b_idx], greenlist_token_ids=greenlist_ids)
            scores[b_idx][green_tokens_mask] = scores[b_idx][green_tokens_mask] + delta.half()

        return scores


class TSW(BaseWatermark):
    def __init__(self, transformers_config):
        self.config = TSWConfig(transformers_config)
        self.prefix_len = self.config.prefix_length
        self.logits_processor = TSWLogitsProcessor(self.config)

        self.tgt_tokenizer = self.config.generation_tokenizer
        self.opt_tokenizer = self.config.opt_tokenizer
        self.same_tokenizer = True
        if self.tgt_tokenizer.vocab_size != self.opt_tokenizer.vocab_size:
            self.same_tokenizer = False
            self.vocab_size = len(self.tgt_tokenizer)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to('cuda')
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = \
            self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.logits_processor.gamma_list * (1 - self.logits_processor.gamma_list))
        mean = torch.sum(self.logits_processor.gamma_list)
        z = (observed_count - mean)/torch.sqrt(var)
        return z

    def _score_sequence(self, input_ids: Tensor):
        num_tokens_scored = len(input_ids) - self.prefix_len
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.prefix_len} tokens required by the seeding scheme."
                )
            )
        green_token_count, green_token_mask = 0, []
        for idx in range(self.prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            if self.same_tokenizer:
                greenlist_ids, gamma, delta = self.logits_processor._get_greenlist_ids(input_ids[:idx])
            else:
                llama_str = self.tgt_tokenizer.decode(input_ids[max(idx - 5, 0):idx], add_special_tokens=False)
                ids_opt = self.opt_tokenizer(llama_str, add_special_tokens=False)['input_ids']
                if len(ids_opt) == 0:
                    continue
                greenlist_ids, gamma, delta = self.logits_processor._get_greenlist_ids(torch.tensor(ids_opt).cuda())

            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_mask.append(True)
            else:
                green_token_mask.append(False)

        return self._compute_z_score(green_token_count, num_tokens_scored)

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        if self.tgt_tokenizer is not None:
            tokenized_text = self.tgt_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][
                0].cuda()
            if tokenized_text[0] == self.tgt_tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            assert self.opt_tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer_opt ",
                "that was used at generation time.",
            )
            tokenized_text = self.opt_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][
                0].cuda()
            if tokenized_text[0] == self.opt_tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]

        # Compute z_score using a utility method
        self.logits_processor.reset_gamma_delta_list()
        z_score = self._score_sequence(tokenized_text)
        z_score = z_score.item()

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
