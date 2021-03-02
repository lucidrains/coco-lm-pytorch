import math
from functools import reduce

import torch
from torch import nn, einsum
import torch.nn.functional as F

# helpers

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

# hidden layer extractor class, for magically adding adapter to language model to be pretrained

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# main electra class

class COCO(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        *,
        discr_dim,
        num_tokens = None,
        discr_layer = -1,
        mask_prob = 0.15,
        replace_prob = 0.85,
        random_token_prob = 0.,
        pad_token_id = 0,
        cls_token_id = 1,
        mask_token_id = 2,
        mask_ignore_token_ids = [],
        disc_weight = 50.,
        gen_weight = 1.,
        cl_weight = 1.,
        temperature = 1.,
        crop_percentage = 0.5
    ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.discriminator = HiddenLayerExtractor(discriminator, layer = discr_layer)
        self.to_correction_logits = nn.Linear(discr_dim, 1)

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id, cls_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight
        self.cl_weight = cl_weight

        self.cl_temperature = nn.Parameter(torch.tensor(1.))

        self.crop_percentage = crop_percentage

    def forward(self, input, **kwargs):
        b, t, device = *input.shape, input.device
        assert b > 1, 'batch size need to be bigger than 1 for contrastive learning'

        cls_tokens = torch.empty(b, 1, dtype = torch.long).fill_(self.cls_token_id)

        input = torch.cat((cls_tokens, input), dim = 1)
        input = input[:, :-1]

        replace_prob = prob_mask_like(input, self.replace_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get random cropped input for contrastive learning
        random_crop = get_mask_subset_with_prob(~no_mask, self.crop_percentage)
        crop_length = int(t * self.crop_percentage)
        cropped_input = input.masked_select(random_crop).reshape(b, crop_length)
        cropped_input = torch.cat((cls_tokens, cropped_input), dim = 1)
        cropped_input = F.pad(cropped_input, (0, t - crop_length - 1), value = self.pad_token_id)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            random_token_prob = prob_mask_like(input, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input.shape, device=input.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]

        # [mask] input
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)

        # set inverse of mask to padding tokens for labels
        gen_labels = input.masked_fill(~mask, self.pad_token_id)

        # get generator output and get mlm loss
        logits = self.generator(masked_input, **kwargs)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index = self.pad_token_id
        )

        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = gumbel_sample(sample_logits, temperature = self.temperature)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float().detach()

        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # get discriminator output and binary cross entropy loss
        disc_embeddings_correction = self.discriminator(disc_input, **kwargs)

        correction_logits = self.to_correction_logits(disc_embeddings_correction)
        disc_logits = correction_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        # contrastive loss
        disc_embeddings_cropped = self.discriminator(cropped_input, **kwargs)

        cls_tokens_corrected = disc_embeddings_correction[:, 0]
        cls_tokens_cropped = disc_embeddings_cropped[:, 0]

        cl_temperature = self.cl_temperature.exp()
        sim = einsum('i d, j d -> i j', cls_tokens_corrected, cls_tokens_cropped) * cl_temperature
        labels = torch.arange(b, device = device)

        cl_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) * 0.5

        # weight all losses
        weighted_loss = self.cl_weight * cl_loss + self.gen_weight * mlm_loss + self.disc_weight * disc_loss
        return weighted_loss
