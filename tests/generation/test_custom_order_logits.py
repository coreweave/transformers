import copy
from inspect import signature
import random
import types

import torch
from torch import nn

from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_torch, torch_device
from ..test_modeling_common import ids_tensor

from transformers.generation import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import lovely_tensors as lt
lt.monkey_patch()


PROCESSORS = (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def verbosify(obj):
    old_call = obj.__call__
    def wrapped_call(*args, **kargs):
        print(obj.__name__)
        old_call(*args, **kargs)
    obj.__call__ = wrapped_call
    return obj



#I feel like this should be more easily doable with param or something like that?
class Parameterization:
    # which parameters are currently supported by this object
    _supported = [
        'top_k',
        'top_p',
        'temperature',
    ]

    def __init__(self):
        self.params={}

    def __getitem__(self, item):
        if item not in self.params:
            self.sample(item)
        return self.params[item]

    def sample(self, item):
        func_name = "_sample_" + item
        if not hasattr(self, func_name):
            raise NotImplementedError
        self.params[item] = getattr(self, func_name)()

    def __str__(self):
        return str(self.params)

    ##################################

    def _sample_top_k(self):
        return random.randint(3,10)
    def _sample_top_p(self):
        return random.random()
    def _sample_temperature(self):
        return random.random()


def select_processors(n: int):
    return random.sample(PROCESSORS, n)

# def select_ordering(n: int):
#     return random.sample(list(range(n)), n)

#
# @require_torch
# class TestLogitsProcessorOrderings:
#     def test_random_ordering(self, n=3):
#         params = Parameterization()
#         ordering_0 = []
#         for proc in select_processors(n):
#             sig = signature(proc)
#             kwargs = {k:params[k] for k in sig.parameters if k in params._supported}
#             ordering_0.append(proc(**kwargs))
#         #ordering_1 = [ordering_0[i] for i in select_ordering(n)]
#         ordering_1 = ordering_0[::-1] # we can ensure a different ordering by just reversing it
#
#         model_name = "hf-internal-testing/tiny-random-gpt2" # "EleutherAI/pythia-160m-deduped"
#         model = AutoModelForCausalLM.from_pretrained(model_name).to(torch_device)
#         input_ids = ids_tensor((1,5), vocab_size=model.config.vocab_size).to(torch_device)
#
#         outv = model.generate(
#             input_ids,
#             max_new_tokens=1,
#             do_sample=False,
#             output_scores=True,
#             output_logits=True,
#             return_dict_in_generate=True,
#         )
#
#         pred_ids = outv.sequences
#         logits_greedy = outv.scores[0] # unpack tuple...
#         logits_ordering_0 = copy.deepcopy(logits_greedy)
#         logits_ordering_1 = copy.deepcopy(logits_greedy)
#
#         # NB: current implementation modifies logits IN PLACE
#         assert ordering_0 != ordering_1
#
#         for proc in ordering_0:
#             logits_ordering_0 = proc(input_ids=pred_ids, scores=logits_ordering_0)
#         for proc in ordering_1:
#             logits_ordering_1 = proc(input_ids=pred_ids, scores=logits_ordering_1)
#         assert not torch.equal(logits_greedy, logits_ordering_0)     # Pass
#         assert not torch.equal(logits_ordering_0, logits_ordering_1) # Pass/Fail :(



#############################

from transformers import LogitsProcessorList, LogitsWarper, LogitsProcessor

List = list
Tuple = tuple

class LogitBiasProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` adding bias to specific tokens

    Args:
        logit_biases (:obj:`List[Tuple[int, float]]`):
            Adds a float bias to the given token's logit.
    """

    def __init__(self, logit_bias: List[Tuple[int, float]] = []):
        if not isinstance(logit_bias, list) and len(logit_bias) > 0:
            raise ValueError("`logit_bias` has to be a non-empty list")
        self.logit_bias = logit_bias
        self.bias = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bias is None:
            self.bias = torch.zeros(scores.shape[1]).float()
            logit_bias = torch.tensor(self.logit_bias)
            self.bias.scatter_(0, logit_bias[:, 0].long(), logit_bias[:, 1].float())
            self.bias = self.bias.to(scores.dtype).to(scores.device).unsqueeze(0)
        return scores + self.bias


class TypicalLogitsWarper(LogitsWarper):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float = 1.0, m=3.33, penalize_last=250, alpha_frequency=None, alpha_presence=None,
                 whitelist=None):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = 1.0 if penalty < 1.0 else penalty
        self.raw_penalty = penalty
        self.penalize_last = None
        if not m is None and not penalize_last is None and penalize_last >= 1:
            self.penalty = (torch.arange(penalize_last) / (penalize_last - 1)) * 2. - 1
            self.penalty = (m * self.penalty) / (1 + torch.abs(self.penalty) * (m - 1))
            self.penalty = 1 + ((self.penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
            self.penalize_last = penalize_last
        self.alpha_frequency = alpha_frequency if alpha_frequency is not None else None
        self.alpha_presence = alpha_presence if alpha_presence is not None else None
        self.alpha_enable = self.alpha_frequency is not None or self.alpha_presence is not None
        self.whitelist = None
        self.whitelist_list = None
        if whitelist is not None:
            self.whitelist_list = whitelist

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.whitelist is None and self.whitelist_list is not None:
            self.whitelist_list = list(filter(lambda x: x >= 0 and x < scores.shape[1], self.whitelist_list))
            if len(self.whitelist_list) > 0:
                self.whitelist = torch.tensor(self.whitelist_list).long().sort()[0]
                self.whitelist = self.whitelist.to(input_ids.device)
        if self.whitelist is not None:
            unpenalized = scores.gather(1, self.whitelist.view(1, -1))

        if self.raw_penalty > 1.0:
            if not self.penalize_last is None:
                penality_len = min(input_ids.shape[1], self.penalize_last)
                input_ids = input_ids[:, -penality_len:]
            score = torch.gather(scores, 1, input_ids)

            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            if not self.penalize_last is None:
                penalty = self.penalty.type(score.dtype).to(score.device)
                score = torch.where(score < 0, score * penalty[:, -penality_len:], score / penalty[:, -penality_len:])
            else:
                score = torch.where(score < 0, score * self.penalty, score / self.penalty)

            scores.scatter_(1, input_ids, score)

        if self.alpha_enable:
            c = torch.zeros(scores.shape).long().to(input_ids.device)
            # unique only returns counts for first item in batch, so manually iterate
            for i in range(input_ids.shape[0]):
                if self.penalize_last is not None:
                    token_input_ids, counts = torch.unique(input_ids[i, -self.penalize_last:], sorted=True,
                                                           return_counts=True, dim=-1)
                else:
                    token_input_ids, counts = torch.unique(input_ids[i], sorted=True, return_counts=True, dim=-1)
                c[i].scatter_(0, token_input_ids, counts)
            if self.alpha_frequency:
                scores -= c * self.alpha_frequency
            if self.alpha_presence:
                scores[c > 0] -= self.alpha_presence

        if self.whitelist is not None:
            scores.scatter_(1, self.whitelist.view(1, -1), unpenalized)

        return scores


class TailFreeSamplingLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs tail free sampling according to:
        https://trentbrick.github.io/Tail-Free-Sampling/#tail-free-sampling-algorithm

    Args:
        threshold (:obj:`float`):
            This sets the threshold z. A reasonable value is 0.95.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
    """

    def __init__(self, threshold: float, filter_value: float = -float("inf")):
        if not isinstance(threshold, float) or (threshold < 0 or threshold > 1.0):
            raise ValueError(f"`threshold` has to be a float > 0 and < 1, but is {threshold}")

        self.z = threshold
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        d = sorted_logits.softmax(dim=-1)
        d = d[:, 1:] - d[:, :-1]
        d = d[:, 1:] - d[:, :-1]
        d = d.abs()
        d = d / d.sum(dim=-1).view(1, -1).T
        cumulative_probs = d.cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = torch.zeros(sorted_indices.shape).bool().to(scores.device)
        sorted_indices_to_remove[:, 1:-1] = (cumulative_probs > self.z)[:, :]

        # Always remove last token
        sorted_indices_to_remove[:, -1:] = True

        # Always keep the first token
        sorted_indices_to_remove[:, 0] = False

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs tail free sampling according to:
        https://trentbrick.github.io/Tail-Free-Sampling/#tail-free-sampling-algorithm

    Args:
        threshold (:obj:`float`):
            This sets the threshold z. A reasonable value is 0.95.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
    """

    def __init__(self, threshold: float, filter_value: float = -float("inf")):
        if not isinstance(threshold, float) or (threshold < 0 or threshold > 1.0):
            raise ValueError(f"`threshold` has to be a float > 0 and < 1, but is {threshold}")

        self.z = threshold
        self.filter_value = filter_value
        print(threshold)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores, dim=-1)
        limit = torch.pow(torch.max(probs), 2.0) * self.z
        # print(probs)
        # print(limit)
        # amount = 0
        indices_to_remove = probs < limit
        # print(indices_to_remove)
        # print(scores.shape)
        '''
        for x in indices_to_remove[0]:
            if x == True:
                amount += 1

        print(amount-143)
        '''
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


# looks like the current implementation of GenerationMixin leverages a very different method which also has a very different signature.
# see ._get_logits_processor()

def _old_OVERRIDE_get_logits_warper(
        self, top_k: int = None, top_p: float = None, top_a: float = None,
        tfs: float = None, temperature: float = None,
        typical_p: float = None,
        #num_beams: int = None,
        num_beams: int = 1,
        order: tuple[int, int, int, int, int] = None
) -> LogitsProcessorList:
    """
    This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
    """

    # init warp parameters
    # self here is apparently going to end up being a GenerationConfig object? Is that right?
    #top_k = top_k if top_k is not None else self.config.top_k
    top_k = top_k if top_k is not None else self.top_k
    #top_p = top_p if top_p is not None else self.config.top_p
    top_p = top_p if top_p is not None else self.top_p
    top_a = top_a
    tfs = tfs
    #temperature = temperature if temperature is not None else self.config.temperature
    temperature = temperature if temperature is not None else self.temperature
    # instantiate warpers list
    warpers = []

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    else:
        warpers.append(None)

    if top_k is not None and top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(
            2 if num_beams > 1 else 1)))
    else:
        warpers.append(None)

    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(
            2 if num_beams > 1 else 1)))
    else:
        warpers.append(None)

    if tfs is not None and tfs < 1.0:
        warpers.append(TailFreeSamplingLogitsWarper(threshold=tfs))
    else:
        warpers.append(None)

    if top_a is not None and top_a < 1.0:
        warpers.append(TopALogitsWarper(threshold=top_a))
    else:
        warpers.append(None)

    if typical_p is not None and typical_p < 1.0:
        warpers.append(TypicalLogitsWarper(mass=typical_p,
                                           min_tokens_to_keep=(2 if (
                                                   num_beams is not None and num_beams > 1) else 1)))
    else:
        warpers.append(None)

    # this is insanity and I don't like it.
    # can we just do this with enums instead?
    print("hi there")
    if order is not None and len(order) == 6 and all(
            [x in order for x in (0, 1, 2, 3, 4, 5)]):
        reordered = []
        for i in order:
            w = warpers[i]
            w = verbosify(w)
            reordered.append(w)
        warpers = reordered
    else:
        print("order not received")

    warpers = list(filter(lambda x: x is not None, warpers))

    lpl = LogitsProcessorList()
    for warper in warpers:
        lpl.append(warper)

    return lpl

###########################################################################################

# We can have an "order" tuple coming in from the servicer, but I want the solution on this end
# to support arbitrary processors. Let's do this:
    # 1. assume some kind of default "processor priority" for all processors
    # 2. allow users to pass in a custom priority mapping
    # 3. after collecting all processors into the LogitsProceessorList, ensure the mapping is satisfied
# one benefit of this approach will be that we can just call  super()._get_logits_processor and
# slap this extra step on after


def OVERRIDE_get_logits_processor(
    self,
    generation_config, #: GenerationConfig,
    input_ids_seq_length, #: int,
    encoder_input_ids, #: torch.LongTensor,
    prefix_allowed_tokens_fn, #: Callable[[int, torch.Tensor], List[int]],
    logits_processor=None, #: Optional[LogitsProcessorList],
    model_kwargs=None, #: Optional[Dict[str, Any]] = None,
    negative_prompt_ids=None, #: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask=None, #: Optional[torch.Tensor] = None,
) -> LogitsProcessorList:
    processors = self._OLD_get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=encoder_input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    # todo: backfill custom processors here
    # ...
    # ... actually this might not even be necessary? it looks like OVERRIDE_get_logits_warper is actually being called

    if 'order' in model_kwargs:
        print("applying an ordering")
        processors = order_processors(processors, **model_kwargs)
    else:
        print("for real? wtf are those model args...")
        print(model_kwargs)
    return processors

def OVERRIDE_get_logits_warper(
        self,
        generation_config, #: GenerationConfig,
    ) -> LogitsProcessorList:
    processors = self._OLD_get_logits_warper(generation_config=generation_config)
    print(f"generation_config: {generation_config}")
    return processors



def get_processor_priority(processor, priority: dict, default=500):
    pname = processor.__name__
    outv = priority.get(pname, default)
    print((pname, outv))
    return outv

def order_processors(processors, **kargs):
    default_priority=500
    prioritized = []
    for proc in processors:
        rec = (proc, get_processor_priority(proc, kwargs['order']))
        prioritized.append(rec)
        default_priority += 1 # to conserve huggingface-provided ordering when unspecified in case it's important
    return LogitsProcessorList([item[0] for item in sorted(prioritized, key=lambda x: x[1])])

class TestClassOverride:
    """
    Can I make this work by just overriding a base transformers class?
    ... because of all the "automodel" stuff, this seems unlikely to be trivial.
    Or rather, it's probably trivial, but figuring out *where* to apply it is non-trivial.

    As an alternative... maybe I can load the model as I normally would, and then just over-write
    self._get_logits_warper() ?
    """

    def test_sanity(self):
        """
        manipulate individual logits to demonstrate that changing order impacts score as expected
        """
        p = torch.rand(1,1000)
        probs = p / p.sum()
        logprobs1 = probs.log()
        logprobs2 = logprobs1.clone()

        top_p = TopPLogitsWarper(0.8)
        temp = TemperatureLogitsWarper(0.8)

        assert torch.equal(top_p(None, logprobs1), top_p(None, logprobs2))
        assert torch.equal(temp(None, logprobs1), temp(None, logprobs2))
        assert not torch.equal(top_p(None, logprobs1), temp(None, logprobs2))
        g1 = top_p(None, logprobs1)
        h1 = temp(None, g1)
        g2 = temp(None, logprobs2)
        h2 = top_p(None, g2)
        assert not torch.equal(h1, h2)


    def test_method_override(self):
        model_name = "mistralai/Mistral-7b-v0.1" #"EleutherAI/pythia-160m-deduped" # "hf-internal-testing/tiny-random-gpt2"  # "EleutherAI/pythia-160m-deduped"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(torch_device)
        #model._get_logits_warper = OVERRIDE_get_logits_warper # godspeed 🫡
        model._OLD_get_logits_warper = model._get_logits_warper
        model._get_logits_warper = types.MethodType(OVERRIDE_get_logits_warper, model)

        model._validate_model_kwargs = lambda x: True

        # hmmm
        # apparently monkeypatching is non-trivial because of inco
        model._OLD_get_logits_processor = model._get_logits_processor
        # This is to correctly bind "self"
        model._get_logits_processor = types.MethodType(OVERRIDE_get_logits_processor, model)

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)

        #ORDER = (0,1,2,3,4,5)

        seed = random.randint(0, int(1e9))
        torch.manual_seed(seed)
        outv0 = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            #top_k = 5,
            top_p =0.7,
            temperature = 0.7,
            #order=ORDER
            order = {'TemperatureLogitsWarper':0}
        )

        torch.manual_seed(seed)
        outv1 = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            #top_k=5,
            top_p=0.7,
            temperature=0.7,
            #order = ORDER[::-1]
            order={'TemperatureLogitsWarper': 1000}
        )

        #print(outv0.scores[0].shape) #torch.Size([1, 32000])
        #print(len(outv0.scores)) # 10
        scores0 = torch.cat(outv0.scores, axis=0)
        scores1 = torch.cat(outv1.scores, axis=0)
        logits0 = torch.cat(outv0.logits, axis=0)
        logits1 = torch.cat(outv1.logits, axis=0)
        #print(scores0.shape) # torch.Size([10, 32000])

        assert torch.equal(logits0, logits1)
        assert not torch.equal(scores0, scores1) # still fails


