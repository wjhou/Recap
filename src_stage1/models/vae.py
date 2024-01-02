import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modeling_bart import shift_tokens_right
from torch.autograd import Variable

from models.activations import ACT2FN
from models.layers import SinkhornDistance
from models.von_mises_fisher import VonMisesFisher


def word_avg(encoder_hidden_states, attention_mask):
    if attention_mask is not None:
        sum_vecs = (encoder_hidden_states *
                    attention_mask.unsqueeze(-1)).sum(1)
        avg_vecs = sum_vecs / attention_mask.sum(1, keepdim=True)
    else:
        avg_vecs = encoder_hidden_states.mean(1)
    return avg_vecs


def kl_loss(mean, var):
    """
    KL(p||N(0,1))
    """
    return -0.5 * torch.mean(torch.mean(1 + var - mean.pow(2) - var.exp(), 1))


class VecDecoder(nn.Module):
    def __init__(self, config):
        super(VecDecoder, self).__init__()
        self.config = config
        bow_head = []
        for i in range(config.bow_layers):
            if i == 0:
                inp_dim = config.d_mean
                out_dim = config.d_mean_var
            elif i == config.bow_layers - 1:
                inp_dim = config.d_mean_var
                out_dim = config.d_model
            else:
                inp_dim = out_dim = config.d_mean_var
            layer = nn.Linear(inp_dim, out_dim)
            bow_head.append(layer)
            bow_head.append(nn.Dropout(config.dropout))
            if i < config.bow_layers - 1:
                bow_head.append(ACT2FN[config.activation_function])
            else:
                bow_head.append(nn.Tanh())
        self.bow_head = nn.Sequential(*bow_head)

    def forward(self, z, labels=None):
        vae_loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            logits = self.bow_head(z)
            vae_loss = loss_fct(
                logits,
                labels,
            )
        return vae_loss


class BoWDecoder(nn.Module):
    def __init__(self, config, init_inp_dim=None):
        super(BoWDecoder, self).__init__()
        self.config = config
        bow_head = []
        for i in range(config.bow_layers):
            if i == 0:
                inp_dim = config.d_mean if init_inp_dim is None else init_inp_dim
                out_dim = config.d_mean_var
            elif i == config.bow_layers - 1:
                inp_dim = config.d_mean_var
                out_dim = config.vocab_size
            else:
                inp_dim = out_dim = config.d_mean_var
            layer = nn.Linear(inp_dim, out_dim)
            bow_head.append(layer)
            bow_head.append(nn.Dropout(config.dropout))
            if i < config.bow_layers - 1:
                bow_head.append(ACT2FN[config.activation_function])
        self.bow_head = nn.Sequential(*bow_head)

    def forward(self, z, labels=None):
        vae_loss = None
        if labels is not None:
            seq_len = labels.size(1)
            loss_fct = nn.CrossEntropyLoss()
            logits = self.bow_head(z)
            expanded_logits = logits.unsqueeze(1).expand(-1, seq_len, -1)
            vae_loss = loss_fct(
                expanded_logits.reshape(-1, self.config.vocab_size),
                labels.view(-1),
            )
        return vae_loss


class RNNDecoder(nn.Module):
    def __init__(self, config):
        super(RNNDecoder, self).__init__()
        self.config = config
        self.embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.decoder = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_mean,
            num_layers=1,
            batch_first=True,
        )
        self.lm_head = nn.Linear(config.d_mean, config.vocab_size)

    def forward(self, init_hidden, labels):
        input_ids = shift_tokens_right(
            input_ids=labels,
            pad_token_id=self.config.pad_token_id,
            decoder_start_token_id=self.config.bos_token_id,
        )
        embed_tokens = self.embed(input_ids)
        hidden_states, _ = self.decoder(embed_tokens, init_hidden.unsqueeze(0))
        logits = self.lm_head(hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
        )
        return lm_loss


class VAE(nn.Module):
    def __init__(self, config, is_sem=False):
        super(VAE, self).__init__()
        self.mean = nn.Linear(config.d_model, config.d_mean)
        d_var = config.d_var if not is_sem else 1
        self.var = nn.Linear(config.d_model, d_var)
        self.decoder = None

    def process(
        self,
        encoder_hidden_states,
        attention_mask,
        is_sem=False,
    ):
        mean_hidden_states = self.mean(encoder_hidden_states)
        if is_sem:
            mean_hidden_states = mean_hidden_states / mean_hidden_states.norm(
                dim=-1, keepdim=True)
        mean_state = word_avg(mean_hidden_states, attention_mask)
        var_hidden_states = self.var(encoder_hidden_states)
        if is_sem:
            var_hidden_states = F.softplus(var_hidden_states) + 100
        var_state = word_avg(var_hidden_states, attention_mask)
        return mean_state, var_state

    def forward(
        self,
        encoder_hidden_states,
        attention_mask,
        labels=None,
    ):
        mean_hidden_states, var_hidden_states = self.process(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        z = self.sample_gaussian(
            mean=mean_hidden_states,
            var=var_hidden_states,
        )
        # assert self.decoder is None, "Please initialize decoder"
        rl = self.decoder(z, labels)
        kl = kl_loss(mean_hidden_states, var_hidden_states)
        return rl, kl, mean_hidden_states

    def sample_gaussian(self, mean, var):
        sample = mean + torch.exp(0.5 * var) * Variable(
            var.data.new(var.size()).normal_())
        return sample


class TextVAE(VAE):
    def __init__(self, config):
        super(TextVAE, self).__init__(config)
        self.decoder = BoWDecoder(config)
        # self.decoder = RNNDecoder(config)


class VGVAE(nn.Module):
    def __init__(self, config):
        super(VGVAE, self).__init__()
        self.semantic_vae = VAE(config, is_sem=True)
        self.syntactic_vae = VAE(config)
        self.decoder = BoWDecoder(config, init_inp_dim=config.d_mean * 2)
        self.pos_decoder = nn.Linear(
            config.d_var + config.d_model,
            config.max_position_embeddings,
        )
        self.max_position = config.max_position_embeddings

    def wpl_fn(self, hidden_states, labels):
        # word position loss
        max_position = labels.size(1)
        pos_labels = torch.arange(
            0,
            max_position,
            device=hidden_states.device,
        ).masked_fill(labels == -100, -100)
        pos_hidden_states = self.pos_decoder(hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        wpl = loss_fct(
            pos_hidden_states.view(-1, self.max_position),
            pos_labels.view(-1),
        )
        return wpl

    def forward(
        self,
        encoder_hidden_states,
        attention_mask,
        labels=None,
    ):
        sem_mean, sem_var = self.semantic_vae.process(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            is_sem=True,
        )
        syn_mean, syn_var = self.syntactic_vae.process(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            is_sem=False,
        )
        sem_dist = VonMisesFisher(sem_mean, sem_var)
        sem_z = sem_dist.rsample()
        syn_z = self.syntactic_vae.sample_gaussian(syn_mean, syn_var)
        z = torch.cat((sem_z, syn_z), dim=-1)
        rl = self.decoder(z, labels)
        sem_kl = sem_dist.kl_div().mean()
        syn_kl = kl_loss(syn_mean, syn_var)
        wpl = self.wpl_fn(
            torch.cat(
                (encoder_hidden_states, syn_z.unsqueeze(1).expand(
                    -1,
                    encoder_hidden_states.size(1),
                    -1,
                )),
                dim=-1,
            ),
            labels,
        )
        return rl, syn_kl, wpl, sem_kl, sem_mean


class VisualVAE(VAE):
    def __init__(self, config):
        super(VisualVAE, self).__init__(config)
        self.decoder = VecDecoder(config)

    def forward(self, encoder_hidden_states, labels):
        avg_hidden_states = word_avg(encoder_hidden_states, None)
        mean_hidden_states = self.mean(avg_hidden_states)
        var_hidden_states = self.var(avg_hidden_states)
        z = self.sample_gaussian(
            mean=mean_hidden_states,
            var=var_hidden_states,
        )
        rl = self.decoder(z, labels)
        kl = kl_loss(mean_hidden_states, var_hidden_states)
        return rl, kl, mean_hidden_states


class DisentanglementModel(nn.Module):
    def __init__(self, config):
        super(DisentanglementModel, self).__init__()
        self.text_vae = TextVAE(config)
        # self.text_vae = VGVAE(config)
        self.visual_vae = VisualVAE(config)
        self.margin = 1.

    def forward(
        self,
        visual_hidden_states,
        pos_hidden_states,
        neg_hidden_states,
        pos_attention_mask,
        neg_attention_mask,
        visual_labels,
        pos_labels,
        neg_labels,
    ):
        pos_text_rl, pos_sem_kl, pos_states = self.text_vae(
            pos_hidden_states,
            pos_attention_mask,
            pos_labels,
        )
        neg_text_rl, neg_sem_kl, neg_states = self.text_vae(
            neg_hidden_states,
            neg_attention_mask,
            neg_labels,
        )
        # pos_text_rl, pos_syn_kl, pos_wpl, pos_sem_kl, pos_states = self.text_vae(
        #     pos_hidden_states.detach(),
        #     pos_attention_mask,
        #     pos_labels,
        # )
        # neg_text_rl, neg_syn_kl, neg_wpl, neg_sem_kl, neg_states = self.text_vae(
        #     neg_hidden_states.detach(),
        #     neg_attention_mask,
        #     neg_labels,
        # )
        visual_rl, visual_kl, anchor_states = self.visual_vae(
            visual_hidden_states,
            visual_labels.mean(dim=1).detach(),
        )
        pos_cos = F.cosine_similarity(pos_states, anchor_states)
        neg_cos = F.cosine_similarity(neg_states, anchor_states)
        dl = F.relu(self.margin - pos_cos + neg_cos).mean()
        # pos_loss = self.margin - F.mse_loss(pos_states, anchor_states)
        # neg_loss = self.margin - F.mse_loss(neg_states, anchor_states)
        # dl = 0.5 * F.relu(pos_loss).mean() + 0.5 * F.relu(neg_loss).mean()
        text_beta = 1e-3
        visual_beta = 1e-3
        # text_beta = visual_beta = 1.
        # print("pos_text_rl", pos_text_rl.item())
        print("pos_text_kl", pos_sem_kl.item())
        # print("neg_text_rl", neg_text_rl.item())
        print("neg_text_kl", neg_sem_kl.item())
        # print("visual_rl", visual_rl.item())
        print("visual_kl", visual_kl.item())
        # print("dl", dl.item())
        loss = 0
        # positive report vae
        loss = loss + pos_text_rl + text_beta * pos_sem_kl  #+ text_beta * pos_syn_kl + pos_wpl
        # negative report vae
        loss = loss + neg_text_rl + text_beta * neg_sem_kl  #+ text_beta * neg_syn_kl + neg_wpl
        # image vae
        loss = loss + visual_rl + visual_beta * visual_kl
        # discriminative loss
        # sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        # dl, P, C = sinkhorn(pos_states, anchor_states)
        loss = loss + dl
        return loss
