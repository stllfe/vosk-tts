import sys

sys.path.append("training")

from torch.utils.data import DataLoader

from training.models import AVAILABLE_FLOW_TYPES, SynthesizerTrn
from training.data_utils import TextAudioCollate, TextAudioLoader
from training.text.charbert import tokenizer
from training.utils import get_hparams_from_file

hps = get_hparams_from_file(
    "training/configs/mb_istft_vits2_charbert_single__mini.json"
)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

collate_fn = TextAudioCollate()
dataset = TextAudioLoader(hps.data.validation_files, hps.data)
loader = DataLoader(
    dataset,
    num_workers=1,
    shuffle=False,
    batch_size=hps.train.batch_size,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_fn,
)
# some of these flags are not being used in the code and directly set in hps json file.
# they are kept here for reference and prototyping.
if (
    "use_transformer_flows" in hps.model.keys()
    and hps.model.use_transformer_flows == True
):
    use_transformer_flows = True
    transformer_flow_type = hps.model.transformer_flow_type
    print(f"Using transformer flows {transformer_flow_type} for VITS2")
    assert (
        transformer_flow_type in AVAILABLE_FLOW_TYPES
    ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
else:
    print("Using normal flows for VITS1")
    use_transformer_flows = False

if (
    "use_spk_conditioned_encoder" in hps.model.keys()
    and hps.model.use_spk_conditioned_encoder == True
):
    if hps.data.n_speakers == 0:
        raise ValueError(
            "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
        )
    use_spk_conditioned_encoder = True
else:
    print("Using normal encoder for VITS1")
    use_spk_conditioned_encoder = False

if (
    "use_noise_scaled_mas" in hps.model.keys()
    and hps.model.use_noise_scaled_mas == True
):
    print("Using noise scaled MAS for VITS2")
    use_noise_scaled_mas = True
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6
else:
    print("Using normal MAS for VITS1")
    use_noise_scaled_mas = False
    mas_noise_scale_initial = 0.0
    noise_scale_delta = 0.0


net_g = SynthesizerTrn(
    tokenizer.vocab_size,
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    mas_noise_scale_initial=mas_noise_scale_initial,
    noise_scale_delta=noise_scale_delta,
    **hps.model,
).cuda()

for batch_idx, (
    x,
    x_lengths,
    spec,
    spec_lengths,
    y,
    y_lengths,
) in enumerate(loader):
    x, x_lengths = (
        x.cuda(non_blocking=True),
        x_lengths.cuda(non_blocking=True),
    )
    spec, spec_lengths = (
        spec.cuda(non_blocking=True),
        spec_lengths.cuda(non_blocking=True),
    )
    y, y_lengths = (
        y.cuda(non_blocking=True),
        y_lengths.cuda(non_blocking=True),
    )
    (
        y_hat,
        y_hat_mb,
        l_length,
        attn,
        ids_slice,
        x_mask,
        z_mask,
        (z, z_p, m_p, logs_p, m_q, logs_q),
        (hidden_x, logw, logw_),
    ) = net_g(x, x_lengths, spec, spec_lengths)
    break

print("Debug is done. It will cost you $300.")
