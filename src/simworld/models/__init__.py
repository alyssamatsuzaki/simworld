"""GraphRSSM emulator components (§10 Stages 6+7).

`rssm.py` holds the macro path (discrete-latent RSSM), `gnn.py` the micro path
(heterogeneous message passing + per-node recurrence), `encoder.py` the pooled
observation encoder, `heads.py` the decoders, and `world_model.py` the composite
that the training loop and `EmulatorEnv` consume.
"""
