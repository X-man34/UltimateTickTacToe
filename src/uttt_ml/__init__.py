"""
uttt_ml — training pipeline and data tooling for the Ultimate Tic Tac Toe bot.

Includes:
    - nd4j_loader: pure-Python reader for the legacy ND4J DataSet .bin files
      produced by the original Java project
    - convert_bin_to_npz: one-time migration tool for legacy training data
    - dataset: PyTorch Dataset classes over the .npz format (shared by both
      converted legacy data and newly generated self-play data)
    - self_play: CLI for generating new training data via MCTS self-play
    - train_value / train_policy: CLI trainers for the value and policy nets
"""
