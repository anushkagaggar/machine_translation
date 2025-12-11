from src.utils.config import load_yaml_config

cfg = load_yaml_config("configs/train.yaml")

print("---- RAW LOADED CONFIG ----")
print(cfg)

print("\n---- TYPES ----")
print("lr:", cfg["train"]["optimizer"]["lr"], type(cfg["train"]["optimizer"]["lr"]))
print("betas:", cfg["train"]["optimizer"]["betas"], type(cfg["train"]["optimizer"]["betas"]))
print("eps:", cfg["train"]["optimizer"]["eps"], type(cfg["train"]["optimizer"]["eps"]))
