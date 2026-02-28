from pylate import models

model = models.ColBERT(
    model_name_or_path="/opt/home/nohtow/mtrl_pylate/pylate/examples/train/output/GTE-ModernColBERT-MatryoshkaDocTokens-3e-05-lr-3-epochs/final"
)
model.push_to_hub("lightonai/MTRL_ColBERT", private=True)
