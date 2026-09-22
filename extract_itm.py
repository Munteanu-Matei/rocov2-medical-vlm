import torch, warnings, traceback
warnings.filterwarnings("ignore")
try:
    from transformers import Blip2ForImageTextRetrieval
    print("Downloading/loading blip2-itm-vit-g (~4 GB, cached after first run)...", flush=True)
    itm = Blip2ForImageTextRetrieval.from_pretrained(
        "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # locate the Q-Former text embeddings (word + position)
    emb = None
    for name, mod in itm.named_modules():
        if hasattr(mod, "word_embeddings") and hasattr(mod, "position_embeddings"):
            emb = mod; print("text embeddings module:", name, flush=True); break
    assert emb is not None, "could not find text embeddings"

    layers = itm.qformer.encoder.layer
    save = {
        "word_embeddings.weight":     emb.word_embeddings.weight.detach().float().clone(),
        "position_embeddings.weight": emb.position_embeddings.weight.detach().float().clone(),
        "num_layers": len(layers),
        "text_ffn": [],
    }
    for i, l in enumerate(layers):
        assert hasattr(l, "intermediate") and hasattr(l, "output"), f"layer {i} lacks text FFN"
        save["text_ffn"].append({
            "intermediate": {k: v.detach().float().clone() for k, v in l.intermediate.state_dict().items()},
            "output":       {k: v.detach().float().clone() for k, v in l.output.state_dict().items()},
        })
    out = "/home/matei/blip2_qformer_text_weights.pt"
    torch.save(save, out)
    print(f"SAVED {out}", flush=True)
    print("word:", tuple(save['word_embeddings.weight'].shape),
          "pos:", tuple(save['position_embeddings.weight'].shape),
          "ffn layers:", len(save['text_ffn']),
          "ffn keys:", list(save['text_ffn'][0]['intermediate'].keys()), list(save['text_ffn'][0]['output'].keys()),
          flush=True)
except Exception:
    traceback.print_exc()
    raise
