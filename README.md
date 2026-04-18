## Hackathon : Benchmarking Small Language Models in the Real World

Team : "Coffee is Life"

## Links

Synthetic dataset generation:
  - https://huggingface.co/spaces/HuggingFaceFW/finephrase

- Submission examples: https://github.com/Nisseya/submission_example
- Code behind the polarsbench.net website: https://github.com/Nisseya/hack_apr/tree/master
- Training Gemma 4 with unsloth:
  - https://unsloth.ai/docs/models/gemma-4/train
  - https://unsloth.ai/docs/models/gemma-4/train#unsloth-core-code-based-guide
  - Sample data formats for this repo: `docs/DATA_FORMATS.md`

## GPU / Runpod access

JupyterLab: https://30y88pd4dxzb6q-8888.proxy.runpod.net/lab

polarsbench: https://30y88pd4dxzb6q-8888.proxy.runpod.net/

SSH:

```
ssh 30y88pd4dxzb6q-64411fd0@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The SSH private key is **not** committed. Grab it from the team password
manager and place it at `~/.ssh/id_ed25519` (`chmod 600`). Rotate it on Runpod
if it ever leaks.
