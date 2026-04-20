# On the Cultural Anachronism and Temporal Reasoning in Vision Language Models

Code and benchmark release for the paper *On the Cultural Anachronism and Temporal Reasoning in Vision Language Models*.

## Repository layout

```
tab-vlm/
├── data/
│   ├── temporal_anomaly_benchmark.json   # benchmark questions + metadata
│   └── benchmark_data.csv                # flat CSV view of the benchmark
├── src/
│   └── eval/           # evaluation pipeline (API + HF models)
│       ├── main.py
│       ├── inference.py
│       └── data_prep.py
├── app/
│   ├── download_image.py   # download benchmark images
│   └── human_eval.py       # Gradio app for human evaluation
├── utils/
│   ├── benchmark_to_csv.py
│   ├── data_analysis.py
│   ├── fix_json_responses.py
│   └── modify_json_question.py
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in API keys for GPT/Claude/Gemini models
```

## Download benchmark images

Benchmark images are not shipped with the code. Download them with:

```bash
python app/download_image.py
```

## Run evaluation

```bash
cd src/eval
python main.py \
    --benchmark ../../data/temporal_anomaly_benchmark.json \
    --model qwen \
    --output_dir results \
    --data_dir data
```

Supported `--model` values: `gpt4o`, `gpt4o-mini`, `claude`, `claude4`, `qwen`, `internvl`, `deepseek`, `gemma3`.

## Human evaluation UI

```bash
python app/human_eval.py
```

## Citation

```bibtex
@inproceedings{cultural_anachronism_vlm_2026,
  title  = {On the Cultural Anachronism and Temporal Reasoning in Vision Language Models},
  year   = {2026},
}
```
