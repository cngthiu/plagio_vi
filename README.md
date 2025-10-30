```
plagio_vi/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ default.yaml
│  ├─ hardware.cpu.yaml
│  ├─ hardware.gpu1650.yaml
│  ├─ indexing.faiss.hnsw.yaml
│  └─ thresholds.accuracy.yaml
├─ data/
│  ├─ dictionaries/{stop_phrases_vi.txt,synonyms_vi.tsv,abbreviations.tsv,gazetteer_entities.tsv}
│  └─ patterns/{boilerplate_vi.txt,section_headers.txt}
├─ models/{bi_encoder/,cross_encoder/,onnx/}
├─ src/
│  ├─ cli/{compare_docs.py,index_corpus.py}
│  ├─ io/{docx_reader.py,pdf_reader.py,writer_report.py}
│  ├─ preprocess/{normalize_vi.py,sentence_seg.py,tokenizer_vi.py,chunker.py}
│  ├─ dicts/{loader.py,filters.py}
│  ├─ candidate/{bm25_index.py,simhash.py,ann_index.py}
│  ├─ models/{embedder.py,cross_encoder.py}
│  ├─ ranking/{features.py,rerank.py}
│  ├─ alignment/{tiling.py,sw_align.py}
│  ├─ report/{aggregate.py,highlight.py}
│  └─ utils/{text_map.py,timing.py,hardware.py,logging.py}
└─ webui/{app.py,assets/}

```

#RUN 
python3 -m src.cli.compare_docs \
  --config configs/default.yaml \
  --hardware configs/hardware.gpu1650.yaml \
  --a /home/congthieu/Documents/AI1.docx \
  --b /home/congthieu/Documents/AI4.docx \
  --out outputs/run_A_B
#API
uvicorn src.service.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
        "a": "/path/to/A.docx",
        "b": "/path/to/B.docx",
        "out": "outputs/api_run",
        "config": "configs/default.yaml",
        "hardware": "configs/hardware.gpu1650.yaml"
      }'

