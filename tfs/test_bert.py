from pathlib import Path
from typing import List
from transformers import pipeline
from loguru import logger
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.convert_graph_to_onnx import convert

from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers    
from os import environ
from psutil import cpu_count
from transformers import BertTokenizerFast
from transformers import BertModel
import numpy as np


def test_bert_model_load():
    config = BertConfig()
    # model = BertModel(config)
    model = BertModel.from_pretrained("bert-base-cased", mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    print(config)
    encoded_sequences = [
        [101, 7592, 999, 102],
        [101, 4658, 1012, 102],
        [101, 3835, 999, 102],
    ]   
    model_inputs = torch.tensor(encoded_sequences)
    output = model(model_inputs)
    logger.info("model output {}", output)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased",  mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    tokens = tokenizer("Using a Transformer network is simple", padding=True, truncation=True, return_tensors="pt")
    logger.info("tokens {}", tokens)
    output = model(**tokens)
    logger.info("model output {}", output)

    

def test_convert_to_onnx():
    # convert bert model to onnx format
    convert(framework="pt", model="bert-base-cased", output=Path("onnx/bert-base-cased.onnx"), opset=11)
    
    # simplify bert model
    # onnxsim.simplify("onnx/bert-base-cased.onnx", "onnx/bert-base-cased_sim.onnx")
    
     # # An optional step unless
    # # you want to get a model with mixed precision for perf accelartion on newer GPU
    # # or you are working with Tensorflow(tf.keras) models or pytorch models other than bert

    # !pip install onnxruntime-tools
    # from onnxruntime_tools import optimizer

    # # Mixed precision conversion for bert-base-cased model converted from Pytorch
    # optimized_model = optimizer.optimize_model("bert-base-cased.onnx", model_type='bert', num_heads=12, hidden_size=768)
    # optimized_model.convert_model_float32_to_float16()
    # optimized_model.save_model_to_file("bert-base-cased.onnx")

    # # optimizations for bert-base-cased model converted from Tensorflow(tf.keras)
    # optimized_model = optimizer.optimize_model("bert-base-cased.onnx", model_type='bert_keras', num_heads=12, hidden_size=768)
    # optimized_model.save_model_to_file("bert-base-cased.onnx")


    # optimize transformer-based models with onnxruntime-tools
    # from onnxruntime_tools import optimizer
    # from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions

    # # disable embedding layer norm optimization for better model size reduction
    # opt_options = BertOptimizationOptions('bert')
    # opt_options.enable_embed_layer_norm = False

    # opt_model = optimizer.optimize_model(
    #     'onnx/bert-base-cased.onnx',
    #     'bert', 
    #     num_heads=12,
    #     hidden_size=768,
    #     optimization_options=opt_options)
    # opt_model.save_model_to_file('bert.opt.onnx')


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession: 
  
  assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

  # Few properties that might have an impact on performances (provided by MS)
  options = SessionOptions()
  options.intra_op_num_threads = 1
  options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

  # Load the model as a graph and prepare the CPU backend 
  session = InferenceSession(model_path, options, providers=[provider])
  session.disable_fallback()
    
  return session


@contextmanager
def track_infer_time(buffer: List[int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


@dataclass
class OnnxInferenceResult:
  model_inference_time: List[int]  
  optimized_model_path: str
  
def test_infer_with_onnx():

    # Constants from the performance optimization available in onnxruntime
    # It needs to be done before importing onnxruntime
    environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
    environ["OMP_WAIT_POLICY"] = 'ACTIVE'

    
    # logger.info("Available Providers: {}", get_all_providers())
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    cpu_model = create_model_for_provider("onnx/bert-base-cased.onnx", "CPUExecutionProvider")

    # Inputs are provided through numpy array
    model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    # Run the model (None = get all the outputs)
    sequence, pooled = cpu_model.run(None, inputs_onnx)

    # Print information about outputs

    print(f"Sequence output: {sequence.shape}, Pooled output: {pooled.shape}")


def test_bert_gpu_vs_cpu():

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # Inputs are provided through numpy array
    model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    PROVIDERS = {
    ("CPUExecutionProvider", "ONNX CPU"),
    #  Uncomment this line to enable GPU benchmarking
    ("CUDAExecutionProvider", "ONNX GPU")
    }

    results = {}
    for provider, label in PROVIDERS:
        # Create the model with the specified provider
        # model = create_model_for_provider("onnx/bert-base-cased.onnx", provider)
        model = create_model_for_provider("onnx/bert-base-cased_sim.onnx", provider)

        # Keep track of the inference time
        time_buffer = []

        # Warm up the model
        model.run(None, inputs_onnx)

        # Compute 
        for _ in trange(100, desc=f"Tracking inference time on {provider}"):
            with track_infer_time(time_buffer):
                model.run(None, inputs_onnx)

        # Store the result
        results[label] = OnnxInferenceResult(
        time_buffer,
        model.get_session_options().optimized_model_filepath
        )
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
    logger.info("result {}", time_results)
        
if __name__ == "__main__":
    # test_bert_model_load()
    # test_convert_to_onnx()
    # test_infer_with_onnx()
    test_bert_gpu_vs_cpu()