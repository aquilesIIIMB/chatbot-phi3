import argparse
import logging
import os
import time
from pathlib import Path

import torch
from optimum.exporters.onnx import OnnxConfig, export
from optimum.exporters.onnx.model_configs import NormalizedConfig
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phi3NormalizedConfig(NormalizedConfig):
    """
    Normalized config for a Phi3-based causal language model.
    Properly extracts and provides access to all necessary model parameters.
    """
    def __init__(self, config):
        super().__init__(config)
        self._num_layers = getattr(config, "num_hidden_layers", None)
        self._hidden_size = getattr(config, "hidden_size", None)
        self._num_attention_heads = getattr(config, "num_attention_heads", None)
        self._model_type = getattr(config, "model_type", "phi3")
        self._vocab_size = getattr(config, "vocab_size", None)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self._num_attention_heads

    @property
    def model_type(self) -> str:
        return self._model_type
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

class Phi3CustomOnnxConfig(OnnxConfig):
    """
    OnnxConfig for Phi3-based causal LMs.
    Specifies input/output configurations with proper dimension names.
    """
    NORMALIZED_CONFIG_CLASS = Phi3NormalizedConfig

    @property
    def inputs(self):
        return {
            "input_ids": {
                0: "batch_size",
                1: "sequence_length"
            },
            "attention_mask": {
                0: "batch_size",
                1: "sequence_length"
            },
        }

    @property
    def outputs(self):
        # Now properly includes vocab_size dimension
        return {
            "logits": {
                0: "batch_size",
                1: "sequence_length",
                2: "vocab_size"
            },
        }

    @property
    def default_onnx_opset_version(self) -> int:
        return 15

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        """
        Provide dummy tensors to trace the model.
        Now explicitly places tensors on the same device as the model.
        """
        import torch

        batch_size = kwargs.get("batch_size", 1)
        seq_length = kwargs.get("seq_length", 4)
        
        # Important: Get device from kwargs instead of assuming
        # We need to match exactly the model's device
        device = kwargs.get("device", "cpu")
        
        # Log the device being used for dummy inputs
        print(f"Generating dummy inputs on device: {device}")
        
        # Use actual vocab size from config for better compatibility
        vocab_size = self._normalized_config.vocab_size or 32000
        
        # Create tensors on CPU first
        dummy_input_ids = torch.randint(
            low=0,
            high=vocab_size - 1,
            size=(batch_size, seq_length),
            dtype=torch.long
        )
        dummy_attention_mask = torch.ones(
            (batch_size, seq_length),
            dtype=torch.long
        )
        
        # Then explicitly move to the target device
        dummy_input_ids = dummy_input_ids.to(device)
        dummy_attention_mask = dummy_attention_mask.to(device)
        
        return {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask
        }

def get_gpu_memory_info():
    """Get GPU memory information if available."""
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert to GB
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        
        return {
            "device_name": gpu_properties.name,
            "total_memory_gb": total_memory,
            "allocated_memory_gb": allocated_memory,
            "free_memory_gb": total_memory - allocated_memory
        }
    return None

def validate_model(model_path, tokenizer, original_model=None, device="cuda"):
    """
    Validate the exported/quantized model against the original model.
    Compares prediction outputs to ensure model quality is maintained.
    """
    from onnxruntime import InferenceSession
    import numpy as np
    
    logger.info(f"Validating model at {model_path}")
    
    # Create ONNX inference session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    try:
        session = InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        logger.error(f"Failed to load ONNX model for validation: {e}")
        return False
    
    # Prepare input text
    input_text = "Hello, my name is"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # If we have the original model, compare the outputs
    if original_model is not None:
        with torch.no_grad():
            original_outputs = original_model(input_ids=input_ids, attention_mask=attention_mask)
            original_logits = original_outputs.logits.cpu().numpy()
    
    # Run the ONNX model
    onnx_inputs = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy()
    }
    
    try:
        onnx_outputs = session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]
    except Exception as e:
        logger.error(f"Failed to run inference on ONNX model: {e}")
        return False
    
    # If we have the original model, compare the outputs
    if original_model is not None:
        # Calculate the difference
        max_diff = np.max(np.abs(original_logits - onnx_logits))
        logger.info(f"Maximum absolute difference in logits: {max_diff}")
        
        # Check if the argmax predictions match
        original_preds = np.argmax(original_logits, axis=-1)
        onnx_preds = np.argmax(onnx_logits, axis=-1)
        matches = np.array_equal(original_preds, onnx_preds)
        logger.info(f"Prediction matches: {matches}")
        
        return max_diff < 1e-3 and matches  # Slightly relaxed tolerance for float16
    
    # If no original model is provided, just make sure the ONNX model runs
    return True

def export_and_quantize_phi3(
    model_id: str, 
    onnx_dir: Path, 
    quantized_dir: Path, 
    batch_size: int = 1, 
    seq_length: int = 4,
    use_gpu: bool = True,
    optimize_memory: bool = True,
    validate: bool = True,
    save_fp16: bool = False,
    force_cpu: bool = False
):
    """
    Export and quantize a Phi-3 model with comprehensive error handling and optimizations.
    
    Args:
        model_id: Hugging Face model ID
        onnx_dir: Path to save the ONNX model
        quantized_dir: Path to save the quantized model
        batch_size: Batch size for dummy inputs
        seq_length: Sequence length for dummy inputs
        use_gpu: Whether to use GPU for export and quantization
        optimize_memory: Whether to optimize memory usage
        validate: Whether to validate the exported and quantized models
        save_fp16: Whether to save an FP16 model in addition to the quantized model
        force_cpu: Whether to force CPU usage even if GPU is available
    """
    start_time = time.time()
    
    # Create output directories
    onnx_dir.mkdir(parents=True, exist_ok=True)
    quantized_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability and respect force_cpu flag
    if force_cpu:
        logger.info("CPU-only mode explicitly requested. Using CPU regardless of GPU availability.")
        use_gpu = False
        device = "cpu"
    elif use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but not available. Falling back to CPU.")
        use_gpu = False
        device = "cpu"
    else:
        device = "cuda" if use_gpu else "cpu"
    
    # Log GPU info if using GPU
    if use_gpu:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(f"Using GPU: {gpu_info['device_name']}")
            logger.info(f"GPU memory: {gpu_info['free_memory_gb']:.2f}GB free out of {gpu_info['total_memory_gb']:.2f}GB")
            
            # Check if there's enough GPU memory (minimum 4GB required)
            if gpu_info['free_memory_gb'] < 4.0:
                logger.warning(f"Less than 4GB of GPU memory available ({gpu_info['free_memory_gb']:.2f}GB). This may cause OOM errors.")
                logger.warning("Consider using --cpu_only flag if export fails.")
    else:
        logger.info("Using CPU for all operations.")
    
    # 1. Load the model and config
    logger.info(f"Loading model {model_id}")
    try:
        # Load model with consistent device placement
        # The key issue is to avoid device_map="auto" which creates mixed device placement
        try:
            # First try loading with reduced precision if memory optimization is requested
            if optimize_memory and use_gpu:
                logger.info("Using 16-bit precision to optimize memory usage")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    # Don't use device_map="auto" to avoid mixed device placement
                )
                # Move the entire model to GPU
                model = model.to(device)
                logger.info(f"Model loaded and placed on {device}")
            else:
                # Load the model and move it entirely to a single device
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                model = model.to(device)
                logger.info(f"Model loaded and placed on {device}")
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and use_gpu:
                logger.warning("GPU out of memory. Falling back to CPU.")
                use_gpu = False
                device = "cpu"
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                logger.info("Model loaded on CPU due to GPU memory constraints")
                
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # 2. Create custom ONNX config
    onnx_config = Phi3CustomOnnxConfig(config)
    
    # 3. Export the model to ONNX
    logger.info(f"Exporting model to ONNX format at {onnx_dir}")
    try:
        onnx_path = onnx_dir / "model.onnx"
        
        # Set torch.jit.optimize_for_inference to True
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        
        # For larger models, we need to handle external data differently
        # And ensure consistent device usage
        logger.info(f"Beginning ONNX export with device: {device}")
        
        # Make sure model is in eval mode
        model.eval()
        
        # Double-check all model parameters are on the correct device
        actual_device = next(model.parameters()).device
        logger.info(f"Model parameters are on device: {actual_device}")
        
        if str(actual_device) != device and device == "cuda":
            logger.warning(f"Device mismatch! Requested {device} but model is on {actual_device}. Fixing...")
            model = model.to(device)
            actual_device = next(model.parameters()).device
            logger.info(f"Model moved to device: {actual_device}")
        
        # Pass the actual device to export function and dummy input generation
        device_to_use = str(actual_device)
        
        # Generate dummy inputs with explicit device
        dummy_inputs = onnx_config.generate_dummy_inputs(device=device_to_use, batch_size=batch_size, seq_length=seq_length)
        
        # Check devices of dummy inputs
        for name, tensor in dummy_inputs.items():
            logger.info(f"Dummy input '{name}' is on device: {tensor.device}")
        
        # Export with verified device settings
        export(
            model=model,
            config=onnx_config,
            output=onnx_path,
            opset=onnx_config.default_onnx_opset_version,
            device=device_to_use,  # Use the verified device
            input_shapes={
                "input_ids": {0: batch_size, 1: seq_length},
                "attention_mask": {0: batch_size, 1: seq_length}
            }
        )
        
        # Save tokenizer & config
        tokenizer.save_pretrained(onnx_dir)
        config.save_pretrained(onnx_dir)
        logger.info(f"Model exported to ONNX in {time.time() - start_time:.2f} seconds")
        
        # Optionally save a FP16 version
        if save_fp16:
            import onnx
            from onnxmltools.utils.float16_converter import convert_float_to_float16
            logger.info("Converting model to FP16...")
            fp16_dir = Path(str(onnx_dir) + "_fp16")
            fp16_dir.mkdir(parents=True, exist_ok=True)
            fp16_path = fp16_dir / "model.onnx"
            
            # Load and convert model
            onnx_model = onnx.load(onnx_path)
            onnx_model = convert_float_to_float16(onnx_model)
            
            # Check if the version supports save_as_external_data
            import inspect
            save_params = inspect.signature(onnx.save).parameters
            
            # Save with appropriate parameters
            if "save_as_external_data" in save_params:
                onnx.save(onnx_model, fp16_path, save_as_external_data=True)
            else:
                # For older onnx versions
                onnx.save(onnx_model, fp16_path)
            
            # Save tokenizer & config
            tokenizer.save_pretrained(fp16_dir)
            config.save_pretrained(fp16_dir)
            logger.info(f"FP16 model saved to {fp16_dir}")
        
        # Validate the exported model
        if validate:
            logger.info("Validating exported ONNX model...")
            validation_result = validate_model(onnx_path, tokenizer, model, device)
            if not validation_result:
                logger.warning("ONNX model validation failed or showed significant differences")
            else:
                logger.info("ONNX model validation successful")
            
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise
    
    # 4. Quantize the model
    logger.info(f"Quantizing model to {quantized_dir}")
    try:
        quantization_start_time = time.time()
        
        # Determine the best providers based on hardware
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider = "CUDAExecutionProvider"
        else:
            providers = ["CPUExecutionProvider"]
            provider = "CPUExecutionProvider"
        
        # Flag to try direct ONNX quantization if optimum fails
        try_direct_onnx_quantization = False
        
        # Create appropriate quantization configuration based on available methods
        try:
            import platform
            
            # Try to detect if we're on an AVX512-capable CPU 
            has_avx512 = False
            try:
                cpu_info = platform.processor()
                
                if hasattr(platform, "cpuinfo") and hasattr(platform.cpuinfo, "is_avx512_compatible"):
                    has_avx512 = platform.cpuinfo.is_avx512_compatible()
                # Fall back to checking /proc/cpuinfo on Linux
                elif os.path.exists('/proc/cpuinfo'):
                    with open('/proc/cpuinfo', 'r') as f:
                        cpu_info_text = f.read()
                    has_avx512 = 'avx512' in cpu_info_text.lower()
            except Exception as e:
                logger.warning(f"Error detecting CPU features: {e}")
                has_avx512 = False
                
            # Try different methods in order of preference
            if has_avx512:
                logger.info("Detected AVX512 support, using optimized configuration")
                quant_config = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)
            elif hasattr(AutoQuantizationConfig, "default"):
                logger.info("Using default quantization configuration")
                quant_config = AutoQuantizationConfig.default(is_static=False, per_channel=True)
            elif hasattr(AutoQuantizationConfig, "arm64"):
                logger.info("Using ARM64 quantization configuration")
                quant_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
            elif hasattr(AutoQuantizationConfig, "avx2"):
                logger.info("Using AVX2 quantization configuration")
                quant_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
            else:
                # If none of the specific methods are available, create a basic config
                from optimum.onnxruntime.configuration import QuantizationConfig
                logger.info("Creating basic quantization configuration (no specific optimizations available)")
                quant_config = QuantizationConfig(
                    is_static=False,
                    per_channel=True,
                    reduce_range=False,
                    operators_to_quantize=["MatMul", "Attention"]
                )
        except Exception as e:
            logger.error(f"Failed to create quantization configuration: {e}")
            raise
        
        # Create the quantizer with version-compatible parameters
        try:
            # Check which parameters are supported
            import inspect
            quantizer_params = inspect.signature(ORTQuantizer.from_pretrained).parameters
            
            # Build parameters dict based on what's supported
            init_params = {}
            if "onnx_model_path" in quantizer_params:
                # Older versions use this name
                init_params["onnx_model_path"] = onnx_dir
            else:
                # Newer versions use this or just the path
                init_params["model"] = onnx_dir
                
            # Only add providers if supported
            if "providers" in quantizer_params:
                init_params["providers"] = providers
                
            logger.info(f"Initializing quantizer with params: {init_params}")
            quantizer = ORTQuantizer.from_pretrained(**init_params)
            
        except Exception as e:
            logger.error(f"Failed to initialize quantizer: {e}")
            logger.warning("Trying fallback to direct ONNX quantization...")
            try_direct_onnx_quantization = True
        
        # Check if the version supports use_external_data_format
        if not try_direct_onnx_quantization:
            import inspect
            quantize_params = inspect.signature(quantizer.quantize).parameters
            
            # Prepare quantization parameters
            quantize_kwargs = {
                "quantization_config": quant_config,
                "save_dir": quantized_dir,
            }
            
            # Add use_external_data_format if supported
            if "use_external_data_format" in quantize_params:
                quantize_kwargs["use_external_data_format"] = True
                logger.info("Using external data format for quantization")
            else:
                logger.info("External data format not supported in this version of optimum")
                
            # Quantize the model with appropriate parameters and better error handling
            try:
                logger.info("Starting quantization process...")
                quantizer.quantize(**quantize_kwargs)
                logger.info("Quantization completed successfully")
            except Exception as e:
                logger.error(f"Quantization failed: {e}")
                
                # Try with more basic settings
                error_str = str(e)
                if "operators_to_quantize" in error_str or "per_channel" in error_str:
                    logger.warning("Trying again with basic quantization settings...")
                    from optimum.onnxruntime.configuration import QuantizationConfig
                    basic_config = QuantizationConfig(
                        is_static=False,
                        per_channel=False,  # Use simpler per-tensor quantization
                        reduce_range=True,  # Sometimes helps with compatibility
                    )
                    
                    try:
                        # Update kwargs with new config
                        quantize_kwargs["quantization_config"] = basic_config
                        quantizer.quantize(**quantize_kwargs)
                        logger.info("Quantization succeeded with basic settings")
                    except Exception as retry_e:
                        logger.error(f"Basic quantization also failed: {retry_e}")
                        try_direct_onnx_quantization = True
                else:
                    try_direct_onnx_quantization = True
        
        # Fallback to direct ONNX quantization if needed
        if try_direct_onnx_quantization:
            logger.warning("Attempting fallback to direct ONNX quantization...")
            try:
                import onnx
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                model_path = str(onnx_dir / "model.onnx")
                output_path = str(quantized_dir / "model.onnx")
                
                # Make sure the directory exists
                os.makedirs(quantized_dir, exist_ok=True)
                
                # Copy tokenizer and config files
                if os.path.exists(onnx_dir / "tokenizer_config.json"):
                    import shutil
                    for file in os.listdir(onnx_dir):
                        if file != "model.onnx" and not file.endswith(".onnx_data"):
                            src = onnx_dir / file
                            dst = quantized_dir / file
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                
                # Check which parameters are supported by quantize_dynamic
                import inspect
                quant_params = inspect.signature(quantize_dynamic).parameters
                
                # Build parameters dict based on what's supported
                quant_args = {
                    "model_input": model_path,
                    "model_output": output_path,
                    "per_channel": False,
                    "reduce_range": True,
                    "weight_type": QuantType.QUInt8
                }
                
                # Add optimize_model if supported
                if "optimize_model" in quant_params:
                    quant_args["optimize_model"] = True
                
                # Most importantly - add use_external_data_format for large models
                quant_args["use_external_data_format"] = True
                    
                # Perform direct quantization with compatible parameters
                logger.info("Quantizing with external data format to handle large model size")
                quantize_dynamic(**quant_args)
                
                logger.info("Direct ONNX quantization completed successfully")
                
            except Exception as onnx_e:
                logger.error(f"Direct ONNX quantization also failed: {onnx_e}")
                raise
        
        logger.info(f"Model quantized in {time.time() - quantization_start_time:.2f} seconds")
        
        # Validate the quantized model
        if validate:
            logger.info("Validating quantized model...")
            validation_result = validate_model(quantized_dir / "model.onnx", tokenizer, model, device)
            if not validation_result:
                logger.warning("Quantized model validation failed or showed significant differences")
            else:
                logger.info("Quantized model validation successful")
            
    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        raise
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    return {
        "onnx_dir": onnx_dir,
        "quantized_dir": quantized_dir,
        "total_time_seconds": total_time
    }

def main():
    """Main function to parse arguments and run the export and quantization."""
    parser = argparse.ArgumentParser(description="Export and quantize Phi-3 model to ONNX")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-3-mini-4k-instruct",
                        help="Hugging Face model ID")
    parser.add_argument("--onnx_dir", type=str, default="./phi3_mini_onnx",
                        help="Path to save the ONNX model")
    parser.add_argument("--quantized_dir", type=str, default="./phi3_mini_quantized",
                        help="Path to save the quantized model")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for dummy inputs")
    parser.add_argument("--seq_length", type=int, default=4,
                        help="Sequence length for dummy inputs")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Use CPU only for export and quantization")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage even if GPU is available (stronger than --cpu_only)")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Optimize memory usage")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip validation")
    parser.add_argument("--save_fp16", action="store_true",
                        help="Save an additional FP16 model")
    
    args = parser.parse_args()
    
    onnx_dir = Path(args.onnx_dir)
    quantized_dir = Path(args.quantized_dir)
    
    # Use either cpu_only or force_cpu
    use_cpu = args.cpu_only or args.force_cpu
    
    export_and_quantize_phi3(
        model_id=args.model_id,
        onnx_dir=onnx_dir,
        quantized_dir=quantized_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        use_gpu=not use_cpu,
        optimize_memory=args.optimize_memory,
        validate=not args.no_validate,
        save_fp16=args.save_fp16,
        force_cpu=args.force_cpu
    )

if __name__ == "__main__":
    main()
