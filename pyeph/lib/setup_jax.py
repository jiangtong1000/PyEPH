"""
JAX backend configuration utilities
"""
import os
import jax
import psutil

# Enable 64-bit precision immediately when this module is imported
os.environ['JAX_ENABLE_X64'] = '1'
jax.config.update("jax_enable_x64", True)

def configure_jax_backend(verbose=True):
    """
    Configure JAX based on hardware.
    Returns:
        dict: Configuration summary
    """
    
    if verbose:
        print("Configuring JAX backend...", flush=True)
    
    backend = jax.default_backend()
    devices = jax.devices()
    
    config_info = {
        'backend': backend,
        'device_count': len(devices),
        'devices': [str(d) for d in devices]
    }
    if verbose:
        print(f"JAX backend: {backend}")
        print(f"JAX devices: {devices}")
    if backend == 'gpu':
        raise ValueError("GPU backend not supported yet")
    elif backend == 'cpu':
        cpu_config = _configure_cpu(verbose)
        config_info.update(cpu_config)
        config_info['optimization'] = 'cpu'
    else:
        raise ValueError(f"Backend '{backend}' not checked yet")
    return config_info

def _configure_cpu(verbose=True):
    """Configure JAX for high-performance CPU training."""
    cpu_count = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count))
    effective_cpus = min(slurm_cpus, cpu_count)
    
    # these are to let jax xla handle everything, avoid conflict
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    if 'XLA_FLAGS' not in os.environ:
        os.environ['XLA_FLAGS'] = ''
    
    xla_flags = [
        '--xla_cpu_multi_thread_eigen=true',
        '--xla_cpu_enable_xla_runtime_executable=true',
        f'--xla_force_host_platform_device_count={effective_cpus}'
    ]
    
    for flag in xla_flags:
        if flag.split('=')[0] not in os.environ['XLA_FLAGS']:
            os.environ['XLA_FLAGS'] += f' {flag}'
    
    if verbose:
        print(f"CPU configuration: {effective_cpus} CPUs available")
        print(f"BLAS threading disabled for JAX parallelization")
        print(f"XLA_FLAGS: {os.environ['XLA_FLAGS']}")
    
    return {
        'cpu_count': cpu_count,
        'physical_cores': physical_cores,
        'slurm_cpus': slurm_cpus,
        'effective_cpus': effective_cpus,
        'threading_mode': 'jax_only'
    }

def get_memory_info():
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent
    }
    
def setup_jax_environment(verbose=True):
    """Complete JAX setup with config and memory reporting."""
    config_info = configure_jax_backend(verbose=verbose)
    memory_info = get_memory_info()

    if verbose:
        print(f"Memory: {memory_info['available_gb']:.1f}GB available / {memory_info['total_gb']:.1f}GB total", flush=True)

    return config_info, memory_info