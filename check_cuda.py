import subprocess

# Check Cuda Version
cuda_version = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE).stdout.decode('utf-8')
assert 'cuda_11.6' in cuda_version, "Doesn't use CUDA 11.6"
