#!/usr/bin/env python3
import json

def print_results_table(json_file: str):
    # Leggiamo il file JSON
    with open(json_file, "r") as f:
        data = json.load(f)  # data è una lista di dizionari

    # PRIMA TABELLA: dati "serial" e "cuda"
    print(" MAIN RESULTS (SERIAL / CUDA) ".center(80, "="))
    header = f"{'Matrix Name':30} | {'SerTime':>10} | {'SerFLOPS':>10} | {'CudaTime':>10} | {'CudaFLOPS':>10}"
    print(header)
    print("-" * len(header))

    for mat in data:
        matrix_name = mat.get("matrix_name", "")

        ser_time  = mat["serial"]["time"]
        ser_flops = mat["serial"]["flops"]

        # Questi due campi (cuda_time, cuda_flops) rimangono a 0.0
        # perché nel codice C non viene assegnato nulla a results[i].cuda.time e flops
        cuda_time  = mat["cuda"]["time"]
        cuda_flops = mat["cuda"]["flops"]

        line = f"{matrix_name:30} | {ser_time:10.5f} | {ser_flops:10.5f} | {cuda_time:10.5f} | {cuda_flops:10.5f}"
        print(line)

    # SECONDA TABELLA: dati OpenMP (array "openmp")
    print("\n OPENMP RESULTS ".center(80, "="))
    header_omp = f"{'Matrix Name':30} | {'Threads':>7} | {'Time':>10} | {'FLOPS':>10}"
    print(header_omp)
    print("-" * len(header_omp))

    for mat in data:
        matrix_name = mat.get("matrix_name", "")
        # In JSON si chiama "openmp" o "openmp_results"? Dipende da come l’hai chiamato nel writer
        # Se l’hai chiamato "openmp" allora mat.get("openmp", [])
        openmp_list = mat.get("openmp", [])
        for omp in openmp_list:
            threads = omp["threads"]
            time    = omp["time"]
            flops   = omp["flops"]
            line = f"{matrix_name:30} | {threads:7d} | {time:10.5f} | {flops:10.5f}"
            print(line)

    # TERZA TABELLA: dati CUDA CSR
    print("\n CUDA CSR KERNELS ".center(80, "="))
    header_csr = f"{'Matrix Name':30} | {'KernelName':>30} | {'Time':>10} | {'GFLOPS':>10}"
    print(header_csr)
    print("-" * len(header_csr))

    for mat in data:
        matrix_name = mat.get("matrix_name", "")
        cuda_csr_list = mat.get("cuda_csr", [])  # array di kernel
        for kernel in cuda_csr_list:
            kname  = kernel.get("kernel_name", "")
            ktime  = kernel.get("time", 0.0)
            kgflops= kernel.get("gflops", 0.0)
            line = f"{matrix_name:30} | {kname:30} | {ktime:10.5f} | {kgflops:10.5f}"
            print(line)

    # QUARTA TABELLA: dati CUDA HLL
    print("\n CUDA HLL KERNELS ".center(80, "="))
    header_hll = f"{'Matrix Name':30} | {'KernelName':>20} | {'Time':>10} | {'GFLOPS':>10}"
    print(header_hll)
    print("-" * len(header_hll))

    for mat in data:
        matrix_name = mat.get("matrix_name", "")
        cuda_hll_list = mat.get("cuda_hll", [])  # array di kernel
        for kernel in cuda_hll_list:
            kname  = kernel.get("kernel_name", "")
            ktime  = kernel.get("time", 0.0)
            kgflops= kernel.get("gflops", 0.0)
            line = f"{matrix_name:30} | {kname:20} | {ktime:10.5f} | {kgflops:10.5f}"
            print(line)

if __name__ == "__main__":
    print_results_table("../build/results.json")
