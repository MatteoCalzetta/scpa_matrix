import matplotlib.pyplot as plt
import json
import numpy as np

def plot_gflops_and_metrics_from_json(json_file):
    # Carica i dati dal file JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    categories = ['serial_csr', 'serial_hll', 'cuda_csr', 'cuda_hll', 'openmp_csr', 'openmp_hll']
    
    for category in categories:
        matrix_names = []
        gflops_data = []

        # Crea una lista per i grafici
        for matrix in data:
            matrix_names.append(matrix['matrix_name'])

            if category in ['serial_csr', 'serial_hll']:
                gflops_data.append([matrix[category]['flops']])
            elif category in ['cuda_csr', 'cuda_hll']:
                gflops_data.append([kernel['gflops'] for kernel in matrix[category]])
            elif category in ['openmp_csr', 'openmp_hll']:
                gflops_data.append([item['flops'] for item in matrix[category]])

        # Numero massimo di kernel/thread
        num_matrices = len(matrix_names)
        max_kernels = max(len(kernels) for kernels in gflops_data) if gflops_data else 1  

        # Larghezza delle barre e posizioni
        width = 0.15
        x = np.arange(num_matrices)

        # Creazione della figura
        fig, ax = plt.subplots(figsize=(12, 8))

        # Assegna le etichette della leggenda
        if category in ['serial_csr', 'serial_hll']:
            legend_labels = ["Seriale"]
        elif category in ['cuda_csr', 'cuda_hll']:
            legend_labels = [f'Kernel {i+1}' for i in range(max_kernels)]
        elif category in ['openmp_csr', 'openmp_hll']:
            thread_counts = [2, 4, 8, 16, 32, 40]
            legend_labels = [f"{t} thread" for t in thread_counts[:max_kernels]]

        # Posizionamento delle barre
        for i in range(max_kernels):
            kernel_gflops = [gflops[i] if i < len(gflops) else 0 for gflops in gflops_data]
            ax.bar(x + i * width, kernel_gflops, width, label=legend_labels[i])
        
        # Etichette e titolo
        ax.set_title(f'GFLOPS per {category.upper()}', fontsize=16)
        ax.set_xlabel('Matrice', fontsize=14)
        ax.set_ylabel('GFLOPS', fontsize=14)
        ax.set_xticks(x + (max_kernels - 1) * width / 2)
        ax.set_xticklabels(matrix_names, fontsize=12, rotation=45, ha="right")
        ax.legend(title='Configurazione', fontsize=10)

        # Layout e visualizzazione
        plt.tight_layout()
        plt.show()

    # -------- PLOTTING SPEEDUP & EFFICIENCY --------
    
    def plot_metric(metric_name, category):
        matrix_names = []
        metric_data = []

        for matrix in data:
            matrix_names.append(matrix['matrix_name'])
            metric_data.append([item[metric_name] for item in matrix[category]])

        num_matrices = len(matrix_names)
        num_threads = max(len(metric) for metric in metric_data)

        width = 0.15
        x = np.arange(num_matrices)

        fig, ax = plt.subplots(figsize=(12, 8))

        thread_counts = [2, 4, 8, 16, 32, 40]
        legend_labels = [f"{t} thread" for t in thread_counts[:num_threads]]

        for i in range(num_threads):
            thread_values = [values[i] if i < len(values) else 0 for values in metric_data]
            ax.bar(x + i * width, thread_values, width, label=legend_labels[i])

        ax.set_title(f'{metric_name.capitalize()} per {category.upper()}', fontsize=16)
        ax.set_xlabel('Matrice', fontsize=14)
        ax.set_ylabel(metric_name.capitalize(), fontsize=14)
        ax.set_xticks(x + (num_threads - 1) * width / 2)
        ax.set_xticklabels(matrix_names, fontsize=12, rotation=45, ha="right")
        ax.legend(title='Thread', fontsize=10)

        plt.tight_layout()
        plt.show()

    # Plotta Speedup ed Efficienza per OpenMP
    plot_metric("speedup", "openmp_csr")
    plot_metric("efficienza", "openmp_csr")
    plot_metric("speedup", "openmp_hll")
    plot_metric("efficienza", "openmp_hll")

# Esegui la funzione
plot_gflops_and_metrics_from_json("results.json")
