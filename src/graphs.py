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

    # -------- GRAFICO KERNEL 4 CUDA_HLL vs KERNEL 5 CUDA_CSR --------
    matrix_names = []
    gflops_cuda_hll = []
    gflops_cuda_csr = []

    for matrix in data:
        matrix_names.append(matrix['matrix_name'])

        # Kernel 4 di cuda_hll
        if len(matrix["cuda_hll"]) >= 4:
            gflops_cuda_hll.append(matrix["cuda_hll"][3]["gflops"])  # Kernel 4 (indice 3)
        else:
            gflops_cuda_hll.append(0)

        # Kernel 5 di cuda_csr
        if len(matrix["cuda_csr"]) >= 5:
            gflops_cuda_csr.append(matrix["cuda_csr"][4]["gflops"])  # Kernel 5 (indice 4)
        else:
            gflops_cuda_csr.append(0)

    x = np.arange(len(matrix_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, gflops_cuda_hll, width, label="Kernel 4 CUDA_HLL", color="blue")
    ax.bar(x + width/2, gflops_cuda_csr, width, label="Kernel 5 CUDA_CSR", color="orange")

    ax.set_title("Confronto Kernel 4 CUDA_HLL vs Kernel 5 CUDA_CSR", fontsize=16)
    ax.set_xlabel("Matrice", fontsize=14)
    ax.set_ylabel("GFLOPS", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_names, fontsize=12, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import json
import numpy as np


# Categorie dei range di zeri
zero_bins = ["<10K", "10K-100K", "100K-500K", "500K-1M", "1M-2.5M", "2.5M-10M", "≥10M"]
thread_counts = [2, 4, 8, 16, 32, 40]  # Thread disponibili per OpenMP

def categorize_zeros(num_zeros):
    """Assegna il numero di zeri a una delle 7 categorie definite."""
    if num_zeros < 10000:
        return "<10K"
    elif 10000 <= num_zeros < 100000:
        return "10K-100K"
    elif 100000 <= num_zeros < 500000:
        return "100K-500K"
    elif 500000 <= num_zeros < 1000000:
        return "500K-1M"
    elif 1000000 <= num_zeros < 2500000:
        return "1M-2.5M"
    elif 2500000 <= num_zeros < 10000000:
        return "2.5M-10M"
    else:
        return "≥10M"

def plot_metric_vs_threads(json_file, category, metric):
    """
    Plotta la media di una metrica (GFLOPS, speedup o efficienza) al variare del numero di thread per ogni range di zeri.
    
    :param json_file: File JSON contenente i dati
    :param category: "openmp_csr" o "openmp_hll"
    :param metric: "flops", "speedup" o "efficienza"
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    # Dizionario per accumulare tutti i valori della metrica per ogni range di zeri e numero di thread
    metric_data = {zero_range: {t: [] for t in thread_counts} for zero_range in zero_bins}

    # Analizza ogni matrice nel JSON
    for matrix in data:
        num_zeros = matrix.get("NZ", 0)  # Prende il numero di zeri
        zero_category = categorize_zeros(num_zeros)

        # Estrai i valori della metrica per la configurazione OpenMP specificata
        if category in matrix:
            for run in matrix[category]:
                threads = run.get("threads")
                value = run.get(metric)
                if threads in thread_counts and value is not None:
                    metric_data[zero_category][threads].append(value)

    # Calcola la media della metrica per ogni range e numero di thread (corretto!)
    avg_metric = {
        zero_range: [
            np.mean(metric_data[zero_range][t]) if metric_data[zero_range][t] else 0
            for t in thread_counts
        ]
        for zero_range in zero_bins
    }

    # Plot dei dati
    plt.figure(figsize=(10, 6))
    for zero_range in zero_bins:
        plt.plot(thread_counts, avg_metric[zero_range], marker='o', linestyle='-', label=zero_range)

    metric_label = {"flops": "GFLOPS", "speedup": "Speedup", "efficienza": "Efficienza"}[metric]
    plt.title(f'{metric_label} medi al variare dei thread per {category.upper()}')
    plt.xlabel('Numero di thread')
    plt.ylabel(f'{metric_label} medi')
    plt.legend(title="Range di zeri")
    plt.grid(True)
    plt.xscale("log", base=2)  # Scala logaritmica per evidenziare il trend
    plt.xticks(thread_counts, thread_counts)  # Mostra solo i valori effettivi dei thread
    plt.show()

# Esegui la funzione per OpenMP CSR e OpenMP HLL separatamente per ogni metrica
plot_metric_vs_threads("results.json", "openmp_csr", "flops")       # GFLOPS OpenMP CSR
plot_metric_vs_threads("results.json", "openmp_csr", "speedup")     # Speedup OpenMP CSR
plot_metric_vs_threads("results.json", "openmp_csr", "efficienza")  # Efficienza OpenMP CSR

plot_metric_vs_threads("results.json", "openmp_hll", "flops")       # GFLOPS OpenMP HLL
plot_metric_vs_threads("results.json", "openmp_hll", "speedup")     # Speedup OpenMP HLL
plot_metric_vs_threads("results.json", "openmp_hll", "efficienza")  # Efficienza OpenMP HLL



# Esegui la funzione
plot_gflops_and_metrics_from_json("results.json")



# Categorie dei range di zeri
zero_bins = ["<10K", "10K-100K", "100K-500K", "500K-1M", "1M-2.5M", "2.5M-10M", "≥10M"]

def categorize_zeros(num_zeros):
    """Assegna il numero di zeri a una delle 7 categorie definite."""
    if num_zeros < 10000:
        return "<10K"
    elif 10000 <= num_zeros < 100000:
        return "10K-100K"
    elif 100000 <= num_zeros < 500000:
        return "100K-500K"
    elif 500000 <= num_zeros < 1000000:
        return "500K-1M"
    elif 1000000 <= num_zeros < 2500000:
        return "1M-2.5M"
    elif 2500000 <= num_zeros < 10000000:
        return "2.5M-10M"
    else:
        return "≥10M"

def group_matrices_by_zeros(json_file):
    """Gruppo i nomi delle matrici per range di zeri."""
    with open(json_file, "r") as f:
        data = json.load(f)

    # Dizionario per accumulare i nomi delle matrici per range di zeri
    matrix_names_by_zero_range = {zero_range: [] for zero_range in zero_bins}

    # Raccogli i nomi delle matrici per categoria di zeri
    for matrix in data:
        num_zeros = matrix.get("NZ", 0)  # Prende il numero di zeri
        zero_category = categorize_zeros(num_zeros)
        
        # Aggiungi il nome della matrice alla categoria appropriata
        matrix_names_by_zero_range[zero_category].append(matrix['matrix_name'])

    # Stampa i risultati
    for zero_range, matrix_names in matrix_names_by_zero_range.items():
        print(f"Range di zeri {zero_range}:")
        print(matrix_names)
        print()  # Linea vuota tra le categorie

# Esegui la funzione per stampare i risultati
group_matrices_by_zeros("results.json")





