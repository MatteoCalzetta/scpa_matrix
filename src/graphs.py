import matplotlib.pyplot as plt
import json
import numpy as np

def plot_gflops_from_json(json_file):
    # Carica i dati dal file JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    categories = ['serial', 'cuda_csr', 'cuda_hll', 'openmp_csr', 'openmp_hll']
    
    for category in categories:
        matrix_names = []
        gflops_data = []

        # Crea una lista per i grafici
        for matrix in data:
            matrix_names.append(matrix['matrix_name'])

            if category == 'serial':
                gflops_data.append([matrix['serial']['flops']])
            elif category == 'cuda_csr':
                gflops_data.append([kernel['gflops'] for kernel in matrix['cuda_csr']])
            elif category == 'cuda_hll':
                gflops_data.append([kernel['gflops'] for kernel in matrix['cuda_hll']])
            elif category == 'openmp_csr':
                gflops_data.append([item['flops'] for item in matrix['openmp_csr']])
            elif category == 'openmp_hll':
                gflops_data.append([item['flops'] for item in matrix['openmp_hll']])

        # Crea un grafico con barre multiple
        num_matrices = len(matrix_names)  # Numero di matrici
        max_kernels = max(len(kernels) for kernels in gflops_data)  # Numero massimo di kernel per matrice
        
        # Larghezza delle barre e posizioni
        width = 0.15
        x = np.arange(num_matrices)

        # Creazione della figura e degli assi
        fig, ax = plt.subplots(figsize=(12, 8))

        # Posizionamento delle barre per ciascun kernel
        for i in range(max_kernels):
            kernel_gflops = [gflops[i] if i < len(gflops) else 0 for gflops in gflops_data]
            ax.bar(x + i * width, kernel_gflops, width, label=f'Kernel {i + 1}')
        
        # Etichette e titolo
        ax.set_title(f'GFLOPS per {category.upper()}', fontsize=16)
        ax.set_xlabel('Matrice', fontsize=14)
        ax.set_ylabel('GFLOPS', fontsize=14)
        ax.set_xticks(x + (max_kernels - 1) * width / 2)  # Centra le etichette delle matrici
        ax.set_xticklabels(matrix_names, fontsize=12, rotation=45, ha="right")
        ax.legend(title='Kernels', fontsize=10)

        # Layout e visualizzazione
        plt.tight_layout()
        plt.show()

# Esegui la funzione passando il tuo file JSON
plot_gflops_from_json("results.json")



def plot_gflops_from_json2(json_file):
    with open("results.json", "r") as f:
        data = json.load(f)
    
    categories = {"serial": [], "cuda_csr": [], "cuda_hll": [], "openmp": []}
    gflops_values = {"serial": [], "cuda_csr": [], "cuda_hll": [], "openmp": []}
    matrix_names = []
    
    for matrix in data:
        matrix_names.append(matrix["matrix_name"])
        
        # Serial
        categories["serial"].append("serial")
        gflops_values["serial"].append(matrix["serial"].get("flops", 0))
        
        # CUDA CSR
        for kernel in matrix.get("cuda_csr", []):
            categories["cuda_csr"].append(kernel["kernel_name"])
            gflops_values["cuda_csr"].append(kernel["gflops"])
        
        # CUDA HLL
        for kernel in matrix.get("cuda_hll", []):
            categories["cuda_hll"].append(kernel["kernel_name"])
            gflops_values["cuda_hll"].append(kernel["gflops"])
        
        # OpenMP
        for kernel in matrix.get("openmp", []):
            categories["openmp"].append(f"openmp_{kernel['threads']}t")
            gflops_values["openmp"].append(kernel["flops"])
    
    titles = {"serial": "Serial", "cuda_csr": "CUDA CSR", "cuda_hll": "CUDA HLL", "openmp": "OpenMP"}
    
    for key in ["serial", "cuda_csr", "cuda_hll", "openmp"]:
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(categories[key]))
        plt.bar(x_pos, gflops_values[key], color='blue')
        plt.xticks(x_pos, categories[key], rotation=90)
        plt.xlabel("Categoria")
        plt.ylabel("GFLOPS")
        plt.title(titles[key])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()



def plot_gflops_from_json3(json_file):
    # Carica i dati dal file JSON
    with open("results.json", "r") as f:
        data = json.load(f)

    categories = ['serial', 'cuda_csr', 'cuda_hll', 'openmp']
    
    for category in categories:
        matrix_names = []
        gflops_data = []
        
        # Crea una lista per i grafici
        for matrix in data:
            matrix_names.append(matrix['matrix_name'])
            
            if category == 'serial':
                gflops_data.append([matrix['serial']['flops']])
            elif category == 'cuda_csr':
                gflops_data.append([kernel['gflops'] for kernel in matrix['cuda_csr']])
            elif category == 'cuda_hll':
                gflops_data.append([kernel['gflops'] for kernel in matrix['cuda_hll']])
            elif category == 'openmp':
                gflops_data.append([item['flops'] for item in matrix['openmp']])
        
        # Definisci il numero di kernel per ogni matrice
        num_kernels = max(len(gflops) for gflops in gflops_data)  # Trova il massimo numero di kernel

        # Crea il grafico
        fig, ax = plt.subplots(figsize=(12, 8))  # Imposta la dimensione del grafico
        x = np.arange(len(matrix_names))  # Posizioni delle barre

        # Per ogni matrice, crea una barra per ciascun kernel
        width = 0.15  # Larghezza di ogni barra per il kernel
        offset = (num_kernels - 1) * width / 2  # Per centrare le barre per ogni matrice

        for i, gflops in enumerate(gflops_data):
            ax.bar(x + (i - (len(gflops) - 1) / 2) * width, gflops, width, label=matrix_names[i])
        
        # Etichette e titolo
        ax.set_title(f'GFLOPS per {category.upper()}', fontsize=16)
        ax.set_xlabel('Matrice', fontsize=14)
        ax.set_ylabel('GFLOPS', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_names, fontsize=12, rotation=45, ha="right")
        ax.legend(title='Kernels', fontsize=10)

        # Visualizza il grafico
        plt.tight_layout()  # Impedisce che le etichette si sovrappongano
        plt.show()

# Esegui la funzione passando il tuo file JSON
plot_gflops_from_json("results.json")

