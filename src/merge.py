import json

# Carica i due file JSON (usa i tuoi path)
with open("matrixperf.txt", "r") as f1:
    data1 = json.load(f1)

with open("matrixperf2.txt", "r") as f2:
    data2 = json.load(f2)

# Convertiamo le liste in dizionari indicizzati per matrix_name
dict1 = {item["matrix_name"]: item for item in data1}
dict2 = {item["matrix_name"]: item for item in data2}

# Per comodità, creiamo una struttura di output partendo dal primo dizionario
merged_dict = dict(dict1)  # copia

# Fondere i dati di matrixperf2 in merged_dict
for mtx_name, item2 in dict2.items():
    if mtx_name not in merged_dict:
        # Se non esiste affatto nel primo file, aggiungilo direttamente
        merged_dict[mtx_name] = item2
    else:
        # Se esiste già, andiamo a unire i kernel di cuda_hll
        # 1) Recupera la lista "cuda_hll" esistente
        item1_cuda_hll = merged_dict[mtx_name].get("cuda_hll", [])
        # 2) Creiamo una mappa { kernel_name: dict } per cercare rapidamente duplicati
        cuda_hll_map = {entry["kernel_name"]: entry for entry in item1_cuda_hll}

        # 3) Iteriamo sui kernel di matrixperf2 e aggiungiamo solo se mancano
        item2_cuda_hll = item2.get("cuda_hll", [])
        for hll_entry in item2_cuda_hll:
            kname = hll_entry["kernel_name"]
            if kname not in cuda_hll_map:
                item1_cuda_hll.append(hll_entry)
                cuda_hll_map[kname] = hll_entry

        # 4) Salviamo la nuova lista unita in merged_dict
        merged_dict[mtx_name]["cuda_hll"] = item1_cuda_hll

# Ora abbiamo un dizionario con tutti i dati. Se vuoi una lista finale:
merged_list = list(merged_dict.values())

# Se vuoi salvare su file:
with open("matrixperf_merged.json", "w") as f:
    json.dump(merged_list, f, indent=2)

print("Merge completato! Risultato in 'matrixperf_merged.json'.")