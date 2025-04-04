Repository dedicata al prodotto matrice sparsa per vettore (SpMV) per il corso SCPA.

Per runnare:
1) in /home/utente/scpa_matrix eseguire "mkdir build"
2) cd build
3) mkdir test_matrix
4) copiare le matirci in test_matrix usando scp da locale a ssh o cp se già presenti su macchina locale
5) cmake ..
6) make -j$(nproc)
7) ./progetto.out