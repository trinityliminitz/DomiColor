"""
    KELOMPOK 13 | Proyek Pembelajaran Mesin
    - Timothy Sipahutar 11S19016
    - Edrei Siregar 11S19019
    - Judah Sitorus 11S19040
    - Kevin Sihaloho 11S19044
    - Andreas Pakpahan 11S19047 
"""

# Package yang diperlukan dalam penghitungan DomiColor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils

# Package yang diperlukan dalam pembuatan GUI dan akses direktori
import PySimpleGUI as sg
import os.path

# Algoritma DomiColor
def domi_color(file_path):
    # Mendefinisikan jumlah cluster untuk algoritma KMeans
    clusters = 10

    # Membaca input gambar
    img = cv2.imread(file_path)
    org_img = img.copy() # salinan gambar
    
    # Menampilkan dimensi gambar
    print('Org image shape --> ',img.shape)

    # Mengkonversi ukuran gambar dan menampilkan dimensi hasilnya
    img = imutils.resize(img,height=200)
    print('After resizing shape --> ',img.shape)

    # Meratakan gambar menjadi satu kolom sepanjang jumlah pixels gambar
    flat_img = np.reshape(img,(-1,3))
    print('After Flattening shape --> ',flat_img.shape)
    
    # Menyiapkan K-Means Clustering dengan jumlah clusters yang telah dideklarasikan sebelumnya
    kmeans = KMeans(n_clusters=clusters,random_state=0)
    
    """
    Gambar yang telah diratakan kemudian disesuaikan sebagai larik yang berisi semua warna pada pixels gambar 
    yang dimana akan dikelompokan sebanyak jumlah clusters yang telah dideklarasikan
    dan kelompok ini akan dianggap sebagai warna utama/dominan pada gambar
    """
    kmeans.fit(flat_img)
    
    # Mengambil warna utama/dominan pada gambar
    dominant_colors = np.array(kmeans.cluster_centers_,dtype='uint')

    """
    Berdasarkan warna utama/dominan pada gambar, setiap kelompok akan dihitung persentase kemunculannya dimana;
        np.unique(kmeans.labels_,return_counts=True
    akan mengembalikan array dua dimensi; dimensi pertama menjelaskan setiap pixel merujuk pada suatu kelompok, dan dimensi kedua menunjukan jumlah pixel yang berada pada suatu kelompok
    pada kasus ini, digunakan dimensi kedua dan dibagi oleh jumlah pixels gambar keseluruhan dan menghasilkan persentase pada setiap kelompok
    """
    print(np.unique(kmeans.labels_,return_counts=True)[1])
    percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]
    
    # Menggabungkan (zip) persentase dengan kode warna setiap kelompok
    p_and_c = zip(percentages,dominant_colors)
    
    # Mengurutkan objek zip secara menurun (descending) sehingga elemen pertama adalah kelompok yang memiliki persentase tertinggi (warna terbanyak)
    p_and_c = sorted(p_and_c,reverse=True)

    # Membuat blocks-plot dari kelompok warna
    block = np.ones((50,50,3),dtype='uint')
    plt.figure(figsize=(12,8))
    for i in range(clusters):
        plt.subplot(1,clusters,i+1)
        block[:] = p_and_c[i][1][::-1] # Konversi bgr(opencv) menjadi rgb(matplotlib) 
        plt.imshow(block)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(round(p_and_c[i][0]*100,2))+'%')

    # Membuat bar-plot dari kelompok warna
    bar = np.ones((50,500,3),dtype='uint')
    plt.figure(figsize=(12,8))
    plt.title('Proportions of colors in the image')
    start = 0
    i = 1
    for p,c in p_and_c:
        end = start+int(p*bar.shape[1])
        if i==clusters:
            bar[:,start:] = c[::-1]
        else:
            bar[:,start:end] = c[::-1]
        start = end
        i+=1
    plt.imshow(bar)
    plt.xticks([])
    plt.yticks([])

    # Menyiapkan laporan akhir
    rows = 1000
    cols = int((org_img.shape[0]/org_img.shape[1])*rows)
    img = cv2.resize(org_img,dsize=(rows,cols),interpolation=cv2.INTER_LINEAR)
    copy = img.copy()
    cv2.rectangle(copy,(rows//2-250,cols//2-90),(rows//2+250,cols//2+110),(255,255,255),-1)
    final = cv2.addWeighted(img,0.1,copy,0.9,0)
    cv2.putText(final,'Most Dominant Colors in the Image',(rows//2-230,cols//2-40),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
    start = rows//2-220
    for i in range(5):
        end = start+70
        final[cols//2:cols//2+70,start:end] = p_and_c[i][1]
        cv2.putText(final,str(i+1),(start+25,cols//2+45),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1,cv2.LINE_AA)
        start = end+20
    plt.show()

    # Menyimpan laporan akhir sebagai output gambar
    cv2.imshow('img',final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output.png',final)


# GUI
# Layout kiri, untuk meminta direktori, menampilkan daftar file, dan menjalankan algoritma DomiColor
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
        sg.Button("Classify Image", key="-CLASSIFY-", visible=False),
    ],
]

# Layout kanan, menampilkan quick-preview dari gambar yang dipilih
image_viewer_column = [
    [sg.Text("Choose an image from list on left:", key="-GUIDE-")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        # Layout kiri
        sg.Column(file_list_column),
        # Garis pembatas
        sg.VSeperator(),
        #Layout kanan
        sg.Column(image_viewer_column),
    ]
]

# Judul aplikasi
window = sg.Window("DomiColor", layout)

#Event Loop | Main Activity
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Jika folder telah dipilih, siapkan daftar file nya pada list
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Mengambil daftar file
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png"))
        ]
        window["-FILE LIST-"].update(fnames)
    # Jika sebuah file dipilih dari list
    elif event == "-FILE LIST-":
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window['-CLASSIFY-'].update(visible=True)
            window['-GUIDE-'].update(visible=False)
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass
    # Jika file telah dipilih dan dilanjutkan dengan menekan tombol untuk menjalankan algoritma DomiColor
    elif event == "-CLASSIFY-":
        print('CI Pressed')
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            domi_color(filename)
            print('CI Pressed - Berhasil')
        except:
            print('CI Pressed - Gagal')
            pass

window.close()