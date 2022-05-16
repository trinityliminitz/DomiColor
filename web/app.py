import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# Package yang diperlukan dalam penghitungan DomiColor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png'])
BAR_PLOT = 'bar_plot.png'
BLOCK_PLOT = 'block_plot.png'
OUTPUT_FINAL = 'output_final.png'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/tentang/')
def tentang():
    anggota = ['Timothy Sipahutar | 11S19016',
               'Edrei Siregar | 11S19019',
               'Judah Sitorus | 11S19040',
               'Kevin Sihaloho | 11S19044',
               'Andreas Pakpahan | 11S19047'
               ]

    return render_template('tentang.html', anggota=anggota)


@app.route('/klasifikasi/')
def klasifikasi():
    return render_template('klasifikasi.html')


@app.route('/klasifikasi/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('[Err:1]Belum ada gambar yang dipilih.')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('[Err:2]Belum ada gambar yang dipilih.')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        domi_color(UPLOAD_FOLDER+filename)
        print('upload_image filename: ' + filename)
        flash('Gambar berhasil diupload dan hasil ditampilkan dibawah.')
        return render_template('klasifikasi.html', filename=filename, bar=BAR_PLOT, block=BLOCK_PLOT, out=OUTPUT_FINAL)
    else:
        flash('Mohon upload gambar dengan format .PNG')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Algoritma DomiColor
def domi_color(file_path):
    # Mendefinisikan jumlah cluster untuk algoritma KMeans
    clusters = 10

    # Membaca input gambar
    img = cv2.imread(file_path)
    org_img = img.copy()  # salinan gambar
    path = 'static/uploads' # Lokasi gambar akan disimpan

    # Menampilkan dimensi gambar
    print('Org image shape --> ', img.shape)

    # Mengkonversi ukuran gambar dan menampilkan dimensi hasilnya
    img = imutils.resize(img, height=200)
    print('After resizing shape --> ', img.shape)

    # Meratakan gambar menjadi satu kolom sepanjang jumlah pixels gambar
    flat_img = np.reshape(img, (-1, 3))
    print('After Flattening shape --> ', flat_img.shape)

    # Menyiapkan K-Means Clustering dengan jumlah clusters yang telah dideklarasikan sebelumnya
    kmeans = KMeans(n_clusters=clusters, random_state=0)

    """
    Gambar yang telah diratakan kemudian disesuaikan sebagai larik yang berisi semua warna pada pixels gambar 
    yang dimana akan dikelompokan sebanyak jumlah clusters yang telah dideklarasikan
    dan kelompok ini akan dianggap sebagai warna utama/dominan pada gambar
    """
    kmeans.fit(flat_img)

    # Mengambil warna utama/dominan pada gambar
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')

    """
    Berdasarkan warna utama/dominan pada gambar, setiap kelompok akan dihitung persentase kemunculannya dimana;
        np.unique(kmeans.labels_,return_counts=True
    akan mengembalikan array dua dimensi; dimensi pertama menjelaskan setiap pixel merujuk pada suatu kelompok, dan dimensi kedua menunjukan jumlah pixel yang berada pada suatu kelompok
    pada kasus ini, digunakan dimensi kedua dan dibagi oleh jumlah pixels gambar keseluruhan dan menghasilkan persentase pada setiap kelompok
    """
    print(np.unique(kmeans.labels_, return_counts=True)[1])
    percentages = (np.unique(kmeans.labels_, return_counts=True)
                   [1])/flat_img.shape[0]

    # Menggabungkan (zip) persentase dengan kode warna setiap kelompok
    p_and_c = zip(percentages, dominant_colors)

    # Mengurutkan objek zip secara menurun (descending) sehingga elemen pertama adalah kelompok yang memiliki persentase tertinggi (warna terbanyak)
    p_and_c = sorted(p_and_c, reverse=True)

    # Menyiapkan laporan akhir
    rows = 1000
    cols = int((org_img.shape[0]/org_img.shape[1])*rows)
    img = cv2.resize(org_img, dsize=(rows, cols),
                     interpolation=cv2.INTER_LINEAR)
    copy = img.copy()
    cv2.rectangle(copy, (rows//2-250, cols//2-90),
                  (rows//2+250, cols//2+110), (255, 255, 255), -1)
    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, '  Warna paling dominan adalah', (rows//2-230,
                cols//2-40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    start = rows//2-220
    for i in range(5):
        end = start+70
        final[cols//2:cols//2+70, start:end] = p_and_c[i][1]
        cv2.putText(final, str(i+1), (start+25, cols//2+45),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        start = end+20

    # Menyimpan laporan akhir sebagai output gambar
    cv2.imwrite(os.path.join(path , OUTPUT_FINAL), final)
