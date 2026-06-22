# Roadmap

## Faz 0: Arastirma ve Tasarim

- Hedef GPU ve ROCm surumleri belirlenecek.
- OpenCV CUDA API yuzeyi incelenecek.
- RPP, rocAL ve MIVisionX ile cakisan alanlar netlestirilecek.
- MVP fonksiyon listesi kesinlestirilecek.

## Faz 1: Cekirdek Altyapi

- `GpuMat` veri yapisi
- HIP memory allocation
- `upload` / `download`
- `Stream` ve async calisma modeli
- Hata yonetimi
- CMake build sistemi
- C++ smoke test

## Faz 2: Ilk Goruntu Fonksiyonlari

- `cvtColor`
- `resize`
- `threshold`
- `blur`
- `gaussianBlur`

Her fonksiyon icin OpenCV CPU ciktisiyla dogruluk testi yazilacak.

## Faz 3: Pipeline ve Benchmark

- Tek fonksiyon benchmarklari
- Zincirlenmis pipeline benchmarklari
- 720p, 1080p ve 4K testleri
- Upload/download maliyeti ayri olculecek

Ornek pipeline:

```text
upload -> resize -> cvtColor -> gaussianBlur -> download
```

## Faz 4: Python API

- pybind11 entegrasyonu
- NumPy array giris/cikis destegi
- Basit Python ornekleri
- Paketleme denemesi

## Faz 5: OpenCV Uyumlulugu

- `cv::Mat` ve `hipcv::GpuMat` donusumleri
- OpenCV kullanan projelere dusuk maliyetli gecis
- `cv2.cuda` benzeri isimlendirme stratejisinin degerlendirilmesi

