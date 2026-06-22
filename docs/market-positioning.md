# Market Positioning

## Kisa Konumlandirma

AMD GPU'larda OpenCV CUDA benzeri, kolay kullanilan, Python dostu goruntu
isleme katmani.

## Ne Degil?

- CUDA klonu degil.
- ROCm/HIP yerine gecmez.
- Ilk asamada DNN inference framework'u degil.
- OpenCV'nin tam alternatifi degil.

## Ne?

- HIP uzerinde calisan ust seviye goruntu isleme API'si.
- OpenCV kullanan gelistiriciler icin tanidik fonksiyonlar.
- AMD GPU'da veri GPU uzerindeyken islemleri zincirleme imkani.
- Benchmark ve dogruluk testleriyle guven veren kucuk bir cekirdek.

## Mevcut Cozumlerden Farki

| Cozum | Guclu Yani | hipcv'nin Farki |
| --- | --- | --- |
| ROCm/HIP | Dusuk seviye GPU programlama | Daha ust seviye OpenCV benzeri API |
| OpenCV CUDA | Olgun ve kolay CUDA deneyimi | AMD GPU hedefi |
| OpenCV UMat/OpenCL | Tasima kabiliyeti | Daha belirgin HIP backend ve performans kontrolu |
| RPP | Performans primitive'leri | Daha sade uygulama gelistirici API'si |
| rocAL | ML veri pipeline'i | Genel amacli goruntu isleme |
| MIVisionX | Kapsamli OpenVX toolkit | Daha hafif ve OpenCV/Python odakli deneyim |

