# hipcv

AMD GPU'larda OpenCV CUDA benzeri bir gelistirici deneyimi sunmayi hedefleyen
HIP tabanli goruntu isleme kutuphanesi.

## Amac

`hipcv`, CUDA'yi AMD icin bastan yazmaya calismaz. Bunun yerine AMD'nin
ROCm/HIP ekosistemi uzerinde, OpenCV kullanan gelistiricilerin kolayca
anlayabilecegi daha ust seviye bir API hedefler.

Hedeflenen kullanim:

```python
import hipcv as hcv

gpu = hcv.upload(img)
gray = hcv.cvtColor(gpu, hcv.COLOR_BGR2GRAY)
blur = hcv.gaussianBlur(gray, (5, 5), 1.2)
out = blur.download()
```

## Ilk MVP Kapsami

- `GpuMat`
- `upload` / `download`
- `Stream`
- `cvtColor`
- `resize`
- `threshold`
- `blur`
- `gaussianBlur`
- Python binding
- OpenCV CPU ile dogruluk testleri
- Temel benchmark araci

## Piyasadaki Bosluk

Bugun AMD tarafinda ROCm, HIP, RPP, rocAL ve MIVisionX gibi guclu parcaciklar
var. Fakat OpenCV CUDA'daki `GpuMat` + `cv::cuda` deneyimine benzeyen, sade,
Python dostu ve genel amacli bir AMD GPU goruntu isleme katmani yeterince net
degil.

`hipcv` bu bosluga odaklanir:

- HIP'in dusuk seviye gucunu saklar.
- OpenCV'ye benzer fonksiyon isimleri ve veri akisi sunar.
- Mumkun oldugunda ROCm/RPP gibi mevcut AMD bilesenlerinden yararlanir.
- Tek tek fonksiyonlardan cok gercek zamanli pipeline performansini onemser.

## Ilk Hedef Platform

- Linux
- ROCm destekli AMD GPU
- C++17
- HIP
- Python 3.10+
- OpenCV CPU baseline

Windows destegi daha sonraki asamada degerlendirilecektir.

## Proje Durumu

Bu repo erken tasarim ve MVP planlama asamasindadir.

