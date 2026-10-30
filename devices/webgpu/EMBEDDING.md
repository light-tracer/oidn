# Встраивание в Light Tracer

Реализовано 29 октября 2026 года поверх OIDN 2.5.1 (`6602ee2c`),
отчёт обновлён 30 октября.
OIDN собирается статически и использует Dawn приложения. Viewport Light Tracer
передаёт GPU изображения и записывает denoise в текущий encoder Sokol.
Загрузка, readback и ожидание изображений через CPU из этого пути удалены.

## Сборка

Пример подключения из родительского CMake проекта:

```cmake
# app_dawn должен предоставлять те же headers, library и WebGPU bindings,
# которые использует приложение.
set(OIDN_STATIC_LIB ON CACHE BOOL "" FORCE)
set(OIDN_DEVICE_WEBGPU ON CACHE BOOL "" FORCE)
foreach(backend CPU SYCL CUDA HIP METAL)
  set(OIDN_DEVICE_${backend} OFF CACHE BOOL "" FORCE)
endforeach()
set(OIDN_WEBGPU_PROVIDER EXTERNAL CACHE STRING "" FORCE)
set(OIDN_WEBGPU_PROVIDER_TARGET app_dawn CACHE STRING "" FORCE)
foreach(option APPS FILTER_RT FILTER_RTLIGHTMAP ENABLE_RTLIGHTMAP
               WEBGPU_STANDALONE WEBGPU_TESTS LIBRARY_VERSIONED)
  set(OIDN_${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(oidn)
target_link_libraries(app PRIVATE OIDN::WebGPU_RT app_dawn)
```

Профиль не скачивает wgpu-native, не включает CPU/TBB/ISPC, встроенные модели,
RTLightmap, отдельную web-обвязку или Asyncify. `FILTER_RT=OFF` отключает
встроенные веса RT; код RT остаётся для внешней модели.

Проверен ABI Dawn Light Tracer `fccbcdae2ba4d3986c82310a1bb374d41ad07495`
и браузерные bindings **Emscripten 3.1.74** (`USE_WEBGPU=1`). Это конкретная
совместимая конфигурация, а не обещание поддержки любой версии Dawn.
Самостоятельная конфигурация OIDN с современным emdawnwebgpu остаётся отдельной.
В LT CMake адаптер находится в `cmake/WebGPUOIDN.cmake`, инструкции —
`docs/WEBGPU_OIDN.md`. Для LT включаются `LT_OIDN_SOURCE_DIR` и
`LT_OIDN_WEIGHTS_DIR`; основной checkout приложения менять не требуется.

## API и владение

Публичный C API: `include/OpenImageDenoise/oidn_webgpu.h`.

```cpp
OIDNWebGPUContextDesc context{sizeof(context), appDevice, appQueue};
OIDNDevice device = oidnNewWebGPUDevice(&context);
oidnCommitDevice(device);
OIDNFilter filter = oidnNewFilter(device, "RT");

OIDNWebGPUBufferDesc inputDesc{sizeof(inputDesc), appInputBuffer, 0, 0};
OIDNWebGPUBufferDesc outputDesc{sizeof(outputDesc), appOutputBuffer, 0, 0};
OIDNBuffer input = oidnNewSharedBufferFromWebGPU(device, &inputDesc);
OIDNBuffer output = oidnNewSharedBufferFromWebGPU(device, &outputDesc);
oidnSetFilterImage(filter, "color", input, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
oidnSetFilterImage(filter, "output", output, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
oidnSetFilterBool(filter, "srgb", true);
oidnSetSharedFilterData(filter, "weights", modelBytes, modelSize);
if (!oidnPrepareFilterWebGPU(filter)) { /* inspect oidnGetDeviceError */ }

// Outside any open render/compute pass, after producers and before consumers:
if (!oidnEncodeFilterWebGPU(filter, appEncoder)) { /* discard this encoder */ }
// The application finishes and submits appEncoder in its usual frame flow.

oidnReleaseBuffer(input);
oidnReleaseBuffer(output);
oidnReleaseFilter(filter);
oidnReleaseDevice(device);
```

Пример использует внешние float32 RGB storage buffers и sRGB color-only модель
`rt_ldr.tza`. Для RGBA задаётся pixel stride 16, для подизображений — offsets и
row stride. `byteSize=0` означает остаток импортированного буфера. Usages,
размеры и диапазоны проверяются. Буферы приложения должны иметь `Storage` usage.

`oidnSetFilterImageFromWebGPUTexture` принимает `WGPUTexture` вместе с mip,
array layer и прямоугольником; opaque view не позволяет восстановить эти данные.
Input требует `TextureBinding`, output — `StorageBinding` и формат rgba8unorm,
rgba16float или rgba32float. Внешний float4 conversion buffer создаётся и
переиспользуется на GPU. Для исходного alpha задаётся `alphaSource`, иначе
записывается 1. Это отсутствие CPU копирования, но GPU преобразования остаются.
Упакованные integer guides распаковывает shader адаптера LT.

Device, queue, buffers и textures удерживаются ссылками. Импортированные
ресурсы OIDN никогда не уничтожает через `Destroy`. Приложение обязано передавать
ресурсы одного device и совместимого runtime и обрабатывать его validation и
device-loss события. Ссылку приложения можно отпустить после импорта; вызывать
`Destroy` до завершения GPU работы нельзя.

Подготовка может загрузить веса через общую queue и создать pipelines; она не
ждёт GPU. Веса `SetSharedFilterData` должны оставаться валидными согласно
стандартному shared-data контракту OIDN. Кэш GPU весов привязан к снимку их байтов,
поэтому повторно использованный CPU адрес не выбирает устаревшую модель.
Resize и смена модели в LT сохраняют device и его кэши.

Encode не делает finish, submit, map/readback или queue wait. Его encoder
принадлежит приложению. Последовательные encodes одного контекста должны
исполняться в том же порядке на одной queue: scratch переиспользуется.
Uniform pages выделяются отдельно на encode и удерживаются записанными командами.
Можно подготовить новый размер/модель и записать второй encoder до отправки
первого. При ошибке encode приложение отбрасывает весь этот encoder.
Изменение фильтра и encode не должны выполняться конкурентно из разных потоков.

HDR autoexposure остаётся на GPU. Явный экспорт и baking в LT используют отдельный
асинхронный job с upload/readback CPU изображения; viewport от него не зависит.
Веб не требует Asyncify. Неблокирующее освобождение сохраняет lifetime уже
записанных команд; за завершение работы общей queue отвечает приложение.

## Веса и размер поставки

Минимальная viewport модель `rt_ldr_calb_cnrm.tza` занимает **1 841 160 байт**.
Она сохраняет текущую семантику LT: sRGB, cleanAux и color/albedo/normal.
Опционально поставляются `rt_ldr_alb.tza` (1 835 976 байт), `rt_ldr.tza` и
`rt_hdr.tza` (по 1 830 792 байта). Файлы загружаются при первом запросе режима,
в wasm веса не встраиваются. Пустой `LT_OIDN_OPTIONAL_MODELS` в свежем build
оставляет только обязательную модель; такой пакет не содержит режимы без guides,
с одним guide, baking и HDR export. Модель другого профиля не подставляется.
Рядом поставляются `models.json`, SHA256SUMS и лицензии OIDN/весов.

Сравнение одной dev/test конфигурации Light Tracer, Emscripten 3.1.74 Release,
pthread и SIMD, gzip level 9, mtime 0:

| Файл | Без интеграции, байт | С интеграцией, байт | Прирост |
| --- | ---: | ---: | ---: |
| wasm | 12 890 818 | 13 203 592 | **312 774 (305,4 КиБ)** |
| wasm gzip | 4 129 012 | 4 214 765 | **85 753 (83,7 КиБ)** |
| JS | 277 984 | 282 473 | 4 489 |
| JS gzip | 60 808 | 61 455 | 647 |

Это прирост монолитной тестовой сборки с адаптером, а не размер отдельного OIDN
wasm. Baseline собран отключением `LT_OIDN_SOURCE_DIR` в том же worktree до
финальных небольших изменений feature negotiation и test bindings; production
пакет и Brotli отдельно не измерялись. Локальные артефакты и SHA256 сохранены в
`lighttracer/build-oidn-web/{with-oidn,without-oidn}/sizes.json`.

## Проверка и производительность

Внешний Dawn harness описан в `tests/external/README.md`. На RTX 3060 / Windows:

- Шесть native CTest проходят: buffer/texture, 333×197 sRGB/HDR и tiled HDR
  1024×1024. FP32 средняя относительная ошибка 3,4e-7 — 5,4e-7, выбросов нет.
- В браузере FP16 проходят 24 сравнения: те же три изображения, buffer/texture,
  in-place и разные C handles одного JS объекта, по два encoder до submit.
  Средняя ошибка 0,000284 — 0,000457, выбросов нет.
- Браузерный FP32 fallback также проходит все 24 сравнения, включая 1024×1024
  и in-place. Средняя ошибка 3,38e-7 — 5,37e-7, выбросов нет.
- Проверены alpha/padding, texture subregions, ошибочные buffer ranges/usages,
  освобождение OIDN объектов до submit и смена memory budget 512 → 64 MB между
  двумя ещё не отправленными encoder.
- Отдельный native стресс-тест проверяет по 1100 различных uniforms в каждом
  из двух неотправленных encoder, включая переход на следующую uniform page.
- Light Tracer native и web проходят по восемь assertions: включение denoise,
  изменение изображения, 0/1/2 guides, resize, baking и HDR job.
- Самостоятельная native конфигурация wgpu-native также компилируется после
  изменений. Полный набор её старых CTest этим прогоном не повторялся.

В массовом браузерном harness каждая transport-группа создаёт один device
приложения, выполняет на нём все три размера/режима и освобождает device перед
следующей группой. Первоначальный FP32 прогон всех групп на одном device
исчерпывал память браузерного процесса к концу серии. Освобождение C handles
не задаёт момент GC браузерных GPU объектов; длительный resize stress на одном
device остаётся отдельной проверкой. Запись двух encoder до submit проверяется
внутри каждого случая; библиотека не создаёт дополнительный device.

Метрика качества: `abs(gpu-cpu)/(abs(gpu)+abs(cpu)+0.01)`, среднее по RGB.
Порог среднего 0,003; выброс одновременно превышает absolute 0,02 и relative 0,05.
Это проверка численного соответствия OIDN, а не отдельная оценка качества модели.

GPU timestamps браузерного harness, FP16 color-only, texture path: 333×197
примерно **20,6–21,1 мс**; 1024×1024 с budget 512 MB **302 мс**, с 64 MB и tiling
**687 мс**. CPU encode в этом прогоне примерно 0,3–0,5 мс и 1,8 мс для tiled случая.
Диапазон включает GPU texture conversions и OIDN, исключает загрузку весов,
подготовку, warmup, тестовый readback и восстановление in-place входа.
Это единичный локальный прогон, не p50/p95 и не полное время кадра Light Tracer.
Старые timings до исправления submission использовать для сравнения нельзя.

## Сравнение со старым нативным LT

Отдельный прогон 30 октября сравнил реальный viewport путь LT: поставляемую
OIDN **2.3.2 / CUDA** и новую **2.5.1 / WebGPU FP32** на RTX 3060. При одинаковом
16 SPP кадре и двух guides медиана пяти повторов составила соответственно
133,3 / 83,3 мс для 333×197, 133,3 / 166,7 мс для 512² и 150,1 / 663,0 мс
для 1024². Это вся задержка до GPU результата с общим диагностическим readback,
а не чистое время kernels. Новый путь не заменяет CUDA по скорости крупных
нативных кадров.

Все входные PNG совпали побитно. Разница выходов — максимум один RGB уровень
из 255; PSNR 65,8–66,2 dB. На этой сцене качество практически одинаково.
Сценарий, полные времена, оценка к 1024 SPP и ограничения приведены в LT
`docs/OIDN_NATIVE_COMPARISON.md`, численные данные — в
`docs/benchmarks/oidn-native-2026-10-30.json`.

## Границы текущей реализации

Fast/другие model profiles не поставляются автоматически. Внешние half-format
buffer images не добавлены: вход/выход float32, внутренние tensors/weights FP16
при shader-f16, иначе FP32. Texture I/O пока использует промежуточный GPU buffer;
fusion с network I/O и дальнейшая оптимизация convolution остаются отдельными
задачами. Принудительный device loss, смешивание чужих devices и другие GPU /
браузеры не прошли отдельный regression matrix. Рабочие каталоги оставлены для
review; изменения в основные ветки и публикация не выполнялись.
