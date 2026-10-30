# План интеграции OIDN WebGPU в Light Tracer

Статус на 29 октября 2026: базовая интеграция реализована в worktrees OIDN
`codex/lighttracer-webgpu` и Light Tracer `codex/oidn-webgpu`. Native и web
используют Dawn LT, общий encoder и GPU изображения. Реальные API, команды
сборки, размеры, результаты проверок и ограничения описаны в [EMBEDDING.md](EMBEDDING.md).
Ниже сохранён исходный план; его проектируемые интерфейсы следует сверять с отчётом.

Цель — подключить WebGPU порт OIDN как статическую библиотеку к нативному и веб приложению Light Tracer. Денойзер использует уже созданный device приложения, принимает его GPU текстуры и буферы, записывает вычисления в общий поток команд и возвращает результат на GPU. Веса и код включаются только для выбранных режимов фильтра.

База реализации — ветка `codex/webgpu-upstream`, коммит `4623eb83`, поверх upstream OIDN 2.5.1. План относится к двум рабочим каталогам: `D:\projects\oidn` и `C:\projects\lighttracer`. Дата — 29 октября 2026 года.

## Целевые решения

| Область | Решение |
| --- | --- |
| Библиотека | Статические core и WebGPU backend, общий с Light Tracer WebGPU runtime |
| Контекст | Device и queue предоставляет Light Tracer; OIDN не выбирает адаптер и не создаёт второй device |
| Команды | OIDN записывает работу в переданный `WGPUCommandEncoder`; finish и submit выполняет Light Tracer |
| Вход и выход | Импорт `WGPUBuffer` и `WGPUTextureView`, включая размеры, формат, offsets и strides |
| Передача изображений | Без CPU staging и readback; преобразования форматов выполняются на GPU |
| Веса | Встроенные наборы OIDN отключены; приложение передаёт только выбранный набор |
| Веб | Один wasm модуль Light Tracer и одна таблица WebGPU handles; без отдельного `createOIDN` |
| Совместимость | Сохраняется полноценная конфигурация OIDN для тестов против CPU |

Отсутствие CPU копирования не означает отсутствие преобразований на GPU. Сеть использует внутренние HWC тензоры, а Light Tracer — текстуры и упакованные вспомогательные карты. Чтение текстур, распаковка и запись результата остаются GPU операциями. Загрузка весов из памяти или по сети допускается при подготовке, но изображения кадра через CPU не проходят.

## Текущее состояние и обязательные изменения

| Сейчас | Изменение |
| --- | --- |
| `WebGPUDevice::init` создаёт instance, adapter, device и queue | Добавить создание OIDN device из контекста приложения |
| Нативный OIDN загружает wgpu-native, Light Tracer использует Dawn | Выделить внешний provider и линковаться с тем же Dawn, что и приложение |
| OIDN web использует emdawnwebgpu, Light Tracer — `USE_WEBGPU` и собственный `library_webgpu.js` | Согласовать C API и единую реализацию браузерных bindings |
| `WebGPUBuffer` управляет собственными буферами; shared import не реализован | Добавить обёртку над существующим `WGPUBuffer` без выделения и копирования данных |
| Изображения WebGPU backend представлены буферами float32 | Добавить texture storage в описание изображений и texture варианты I/O kernels |
| `flush` завершает и отправляет собственный encoder | Разделить автономное выполнение и запись в encoder приложения |
| `commit`, sync и освобождение ресурсов могут ждать GPU | Подготовить неблокирующий жизненный цикл для общей очереди и веба |
| Light Tracer читает color и guides через `FetchImageData`, затем передаёт массивы в `OIDenoiser` | Заменить viewport путь на передачу GPU ресурсов и запись команд |
| Результат возвращается через `GetOutput` и `sg_update_image` | Использовать выходной GPU ресурс непосредственно в последующем проходе |

Передача handles между текущими wgpu-native и Dawn недопустима, даже если они работают на одной видеокарте. Нужны один device, одна реализация C API и совместимая версия заголовков. В вебе дополнительно должна совпадать таблица, преобразующая handles wasm в JS объекты.

## Минимальная статическая сборка

Использовать существующие upstream опции: `OIDN_STATIC_LIB=ON`, `OIDN_DEVICE_WEBGPU=ON`, остальные device backends отключить, `OIDN_APPS=OFF`, `OIDN_LIBRARY_VERSIONED=OFF`, `OIDN_FILTER_RT=OFF`, `OIDN_FILTER_RTLIGHTMAP=OFF`.

Последние две опции отключают встроенные веса, а не весь код фильтров. Сейчас `core/CMakeLists.txt` всё равно включает RTLightmap, а корневой CMake автоматически добавляет `web/` при Emscripten. Для встраивания это нужно разделить явно.

Добавить следующие настройки сборки. Названия ниже проектируемые, в текущей версии их нет.

| Новая настройка | Назначение |
| --- | --- |
| `OIDN_WEBGPU_PROVIDER=EXTERNAL` | Не скачивать и не линковать собственный wgpu-native |
| `OIDN_WEBGPU_PROVIDER_TARGET` | Получать заголовки, compile options и link dependencies из target Light Tracer |
| `OIDN_WEBGPU_STANDALONE=OFF` | Не создавать отдельный wasm, JS обвязку, демостраницу и автономный device path |
| `OIDN_WEBGPU_TESTS=OFF` | Не включать GTest и CPU reference в поставку приложения |
| `OIDN_ENABLE_RTLIGHTMAP=OFF` | Исключать реализацию RTLightmap и соответствующую ветку API, сохраняя upstream поведение по умолчанию |

Предоставить агрегирующий target, например `OIDN::WebGPU_RT`, для `target_link_libraries`. Все библиотеки OIDN в этом профиле статические; WebGPU runtime наследуется от приложения. Если собственный Dawn Light Tracer поставляется DLL, статический OIDN использует эту уже существующую DLL.

Для Windows проверить `OIDN_WEBGPU_API`: текущий private header использует `dllimport` вне сборки DLL и требует корректного режима для static linking. В минимальном профиле исключить экспорт внутренних engine символов для тестов.

Собирать весь граф зависимостей совместимыми инструментами и настройками: MSVC runtime, Emscripten SDK, pthread/shared memory, исключения и LTO. Light Tracer в вебе уже использует `-pthread` и `-fexceptions`; OIDN должен наследовать их из общей конфигурации. На Windows учитывать `_HAS_EXCEPTIONS=0` в части проекта Light Tracer: граница интеграции должна быть C API с POD дескрипторами и кодами ошибок, без передачи STL объектов и исключений между библиотеками.

После работающей интеграции включить dead stripping и согласованный LTO. В поставку не должны попадать CPU, ISPC, TBB, прочие GPU backends, sample programs, тесты, диагностические выгрузки и все модели сразу. Graph, allocator и нужные RT kernels сохраняются: они непосредственно выполняют сеть.

## Общий контекст и интерфейс ресурсов

Расширение оформить отдельным публичным C заголовком, например `oidn_webgpu.h`. Приложение не должно подключать private `webgpu_engine.h` или вызывать `getEngine`.

Предлагаемые операции API:

```c
OIDNDevice oidnNewWebGPUDevice(const OIDNWebGPUContextDesc* context);
OIDNBuffer oidnNewSharedBufferFromWebGPU(
    OIDNDevice device, const OIDNWebGPUBufferDesc* buffer);
void oidnSetFilterWebGPUTexture(
    OIDNFilter filter, const char* name, const OIDNWebGPUTextureDesc* texture);
OIDNWebGPUResult oidnPrepareFilterWebGPU(OIDNFilter filter);
OIDNWebGPUResult oidnEncodeFilterWebGPU(
    OIDNFilter filter, WGPUCommandEncoder encoder);
```

Это проект интерфейса, а не готовые объявления. Перед реализацией зафиксировать версионирование дескрипторов через размер структуры и точную схему завершения асинхронной подготовки.

Дескриптор контекста содержит `WGPUDevice` и его `WGPUQueue`. Instance передаётся только при необходимости обработки callbacks; adapter не требуется для выбора устройства внутри OIDN. Возможности, включая shader-f16 и limits, определяются по уже созданному device. Light Tracer заранее включает необходимые features. Fallback на float32 работает на том же device.

Правила владения:

- Light Tracer владеет device, queue, внешними текстурами, буферами и encoder. OIDN удерживает необходимые ссылки, но не вызывает `Destroy` внешних объектов.
- OIDN владеет весами на GPU, scratch tensors, pipelines, bind groups и метаданными исполнения.
- Импорт буфера описывает handle, доступный диапазон, offsets, pixel/row strides и логический формат. Он не создаёт второй буфер и не трактует handle как CPU указатель.
- Импорт текстуры описывает view, фактический формат view, mip/layer, размеры, тип доступа и интерпретацию каналов. На первом этапе поддерживаются обычные 2D изображения без MSAA.
- Ресурсы принадлежат одному device. Ошибки устройства, формата, usage и диапазона возвращаются явно; скрытого перехода через CPU нет.

Для текстур потребуется расширить общую модель `Image`, которая сейчас хранит host pointer или buffer. Ввести отдельное backend storage для texture view с обычным описанием размеров и каналов. Планирование сети продолжает использовать это описание, а WebGPU I/O выбирает buffer или texture kernel. CPU accessor не должен получать texture storage.

Проверку пересечения изображений строить по идентичности ресурса и диапазонам. Для импортированных буферов это underlying `WGPUBuffer` и byte ranges; для текстур — texture и subresource. Одних `getPtr()` и `isShared()` недостаточно. При реальном перекрытии входа и выхода сохраняется необходимый GPU временный результат.

## Запись команд и отсутствие ожиданий в кадре

Основной режим — запись в encoder Light Tracer после завершения render pass и перед проходом, использующим результат. OIDN не завершает, не отправляет и не освобождает переданный encoder. Нельзя вызывать `sg_commit` из денойзера.

Предполагаемый порядок:

```text
Render color and helper maps
  -> GPU unpack of helper maps when needed
  -> OIDN autoexposure, input processing, network, output processing
  -> GPU output conversion when needed
  -> display or subsequent processing
  -> application finish and submit
```

Sokol уже предоставляет `sg_wgpu_command_encoder`, `sg_wgpu_device`, `sg_wgpu_queue`, `sg_wgpu_query_image_info` и `sg_wgpu_query_buffer_info`. Вызов интеграции выполняется вне открытого render/compute pass. Нужно сохранить существующий порядок frame encoder и commit listeners, включая ограничения Sokol GL из `CLAUDE.md` Light Tracer.

Существующий uniform arena при переполнении вызывает `flush`. В режиме внешнего encoder это запрещено. Заменить такой механизм страницами параметров на исполнение либо другим размещением, которое не перезаписывает данные до завершения использующих их команд. Учитывать и encoder, который приложение ещё не отправило. Ссылки на страницы и временные ресурсы освобождаются по completion callback, без ожидания CPU в кадре.

Один scratch set нельзя одновременно использовать несколькими перекрывающимися исполнениями. Для первой версии выбрать последовательные исполнения на одной очереди, а для нескольких кадров в полёте — отдельные execution slots с собственным scratch и параметрами. Веса и pipelines разделяются в пределах device.

Подготовку сети отделить от кодирования кадра. Проверить все пути `UNetFilter::commit`, `Device::execute`, переаллокации scratch, уничтожения фильтра и context shutdown: они не должны незаметно вызывать blocking wait общей очереди. При необходимости подготовка возвращает `Pending`, а готовность сообщается через callback приложения. Загрузка весов должна быть упорядочена перед первым использованием.

Веб интеграция не должна требовать добавления Asyncify ко всему Light Tracer ради обычного денойзинга. Для этого недостаточно убрать создание adapter: нужно удалить или изолировать все `WaitAny`, blocking map и sync пути встроенного режима. Сохранение изображения может иметь отдельный асинхронный readback API. Завершение работы выполняется после освобождения ресурсов приложения или через неблокирующую retirement очередь.

## Текстуры и форматы Light Tracer

Сначала сохранить текущую семантику viewport: LDR, `srgb=true`, `cleanAux=true`, albedo и normal. Не переводить viewport на HDR одновременно с заменой транспорта данных: иначе изменится и изображение, и модель, что затруднит проверку качества.

Сейчас LDR текстуры создаются как RGBA8 в вебе и RGBA16 на desktop. Прежний CPU путь нормализует каналы, а alpha переносит отдельно. Texture input должен получать те же значения с учётом аппаратного sRGB decoding конкретного формата. Интерпретация цвета задаётся явно; нельзя применять transfer function дважды.

Вспомогательные карты упакованы. На GPU воспроизвести `UnpackOutAlbedo`, существующий `sqrt` для albedo в viewport и `UnpackOutNormal`. Для первой версии это shader адаптера Light Tracer, создающий стандартные GPU ресурсы OIDN. Общая библиотека OIDN не должна зависеть от частного packing Light Tracer. Объединение распаковки с input processing — последующая оптимизация после проверки затрат.

Для I/O kernels поддержать два варианта: чтение strided buffers и чтение texture views через `textureLoad`. Сеть и веса при этом общие. Запись напрямую в storage texture разрешать только для совместимых форматов и usages уже созданного ресурса.

Если текущая display texture не поддерживает нужную storage запись, использовать выходной GPU buffer либо RGBA16F texture и дополнительный GPU проход в существующий render target. Это сохраняет отсутствие CPU копирования и позволяет оставить форматы Light Tracer. Не предполагать одинаковую storage поддержку RGBA8 и RGBA16; подтвердить её на выбранном Dawn и целевых браузерах.

Alpha, ориентация изображения, viewport extent и сохранение пикселей вне записываемой области входят в контракт. Форматы и usage flags назначаются при создании ресурсов; OIDN не может добавить недостающий `StorageBinding` существующей текстуре. Ошибка usage должна обнаруживаться до выполнения сети.

## Выбор весов

Базовый профиль поставки — текущий viewport с albedo и normal, High и cleanAux. Для него достаточно `rt_ldr_calb_cnrm.tza`. Все builtin weights OIDN остаются отключёнными; native приложение может встроить этот один blob в собственный target адаптера, а веб приложение загружает его при включении денойзинга.

| Возможность | Модель | Размер без сжатия |
| --- | --- | --- |
| Viewport LDR с clean guides | `rt_ldr_calb_cnrm.tza` | 1 841 160 байт |
| Viewport Fast с clean guides | `rt_ldr_calb_cnrm_small.tza` | 641 480 байт |
| HDR с clean guides в Balanced | `rt_hdr_calb_cnrm.tza` | 1 841 160 байт |
| HDR с clean guides в High | `rt_hdr_calb_cnrm_large.tza` | 7 698 439 байт |
| HDR с обычными guides в High | `rt_hdr_alb_nrm.tza` | 1 841 160 байт |
| LDR только color | `rt_ldr.tza` | 1 830 792 байта |

Fast, HDR экспорт, baking и режимы без guides не включать автоматически в обязательную загрузку viewport. Добавлять их отдельными профилями или загружать нужную модель по запросу. Если UI сохраняет режимы None и Albedo, для них нужны соответствующие веса либо явно выбранное поведение при их отсутствии. Использовать модель с другим числом входных каналов нельзя.

Существующие режимы Light Tracer сохраняются в итоговой интеграции. Одна обязательная модель означает минимальный стартовый набор, а не удаление HDR экспорта, baking или настроек guides. Веб загружает нужные модели по запросу с состоянием подготовки; нативная поставка может хранить их отдельными файлами для работы без сети. Уменьшение размера исполняемого файла учитывать отдельно от размера всей поставки с этими assets.

Нельзя уменьшать размер High HDR с cleanAux незаметной заменой large модели на base или изменением cleanAux: это изменение качества и режима фильтра. У каждого набора нужен manifest с именем, digest, числом входов и допустимыми настройками. При передаче пользовательского weights blob OIDN уже не выбирает модель автоматически по quality.

Разбор, reorder и upload выполнять один раз на модель и device, а не на кадр или tile. Предусмотреть разделение GPU весов между фильтрами одинаковой модели и precision. Сейчас upstream cache для пользовательских весов ограничен, поэтому загрузка blob сама по себе ещё не гарантирует такой cache.

Предварительно упакованные веса и удаление TZA parser рассматривать после измерения размера. Сначала сохранить стандартные веса и совпадение с CPU; собственный формат не нужен для первого рабочего подключения.

## Этапы реализации

| Этап | Изменения | Условие завершения |
| --- | --- | --- |
| 1 Общий runtime и сборка | Внешний WebGPU provider, минимальный static target, отключение standalone web target; согласование Dawn C API и browser bindings | Минимальный native и wasm пример используют device Light Tracer; нет второго WebGPU runtime и лишних backends |
| 2 Импорт буферов | Новый C API контекста и внешнего buffer, корректные владение и overlap | Фильтр использует буферы приложения; в трассе нет image upload/readback или создания второго device |
| 3 Общий encoder | Режим записи команд, неблокирующая подготовка, execution slots и безопасные uniforms | Одна очередь и submit приложения; resize, смена модели и несколько кадров в полёте проходят validation |
| 4 Текстуры | Texture storage в Image, I/O shader variants, проверка usages и GPU преобразования | На вход передаются texture views приложения; результат доступен следующему GPU проходу |
| 5 Viewport Light Tracer | Адаптер вместо CPU интерфейса, GPU unpack guides, alpha и display conversion | Удалены `FetchImageData`, CPU pixel loops, `GetOutput` и `sg_update_image` из штатного viewport пути денойзинга |
| 6 Веса и размер | Одна обязательная модель, manifest/cache, optional profiles, dead stripping и LTO | Измерен прирост итогового Light Tracer wasm и загрузки; отсутствуют все лишние weights blobs |
| 7 Дополнительные режимы | HDR экспорт, baking, Fast, color-only и albedo-only по необходимости продукта | Для каждого добавленного режима есть отдельные GPU parity и resource lifetime проверки |

Этап 1 — обязательная проверка совместимости. В Light Tracer есть собственные изменения Dawn и доступ к Vulkan для аппаратного ray tracing. Выбор версии нельзя свести к механическому обновлению заголовка. Предпочтительно адаптировать provider слой OIDN к единой поддерживаемой сборке Light Tracer; если нужна миграция Dawn или browser bindings, сначала проверить эти возможности приложения.

## Изменения по файлам

| Репозиторий и файлы | Работа |
| --- | --- |
| OIDN `CMakeLists.txt`, `core/CMakeLists.txt`, `devices/webgpu/CMakeLists.txt`, `external/webgpu/CMakeLists.txt` | Встраиваемая static конфигурация и внешний provider |
| OIDN `include/OpenImageDenoise`, `api/api.cpp` | Публичное WebGPU C расширение и ошибки |
| OIDN `webgpu_device`, `webgpu_buffer`, `webgpu_engine` | Внешний контекст, импорт, encoder, slots, ownership и shutdown |
| OIDN `core/image`, `core/rt_filter`, `core/unet_filter`, `webgpu_image_util` | Texture storage, overlap, планирование и неблокирующая подготовка |
| OIDN `webgpu_input_process`, `webgpu_output_process`, `webgpu_autoexposure`, `webgpu_image_copy` | Texture I/O и сохранение GPU пути для всех операций |
| OIDN `tests/test_webgpu_filter.cpp`, browser regressions | Проверки внешних ресурсов и encoder наряду с существующей CPU parity |
| Light Tracer `CMakeLists.txt`, Windows project, настройки веб сборки | Static link, общий Dawn target, одинаковые SDK и thread/exception настройки |
| Light Tracer `src/renderer/OIDenoiser.hxx/.cxx` | GPU интерфейс адаптера вместо передачи массивов float |
| Light Tracer `src/renderer/Renderer.cxx` | Встраивание команд, ресурсы viewport и отказ от CPU round trip |
| Light Tracer `src/renderer/SokolImpl`, создание display images и GPU shaders | Usage flags, unpack guides, выходной проход и согласование encoder |

## Проверки и измерения

Проверять новый backend сначала отдельным harness с внешним Dawn device, затем в Light Tracer. CPU эталон собирается отдельной тестовой конфигурацией и не входит в продукт.

Обязательные проверки:

- Buffer и texture варианты, 333×197, 512×512, 1024×1024 с несколькими tiles и выбранные рабочие разрешения viewport.
- FP16 и fallback FP32 на одном внешнем device; LDR, sRGB, HDR и разные auxiliary inputs в пределах поставляемых моделей.
- Padding и offsets, разные strides, alpha, in-place, общие underlying resources с разными views и ошибочные ranges/usages.
- Чужой device, потеря device, resize, закрытие сцены, смена параметров и освобождение объектов при работе GPU.
- Два encoder до submit, несколько кадров в полёте и больше 1024 наборов uniforms без перезаписи параметров.
- Нативный Dawn и целевые браузеры, включая фоновые вкладки; validation errors должны проваливать проверку.
- Сравнение с CPU по метрике OIDN `abs(a-b)/(abs(a)+abs(b)+0.01)`: средняя ошибка не выше 0,003; выбросом считается одновременное превышение absolute error 0,02 и relative error 0,05. Alpha и padding проверяются отдельно.

Трассировкой подтвердить отсутствие `ReadBuffer`, `MapRead`, `FetchImageData`, CPU staging изображений и ожиданий общей очереди в штатном кадре. Проверять также отсутствие загрузки весов, компиляции pipelines и пересоздания filter на каждом кадре. Диагностический readback разрешается только внутри теста или явно вызванного экспорта.

Производительность измерять раздельно: CPU encode, GPU timestamps, задержка до готовности результата, память и холодная подготовка. Для внешнего encoder включить команды денойзинга и необходимые GPU преобразования в один измеряемый диапазон. Старые показатели, полученные до исправления sync, для сравнения не использовать.

Размер сравнивать между двумя одинаково собранными Light Tracer: без денойзера и с ним. Отдельно записать delta wasm без сжатия и с Brotli/gzip, JS bindings, весовые assets и GPU memory. Текущие отдельные artifacts OIDN составляют около 1,14 МБ wasm и 0,17 МБ JS без весов; это ориентир исходного состояния, а не обещанный прирост монолитной сборки. Бюджет размера зафиксировать после этапа 1 по link map и фактическому результату.

## Первый рабочий результат

Нативный и веб viewport Light Tracer используют один общий WebGPU context, statically linked OIDN и одну обязательную LDR модель. Color и guides поступают с GPU, denoise кодируется в encoder приложения, результат отображается без возврата изображения на CPU. Нечётные размеры, tiles, alpha, resize и lifetime проходят проверки. Дополнительные модели и режимы подключаются отдельно после этого результата.


### Уточнение после профилирования 30 октября

Целевая архитектура — общий WebGPU backend для native и web; сохранять CUDA как
рабочую альтернативу не планируется. Удалены скрытые staging/readback копии
промежуточных tensors и их утечка, свёртки переведены на implicit GEMM с общей
памятью workgroup. Native LT 1024² ускорен с 663 до 181 мс, качество сохранено.
Дальнейший native FP16 требует проверки обновлённого общего Dawn/Vulkan LT:
текущий pinned Dawn явно блокирует ShaderF16 на NVIDIA. Подробные замеры и
ограничения: [PERFORMANCE.md](PERFORMANCE.md).
