# Анализ предметной области и существующих решений (References)

Данный документ содержит анализ научных публикаций, open-source решений и коммерческих продуктов в сфере потокового машинного перевода речи (Speech-to-Speech Translation, S2ST). Выборка подходов и архитектур опирается на актуальные тренды индустрии и открытого сообщества (см. подборку проектов [speech-to-speech-translation на GitHub](https://github.com/topics/speech-to-speech-translation) [1]).

Цель анализа — обоснование архитектурных решений, принятых при разработке проекта **Real-time Ice Cream Translator**.

## 1. Проблематика и теоретическая база

Традиционно задача голосового перевода решалась с помощью каскадной архитектуры (ASR -> MT -> TTS). Однако современные исследования активно отходят от текстовых промежуточных этапов:

* **Переход к End-to-End архитектурам:** В блоге Google Research ([Real-time speech-to-speech translation](https://research.google/blog/real-time-speech-to-speech-translation/), 2025) [2] описывается потоковая модель на базе AudioLM. Использование связки Streaming Encoder и авторегрессионного Streaming Decoder позволило добиться сквозной задержки около 2 секунд.
* **Unit-based подходы и клонирование голоса:** Для сохранения просодии (интонации) и тембра спикера исследователи переходят на дискретные акустические токены. Проект **[PolyVoice](https://speechtranslation.github.io/polyvoice/)** [3][4] демонстрирует архитектуру, где используются две языковые модели: одна для семантического перевода, вторая (на базе VALL-E X) для синтеза речи с zero-shot клонированием голоса.
* **Методология интеграции:** При проектировании пайплайнов и чанковании (chunking) аудиопотока проект опирается на стандарты экосистемы Hugging Face, описанные в курсе [Audio Course (Chapter 7: Speech-to-Speech Translation)](https://huggingface.co/learn/audio-course/en/chapter7/speech-to-speech) [5].

---

## 2. Архитектурный выбор: Каскад против S2ST

В рамках подготовки MVP мы проанализировали два возможных архитектурных пути для запуска на консьюмерских GPU.

### Вариант А: Модульный пайплайн (Каскад)
Для реализации каскада был проведен отбор среди актуальных открытых моделей по критериям: скорость инференса, вес в VRAM и качество потоковой обработки.

**— Распознавание речи (ASR)**
* *Рассмотрено:* [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) [6] и [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) [7] (высокая точность, но 1.7B избыточна для первого узла); [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) [8] (мощная потоковая модель от Mistral, но 4B параметров создают недопустимую задержку).
* *Выбор для пайплайна:* **[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)** [9] (FunAudioLLM) — обеспечивает SOTA-баланс между скоростью распознавания коротких чанков и мультиязычностью.

**— Машинный перевод (MT)**
* *Рассмотрено:* Тяжелые модели [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) [10] и [translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) [11] были исключены, так как их использование исключительно для перевода коротких реплик создает «бутылочное горлышко» в генерации токенов.
* *Выбор для пайплайна:* **[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)** [12] — оптимальный размер для задачи перевода текста (NMT) "на лету" без потери качества.

**— Синтез речи (TTS)**
* *Рассмотрено:* [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) [13] (отличный синтез, но относительно тяжелая модель).
* *Выбор для пайплайна:* **[CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)** [14] (FunAudioLLM) — выбран благодаря компактности (0.5B) и SOTA-качеству zero-shot клонирования голоса.

*Вывод по каскаду:* Суммарный вес оптимальной сборки составляет ~2.5B параметров. Это дает невероятную гибкость, но из-за необходимости пересылки тензоров между тремя сетями задержка неизбежно превышает эталонные 2-3 секунды.

### Вариант Б: End-to-End модель
В качестве прямого конкурента каскаду рассматривается использование единой модели **[SeamlessM4T-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large)** [15] от Meta (2.3B параметров).
Она позволяет выполнять прямой маппинг признаков из энкодера в вокодер, решая проблему потери просодии и минимизируя рассинхронизацию (Audio Alignment) на уровне внутренней архитектуры, при этом требуя столько же VRAM, сколько и оптимизированный каскад.

---

## 3. Анализ конкурентов и существующих решений на рынке

Для выработки оптимальной стратегии развития был проведен глубокий анализ текущих решений в области перевода медиаконтента. Существующие на рынке продукты можно разделить на три основные категории, каждая из которых имеет свои архитектурные ограничения.

### 3.1. Экосистемные и встроенные решения (Cloud-based)

Наиболее технически зрелыми продуктами на данный момент обладают крупные IT-корпорации, однако их решения жестко привязаны к собственным платформам.

**— Яндекс.Браузер (Закадровый нейроперевод видео)**
Яндекс предоставляет один из самых совершенных продуктов на рынке с точки зрения конечного пользователя (UX). Архитектурно это сложный облачный каскад (ASR -> NMT -> биометрия -> TTS).
* **Преимущества:** Использование биометрии (система определяет пол спикера) и Spatial Audio (качественное приглушение оригинальной дорожки — ducking).
* **Недостатки и архитектурные ограничения:** * **Vendor Lock-in:** Решение работает исключительно внутри экосистемы Яндекс.Браузера.
  * **Проблемы с Live-контентом:** Алгоритм идеально работает с VOD (Video on Demand), где сервер генерирует дорожку заранее. На прямых трансляциях (Twitch, YouTube Live) система использует агрессивное кэширование, из-за чего задержка (latency) достигает 10–20 секунд, разрушая эффект интерактива. Приложение не способно переводить локальные файлы или созвоны.

**— YouTube Native AI Dubbing (Aloud)**
[Проект Aloud](https://blog.google/innovation-and-ai/technology/area-120/aloud/) [16], изначально разработанный в инкубаторе Google Area 120 и ныне интегрированный в YouTube.
* **Анализ:** Это решение ориентировано исключительно на авторов контента (Creator Studio), позволяя генерировать мультиязычные дорожки постфактум, до публикации видео. Для задачи пользовательского потокового (real-time) перевода произвольного контента данный инструмент неприменим, так как не работает в режиме реального времени на стороне зрителя.

### 3.2. Кроссплатформенные браузерные расширения

В ответ на закрытость решения от Яндекса, open-source сообщество создало множество браузерных плагинов (например, популярное расширение [YouTube Subtitle Dubbing](https://chromewebstore.google.com/detail/youtube-subtitle-dubbing/hgjcdbncdjkhpmdijaigkmgbkjecpopj) [17] и его аналоги).

* **Преимущества:** Кроссплатформенность. Подобные расширения успешно функционируют не только в Chromium-based браузерах, но и в Firefox (в том числе Firefox for Android) и Safari, делая их доступными без установки тяжелого ПО.
* **Недостатки и архитектурные уязвимости:**
  * **Зависимость от DOM-дерева:** Расширения работают путем инъекции JavaScript и поиска тегов `<video>` или `<audio>`. Любое A/B тестирование интерфейса со стороны YouTube мгновенно ломает логику захвата звука.
  * **API Latency:** Большинство проектов не имеет локальных вычислительных мощностей и отправляет текст в бесплатные шлюзы (Google Translate TTS). Сетевые издержки делают плавный потоковый перевод невозможным.
  * **Отсутствие системного контроля:** Плагин физически не может перехватить звук извне браузерной песочницы (например, перевести речь собеседника в Discord).

### 3.3. Десктопные и системные утилиты (Desktop Audio Translators)

Проекты, которые запускаются локально и используют системный микшер ОС для захвата звука (Audio Loopback). Яркими представителями этого направления являются open-source проекты на базе Whisper, такие как [WhisperLive](https://github.com/collabora/WhisperLive) [18] или [whisper_streaming](https://github.com/ufal/whisper_streaming) [19].

* **Преимущества:** System-Agnostic подход. Программа захватывает любой звук, проходящий через аудиокарту, что позволяет переводить игры, локальные медиафайлы и защищенные DRM стримы.
* **Недостатки:** Абсолютное большинство таких проектов использует оригинальные или модифицированные (faster-whisper) имплементации Whisper, которые изначально не предназначены для потоковой работы (обрабатывают аудио батчами по 30 секунд). В попытках заставить их работать в real-time, разработчики используют агрессивный VAD для нарезки чанков, что приводит к обрывам слов, потере контекста и суммарной сквозной задержке от 5 до 15 секунд при добавлении этапов MT и TTS.

---

## 4. Конкурентное позиционирование "Real-time Ice Cream Translator"

На основе проведенного анализа были сформулированы ключевые архитектурные принципы нашего проекта. Учитывая необходимость разработки стабильного MVP, было принято решение отказаться от хрупких надстроек над веб-интерфейсами и сосредоточиться на надежной системной интеграции.

1. **System-Agnostic архитектура (Превосходство над расширениями):** Вместо парсинга веб-страниц, проект использует системный захват аудио (Audio Loopback). Это делает продукт независимым от обновлений сайтов, смены браузеров и позволяет переводить абсолютно любой системный звук (включая локальные файлы, видеоигры и мессенджеры).
2. **End-to-End инференс (Превосходство над локальными утилитами и Яндексом):** В то время как десктопные конкуренты (на базе Whisper) используют тяжелые текстовые каскады, накапливающие вычислительные задержки, наш выбор в пользу S2ST-моделей (таких как [SeamlessM4T-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) [15]) позволяет обрабатывать аудиопоток напрямую. Отсутствие промежуточной генерации текстовых токенов радикально снижает Time-to-First-Byte, позволяя приблизить задержку перевода Live-стримов к эталонным показателям.

---

## Список источников

1. GitHub Topics: Speech-to-Speech Translation. https://github.com/topics/speech-to-speech-translation
2. Google Research Blog. "Real-time speech-to-speech translation." https://research.google/blog/real-time-speech-to-speech-translation/
3. PolyVoice Project Page. https://speechtranslation.github.io/polyvoice/
4. Dong, et al. "PolyVoice: Language Models for Speech to Speech Translation." *arXiv preprint arXiv:2306.02982* (2023). https://arxiv.org/abs/2306.02982
5. Hugging Face. "Audio Course: Chapter 7 - Speech-to-Speech Translation." https://huggingface.co/learn/audio-course/en/chapter7/speech-to-speech
6. Qwen3-ASR-1.7B. https://huggingface.co/Qwen/Qwen3-ASR-1.7B
7. Qwen3-ASR-0.6B. https://huggingface.co/Qwen/Qwen3-ASR-0.6B
8. Voxtral-Mini-4B-Realtime-2602. https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
9. SenseVoiceSmall. https://huggingface.co/FunAudioLLM/SenseVoiceSmall
10. Qwen3-4B-Instruct-2507. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
11. translategemma-4b-it. https://huggingface.co/google/translategemma-4b-it
12. Qwen3-1.7B. https://huggingface.co/Qwen/Qwen3-1.7B
13. Qwen3-TTS-12Hz-1.7B-CustomVoice. https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
14. CosyVoice2-0.5B. https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
15. SeamlessM4T-v2-large. https://huggingface.co/facebook/seamless-m4t-v2-large
16. Google Area 120: Project Aloud. https://blog.google/innovation-and-ai/technology/area-120/aloud/
17. YouTube Subtitle Dubbing (Chrome Web Store). https://chromewebstore.google.com/detail/youtube-subtitle-dubbing/hgjcdbncdjkhpmdijaigkmgbkjecpopj
18. WhisperLive (GitHub). https://github.com/collabora/WhisperLive
19. whisper_streaming (GitHub). https://github.com/ufal/whisper_streaming
