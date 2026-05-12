# Running Gemma on Android via LiteRT-LM

Technical notes from `student-app/` — a Kotlin/Compose app that runs a fine-tuned
Gemma model fully on-device. Use this as a starting point for new apps.

## TL;DR architecture

```
App.onCreate
   └── App.modelInference: ModelInference  (lazy, application-scoped singleton)

First-run flow:
   QR scan / paste URL → ModelDownloader (OkHttp, streaming) → filesDir/model.litertlm
                                                              + filesDir/model.id (source URL)
Per-launch:
   ChatScreen → ModelInference.load() (suspend, ~seconds) → Engine ready
   ChatScreen.send() → Engine.createConversation(history) → MessageCallback → callbackFlow<String>
```

Three rules that drive everything else:

1. **Load the `Engine` once, keep it on `Application`.** A `lazy` field in your
   `Application` subclass survives configuration changes and Activity restarts.
   Re-loading takes seconds; doing it on every screen entry is the #1 thing
   that makes the app feel broken.
2. **`Engine` is stateful, but `Conversation` is per-turn.** Create a new
   `Conversation` for each user message and seed it with prior turns via
   `ConversationConfig.initialMessages`. The engine handles tokenization/KV
   caching internally.
3. **Streaming is a `MessageCallback` → `callbackFlow<String>` adapter.** Each
   `onMessage` callback is one chunk. `onDone` closes the flow.

## Versions used (Nov 2026)

| Piece | Version |
|---|---|
| AGP | 8.9.2 |
| Kotlin | 2.3.0 (with `org.jetbrains.kotlin.plugin.compose`) |
| compileSdk / targetSdk / minSdk | 35 / 34 / 28 |
| Java target | 17 |
| Compose BOM | `2025.01.00` |
| LiteRT-LM | `com.google.ai.edge.litertlm:litertlm-android:latest.release` |
| OkHttp | `4.12.0` (model download only) |

Pin LiteRT-LM to a concrete version (e.g. `0.1.0`) once the API stabilizes for
you — `latest.release` is a moving target.

## Gradle wiring

```kotlin
// app/build.gradle.kts
android {
    defaultConfig { minSdk = 28 }                  // LiteRT-LM works comfortably here
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17 }
    kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

    packaging {
        // We don't bundle the .litertlm in assets (it's ~4.8GB), but if you ever
        // try to: do NOT compress it. Compression yields almost nothing on
        // already-packed weights and balloons install time.
        jniLibs.useLegacyPackaging = false
    }
}

dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")
    // ...Compose, OkHttp, etc.
}
```

## Manifest

```xml
<uses-permission android:name="android.permission.INTERNET" />          <!-- download only -->
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

<application
    android:name=".App"
    android:largeHeap="true"        <!-- REQUIRED. Multi-GB model loads. -->
    android:allowBackup="false"     <!-- don't try to back up a 5GB blob -->
    ...>
```

`largeHeap="true"` is non-negotiable for Gemma-sized models. Without it you
will OOM during `Engine.initialize()`.

## Model storage (`ModelStorage`)

Three files in `context.filesDir`:

- `model.litertlm` — the final, complete model
- `model.litertlm.part` — partial during download (atomic rename on success)
- `model.id` — opaque identifier for "which model is this?" (we use the
  download URL). Used to scope on-device data (chat history, settings) to the
  current model so a swap doesn't mix contexts.

```kotlin
object ModelStorage {
    fun modelFile(context: Context) = File(context.filesDir, "model.litertlm")
    fun partialFile(context: Context) = File(context.filesDir, "model.litertlm.part")
    private fun idFile(context: Context) = File(context.filesDir, "model.id")

    fun isModelPresent(context: Context) = modelFile(context).let { it.exists() && it.length() > 0 }
    fun currentModelId(context: Context) = idFile(context).let { if (it.exists()) it.readText().trim() else "" }
    fun writeModelId(context: Context, id: String) = idFile(context).writeText(id)
}
```

Why `filesDir` (not external storage):

- No runtime permission needed
- Survives across launches, cleared on uninstall (correct behavior)
- App-private, so you can't accidentally leak weights via shared storage

## Download flow (`ModelDownloader`)

OkHttp streaming download with progress callbacks throttled to "every 1% or
250ms" — anything more frequent kills the UI thread via recomposition storms.

Key points:

```kotlin
val client = OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .readTimeout(60, TimeUnit.SECONDS)
    .callTimeout(0, TimeUnit.SECONDS)   // 0 = no overall timeout — downloads take minutes
    .build()
```

- Write to `.part`, then `renameTo(target)` — never leave a half-written
  `model.litertlm` that `Engine.initialize()` would later try to load.
- Validate `totalRead == contentLength` before the rename; throw on mismatch.
- Cooperative cancellation via `isCancelled: () -> Boolean` checked inside
  the read loop. Don't rely on coroutine cancellation alone — OkHttp's
  `InputStream.read()` is blocking.
- Persist the source URL as `model.id` after the rename succeeds.

## Engine lifecycle (`ModelInference`)

```kotlin
class ModelInference(private val context: Context) : Closeable {
    @Volatile private var engine: Engine? = null
    val isLoaded: Boolean get() = engine != null

    suspend fun load() = withContext(Dispatchers.Default) {
        if (engine != null) return@withContext
        val modelFile = ModelStorage.modelFile(context)
        val config = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.CPU(),                          // GPU backend exists; CPU is the safe default
            cacheDir = context.cacheDir.absolutePath,         // engine writes precompiled artifacts here
        )
        engine = Engine(config).also { it.initialize() }
    }

    override fun close() {
        try { engine?.close() } catch (_: Throwable) {}
        engine = null
    }
}
```

- Hold this in `Application` as `val modelInference by lazy { ModelInference(applicationContext) }`.
- Call `load()` once on the first screen that needs it. The `LaunchedEffect`
  in your composable can check `isLoaded` and skip if already warm.
- `Dispatchers.Default` is appropriate — initialization is CPU/memory-bound,
  not IO.
- The engine writes cached compiled artifacts under `cacheDir`. On a fresh
  install the first load is slower; subsequent loads reuse the cache.
- **Don't `close()` on every screen exit.** Only close when the user actually
  swaps models or you're ending the app session. The cost of reloading is
  high enough that you want it scoped to the process.

## Streaming inference

The LiteRT-LM Kotlin API exposes streaming via `MessageCallback`. Wrap it in
a `callbackFlow` so callers get a clean `Flow<String>` of incremental chunks.

```kotlin
fun generateStream(history: List<ChatTurn>, userMessage: String): Flow<String> = callbackFlow {
    val activeEngine = engine ?: error("Model not loaded")

    val initial = history.map { turn ->
        if (turn.fromUser) Message.user(turn.text) else Message.model(turn.text)
    }

    val conversationConfig = ConversationConfig(
        systemInstruction = Contents.of(SYSTEM_PROMPT),
        initialMessages = initial,
        samplerConfig = SamplerConfig(topK = 40, topP = 0.9, temperature = 0.5),
    )

    val conversation = activeEngine.createConversation(conversationConfig)

    val callback = object : MessageCallback {
        override fun onMessage(message: Message) {
            // The Kotlin getting-started example just does `print(message)`. That
            // implies Message.toString() gives you the chunk text. If you find that
            // is NOT the case for your build of the lib, extract from message.contents
            // (Content.Text holds the chunk).
            val text = message.toString()
            if (text.isNotEmpty()) trySend(text)
        }
        override fun onDone() { close() }
        override fun onError(throwable: Throwable) { close(throwable) }
    }

    try {
        conversation.sendMessageAsync(userMessage, callback)
    } catch (t: Throwable) {
        close(t)
    }

    awaitClose {
        try { conversation.close() } catch (_: Throwable) {}
    }
}.flowOn(Dispatchers.Default)
```

Patterns worth copying:

- **One `Conversation` per turn.** Cheaper than you'd expect, and it
  sidesteps the question of mid-stream state management. The `initialMessages`
  list re-establishes context.
- **Pass `history` from the caller.** The `Engine` is stateless across
  conversations — chat history lives in your app's data model, not the engine.
  This is what makes "resume an old chat" work: load messages from disk,
  hand them to `generateStream` as history.
- **`awaitClose { conversation.close() }`** is critical. Without it, a cancelled
  flow leaks the conversation's native resources.
- **`.flowOn(Dispatchers.Default)`** keeps generation off the main thread.

## System prompt & sampling

```kotlin
const val SYSTEM_PROMPT = """You are a friendly tutor helping a 4th grade student..."""

private const val TEMPERATURE = 0.5   // tighter than 0.7 default — domain tutor
private const val TOP_P = 0.9
private const val TOP_K = 40
```

For task-specific apps (tutor, summarizer, classifier), lower temperature
(0.3–0.5) materially improves consistency. The defaults in most examples are
calibrated for open-ended chat.

## Chat history pattern (stateless engine + on-disk transcripts)

This is the integration pattern that pays off most when scaling beyond a
prototype:

1. Engine is stateless across conversations.
2. Persist transcripts as plain JSON files under `filesDir/chats/<id>.json`.
3. Tag each transcript with the `modelId` it was created against
   (`ModelStorage.currentModelId()`).
4. On resume: load JSON → reconstruct `List<ChatTurn>` → pass to
   `generateStream` as `history`.
5. Only allow resuming chats whose `modelId` matches the currently-loaded
   model. Show others as read-only / unavailable.

Why scope chats to the model: a "Mrs. Henderson fractions" model and a
"Mr. Patel poetry" model produce wildly different completions for the same
input. Letting users resume a chat with the wrong model gives them
hallucinated nonsense.

## Gotchas

- **OOM on load.** If you skipped `largeHeap="true"` or you're on a low-RAM
  device, `Engine.initialize()` throws. Surface this as a recoverable error
  in your onboarding flow.
- **Engine reload on Activity recreation.** If you don't application-scope
  the engine, every rotation pays the load cost. Use `Application` or a
  process-level holder.
- **`Message.toString()` may not be the chunk text** depending on the
  LiteRT-LM version. If you see empty strings or weird formatting, switch
  to extracting from `message.contents`.
- **Cancellation mid-generation** must close the `Conversation`. The
  `awaitClose` block above handles flow cancellation; you also need to handle
  the "user navigated away" case by cancelling the coroutine scope that
  collected the flow.
- **Cache invalidation.** When you replace the model file, also clear or
  invalidate `context.cacheDir` — the engine may have stale compiled artifacts
  for the previous weights. Easiest: wipe the cacheDir on model swap.
- **Download resumability** is *not* implemented here. For a multi-GB model
  on school WiFi, consider adding HTTP `Range` requests and resuming from the
  `.part` file length on retry.

## Reference

- Official Kotlin getting-started:
  <https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md>
- LiteRT-LM Android guide: <https://ai.google.dev/edge/litert-lm/android>

## File map (copy these as a starting point)

```
app/src/main/java/.../
├── App.kt                       # Application; holds modelInference lazy
├── MainActivity.kt              # Routes between onboarding / chat
├── model/
│   ├── ModelStorage.kt          # filesDir paths + model.id tracking
│   ├── ModelDownloader.kt       # OkHttp streaming download
│   ├── ModelInference.kt        # Engine lifecycle + generateStream flow
│   └── ChatStore.kt             # JSON-on-disk chats, scoped by modelId
└── ui/
    ├── OnboardingScreen.kt      # QR / paste URL → download
    └── ChatScreen.kt            # Drawer for history, new chat, streaming bubbles
```
