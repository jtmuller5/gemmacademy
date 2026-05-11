# Gemmacademy — Android Student App Instructions

> Build the student-facing Android app for Gemmacademy. The app downloads a fine-tuned Gemma 4 model from Hugging Face Hub once over WiFi and runs it offline forever after — letting a kid do homework at the kitchen table with no internet.
>
> **Scope is deliberately small. Do not invent features.** The whole app is one chat screen. Everything else is a distraction from the demo.

---

## Context (read this first)

Gemmacademy fine-tunes Gemma 4 E2B on a teacher's lesson content and produces a `.litertlm` file (~4.8 GB) that runs on Android via Google's LiteRT-LM Kotlin API. The model lives on Hugging Face Hub. The teacher gives students a QR code that encodes the download URL.

Your app is what the student opens. Three jobs:

1. First launch: scan the QR code (or paste the URL), download the model, store it locally
2. Subsequent launches: open straight to a chat with the locally-stored model
3. The chat itself: text in, streaming response out, fully offline

There are no accounts, no auth, no settings, no analytics, no multi-class support, no model versioning. One app, one model at a time, one chat. **The teacher's specific catchphrases coming out of the model is the whole demo** — anything that doesn't serve that moment is out of scope.

The hackathon judges will see this in a 3-minute video. The shot they will remember is: phone in airplane mode, kid types a fractions question, model streams back an answer that mentions Mrs. Henderson's specific Pizza Method. Everything you build serves that shot.

---

## Stack — non-negotiable

- **Native Android, Kotlin, Jetpack Compose** for UI
- **Android Studio** (latest stable)
- **`minSdk = 28`** (Android 9, 2018) — covers cheap-Android-class devices
- **`targetSdk = 34`** or higher
- **LiteRT-LM Kotlin API** — the on-device inference runtime
- **OkHttp** for the model download (built-in to Android, no extra dep)
- **CameraX + ML Kit Barcode Scanning** for the QR code scanner
- **Material 3** components

**Do not add:** Dagger/Hilt, RxJava, Room, WorkManager, Firebase, analytics, crash reporting, third-party design systems, Compose Navigation Compose. The app has two screens; you do not need a routing library.

**Do not add:** Flutter, React Native, KMM, or any cross-platform layer. We've already decided on native Kotlin.

If you find yourself wanting to add a dependency, ask first.

---

## Project structure

Create the project at `~/Dev/sapid/work/hackathons/gemma-4-good/gemmacademy/student-app/`.

Use Android Studio's "Empty Compose Activity" template, then organize as:

```
student-app/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── java/com/gemmacademy/student/
│   │       │   ├── MainActivity.kt
│   │       │   ├── ui/
│   │       │   │   ├── theme/                # Material 3 theme
│   │       │   │   ├── OnboardingScreen.kt   # First-launch: download model
│   │       │   │   ├── ChatScreen.kt         # The main screen
│   │       │   │   ├── ChatMessage.kt        # Message bubble composable
│   │       │   │   └── QrScannerScreen.kt    # Camera-based QR scan
│   │       │   ├── model/
│   │       │   │   ├── ModelDownloader.kt    # OkHttp + progress callbacks
│   │       │   │   ├── ModelInference.kt     # LiteRT-LM wrapper
│   │       │   │   └── ModelStorage.kt       # Where the .litertlm lives
│   │       │   └── App.kt                    # AndroidApplication subclass
│   │       └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
└── settings.gradle.kts
```

That's it. Three "screens" and three "model" classes. If your file tree gets larger than this, you've added something out of scope.

---

## The two app states

The app has effectively one piece of state: *is the model downloaded?*

- **No model:** show `OnboardingScreen` → `QrScannerScreen` → progress UI → on success, transition to `ChatScreen`
- **Yes model:** open directly to `ChatScreen`

Don't build a sophisticated state machine. A single `MutableState<AppState>` in MainActivity that's either `NeedsModel`, `Downloading(progress)`, or `Ready` is enough. No ViewModels needed for an app this small — composables can hold their own state and call into the model classes directly.

Persist whether-the-model-exists by checking for the file at app startup. No SharedPreferences, no database.

---

## Screen 1: Onboarding

Shown only on first launch, when no model file exists locally.

**Three states the screen cycles through:**

### 1a. Welcome / scan prompt
- App icon (ask the human for one — for now, a placeholder pizza emoji or fraction symbol is fine)
- Heading: *"Get your teacher's lessons"*
- Body copy: *"Your teacher will give you a QR code. Scan it to download today's lessons. You only need to do this once — after that, the app works without internet."*
- Big primary button: **"Scan QR code"**
- Below it, a smaller text link: *"Or paste a URL"* (for the demo, this is the easier path — no camera needed)

### 1b. QR scanner OR URL paste dialog
- If scan: full-screen camera preview with a viewfinder overlay. ML Kit Barcode Scanning detects the QR. On detection, immediately advance to download.
- If paste: a simple dialog with a text field and a "Download" button. Pasted URL goes straight to download.

### 1c. Download progress
- Heading: *"Downloading your lessons…"*
- A linear progress indicator showing percentage
- Below: file size info: *"4.8 GB · downloading over WiFi"*
- Sub-copy: *"Please keep the app open. This takes a few minutes on school WiFi."*
- A "Cancel" button (returns to 1a, deletes any partial file)

When the download completes:
- 1-second confirmation animation (checkmark + "Lessons ready!")
- Auto-advance to ChatScreen

**Failure handling:** if the download fails, show an error message ("Couldn't download — check your WiFi") and a "Try again" button. Don't be clever; the demo will be on a known-good WiFi.

---

## Screen 2: Chat

The main screen. The whole app is really this one screen. It needs to feel **calm, focused, and like talking to a patient teacher**.

### Layout (top to bottom)
1. **Top app bar** — small, minimal. Class name on the left (e.g., "Mrs. Henderson · Fractions"). Three-dot menu on the right with one option: "About" (opens a small dialog with app info; absolutely no settings).
2. **Message list** — fills the screen. Newest at the bottom. Auto-scroll on new messages. Older messages stay visible. Empty state: *"Ask me anything about today's lesson!"* with a soft prompt suggestion or two ("Try: How do I show 3/8 with the Pizza Method?").
3. **Input bar** — pinned to the bottom. Multi-line text field with placeholder "Ask a question..." and a send button (paper-airplane icon). Send disabled when the field is empty or while the model is generating a response.

### Message bubble design
- **Student messages** (right-aligned, primary color background): rounded rectangle, white text
- **Tutor messages** (left-aligned, light neutral background): rounded rectangle, dark text. While streaming, show a small blinking cursor at the end of the message.

### Streaming behavior

This is the most important part. Inference is slow on a phone — likely 5-15 tokens per second. **You must stream tokens to the UI as they arrive.** A response that takes 15 seconds total feels acceptable when you can see it being typed; the same response delivered all at once feels broken.

LiteRT-LM's Kotlin API supports streaming via callbacks. Connect that callback to a `MutableState<String>` for the in-progress message; recompose on each token. When generation ends, the state freezes and a new message slot becomes available.

### What to do during inference
- Disable the send button (greyed out, not hidden)
- Don't show a separate "thinking…" indicator — the streaming tokens *are* the indicator

### System prompt

Hardcode this — there's no settings screen and the teacher doesn't configure anything per-class for v1. Use:

```
You are a friendly tutor helping a 4th grade student with fractions. The student
just learned about fractions in Mrs. Henderson's class today. Use the specific
methods and examples Mrs. Henderson teaches: the Pizza Method (drawing pizzas
with equal slices), the rule "equal slices, equal fractions," and procedures
like "the cut goes on the bottom, the count goes on the top."

Be patient. Use simple language a 9-year-old understands. When a student is
confused, help them work through it step by step. If they ask about something
outside today's lesson on fractions, gently redirect them: tell them you're
focused on fractions today and ask if you can help with that instead.

Keep answers short — 2 to 4 sentences usually. End with a question or
encouragement when it feels natural.
```

This system prompt prepends to every conversation. Pass it as the first message with role `system` (or `model` if Gemma 4's chat template uses that role name — check what the LiteRT-LM Kotlin examples use).

### Sampling settings

Use **temperature 0.5, top-p 0.9, top-k 40**. We tested greedy in eval and it produces flatter, more repetitive output than the model is capable of. These slightly-warmer settings will let Mrs. Henderson's voice come through more naturally on camera.

These should be hardcoded constants in `ModelInference.kt`. No settings screen.

---

## Model handling specifics

### Storage location
Save the model to app-private internal storage:

```kotlin
val modelFile = File(context.filesDir, "model.litertlm")
```

This is the right choice because:
- App-private storage doesn't require runtime permissions
- It's automatically deleted when the user uninstalls
- It's not visible in the gallery or other apps
- 4.8 GB is fine; modern Android allows app-private storage of any size up to the device's limit

### Download

OkHttp + a simple progress callback. The HF URL will look like:

```
https://huggingface.co/joemuller/gemmacademy-fractions-v1/resolve/main/gemmacademy-fractions-v1-wi8.litertlm
```

Stream the response body to disk in chunks and call back to the UI with progress every ~1% (don't call back on every chunk — Compose recomposition will kill you). Verify the download didn't truncate by checking final file size against `Content-Length`.

### LiteRT-LM Kotlin integration

Reference the official LiteRT-LM Android getting-started guide: https://ai.google.dev/edge/litert-lm/get-started/android

The Gradle dependency is something like (verify the exact coordinates in the docs since they may have changed):

```kotlin
implementation("com.google.ai.edge.litertlm:litertlm:0.11.0+")
```

The basic flow inside `ModelInference.kt`:

1. On first use after download, call `Engine.load(modelFile)` — this can take several seconds. Show a loading indicator on first ChatScreen entry.
2. For each user message, build a conversation with the system prompt + chat history + new message
3. Call `engine.generate(...)` with a streaming callback
4. Callback fires per token (or per chunk); update the in-progress message state
5. When done, append final message to history

Check Google's AI Edge Gallery app on GitHub for a known-working reference implementation: https://github.com/google-ai-edge/gallery — it uses LiteRT-LM Kotlin and is what their docs effectively point at as a sample.

### Inference threading

LiteRT-LM inference is CPU-bound (or GPU-bound on devices that support GPU delegation). It must run off the main thread. Use coroutines:

```kotlin
viewModelScope.launch(Dispatchers.Default) {
    engine.generate(...)
}
```

Or just `lifecycleScope` from the activity if you're not using ViewModels.

### Memory note
Gemma 4 E2B's `.litertlm` weights are 4.8 GB on disk but the runtime needs ~2-3 GB of RAM during inference (KV cache, activations). On a 4 GB RAM phone this is tight but workable. If it OOMs, the runtime will throw — handle the exception gracefully with an error message.

---

## Visual design

Calm, warm, school-appropriate. Not a "tech app." Not a chatbot product.

- **Color:** Use Material 3 with a custom primary that matches the dashboard's primary color (whichever the dashboard agent picked — check `FRONTEND_INSTRUCTIONS.md`). Default to a deep teal if you can't tell.
- **Typography:** Material 3 default. Slightly larger body size than usual (16sp) — kids appreciate this.
- **Spacing:** Generous padding around message bubbles. No cramped UIs.
- **Motion:** Subtle. Tokens streaming in is plenty of motion; don't add bouncing dots, slide-in animations on new messages, etc.
- **Iconography:** Material Icons. Sparse — paper-airplane for send, three-dot menu, that's it.
- **Dark mode:** out of scope. Lock to light mode for the demo.

**One specific design call:** the demo footage is going to be filmed of someone actually using this app. Make sure the screen recording / over-the-shoulder camera angle reads cleanly. Big touch targets, high-contrast text, no fiddly small UI. A 9-year-old's hands need to be able to use this confidently on camera.

---

## Implementation order

1. **Set up the project.** Create the Android Studio project. Add the LiteRT-LM dependency. Get the app to launch with an empty ChatScreen.
2. **Get the model running locally first.** Skip the download flow entirely. Manually `adb push` a `.litertlm` file to the app's data directory and hardcode the path. Get inference working end-to-end before touching the download UI. This is the riskiest part — derisk it before you build the easy stuff around it.
3. **Build the ChatScreen with streaming.** Mock the inference at first (a coroutine that emits "Hello world" one character at a time), then wire to real inference.
4. **Build the OnboardingScreen.** URL-paste path first; QR scanner last (or skip for the demo if time is tight — paste-from-clipboard works fine for the video).
5. **Polish.** Loading states, error states, empty state for the chat. Walk through the full flow on a real device 5 times.

**Time budget:** if you can't get steps 1-3 working in two days, deprioritize the QR scanner entirely and just use URL paste for the demo. The QR scanner is nice but not load-bearing.

---

## Things you will be tempted to do — don't

- **Don't add multiple model support.** One model at a time. If a new model needs to download, delete the old one first.
- **Don't build a "history" screen.** All chats are ephemeral; closing the app loses the conversation. This is fine for a homework-helper. Future work.
- **Don't add markdown rendering for the responses.** The model outputs plain text; render it as plain text. Don't parse `**bold**` or LaTeX. Plain text is correct for an elementary school context anyway.
- **Don't add voice input or output.** Out of scope.
- **Don't try to optimize the .litertlm load time.** It's 5-10 seconds; show a "warming up..." indicator. Optimization is research; loading once at app start is fine.
- **Don't add a "regenerate response" button.** Keep the UI minimal.
- **Don't add login or account features.** No.
- **Don't try to implement RAG or any retrieval over the lesson materials.** The lesson is *baked into the weights*. That's the whole point. If you're calling out to retrieve anything, you've misunderstood the architecture.
- **Don't bundle the model in the APK.** It's 4.8 GB; the Play Store APK limit is 200 MB and we wouldn't be on Play Store anyway. Always download at first launch.

---

## When you're done

- App launches and either shows onboarding (no model) or chat (model present)
- URL-paste flow downloads a model from the HF URL successfully
- Chat sends a message, streams back a response, response references Mrs. Henderson's content (this is the demo moment — verify it works before claiming done)
- Airplane mode test: enable airplane mode, send a message, response still streams back. **This is the central proof of the project. Test it on real hardware.**
- No crashes during a 5-minute conversation
- Works on at least one cheap-Android-class device (the demo target)

When all of that's true, write a `STUDENT_APP_DONE.md` at the student-app root summarizing:
- Any deviations from the spec and why
- Inference performance you observed (tokens/sec, time to first token, model load time)
- Any LiteRT-LM gotchas you hit (similar in spirit to NOTES.md for the training pipeline)
- Suggested device for the demo footage

---

## Out of scope (for clarity)

- iOS (no LiteRT-LM Swift bindings yet; v2)
- Multi-class support (download multiple models, switch between them)
- A "students" account system  
- Conversation history persistence
- Real-time collaboration features
- Parent/teacher dashboards within the student app
- Voice input/output
- Image input (Gemma 4 is multimodal but we don't need it for fractions)
- Any analytics or crash reporting
- Background download (must be foreground for v1; demo doesn't care)
- Multiple language support