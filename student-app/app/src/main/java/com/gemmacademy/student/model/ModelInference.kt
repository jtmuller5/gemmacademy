package com.gemmacademy.student.model

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * Thin wrapper around LiteRT-LM that exposes initialize() and a streaming generate flow.
 * Reference: https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md
 */
class ModelInference(private val context: Context) : Closeable {

    companion object {
        private const val TAG = "ModelInference"

        const val SYSTEM_PROMPT: String = """You are a friendly tutor helping a 4th grade student with fractions. The student
just learned about fractions in Mrs. Henderson's class today. Use the specific
methods and examples Mrs. Henderson teaches: the Pizza Method (drawing pizzas
with equal slices), the rule "equal slices, equal fractions," and procedures
like "the cut goes on the bottom, the count goes on the top."

Be patient. Use simple language a 9-year-old understands. When a student is
confused, help them work through it step by step. If they ask about something
outside today's lesson on fractions, gently redirect them: tell them you're
focused on fractions today and ask if you can help with that instead.

Keep answers short — 2 to 4 sentences usually. End with a question or
encouragement when it feels natural."""

        private const val TEMPERATURE = 0.5
        private const val TOP_P = 0.9
        private const val TOP_K = 40
    }

    @Volatile
    private var engine: Engine? = null

    val isLoaded: Boolean get() = engine != null

    /** Load the model. Call from a background dispatcher; takes several seconds. */
    suspend fun load() = withContext(Dispatchers.Default) {
        if (engine != null) return@withContext
        val modelFile = ModelStorage.modelFile(context)
        check(modelFile.exists()) { "Model file missing at ${modelFile.absolutePath}" }

        Log.i(TAG, "Loading model from ${modelFile.absolutePath} (${modelFile.length()} bytes)")
        val config = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.CPU(),
            cacheDir = context.cacheDir.absolutePath,
        )
        val newEngine = Engine(config).also { it.initialize() }
        engine = newEngine
        Log.i(TAG, "Model loaded.")
    }

    /**
     * Generate a streaming response for [userMessage] given prior chat [history].
     * Each emitted string is an incremental chunk from the model.
     *
     * A new [Conversation] is created per call, seeded with the system prompt and
     * any prior turns via [ConversationConfig.initialMessages]. This is simpler than
     * holding a long-lived conversation and matches the spec's "no persistence" UX.
     */
    fun generateStream(history: List<ChatTurn>, userMessage: String): Flow<String> = callbackFlow {
        val activeEngine = engine ?: error("Model not loaded")

        val initial = history.map { turn ->
            if (turn.fromUser) Message.user(turn.text) else Message.model(turn.text)
        }

        val conversationConfig = ConversationConfig(
            systemInstruction = Contents.of(SYSTEM_PROMPT),
            initialMessages = initial,
            samplerConfig = SamplerConfig(
                topK = TOP_K,
                topP = TOP_P,
                temperature = TEMPERATURE,
            ),
        )

        val conversation: Conversation = activeEngine.createConversation(conversationConfig)

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
                // The official Kotlin getting-started example simply does `print(message)`,
                // implying Message.toString() yields the streaming text chunk. If you discover
                // that's not the case for the version you pull, switch to extracting text from
                // message.contents (Content.Text holds the chunk).
                val text = message.toString()
                if (text.isNotEmpty()) trySend(text)
            }

            override fun onDone() {
                close()
            }

            override fun onError(throwable: Throwable) {
                close(throwable)
            }
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

    override fun close() {
        try { engine?.close() } catch (_: Throwable) {}
        engine = null
    }
}

data class ChatTurn(
    val fromUser: Boolean,
    val text: String,
)
