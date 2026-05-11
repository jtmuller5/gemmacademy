package com.gemmacademy.student.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

data class StoredMessage(
    val fromUser: Boolean,
    val text: String,
)

data class StoredChat(
    val id: String,
    val modelId: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val messages: List<StoredMessage>,
)

data class ChatSummary(
    val id: String,
    val modelId: String,
    val title: String,
    val updatedAt: Long,
    val messageCount: Int,
)

/**
 * Simple file-backed chat persistence. Each chat is one JSON file under
 * `filesDir/chats/<id>.json`. Chats are tagged with the modelId they were
 * created against so the UI can disallow resuming a chat after the user
 * has swapped to a different teacher's model.
 */
object ChatStore {

    private const val DIR_NAME = "chats"

    private fun dir(context: Context): File =
        File(context.filesDir, DIR_NAME).apply { if (!exists()) mkdirs() }

    private fun fileFor(context: Context, id: String): File =
        File(dir(context), "$id.json")

    fun newId(): String = UUID.randomUUID().toString()

    suspend fun list(context: Context): List<ChatSummary> = withContext(Dispatchers.IO) {
        val files = dir(context).listFiles { f -> f.isFile && f.name.endsWith(".json") }
            ?: return@withContext emptyList()
        files.mapNotNull { f ->
            runCatching {
                val json = JSONObject(f.readText())
                ChatSummary(
                    id = json.getString("id"),
                    modelId = json.optString("modelId", ""),
                    title = json.optString("title", "Untitled chat"),
                    updatedAt = json.optLong("updatedAt", 0L),
                    messageCount = json.optJSONArray("messages")?.length() ?: 0,
                )
            }.getOrNull()
        }.sortedByDescending { it.updatedAt }
    }

    suspend fun load(context: Context, id: String): StoredChat? = withContext(Dispatchers.IO) {
        val f = fileFor(context, id)
        if (!f.exists()) return@withContext null
        runCatching { fromJson(JSONObject(f.readText())) }.getOrNull()
    }

    suspend fun save(context: Context, chat: StoredChat) = withContext(Dispatchers.IO) {
        fileFor(context, chat.id).writeText(toJson(chat).toString())
    }

    suspend fun delete(context: Context, id: String) = withContext(Dispatchers.IO) {
        fileFor(context, id).delete()
    }

    private fun toJson(chat: StoredChat): JSONObject {
        val arr = JSONArray()
        for (m in chat.messages) {
            arr.put(
                JSONObject()
                    .put("fromUser", m.fromUser)
                    .put("text", m.text)
            )
        }
        return JSONObject()
            .put("id", chat.id)
            .put("modelId", chat.modelId)
            .put("title", chat.title)
            .put("createdAt", chat.createdAt)
            .put("updatedAt", chat.updatedAt)
            .put("messages", arr)
    }

    private fun fromJson(json: JSONObject): StoredChat {
        val arr = json.optJSONArray("messages") ?: JSONArray()
        val msgs = ArrayList<StoredMessage>(arr.length())
        for (i in 0 until arr.length()) {
            val m = arr.getJSONObject(i)
            msgs.add(
                StoredMessage(
                    fromUser = m.optBoolean("fromUser", false),
                    text = m.optString("text", ""),
                )
            )
        }
        return StoredChat(
            id = json.getString("id"),
            modelId = json.optString("modelId", ""),
            title = json.optString("title", "Untitled chat"),
            createdAt = json.optLong("createdAt", 0L),
            updatedAt = json.optLong("updatedAt", 0L),
            messages = msgs,
        )
    }
}
