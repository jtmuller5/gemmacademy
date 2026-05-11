package com.gemmacademy.student.model

import android.content.Context
import java.io.File

object ModelStorage {
    private const val FILE_NAME = "model.litertlm"
    private const val PARTIAL_SUFFIX = ".part"
    private const val ID_FILE_NAME = "model.id"

    fun modelFile(context: Context): File = File(context.filesDir, FILE_NAME)

    fun partialFile(context: Context): File = File(context.filesDir, FILE_NAME + PARTIAL_SUFFIX)

    private fun idFile(context: Context): File = File(context.filesDir, ID_FILE_NAME)

    fun isModelPresent(context: Context): Boolean {
        val f = modelFile(context)
        return f.exists() && f.length() > 0
    }

    /**
     * Stable identifier for the currently-installed model. We use the source URL
     * the model was downloaded from. Returns an empty string if unknown (e.g. a
     * model installed before this field existed).
     */
    fun currentModelId(context: Context): String {
        val f = idFile(context)
        return if (f.exists()) f.readText().trim() else ""
    }

    fun writeModelId(context: Context, id: String) {
        idFile(context).writeText(id)
    }

    fun deleteModel(context: Context) {
        modelFile(context).delete()
        partialFile(context).delete()
        idFile(context).delete()
    }
}
