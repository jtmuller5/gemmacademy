package com.gemmacademy.student.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.util.concurrent.TimeUnit

data class DownloadProgress(
    val bytesRead: Long,
    val contentLength: Long,
) {
    val fraction: Float
        get() = if (contentLength > 0) (bytesRead.toFloat() / contentLength.toFloat()).coerceIn(0f, 1f) else 0f
}

class ModelDownloader(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .callTimeout(0, TimeUnit.SECONDS)
        .build()

    /**
     * Downloads the model to internal storage. Calls [onProgress] no more
     * frequently than every 1% of total or 250ms, whichever comes first.
     * Throws [IOException] on failure.
     */
    suspend fun download(
        url: String,
        onProgress: (DownloadProgress) -> Unit,
        isCancelled: () -> Boolean = { false },
    ): Unit = withContext(Dispatchers.IO) {
        val partial = ModelStorage.partialFile(context)
        val target = ModelStorage.modelFile(context)
        partial.delete()

        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("HTTP ${response.code} ${response.message}")
        }
        val body = response.body ?: throw IOException("Empty response body")
        val contentLength = body.contentLength()

        body.byteStream().use { input ->
            partial.outputStream().use { output ->
                val buffer = ByteArray(64 * 1024)
                var totalRead = 0L
                var lastReportFraction = -1f
                var lastReportMs = 0L

                while (true) {
                    if (isCancelled()) {
                        partial.delete()
                        throw IOException("Cancelled")
                    }
                    val n = input.read(buffer)
                    if (n == -1) break
                    output.write(buffer, 0, n)
                    totalRead += n

                    val now = System.currentTimeMillis()
                    val frac = if (contentLength > 0) totalRead.toFloat() / contentLength else 0f
                    if (frac - lastReportFraction >= 0.01f || now - lastReportMs >= 250) {
                        lastReportFraction = frac
                        lastReportMs = now
                        onProgress(DownloadProgress(totalRead, contentLength))
                    }
                }
                output.flush()

                if (contentLength > 0 && totalRead != contentLength) {
                    partial.delete()
                    throw IOException("Truncated download: got $totalRead / $contentLength bytes")
                }
                onProgress(DownloadProgress(totalRead, if (contentLength > 0) contentLength else totalRead))
            }
        }

        if (target.exists()) target.delete()
        if (!partial.renameTo(target)) {
            throw IOException("Failed to move downloaded file into place")
        }
        ModelStorage.writeModelId(context, url)
    }
}
