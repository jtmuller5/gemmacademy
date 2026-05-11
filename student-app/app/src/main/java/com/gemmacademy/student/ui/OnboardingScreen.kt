package com.gemmacademy.student.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gemmacademy.student.model.DownloadProgress
import com.gemmacademy.student.model.ModelDownloader
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

private sealed interface OnboardingPhase {
    data object Welcome : OnboardingPhase
    data object Paste : OnboardingPhase
    data object Scan : OnboardingPhase
    data class Downloading(val url: String, val progress: DownloadProgress?) : OnboardingPhase
    data object Done : OnboardingPhase
    data class Error(val message: String, val url: String?) : OnboardingPhase
}

@Composable
fun OnboardingScreen(onModelReady: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var phase by rememberSaveable(stateSaver = OnboardingPhaseSaver) {
        mutableStateOf<OnboardingPhase>(OnboardingPhase.Welcome)
    }
    val downloader = remember { ModelDownloader(context) }
    var cancelFlag by remember { mutableStateOf(false) }

    fun startDownload(url: String) {
        cancelFlag = false
        phase = OnboardingPhase.Downloading(url, null)
        scope.launch {
            try {
                downloader.download(
                    url = url,
                    onProgress = { p ->
                        phase = OnboardingPhase.Downloading(url, p)
                    },
                    isCancelled = { cancelFlag },
                )
                phase = OnboardingPhase.Done
                onModelReady()
            } catch (e: CancellationException) {
                phase = OnboardingPhase.Welcome
                throw e
            } catch (t: Throwable) {
                phase = OnboardingPhase.Error(t.message ?: "Download failed", url)
            }
        }
    }

    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        when (val p = phase) {
            is OnboardingPhase.Welcome -> WelcomeView(
                onScan = { phase = OnboardingPhase.Scan },
                onPaste = { phase = OnboardingPhase.Paste },
            )

            is OnboardingPhase.Paste -> PasteUrlDialog(
                onCancel = { phase = OnboardingPhase.Welcome },
                onDownload = { url -> startDownload(url) },
            )

            is OnboardingPhase.Scan -> QrScannerScreen(
                onUrlDetected = { url -> startDownload(url) },
                onCancel = { phase = OnboardingPhase.Welcome },
            )

            is OnboardingPhase.Downloading -> DownloadingView(
                progress = p.progress,
                onCancel = { cancelFlag = true },
            )

            is OnboardingPhase.Done -> DoneView()

            is OnboardingPhase.Error -> ErrorView(
                message = p.message,
                onRetry = { p.url?.let { startDownload(it) } ?: run { phase = OnboardingPhase.Welcome } },
                onBack = { phase = OnboardingPhase.Welcome },
            )
        }
    }
}

@Composable
private fun WelcomeView(onScan: () -> Unit, onPaste: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 28.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Box(
            modifier = Modifier
                .size(120.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primaryContainer),
            contentAlignment = Alignment.Center,
        ) {
            Text(text = "🍕", fontSize = 64.sp)
        }
        Spacer(Modifier.height(28.dp))
        Text(
            "Get your teacher's lessons",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(14.dp))
        Text(
            "Your teacher will give you a QR code. Scan it to download today's lessons. " +
                "You only need to do this once — after that, the app works without internet.",
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(40.dp))
        Button(
            onClick = onScan,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp),
        ) {
            Text("Scan QR code", fontSize = 18.sp)
        }
        Spacer(Modifier.height(12.dp))
        TextButton(onClick = onPaste) {
            Text("Or paste a URL")
        }
    }
}

@Composable
private fun PasteUrlDialog(onCancel: () -> Unit, onDownload: (String) -> Unit) {
    var url by rememberSaveable { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text("Paste model URL") },
        text = {
            OutlinedTextField(
                value = url,
                onValueChange = { url = it },
                placeholder = { Text("https://huggingface.co/...") },
                singleLine = false,
                modifier = Modifier.fillMaxWidth(),
            )
        },
        confirmButton = {
            TextButton(
                onClick = { if (url.isNotBlank()) onDownload(url.trim()) },
                enabled = url.isNotBlank(),
            ) { Text("Download") }
        },
        dismissButton = {
            TextButton(onClick = onCancel) { Text("Cancel") }
        }
    )
}

@Composable
private fun DownloadingView(progress: DownloadProgress?, onCancel: () -> Unit) {
    val frac = progress?.fraction ?: 0f
    val pct = (frac * 100).toInt()
    val mbRead = progress?.bytesRead?.let { it / (1024.0 * 1024.0) } ?: 0.0
    val mbTotal = progress?.contentLength?.takeIf { it > 0 }?.let { it / (1024.0 * 1024.0) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            "Downloading your lessons…",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(28.dp))
        LinearProgressIndicator(
            progress = { frac.coerceIn(0f, 1f) },
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp)
                .clip(RoundedCornerShape(6.dp)),
        )
        Spacer(Modifier.height(16.dp))
        Text("$pct%", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))
        if (mbTotal != null) {
            Text(
                "%.0f MB of %.0f MB · downloading over WiFi".format(mbRead, mbTotal),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            Text("4.8 GB · downloading over WiFi", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(20.dp))
        Text(
            "Please keep the app open. This takes a few minutes on school WiFi.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(36.dp))
        OutlinedButton(onClick = onCancel) { Text("Cancel") }
    }
}

@Composable
private fun DoneView() {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("✓", fontSize = 72.sp, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(12.dp))
        Text("Lessons ready!", style = MaterialTheme.typography.headlineSmall)
    }
}

@Composable
private fun ErrorView(message: String, onRetry: () -> Unit, onBack: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Couldn't download", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        Text(
            "Check your WiFi and try again.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            message,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onRetry) { Text("Try again") }
        Spacer(Modifier.height(8.dp))
        TextButton(onClick = onBack) { Text("Back") }
    }
}

private val OnboardingPhaseSaver = androidx.compose.runtime.saveable.Saver<OnboardingPhase, String>(
    save = {
        when (it) {
            is OnboardingPhase.Welcome -> "welcome"
            is OnboardingPhase.Paste -> "paste"
            is OnboardingPhase.Scan -> "scan"
            is OnboardingPhase.Downloading -> "welcome" // restart on rotation; download is in-flight
            is OnboardingPhase.Done -> "welcome"
            is OnboardingPhase.Error -> "welcome"
        }
    },
    restore = {
        when (it) {
            "paste" -> OnboardingPhase.Paste
            "scan" -> OnboardingPhase.Scan
            else -> OnboardingPhase.Welcome
        }
    }
)
