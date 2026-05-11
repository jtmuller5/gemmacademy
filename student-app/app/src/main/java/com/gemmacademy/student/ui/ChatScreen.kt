package com.gemmacademy.student.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.gemmacademy.student.App
import com.gemmacademy.student.model.ChatStore
import com.gemmacademy.student.model.ChatSummary
import com.gemmacademy.student.model.ChatTurn
import com.gemmacademy.student.model.ModelInference
import com.gemmacademy.student.model.ModelStorage
import com.gemmacademy.student.model.StoredChat
import com.gemmacademy.student.model.StoredMessage
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.util.Date

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen() {
    val context = LocalContext.current
    val app = context.applicationContext as App
    val inference: ModelInference = remember { app.modelInference }
    val scope = rememberCoroutineScope()

    val currentModelId = remember { ModelStorage.currentModelId(context) }

    var loading by remember { mutableStateOf(!inference.isLoaded) }
    var loadError by remember { mutableStateOf<String?>(null) }
    val messages = remember { mutableStateListOf<UiMessage>() }
    var generating by remember { mutableStateOf(false) }
    var input by rememberSaveable { mutableStateOf("") }
    var showAbout by remember { mutableStateOf(false) }
    var chatId by rememberSaveable { mutableStateOf<String?>(null) }
    var chatCreatedAt by rememberSaveable { mutableStateOf(0L) }
    var chatTitle by rememberSaveable { mutableStateOf("") }
    val listState = rememberLazyListState()

    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val chatSummaries = remember { mutableStateListOf<ChatSummary>() }

    suspend fun refreshChats() {
        val list = ChatStore.list(context)
        chatSummaries.clear()
        chatSummaries.addAll(list)
    }

    suspend fun persistCurrentChat() {
        val id = chatId ?: return
        val now = System.currentTimeMillis()
        val storedMsgs = messages.map { StoredMessage(it.fromUser, it.text) }
        val title = chatTitle.ifBlank {
            messages.firstOrNull { it.fromUser }?.text?.take(60).orEmpty().ifBlank { "New chat" }
        }
        chatTitle = title
        ChatStore.save(
            context,
            StoredChat(
                id = id,
                modelId = currentModelId,
                title = title,
                createdAt = if (chatCreatedAt == 0L) now else chatCreatedAt,
                updatedAt = now,
                messages = storedMsgs,
            )
        )
        refreshChats()
    }

    fun startNewChat() {
        chatId = null
        chatCreatedAt = 0L
        chatTitle = ""
        messages.clear()
        input = ""
    }

    fun openChat(summary: ChatSummary) {
        scope.launch {
            val chat = ChatStore.load(context, summary.id) ?: return@launch
            chatId = chat.id
            chatCreatedAt = chat.createdAt
            chatTitle = chat.title
            messages.clear()
            chat.messages.forEachIndexed { i, m ->
                messages.add(UiMessage(id = i.toLong(), fromUser = m.fromUser, text = m.text))
            }
            drawerState.close()
        }
    }

    LaunchedEffect(Unit) {
        if (!inference.isLoaded) {
            try {
                inference.load()
                loading = false
            } catch (t: Throwable) {
                loadError = t.message ?: "Failed to load model"
                loading = false
            }
        } else {
            loading = false
        }
        refreshChats()
    }

    LaunchedEffect(messages.size, messages.lastOrNull()?.text?.length) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.lastIndex)
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ChatHistoryDrawer(
                summaries = chatSummaries,
                currentModelId = currentModelId,
                activeChatId = chatId,
                onNewChat = {
                    startNewChat()
                    scope.launch { drawerState.close() }
                },
                onSelect = { openChat(it) },
                onDelete = { s ->
                    scope.launch {
                        ChatStore.delete(context, s.id)
                        if (s.id == chatId) startNewChat()
                        refreshChats()
                    }
                },
            )
        },
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Text(
                            chatTitle.ifBlank { "Mrs. Henderson · Fractions" },
                            style = MaterialTheme.typography.titleMedium,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Filled.History, contentDescription = "Chat history")
                        }
                    },
                    actions = {
                        IconButton(
                            onClick = { startNewChat() },
                            enabled = !generating,
                        ) {
                            Icon(Icons.Filled.Add, contentDescription = "New chat")
                        }
                        var menuOpen by remember { mutableStateOf(false) }
                        IconButton(onClick = { menuOpen = true }) {
                            Icon(Icons.Filled.MoreVert, contentDescription = "Menu")
                        }
                        DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                            DropdownMenuItem(
                                text = { Text("About") },
                                onClick = {
                                    menuOpen = false
                                    showAbout = true
                                },
                            )
                        }
                    },
                )
            },
            bottomBar = {
                InputBar(
                    value = input,
                    onValueChange = { input = it },
                    enabled = !loading && loadError == null && !generating,
                    onSend = {
                        val text = input.trim()
                        if (text.isEmpty()) return@InputBar
                        if (chatId == null) {
                            chatId = ChatStore.newId()
                            chatCreatedAt = System.currentTimeMillis()
                            chatTitle = text.take(60)
                        }
                        val userMsg = UiMessage(
                            id = System.nanoTime(),
                            fromUser = true,
                            text = text,
                        )
                        val tutorMsg = UiMessage(
                            id = System.nanoTime() + 1,
                            fromUser = false,
                            text = "",
                            streaming = true,
                        )
                        messages.add(userMsg)
                        messages.add(tutorMsg)
                        input = ""
                        generating = true

                        val history = messages
                            .dropLast(2)
                            .map { ChatTurn(fromUser = it.fromUser, text = it.text) }

                        scope.launch {
                            persistCurrentChat()
                            try {
                                val sb = StringBuilder()
                                inference.generateStream(history, text).collect { tok ->
                                    sb.append(tok)
                                    val idx = messages.indexOfLast { it.id == tutorMsg.id }
                                    if (idx >= 0) {
                                        messages[idx] = messages[idx].copy(text = sb.toString())
                                    }
                                }
                                val idx = messages.indexOfLast { it.id == tutorMsg.id }
                                if (idx >= 0) {
                                    messages[idx] = messages[idx].copy(streaming = false)
                                }
                            } catch (t: Throwable) {
                                val idx = messages.indexOfLast { it.id == tutorMsg.id }
                                if (idx >= 0) {
                                    messages[idx] = messages[idx].copy(
                                        text = "Sorry — something went wrong. (${t.message ?: "unknown"})",
                                        streaming = false,
                                    )
                                }
                            } finally {
                                generating = false
                                persistCurrentChat()
                            }
                        }
                    },
                )
            }
        ) { padding ->
            Box(modifier = Modifier.padding(padding).fillMaxSize()) {
                when {
                    loading -> LoadingView()
                    loadError != null -> ErrorBanner(loadError!!)
                    messages.isEmpty() -> EmptyChatHint()
                    else -> LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(vertical = 12.dp),
                    ) {
                        items(items = messages, key = { it.id }) { msg ->
                            ChatMessageBubble(msg)
                        }
                    }
                }
            }
        }
    }

    if (showAbout) {
        AlertDialog(
            onDismissRequest = { showAbout = false },
            confirmButton = {
                TextButton(onClick = { showAbout = false }) { Text("Close") }
            },
            title = { Text("About Gemmacademy") },
            text = {
                Text(
                    "A homework helper running fully on this device. " +
                        "Your teacher's lessons are baked into the model — no internet needed."
                )
            },
        )
    }
}

@Composable
private fun ChatHistoryDrawer(
    summaries: List<ChatSummary>,
    currentModelId: String,
    activeChatId: String?,
    onNewChat: () -> Unit,
    onSelect: (ChatSummary) -> Unit,
    onDelete: (ChatSummary) -> Unit,
) {
    ModalDrawerSheet(modifier = Modifier.fillMaxHeight()) {
        Column(modifier = Modifier.fillMaxHeight()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 18.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Chats",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.weight(1f),
                )
            }
            FilledTonalButton(
                onClick = onNewChat,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
            ) {
                Icon(Icons.Filled.Add, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("New chat")
            }
            Spacer(Modifier.height(12.dp))
            HorizontalDivider()

            val current = summaries.filter { it.modelId == currentModelId }
            val other = summaries.filter { it.modelId != currentModelId }

            LazyColumn(
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(vertical = 8.dp),
            ) {
                if (current.isEmpty() && other.isEmpty()) {
                    item {
                        Text(
                            "No chats yet. Ask your first question to start one.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(20.dp),
                        )
                    }
                }
                if (current.isNotEmpty()) {
                    item { SectionHeader("Today's lessons") }
                    items(current, key = { it.id }) { s ->
                        ChatHistoryRow(
                            summary = s,
                            available = true,
                            active = s.id == activeChatId,
                            onClick = { onSelect(s) },
                            onDelete = { onDelete(s) },
                        )
                    }
                }
                if (other.isNotEmpty()) {
                    item { SectionHeader("Other lessons (not loaded)") }
                    items(other, key = { it.id }) { s ->
                        ChatHistoryRow(
                            summary = s,
                            available = false,
                            active = false,
                            onClick = { /* disabled */ },
                            onDelete = { onDelete(s) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SectionHeader(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp),
    )
}

@Composable
private fun ChatHistoryRow(
    summary: ChatSummary,
    available: Boolean,
    active: Boolean,
    onClick: () -> Unit,
    onDelete: () -> Unit,
) {
    val bg = when {
        active -> MaterialTheme.colorScheme.secondaryContainer
        else -> MaterialTheme.colorScheme.surface
    }
    val rowAlpha = if (available) 1f else 0.5f
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 2.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(bg)
            .let { if (available) it.clickable(onClick = onClick) else it }
            .alpha(rowAlpha)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = summary.title.ifBlank { "Untitled chat" },
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            val subtitle = buildString {
                append(relativeTime(summary.updatedAt))
                append(" · ")
                append(summary.messageCount)
                append(" msg")
                if (!available) append(" · lessons not loaded")
            }
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        IconButton(onClick = onDelete) {
            Icon(
                Icons.Filled.Delete,
                contentDescription = "Delete chat",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private fun relativeTime(epochMs: Long): String {
    if (epochMs <= 0L) return ""
    val now = System.currentTimeMillis()
    val diff = now - epochMs
    val min = diff / 60_000L
    val hr = diff / 3_600_000L
    val day = diff / 86_400_000L
    return when {
        min < 1 -> "just now"
        min < 60 -> "${min}m ago"
        hr < 24 -> "${hr}h ago"
        day < 7 -> "${day}d ago"
        else -> DateFormat.getDateInstance(DateFormat.SHORT).format(Date(epochMs))
    }
}

@Composable
private fun LoadingView() {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(20.dp))
        Text("Warming up your tutor…", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(6.dp))
        Text(
            "This takes a few seconds the first time you open the app.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun ErrorBanner(message: String) {
    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Couldn't start the tutor", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(8.dp))
        Text(
            message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun EmptyChatHint() {
    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            "Ask me anything about today's lesson!",
            style = MaterialTheme.typography.titleLarge,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(12.dp))
        Text(
            "Try: How do I show 3/8 with the Pizza Method?",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun InputBar(
    value: String,
    onValueChange: (String) -> Unit,
    enabled: Boolean,
    onSend: () -> Unit,
) {
    Surface(
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
                .imePadding(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 52.dp),
                placeholder = { Text("Ask a question…") },
                shape = RoundedCornerShape(24.dp),
                enabled = enabled,
                maxLines = 4,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            )
            Spacer(Modifier.width(8.dp))
            FilledIconButton(
                onClick = onSend,
                enabled = enabled && value.isNotBlank(),
                modifier = Modifier
                    .size(52.dp)
                    .clip(CircleShape),
            ) {
                Icon(
                    Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                )
            }
        }
    }
}
